/**
 * @file
 * @brief A capture worker/handler for CRS board packets
 */

#ifndef CRS_16BOARD_CAPTURE_WORKER_HPP
#define CRS_16BOARD_CAPTURE_WORKER_HPP

#include "Config.hpp"
#include "FramePrefetchService.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "crsUtils.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"
#include "util.h"
#include "visUtil.hpp"

#include "json.hpp"

#include <algorithm>
#include <immintrin.h>


class crs16BoardCaptureWorker : public dpdkRXhandler {
public:
    /// Default constructor
    crs16BoardCaptureWorker(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& buffer_container, int worker_id,
                            const std::vector<rte_ring*>& worker_rings);

    /// Processes the incoming packets
    int handle_packet(struct rte_mbuf* mbuf) override;

    /// Update stats, not used by this handler yet.
    virtual void update_stats() override {};

protected:
    const std::vector<rte_ring*>& worker_rings;

    /// The output buffer
    Buffer* out_buf;
    /// The packet recepit bitmap buffer
    Buffer* receipt_bitmap_buf;

    int worker_id;

    /// The packet size
    uint32_t packet_size;

    uint32_t payload_size;

    uint32_t num_expected_stream_ids;

    const uint32_t header_size = 64;
    // Each node processes data from 16 source IDs (16 CRS boards) and 8 stream IDs (1/16 of the
    // total streams)
    const uint32_t num_source_ids = 16;
    const uint32_t num_stream_ids = 8;
    const uint32_t time_samples_per_packet = 16;

    std::vector<uint32_t> stream_ids_expected;

    bool first_run = true;

    uint64_t time_samples_per_frame;
    uint64_t packets_per_frame;

    std::unique_ptr<kotekan::FramePrefetchService> prefetch_service;

    const kotekan::FrameInfo* active_f0 = nullptr;
    const kotekan::FrameInfo* active_f1 = nullptr;

    /// Sample-based FPGA seq monotonicity check. At each sample the seq must
    /// be strictly greater than the previous sample's, otherwise we treat it
    /// as an FPGA reset and shut down. Sampled every @c _seq_check_interval
    /// packets, which at typical CRS packet rates is roughly once per second.
    static constexpr uint64_t _seq_check_interval = 500000;
    uint64_t _last_check_seq = 0;
    uint64_t _seq_check_packet_count = 0;

    inline void packet_copy_to_frame(struct rte_mbuf* mbuf, uint8_t* frame_ptr,
                                     uint64_t relative_seq_num, uint16_t stream_id,
                                     uint16_t source_id);
};

inline crs16BoardCaptureWorker::crs16BoardCaptureWorker(
    kotekan::Config& config, const std::string& unique_name,
    kotekan::bufferContainer& buffer_container, int worker_id,
    const std::vector<rte_ring*>& worker_rings) :
    dpdkRXhandler(config, unique_name, buffer_container, config.get<int>(unique_name, "port")),
    worker_rings(worker_rings),
    out_buf(buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buf"))),
    receipt_bitmap_buf(
        buffer_container.get_buffer(config.get<std::string>(unique_name, "receipt_bitmap_buf"))),
    worker_id(worker_id), packet_size(config.get<uint32_t>(unique_name, "packet_size")),
    payload_size(config.get<uint32_t>(unique_name, "payload_size")),
    num_expected_stream_ids(config.get<uint32_t>(unique_name, "num_expected_stream_ids")) {

    out_buf->register_producer(unique_name);
    receipt_bitmap_buf->register_producer(unique_name);

    if ((out_buf->frame_size % (payload_size * num_source_ids * num_stream_ids)) != 0) {
        throw std::runtime_error("The buffer frame size must be a multiple of the combined payload "
                                 "size of all source and stream IDs for a given time sample.");
    }

    if ((payload_size % 32) != 0) {
        throw std::runtime_error("The packet_size must be a multiple of 32 bytes");
    }

    if (payload_size + header_size != packet_size) {
        throw std::runtime_error("The packet_size must be payload_size + header_size bytes");
    }

    packets_per_frame = out_buf->frame_size / payload_size;
    time_samples_per_frame =
        packets_per_frame * time_samples_per_packet / (num_source_ids * num_stream_ids);
    INFO("crs16BoardCaptureWorker {}: time_samples_per_frame = {}, packets_per_frame = {}",
         unique_name, time_samples_per_frame, packets_per_frame);

    // Number of frames to capture before stopping, 0 = unlimited
    uint64_t capture_n_frames = config.get_default<uint64_t>(unique_name, "capture_n_frames", 0);

    int prefetch_depth = config.get_default<int>(unique_name, "prefetch_depth", 4);
    std::vector<int> cpu_affinity =
        config.get<std::vector<int>>(unique_name, "frame_service_cpu_affinity");
    bool allocate_metadata = config.get_default<bool>(unique_name, "allocate_metadata", false);

    prefetch_service = std::make_unique<kotekan::FramePrefetchService>(
        out_buf, receipt_bitmap_buf, unique_name, port, time_samples_per_frame, prefetch_depth,
        cpu_affinity, capture_n_frames, allocate_metadata);
    prefetch_service->set_log_level(get_log_level());
    prefetch_service->set_log_prefix(unique_name + "_prefetch");
}


inline int crs16BoardCaptureWorker::handle_packet(struct rte_mbuf* mbuf) {

    // Check the packet size, checksum, and the packet cookie
    if (unlikely((mbuf->ol_flags & RTE_MBUF_F_RX_IP_CKSUM_MASK) == RTE_MBUF_F_RX_IP_CKSUM_BAD)) {
        WARN("Port: {:d}, Worker: {:d}; Got bad packet IP checksum", port, worker_id);
        return 0;
    }

    if (unlikely(mbuf->pkt_len != packet_size)) {
        // WARN("Port: {:d}; Got packet with invalid size {:d}, expected {:d}", port, mbuf->pkt_len,
        //      packet_size);
        return 0;
    }

    if (unlikely(get_crs_packet_cookie(mbuf) != CRS_PACKET_COOKIE)) {
        WARN("Port: {:d}, Worker: {:d}; Got packet with invalid cookie {:08X}", port, worker_id,
             get_crs_packet_cookie(mbuf));
        return 0;
    }

    // Print the worker ID and stream ID
    uint16_t stream_id = get_crs_packet_stream_id(mbuf);
    uint16_t source_id =
        get_crs_packet_source_id(mbuf).slot_id + 16 * get_crs_packet_source_id(mbuf).crate_id;
    uint64_t seq_num = get_crs_packet_seq_num(mbuf);

    // Sample-based FPGA seq monotonicity check: at each sample boundary the
    // current seq must be strictly greater than the previous sample's,
    // otherwise the FPGA likely reset.
    if (unlikely(++_seq_check_packet_count >= _seq_check_interval)) {
        if (unlikely(_last_check_seq != 0 && seq_num < _last_check_seq)) {
            FATAL_ERROR("Port: {:d}, Worker: {:d}; CRS FPGA seq went backwards ({:d} -> {:d}), "
                        "controller likely reset, kotekan stopping...",
                        port, worker_id, _last_check_seq, seq_num);
            return -1;
        }
        _last_check_seq = seq_num;
        _seq_check_packet_count = 0;
    }

    if (unlikely(first_run)) {

        // Establish the list of expected stream IDs for this port.
        // Note that in most cases we expect all source ID at every port.
        // In the 16 board configuration, each stream ID should be spaced by 16.
        // Note that this is slightly slow, but only happens once at startup.

        // Check if the stream ID is already in the list, and if not add it.
        if (std::find(stream_ids_expected.begin(), stream_ids_expected.end(), stream_id)
            == stream_ids_expected.end()) {
            stream_ids_expected.push_back(stream_id);
        }

        if (stream_ids_expected.size() < num_expected_stream_ids) {
            return 0; // Wait for more packets to establish the full list
        }

        // Start capturint a future frame to allow time for prefetching
        uint64_t future_seq = seq_num + 400000; // About 2 second in the future.
        uint64_t start_seq = future_seq - (future_seq % time_samples_per_frame);
        prefetch_service->start(start_seq, stream_ids_expected);
        first_run = false;
        INFO("Port: {:d}, Worker: {:d}; Starting prefetch service at sequence number {:d}", port,
             worker_id, start_seq);
        return 0;
    }

    // This should only ever happen if the switch or F-engine is misconfigured.
    // Could possibly be removed at some point if we have other checks in place.
    if (std::find(stream_ids_expected.begin(), stream_ids_expected.end(), stream_id)
        == stream_ids_expected.end()) {
        WARN("Port: {:d}, Worker: {:d}; Got packet with unexpected Stream ID {:d}", port, worker_id,
             stream_id);
        return 0;
    }

    // Print packet details for every 100,000 sequence numbers.
#ifdef DEBUGGING
    if ((seq_num / 16) % 100000 == 0) {
        DEBUG("Port: {:d}, Worker: {:d}; Got packet with Stream ID {:d}, Source ID {:d}, Seq Num "
              "{:d}",
              port, worker_id, stream_id, source_id, seq_num);
    }
#endif

    if (unlikely(!prefetch_service->is_ready())) {
        if (prefetch_service->has_error() || prefetch_service->is_complete())
            return -1;

        if (seq_num >= prefetch_service->get_start_seq()) {
            WARN("Port: {:d}, Worker: {:d}; Dropping packet with sequence number {:d} because "
                 "prefetch service is not ready (start_seq: {:d})",
                 port, worker_id, seq_num, prefetch_service->get_start_seq());
        }
        return 0;
    }

    if (unlikely(active_f0 == nullptr)) {
        active_f0 = prefetch_service->get_frame(0);
        active_f1 = prefetch_service->get_frame(1);

        if (unlikely(active_f0 == nullptr || active_f1 == nullptr)) {
            if (prefetch_service->has_error() || prefetch_service->is_complete())
                return -1;
            // TODO add a metric for out of prefetched frame packet drops.
            active_f0 = nullptr;
            active_f1 = nullptr;
            return 0;
        }
    }

    if (seq_num < active_f0->start_seq
        || seq_num >= active_f1->start_seq + time_samples_per_frame) {

        // Don't warn if we are just waiting for the first frame
        if (seq_num < active_f0->start_seq
            && active_f0->start_seq == prefetch_service->get_start_seq()) {
            return 0;
        }

        // Packet is outside the range of the two active frames, drop it
        ERROR("Port: {:d}, Worker: {:d}; Dropping packet with sequence number {:d} outside active "
              "frame range [{:d}, {:d}]",
              port, worker_id, seq_num, active_f0->start_seq,
              active_f1->start_seq + time_samples_per_frame);
        return -1;
    }

    // If we are at least 160 time samples past the start of the next frame,
    // then advance the frames. The 160 time sample margin ensures that we do not
    // miss any packets that are slightly out of order.
    if (seq_num >= active_f1->start_seq + 160) {
        _mm_sfence(); // Ensure all NT stores are complete before advancing
        prefetch_service->advance();
        active_f0 = prefetch_service->get_frame(0);
        active_f1 = prefetch_service->get_frame(1);

        if (unlikely(active_f0 == nullptr || active_f1 == nullptr)) {
            if (prefetch_service->has_error() || prefetch_service->is_complete())
                return -1;
            // TODO add a metric for dropped packets here.
            active_f0 = nullptr;
            active_f1 = nullptr;
            return 0;
        }
    }


    uint8_t* frame_ptr;
    uint64_t relative_seq_num;
    if (seq_num < active_f1->start_seq) {
        // Packet belongs to the current frame
        frame_ptr = active_f0->frame_ptr;
        relative_seq_num = seq_num - active_f0->start_seq;
    } else {
        // Packet belongs to the next frame
        frame_ptr = active_f1->frame_ptr;
        relative_seq_num = seq_num - active_f1->start_seq;
    }

    packet_copy_to_frame(mbuf, frame_ptr, relative_seq_num, stream_id, source_id);

    // Record which packets were received.
    // The layout of the packet receipt bitmap is:
    // packet_receipt[time_long][source_id][stream_id]
    // with the size (in bits) of:
    // time_long = packets_per_frame / (num_source_ids * num_stream_ids)
    // source_id = num_source_ids
    // stream_id = num_stream_ids
    // For the pathfinder this is [512][16][8] = 65536 bits = 8192 bytes
    active_f0->receipt_bitmap_ptr[(relative_seq_num / time_samples_per_packet) * num_source_ids
                                  + source_id] |= (1 << (stream_id / 16));

    return 0;
}


inline void crs16BoardCaptureWorker::packet_copy_to_frame(struct rte_mbuf* mbuf, uint8_t* frame_ptr,
                                                          uint64_t relative_seq_num,
                                                          uint16_t stream_id, uint16_t source_id) {
    const uint64_t time_long = relative_seq_num / 16;
    const uint64_t element_long = source_id;

    // Base offset calculation
    // stride_time_long = (num_stream_ids * 48) * 2048
    // stride_element_long = 128
    uint64_t base_offset = time_long * (uint64_t)(num_stream_ids * 48) * 2048 + element_long * 128;

    uint8_t* dst_base = frame_ptr + base_offset;

    // Source setup
    struct rte_mbuf* m = mbuf;
    uint8_t* src = rte_pktmbuf_mtod_offset(m, uint8_t*, header_size);
    int rem = rte_pktmbuf_data_len(m) - header_size;

    for (int bin = 0; bin < 48; ++bin) {
        uint64_t freq_bin_idx = stream_id / 16 + 8 * bin;
        uint8_t* dst = dst_base + freq_bin_idx * 2048;

        // Copy 128 bytes
        if (likely(rem >= 128)) {
            // Fast path: AVX non-temporal store with latency hiding
            __m256i r0, r1, r2, r3;

            // Check for alignment
            if (((uintptr_t)src & 0x1F) == 0) {
                r0 = _mm256_load_si256((__m256i*)(src + 0));
                r1 = _mm256_load_si256((__m256i*)(src + 32));
                r2 = _mm256_load_si256((__m256i*)(src + 64));
                r3 = _mm256_load_si256((__m256i*)(src + 96));
            } else {
                r0 = _mm256_loadu_si256((__m256i*)(src + 0));
                r1 = _mm256_loadu_si256((__m256i*)(src + 32));
                r2 = _mm256_loadu_si256((__m256i*)(src + 64));
                r3 = _mm256_loadu_si256((__m256i*)(src + 96));
            }

            _mm256_stream_si256((__m256i*)(dst + 0), r0);
            _mm256_stream_si256((__m256i*)(dst + 32), r1);
            _mm256_stream_si256((__m256i*)(dst + 64), r2);
            _mm256_stream_si256((__m256i*)(dst + 96), r3);

            src += 128;
            rem -= 128;
        } else {
            // Slow path: Handle split across mbuf segments
            int copied = 0;
            while (copied < 128) {
                if (rem == 0) {
                    m = m->next;
                    if (unlikely(m == nullptr))
                        return;
                    src = rte_pktmbuf_mtod(m, uint8_t*);
                    rem = rte_pktmbuf_data_len(m);
                }
                int n = std::min(128 - copied, rem);

                uint8_t* d = dst + copied;
                uint8_t* s = src;
                int left = n;

                while (left > 0) {
                    // Try 32-byte aligned streaming store
                    if (left >= 32 && ((uintptr_t)d & 0x1F) == 0) {
                        _mm256_stream_si256((__m256i*)d, _mm256_loadu_si256((__m256i*)s));
                        d += 32;
                        s += 32;
                        left -= 32;
                        continue;
                    }
                    // Try 16-byte aligned streaming store
                    if (left >= 16 && ((uintptr_t)d & 0xF) == 0) {
                        _mm_stream_si128((__m128i*)d, _mm_loadu_si128((__m128i*)s));
                        d += 16;
                        s += 16;
                        left -= 16;
                        continue;
                    }
                    // Try 8-byte aligned streaming store
                    if (left >= 8 && ((uintptr_t)d & 0x7) == 0) {
                        int64_t val;
                        std::memcpy(&val, s, sizeof(val));
                        _mm_stream_si64((long long*)d, (long long)val);
                        d += 8;
                        s += 8;
                        left -= 8;
                        continue;
                    }
                    // Try 4-byte aligned streaming store
                    if (left >= 4 && ((uintptr_t)d & 0x3) == 0) {
                        int32_t val;
                        std::memcpy(&val, s, sizeof(val));
                        _mm_stream_si32((int*)d, val);
                        d += 4;
                        s += 4;
                        left -= 4;
                        continue;
                    }

                    // Fallback: byte copy (standard store)
                    *d++ = *s++;
                    left--;
                }

                src += n;
                rem -= n;
                copied += n;
            }
        }
    }
}


#endif
