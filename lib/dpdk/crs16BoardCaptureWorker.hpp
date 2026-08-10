/**
 * @file
 * @brief A capture worker/handler for CRS board packets
 */

#ifndef CRS_16BOARD_CAPTURE_WORKER_HPP
#define CRS_16BOARD_CAPTURE_WORKER_HPP

#include "Config.hpp"
#include "FramePrefetchService.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "crsUtils.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"
#include "timeUtil.hpp"
#include "util.h"
#include "visUtil.hpp"

#include "json.hpp"

#include <algorithm>
#include <immintrin.h>


/**
 * @brief Capture worker for the 16 CRS (F-engine) board configuration.
 *
 * Consumes the packets a @c crs16BoardDistributor placed on this worker's ring and copies
 * them into the output voltage buffer, one frame per @c time_samples_per_frame FPGA samples.
 *
 * @par Capture scheduling
 * By default capture starts at the first frame edge a couple of seconds after packets begin
 * to flow.  Set @c capture_start_time_utc to instead hold off until a given wall clock time,
 * and @c capture_n_frames to stop after a fixed number of frames.  Both options can be set on
 * the @c dpdkCore stage itself so that every worker shares them.
 *
 * @conf   out_buf              String. The voltage buffer to write captured frames into.
 * @conf   receipt_bitmap_buf   String. The buffer to write the packet receipt bitmap into.
 * @conf   port                 Int. The NIC port this worker's packets arrive on.
 * @conf   packet_size          Int. Total size of a CRS packet in bytes (header + payload).
 * @conf   payload_size         Int. Size of the CRS packet payload in bytes.
 * @conf   num_expected_stream_ids  Int. Number of stream IDs to see before capture can start.
 * @conf   frame_service_cpu_affinity  Array of Int. CPU cores for the frame prefetch thread.
 * @conf   prefetch_depth       Int. Default 4. Number of frames the prefetch service holds ready.
 * @conf   allocate_metadata    Bool. Default false. Set on exactly one worker per output buffer.
 * @conf   capture_start_time_utc  String. Default "" (start as soon as packets flow).  UTC time
 *                              at which capture should start, e.g. "2026-01-31 18:45:00".  The
 *                              time is converted to an FPGA sequence number using the GPS time
 *                              of the F-engine (never the system clock), and rounded down to the
 *                              nearest frame edge.  It must be at least one minute in the future;
 *                              a start more than 12 hours out is allowed but warned about.
 * @conf   capture_n_frames     Int. Default 0 (unlimited).  Stop capturing after this many
 *                              frames.  The DPDK threads exit but kotekan keeps running, so
 *                              downstream stages can flush what they have.
 */
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

    /// Requested capture start time in ns since the UNIX epoch.
    /// Zero means "start as soon as packets are flowing".
    int64_t capture_start_time_ns = 0;

    /// Number of frames to capture before stopping, 0 = unlimited.
    uint64_t capture_n_frames;

    /// Number of frames handed to the output buffer so far.
    uint64_t frames_captured = 0;

    /// FPGA seq number of the first sample of the first captured frame.
    uint64_t capture_start_seq = 0;

    /// Set once @c capture_start_seq has been worked out from the first packets.
    bool capture_start_seq_known = false;

    /// Instrument (GPS) time at FPGA seq 0, and the length of one seq tick, both in
    /// nanoseconds. Cached from the Telescope when the start seq is computed.
    int64_t time0_ns = 0;
    int64_t seq_length_ns = 0;

    /// How far ahead of @c capture_start_seq the prefetch service is armed.
    uint64_t prefetch_lead_samples = 0;

    /// The seq number at which the next "waiting for the scheduled start" message is logged.
    uint64_t next_countdown_seq = 0;

    /// The prefetch service is armed this long before the capture start, so that the first
    /// frames are ready by the time the first packets belonging to them arrive.
    static constexpr double _prefetch_lead_s = 2.0;

    /// A scheduled start closer than this to the current F-engine time is an error, ...
    static constexpr int64_t _min_start_lead_ns = 60L * 1'000'000'000L;

    /// ... and one further out than this is allowed but warned about.
    static constexpr int64_t _max_start_lead_ns = 12L * 3600L * 1'000'000'000L;

    /// Interval between "waiting for the scheduled start" messages.
    static constexpr int64_t _countdown_interval_ns = 60L * 1'000'000'000L;

    /**
     * @brief Work out, and sanity check, the seq number at which capture starts.
     *
     * Called once, with the seq number of a packet that is arriving now, so that the
     * configured UTC start time can be compared against the F-engine's GPS time.
     *
     * @param current_seq The FPGA seq number of the packet being handled.
     * @return False if the configured start time is unusable, in which case kotekan has
     *         already been asked to shut down and the caller should stop the lcore.
     */
    bool compute_capture_start_seq(uint64_t current_seq);

    /// Instrument (GPS) time in ns of an FPGA seq number, using the cached Telescope values.
    int64_t seq_to_time_ns(uint64_t seq) const {
        return time0_ns + (int64_t)seq * seq_length_ns;
    }

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
    capture_n_frames = config.get_default<uint64_t>(unique_name, "capture_n_frames", 0);
    if (capture_n_frames > 0)
        INFO("crs16BoardCaptureWorker {}: capturing {} frames, after which the DPDK threads "
             "stop but kotekan keeps running",
             unique_name, capture_n_frames);

    // Optional scheduled start time. An empty string means start as soon as packets flow.
    // Parsing failures are raised here so a typo is caught at startup rather than minutes
    // into a run. The time itself is checked against the F-engine's GPS time once packets
    // arrive, see compute_capture_start_seq().
    const std::string capture_start_time_utc =
        config.get_default<std::string>(unique_name, "capture_start_time_utc", "");
    if (!capture_start_time_utc.empty()) {
        capture_start_time_ns = utc_string_to_nanosec_i64(capture_start_time_utc);
        INFO("crs16BoardCaptureWorker {}: capture scheduled to start at {}", unique_name,
             nanosec_i64_to_utc_string(capture_start_time_ns));
    }

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


inline bool crs16BoardCaptureWorker::compute_capture_start_seq(uint64_t current_seq) {

    const Telescope& tel = Telescope::instance();
    time0_ns = tel.to_time_ns(0);
    seq_length_ns = (int64_t)tel.seq_length_nsec();
    capture_start_seq_known = true;

    prefetch_lead_samples = (uint64_t)(_prefetch_lead_s * 1e9 / (double)seq_length_ns);

    if (capture_start_time_ns == 0) {
        // No scheduled start, so begin at the first frame edge past the prefetch lead time.
        const uint64_t future_seq = current_seq + prefetch_lead_samples;
        capture_start_seq = future_seq - (future_seq % time_samples_per_frame);
        return true;
    }

    if (!tel.gps_time_enabled())
        WARN("Port: {:d}, Worker: {:d}; capture_start_time_utc was set, but the telescope has no "
             "GPS time, so FPGA seq numbers are timed against the system clock at kotekan "
             "startup. The capture will not start when requested.",
             port, worker_id);

    // "Now" is the GPS time of the F-engine seq number of the packet we are handling, never
    // the system clock.
    const int64_t now_ns = tel.to_time_ns(current_seq);
    const int64_t wait_ns = capture_start_time_ns - now_ns;

    if (wait_ns < 0) {
        FATAL_ERROR("Port: {:d}, Worker: {:d}; The requested capture_start_time_utc ({:s}) is "
                    "{:s} in the past; the current F-engine time is {:s}. It must be at least "
                    "one minute in the future.",
                    port, worker_id, nanosec_i64_to_utc_string(capture_start_time_ns),
                    format_duration_ns(-wait_ns), nanosec_i64_to_utc_string(now_ns));
        return false;
    }

    if (wait_ns < _min_start_lead_ns) {
        FATAL_ERROR("Port: {:d}, Worker: {:d}; The requested capture_start_time_utc ({:s}) is "
                    "only {:s} away; the current F-engine time is {:s}. It must be at least one "
                    "minute in the future to leave time to set the capture up.",
                    port, worker_id, nanosec_i64_to_utc_string(capture_start_time_ns),
                    format_duration_ns(wait_ns), nanosec_i64_to_utc_string(now_ns));
        return false;
    }

    if (wait_ns > _max_start_lead_ns)
        WARN("Port: {:d}, Worker: {:d}; The requested capture_start_time_utc ({:s}) is {:s} away, "
             "more than 12 hours in the future. Kotekan will sit idle until then.",
             port, worker_id, nanosec_i64_to_utc_string(capture_start_time_ns),
             format_duration_ns(wait_ns));

    // Round the start down to the nearest frame edge. Every worker uses the same frame
    // length, so they all land on the same seq number and produce aligned frames.
    capture_start_seq = tel.to_seq(capture_start_time_ns);
    capture_start_seq -= capture_start_seq % time_samples_per_frame;

    next_countdown_seq = current_seq + (uint64_t)(_countdown_interval_ns / seq_length_ns);

    INFO("Port: {:d}, Worker: {:d}; Capture starts at {:s} (fpga seq {:d}), which is {:s} from "
         "now, rounded down one frame edge from the requested {:s}. Current F-engine time is "
         "{:s}.",
         port, worker_id, nanosec_i64_to_utc_string(seq_to_time_ns(capture_start_seq)),
         capture_start_seq, format_duration_ns(seq_to_time_ns(capture_start_seq) - now_ns),
         nanosec_i64_to_utc_string(capture_start_time_ns), nanosec_i64_to_utc_string(now_ns));

    return true;
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
        // Once the list is full we stop looking, both to keep this out of the hot path and
        // so that a capture waiting on a scheduled start does not keep adding to the list.
        if (stream_ids_expected.size() < num_expected_stream_ids) {

            // Check if the stream ID is already in the list, and if not add it.
            if (std::find(stream_ids_expected.begin(), stream_ids_expected.end(), stream_id)
                == stream_ids_expected.end()) {
                stream_ids_expected.push_back(stream_id);
            }

            if (stream_ids_expected.size() < num_expected_stream_ids) {
                return 0; // Wait for more packets to establish the full list
            }
        }

        // Work out where the capture begins, either a couple of seconds from now or at the
        // scheduled UTC start time.
        if (unlikely(!capture_start_seq_known)) {
            if (!compute_capture_start_seq(seq_num))
                return -1;
        }

        // Hold off arming the prefetch service until the start is close. A capture scheduled
        // hours ahead would otherwise sit on empty buffer frames for the whole wait.
        if (seq_num + prefetch_lead_samples < capture_start_seq) {
            if (worker_id == 0 && seq_num >= next_countdown_seq) {
                INFO("Waiting for the scheduled capture start at {:s}; {:s} remaining",
                     nanosec_i64_to_utc_string(seq_to_time_ns(capture_start_seq)),
                     format_duration_ns(seq_to_time_ns(capture_start_seq)
                                        - seq_to_time_ns(seq_num)));
                next_countdown_seq = seq_num + (uint64_t)(_countdown_interval_ns / seq_length_ns);
            }
            return 0;
        }

        prefetch_service->start(capture_start_seq, stream_ids_expected);
        first_run = false;
        INFO("Port: {:d}, Worker: {:d}; Starting prefetch service at sequence number {:d} ({:s})",
             port, worker_id, capture_start_seq,
             nanosec_i64_to_utc_string(seq_to_time_ns(capture_start_seq)));
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

        // Advancing hands the frame we just filled to the prefetch service, which marks it
        // full for the downstream stages. Once we have handed over the requested number of
        // frames we stop this lcore, but leave the rest of kotekan running so the file
        // writers can flush what they have.
        frames_captured++;
        if (unlikely(capture_n_frames > 0 && frames_captured >= capture_n_frames)) {
            INFO("Port: {:d}, Worker: {:d}; Captured the requested {:d} frames (seq {:d} to "
                 "{:d}), stopping packet capture.",
                 port, worker_id, capture_n_frames, capture_start_seq,
                 capture_start_seq + capture_n_frames * time_samples_per_frame);
            return -1;
        }

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
