/**
 * @file
 * @brief A capture worker/handler for CRS board packets
 */

#ifndef CRS_CAPTURE_WORKER_HPP
#define CRS_CAPTURE_WORKER_HPP

#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"
#include "visUtil.hpp"

#include "json.hpp"


class crs1BoardCaptureWorker : public dpdkRXhandler {
public:
    /// Default constructor
    crs1BoardCaptureWorker(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container, int port,
                           const std::vector<rte_ring*>& worker_rings);

    /// Processes the incoming packets
    int handle_packet(struct rte_mbuf* mbuf) override;

    /// Update stats, not used by this handler yet.
    virtual void update_stats() override {};

protected:
    const std::vector<rte_ring*>& worker_rings;

    /// The output buffer
    Buffer* out_buf;

    uint32_t out_frame_id = 0;

    uint8_t* out_frame = nullptr;
    /// The packet size
    uint32_t packet_size;

    /// The location in the out_frame to put the next packet
    uint32_t packet_location = 0;

    bool first_run = true;

    int64_t total_packets_received = 0;

    double previous_time = 0.0;

    /// Maximum number of frames to capture (used for burst captures), 0 = unlimited
    uint64_t capture_n_frames;

    /// Number of frames captured
    uint64_t num_frames_captured = 0;

    /// Previous seq num
    uint64_t last_seq = 0;

    /// Sample-based FPGA seq monotonicity check. At each sample the seq must
    /// be strictly greater than the previous sample's, otherwise we treat it
    /// as an FPGA reset and shut down. Sampled every @c _seq_check_interval
    /// packets, which at typical CRS packet rates is roughly once per second.
    static constexpr uint64_t _seq_check_interval = 500000;
    uint64_t _last_check_seq = 0;
    uint64_t _seq_check_packet_count = 0;
};

inline crs1BoardCaptureWorker::crs1BoardCaptureWorker(kotekan::Config& config,
                                                      const std::string& unique_name,
                                                      kotekan::bufferContainer& buffer_container,
                                                      int port,
                                                      const std::vector<rte_ring*>& worker_rings) :
    dpdkRXhandler(config, unique_name, buffer_container, port), worker_rings(worker_rings),
    out_buf(buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buf"))),

    packet_size(config.get<uint32_t>(unique_name, "packet_size")) {

    out_buf->register_producer(unique_name);

    if (packet_size > (uint32_t)out_buf->frame_size) {
        throw std::runtime_error("The packet size must be less than the frame size");
    }

    if ((out_buf->frame_size % packet_size) != 0) {
        throw std::runtime_error("The buffer frame size must be a multiple of the packet size");
    }

    if ((packet_size % 32) != 0) {
        throw std::runtime_error("The packet_size must be a multiple of 32 bytes");
    }

    // Number of frames to capture before stopping, 0 = unlimited
    capture_n_frames = config.get_default<uint64_t>(unique_name, "capture_n_frames", 0);
}

inline int crs1BoardCaptureWorker::handle_packet(struct rte_mbuf* mbuf) {

    (void)mbuf;
    // Print the worker ID and stream ID
    // uint16_t stream_id = get_crs_packet_stream_id(mbuf);
    // uint16_t source_id = get_crs_packet_source_id(mbuf);
    uint64_t seq_num = get_crs_packet_seq_num(mbuf);

    // INFO("Worker {:d} got packet from source ID {:d}, stream ID {:d}, seq num {:d}",
    //      port, source_id, stream_id, seq_num);

    // Get the first frame.
    if (first_run) {
        out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            return -1;
        first_run = false;
    }

    // Check the IP checksum
    if (unlikely((mbuf->ol_flags & RTE_MBUF_F_RX_IP_CKSUM_MASK) == RTE_MBUF_F_RX_IP_CKSUM_BAD)) {
        WARN("Port: {:d}; Got bad packet IP checksum", port);
        return 0;
    }

    if (unlikely(packet_size != rte_pktmbuf_pkt_len(mbuf))) {
        WARN("Port: {:d}; Got packet with size {:d}, but expected size was {:d}", port,
             rte_pktmbuf_pkt_len(mbuf), packet_size);
        return 0;
    }

    // Sample-based FPGA seq monotonicity check: at each sample boundary the
    // current seq must be strictly greater than the previous sample's,
    // otherwise the FPGA likely reset.
    if (unlikely(++_seq_check_packet_count >= _seq_check_interval)) {
        if (unlikely(_last_check_seq != 0 && seq_num < _last_check_seq)) {
            FATAL_ERROR("Port: {:d}; CRS FPGA seq went backwards ({:d} -> {:d}), controller "
                        "likely reset, kotekan stopping...",
                        port, _last_check_seq, seq_num);
            return -1;
        }
        _last_check_seq = seq_num;
        _seq_check_packet_count = 0;
    }

    // Copy the packet.
    assert((packet_location + 1) * packet_size <= (uint32_t)out_buf->frame_size);
    int offset = 0;
    copy_block(&mbuf, &out_frame[packet_location * packet_size], packet_size, (int*)&offset);

    packet_location++;
    total_packets_received += 1;

    // Log packet rates.
    if (total_packets_received % 390000 == 0) {
        double time_now = e_time();
        INFO("Worker {:d} has received {:d} packets, seq num {:d}, packet rate: {:d} pps, data "
             "rate: {:f} Gbps, time elapsed: {:f} s",
             port, total_packets_received, seq_num, (int)(390000 / (time_now - previous_time)),
             (double)packet_size * 390000 * 8 / (time_now - previous_time) / 1.0e9,
             (time_now - previous_time));
        previous_time = time_now;
        last_seq = seq_num;
    }

    if (packet_location * packet_size == (uint32_t)out_buf->frame_size) {
        out_buf->mark_frame_full(unique_name, out_frame_id);
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;

        // Check if we have captured enough frames
        num_frames_captured++;
        if (capture_n_frames != 0 && num_frames_captured > capture_n_frames) {
            return -1;
        }

        out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            return -1;

        packet_location = 0;
    }

    return 0;
}

#endif
