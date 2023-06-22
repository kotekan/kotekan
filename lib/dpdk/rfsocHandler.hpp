//
// Created by andre on 20/03/23.
//

#ifndef KOTEKAN_RFSOC_HANDLER_HPP

#include "Config.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"

#include "json.hpp"

/**
 * @brief A simple handler to capture uniformly sized packets into a kotekan buffer
 *
 * This handler doesn't try to account for lost packets, or try to align frames.
 * It is simply for dumping any packets it gets into a buffer frame.
 *
 * @todo This is designed to capture FPGA packets which have a fixed length that is
 *       divisible by 32.  This class could be made more general.
 *
 * @par Buffers
 * @buffer out_buf  Kotekan buffer to place the packets in.
 *                  The frame size must be a multiple of the packet_size
 *       @buffer_format unit8_t array of packet contents
 *       @buffer_metadata none
 *
 * @conf packet_size    Int.  The size of the packet must be divisible by 32.
 *                              Includes Eth/IP/UDP headers.
 *
 * @author Andre Renard
 */
class rfsocHandler : public dpdkRXhandler {
public:
    /// Default constructor
    rfsocHandler(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container, int port);

    /// Processes the incoming packet header
    virtual int handle_packet(struct rte_mbuf* mbuf) override;

    /// Copy the packet
    virtual int worker_copy_packet(struct rte_mbuf* mbuf, uint32_t worker_id) override;

    /// Update stats, not used by this handler yet.
    virtual void update_stats() override{};

protected:
    /// The output buffer
    struct Buffer* out_buf;

    /// The current frame
    uint8_t* out_frame;

    /// The ID of the current frame
    int32_t out_frame_id = 0;

    /// Expected Packet size
    uint32_t packet_size;

    /// last_seq
    uint64_t last_seq = 0;

    /// Lost packets
    uint64_t lost_packets = 0;

    uint64_t total_packets = 0;

    struct timeval last_time;

    /// Flag to setup variables for the first run
    bool first_run = true;
};

inline rfsocHandler::rfsocHandler(kotekan::Config& config, const std::string& unique_name,
                                  kotekan::bufferContainer& buffer_container, int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port) {

    out_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buf"));
    register_producer(out_buf, unique_name.c_str());

    packet_size = config.get<uint32_t>(unique_name, "packet_size");

    if (packet_size > (uint32_t)out_buf->frame_size) {
        throw std::runtime_error("The packet size must be less than the frame size");
    }

    if ((out_buf->frame_size % packet_size) != 0) {
        throw std::runtime_error("The buffer frame size must be a multiple of the packet size");
    }
}

inline int rfsocHandler::handle_packet(struct rte_mbuf* mbuf) {

    if (unlikely((mbuf->ol_flags & RTE_MBUF_F_RX_IP_CKSUM_MASK) == RTE_MBUF_F_RX_IP_CKSUM_BAD)) {
        WARN("Port: {:d}; Got bad packet IP checksum", port);
        return 0;
    }
    // Check the UDP checksum
    if (unlikely((mbuf->ol_flags & RTE_MBUF_F_RX_L4_CKSUM_MASK) == RTE_MBUF_F_RX_L4_CKSUM_BAD)) {
        WARN("Port: {:d}; Got bad packet UDP checksum", port);
        return 0;
    }

    if (unlikely(packet_size != mbuf->pkt_len)) {
        WARN("Port: {:d}; Got packet with size {:d}, but expected size was {:d}", port,
             mbuf->pkt_len, packet_size);
        return 0;
    }

    uint64_t seq_num = *rte_pktmbuf_mtod_offset(mbuf, uint64_t*, 50);


    return 0;
}

inline int rfsocHandler::worker_copy_packet(struct rte_mbuf* mbuf, uint32_t worker_id) {
    (void)mbuf;
    // INFO("Got packet in worker {:d} copy", worker_id);
    total_packets += 1;
    if (total_packets % (1250000 * 1) == 0) {
        struct timeval now;
        gettimeofday(&now, nullptr);
        double elapsed_time = tv_to_double(now) - tv_to_double(last_time);
        INFO("Packet rate: {:.0f} pps, data rate: {:.4f}Gb/s, lost_packets rate: {:.4f}%",
             total_packets / elapsed_time, (double)total_packets * 8224 * 8 / 1e9 / elapsed_time,
             (double)lost_packets / (double)total_packets * 100.0);
        last_time = now;
        total_packets = 0;
    }

    return 0;
}

#define KOTEKAN_RFSOC_HANDLER_HPP
#endif // KOTEKAN_RFSOC_HANDLER_HPP
