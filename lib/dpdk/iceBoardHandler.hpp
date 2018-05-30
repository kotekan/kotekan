#ifndef ICE_BOARD_HANDLER_HPP
#define ICE_BOARD_HANDLER_HPP

#include "dpdkCore.hpp"
#include "fpga_header_functions.h"
#include "prometheusMetrics.hpp"

class iceBoardHandler : public dpdkRXhandler {
public:
    iceBoardHandler(Config &config, const std::string &unique_name,
                    bufferContainer &buffer_container, int port);

    virtual int handle_packet(struct rte_mbuf *mbuf) = 0;

    virtual void update_stats();

protected:

    bool align_first_packet(struct rte_mbuf *mbuf) {
        uint64_t seq = iceBoardHandler::get_mbuf_seq_num(mbuf);
        stream_id_t stream_id = extract_stream_id(iceBoardHandler::get_mbuf_stream_id(mbuf));

        // We allow for the fact we might miss the first packet by upto 100 FPGA frames,
        // if this happens then the missing frames at the start of the buffer frame are filled
        // in as lost packets.
        if ( ((seq % alignment) <= 100) && ((seq % alignment) >= 0 )) {

            INFO("Port %d; Got StreamID: crate: %d, slot: %d, link: %d, unused: %d",
                port, stream_id.crate_id, stream_id.slot_id, stream_id.link_id, stream_id.unused);

            last_seq = seq - seq % alignment;
            cur_seq = seq;
            port_stream_id = stream_id;
            got_first_packet = true;

            return true;
        }

        return false;
    }

    inline uint64_t get_mbuf_seq_num(struct rte_mbuf * cur_mbuf) {
        return (uint64_t)(*(uint32_t *)(rte_pktmbuf_mtod(cur_mbuf, char *) + 54)) +
               (((uint64_t) (0xFFFF & (*(uint32_t *)(rte_pktmbuf_mtod(cur_mbuf, char *) + 50)))) << 32);
    }

    inline uint16_t get_mbuf_stream_id(struct rte_mbuf * cur_mbuf) {
        return *(uint16_t *)(rte_pktmbuf_mtod(cur_mbuf, char *) + 44);
    }

    inline bool check_packet(struct rte_mbuf * cur_mbuf) {
        if (unlikely((cur_mbuf->ol_flags | PKT_RX_IP_CKSUM_BAD) == 1)) {
            WARN("dpdk: Got bad packet checksum on port %d", port);
            rx_crc_errors_total += 1;
            rx_errors_total += 1;
            return false;
        }
        if (unlikely(fpga_packet_size != cur_mbuf->pkt_len)) {
            ERROR("Got packet with incorrect length: %d, expected %d",
                  cur_mbuf->pkt_len, fpga_packet_size);

            // Getting a packet with the wrong length is almost always
            // a configuration/FPGA problem that needs to be addressed.
            // So for now we just exit kotekan with an error message.
            raise(SIGINT);

            rx_packet_len_errors_total += 1;
            rx_errors_total += 1;
            return false;
        }

        // Add too comment stats:
        rx_packets_total += 1;
        rx_bytes_total += cur_mbuf->pkt_len;

        return true;
    }

    inline bool check_order(int64_t diff) {
        if (unlikely(diff < 0)) {
            WARN("Port: %d; Diff %" PRId64 " less than zero, duplicate, bad, or out-of-order packet; last %" PRIu64 "; cur: %" PRIu64 "",
                 port, diff, last_seq, cur_seq);
            rx_out_of_order_errors_total += 1;
            rx_errors_total += 1;
            return false;
        }
        return true;
    }

    inline bool check_for_reset(int64_t diff) {
        if (unlikely(diff < -1000)) {
            ERROR("The FPGAs likely reset, kotekan stopping... (FPGA seq number was less than 1000 of highest number seen.)");
            raise(SIGINT);
            return false;
        }
        return true;
    }

    inline int64_t get_packet_diff() {
        // Since the seq number is actually an unsigned 48-bit numdber, this cast will always be safe.
        return (int64_t)cur_seq - (int64_t)last_seq;
    }

    uint64_t cur_seq = 0;
    uint64_t last_seq = 0;
    stream_id_t port_stream_id;
    bool got_first_packet = false;

    // FPGA parmeters
    uint32_t sample_size;
    uint32_t fpga_packet_size;
    uint32_t samples_per_packet;

    /// This is the value that we will align the first frame too.
    uint64_t alignment;

    // This shouldn't change, so we don't expose this to the config.
    const int32_t header_offset = 58;

    // *** Stats (move into struct?) ***
    uint64_t rx_errors_total = 0;
    uint64_t rx_crc_errors_total = 0;
    uint64_t rx_packet_len_errors_total = 0;
    uint64_t rx_packets_total = 0;
    uint64_t rx_bytes_total = 0;
    uint64_t rx_out_of_order_errors_total = 0;
    uint64_t rx_lost_samples_total = 0;

};

inline iceBoardHandler::iceBoardHandler(Config &config, const std::string &unique_name,
                       bufferContainer &buffer_container, int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port) {

    sample_size = config.get_int_default(unique_name, "sample_size", 2048);
    fpga_packet_size = config.get_int_default(unique_name, "fpga_packet_size", 4928);
    samples_per_packet = config.get_int_default(unique_name, "samples_per_packet", 2);

    alignment = config.get_int_eval(unique_name, "alignment");
}

inline void iceBoardHandler::update_stats() {
    prometheusMetrics &metrics = prometheusMetrics::instance();

    std::string tags = "port=\"" + std::to_string(port) + "\"";

    metrics.add_process_metric("kotekan_dpdk_rx_packets_total",
                                unique_name,
                                rx_packets_total,
                                tags);
    metrics.add_process_metric("kotekan_dpdk_rx_samples_total",
                                unique_name,
                                rx_packets_total * samples_per_packet,
                                tags);

    metrics.add_process_metric("kotekan_dpdk_rx_lost_packets_total",
                                unique_name,
                                (int)(rx_lost_samples_total / samples_per_packet),
                                tags);
    metrics.add_process_metric("kotekan_dpdk_lost_samples_total",
                                unique_name,
                                rx_lost_samples_total,
                                tags);

    metrics.add_process_metric("kotekan_dpdk_rx_bytes_total",
                                unique_name,
                                rx_bytes_total,
                                tags);
    metrics.add_process_metric("kotekan_dpdk_rx_errors_total",
                                unique_name,
                                rx_errors_total,
                                tags);

    metrics.add_process_metric("kotekan_dpdk_rx_crc_errors_total",
                                unique_name,
                                rx_crc_errors_total,
                                tags);
    metrics.add_process_metric("kotekan_dpdk_rx_packet_len_errors_total",
                                unique_name,
                                rx_packet_len_errors_total,
                                tags);
    metrics.add_process_metric("kotekan_dpdk_rx_out_of_order_errors_total",
                                unique_name,
                                rx_out_of_order_errors_total,
                                tags);
}

#endif