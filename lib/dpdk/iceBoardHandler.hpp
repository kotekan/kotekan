/**
 * @file
 * @brief The base class for handlers which work with the McGill ICE FPGA boards
 * - iceBoardHandler : public dpdkRXhandler
 */

#ifndef ICE_BOARD_HANDLER_HPP
#define ICE_BOARD_HANDLER_HPP

#include "Config.hpp"
#include "ICETelescope.hpp"
#include "Telescope.hpp"
#include "dpdkCore.hpp"
#include "prometheusMetrics.hpp"
#include "util.h" // for e_time

#include "json.hpp"

#include <atomic>

/**
 * @brief Abstract class which contains things which are common to processing
 *        packets from the McGill ICE FPGA boards.
 *
 * This needs to be subclassed to actualy do something with the packets, it
 * just provides a common set of functions that are needed for ICEBoard packets
 *
 * @conf   alignment         UInt. Align each output frame of data to this FPGA seq number edge.
 *                                   Note it could be larger than the output frame size
 *                                   (in number of FPGA samples) but must be a multiple of that.
 * @conf   sample_size       Int.  Default 2048. Size of a time samples (unlikely to change)
 * @conf   fpga_packet_size  Int.  Default 4928. Full size of the FPGA packet, including Ethernet,
 *                                                 IP, UDP, and FPGA frame headers, FPGA data
 *                                                 payload, FPGA footer flags, and any padding
 *                                                 (but not the Ethernet CRC).
 * @conf   samples_per_packet Int. Default 2.    The number of time samples per FPGA packet
 * @conf   alignment_startup_delay Float. Default 1.0. The minimum time (in seconds) between the
 *                                                 first packet seen (on any port) and the
 *                                                 alignment edge (FPGA seq number) at which all
 *                                                 ports will start processing data. Increase
 *                                                 this if ports fail to start within one frame
 *                                                 of each other.
 * @conf   status_cadence    Int  Default 0      The time (in seconds between printing port
 *                                                 status) Default 0 == don't print.
 *
 * @par Metrics
 * @metric kotekan_dpdk_rx_packets_total
 *         The number of Rx packets processed since starting
 * @metric kotekan_dpdk_rx_samples_total
 *         The number of timesamples processed since starting
 *         This is basically kotekan_dpdk_rx_packets_total * samples_per_packet
 * @metric kotekan_dpdk_rx_lost_packets_total
 *         The number of lost packets since starting
 * @metric kotekan_dpdk_lost_samples_total
 *         The number of lost time smaples since starting
 * @metric kotekan_dpdk_rx_bytes_total
 *         The number of bytes processed since starting
 * @metric kotekan_dpdk_rx_errors_total
 *         The total number of all errors since starting
 *         (not including packets lost on the wire/NIC)
 * @metric kotekan_dpdk_rx_ip_cksum_errors_total
 *         The total number of IP check sum errors since starting
 * @metric kotekan_dpdk_rx_packet_len_errors_total
 *         The number of packets with incorrect lenght
 * @metric kotekan_dpdk_rx_out_of_order_errors_total
 *         The number of times we got a packet in the wrong order
 *
 * @author Andre Renard
 */
class iceBoardHandler : public dpdkRXhandler {
public:
    /// Default constructor
    iceBoardHandler(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container, int port);

    /// Same abstract function as in @c dpdkRXhandler
    virtual int handle_packet(struct rte_mbuf* mbuf) override = 0;

    /// Update common stats, this should be called by subclasses implementing this function as well
    virtual void update_stats() override;

protected:
    /**
     * @brief Aligns the first packet.
     *
     * This function should only be used at startup to find the first packet to start processing.
     *
     * The first handler (on any port) to see a packet sets a shared target alignment edge
     * which is at least @c alignment_startup_delay seconds in the future, rounded up to the
     * next @c alignment boundary. All handlers then drop packets until they reach that edge,
     * so every port starts processing at exactly the same FPGA seq number.
     *
     * Any gap between the target edge and the first packet a port sees after it is filled
     * in by the lost packet system. If that gap is one full frame (@c alignment) or more,
     * kotekan is stopped with a fatal error, since that indicates the port started too far
     * behind the others (see @c alignment_startup_delay).
     *
     * Should be called by every handler.
     *
     * @param mbuf The packet to check for alignment
     * @return True if the packet is at or past the shared target alignment edge,
     *         False otherwise.
     */
    bool align_first_packet(struct rte_mbuf* mbuf) {
        uint64_t seq = iceBoardHandler::get_mbuf_seq_num(mbuf);

        // Get (or set, if we are the first handler to see a packet) the shared target edge.
        uint64_t target_seq = get_alignment_target(seq);

        // Wait for the target alignment edge to be reached.
        if (seq < target_seq)
            return false;

        // Check that this port didn't start more than one frame past the target edge,
        // otherwise the lost packet fill system would need to back-fill one or more
        // entire frames to catch up.
        if (unlikely(seq >= target_seq + alignment)) {
            const double late_seconds =
                (double)(seq - target_seq) * Telescope::instance().seq_length_nsec() * 1e-9;
            FATAL_ERROR("Port {:d}: first packet (seq num {:d}) arrived {:d} frame(s) "
                        "({:.3f} s) after the target alignment edge {:d}. Try increasing "
                        "alignment_startup_delay (currently {:.3f} s), closing kotekan!",
                        port, seq, (seq - target_seq) / alignment, late_seconds, target_seq,
                        alignment_startup_delay);
            return false;
        }

        ice_stream_id_t stream_id =
            ice_extract_stream_id(iceBoardHandler::get_mbuf_stream_id(mbuf));

        last_seq = target_seq;
        cur_seq = seq;
        port_stream_id = stream_id;
        got_first_packet = true;

        INFO("Port {:d}; Got StreamID: crate: {:d}, slot: {:d}, link: {:d}, unused: {:d}, "
             "start seq num: {:d} current seq num: {:d}",
             port, stream_id.crate_id, stream_id.slot_id, stream_id.link_id, stream_id.unused,
             last_seq, seq);

        return true;
    }

    /**
     * @brief Gets the FPGA seq number from the given packet
     *
     * @param  cur_mbuf  The rte_mbuf containing the packet
     * @return           The FPGA seq number
     */
    inline uint64_t get_mbuf_seq_num(struct rte_mbuf* cur_mbuf) {
        return (uint64_t)(*(uint32_t*)(rte_pktmbuf_mtod(cur_mbuf, char*) + 54))
               + (((uint64_t)(0xFFFF & (*(uint32_t*)(rte_pktmbuf_mtod(cur_mbuf, char*) + 50))))
                  << 32);
    }

    /**
     * @brief Gets the FPGA stream ID from the given packet
     *
     * @param  cur_mbuf  The rte_mbuf containing the packet
     * @return           The encoded streamID
     */
    inline stream_t get_mbuf_stream_id(struct rte_mbuf* cur_mbuf) {
        return {*(uint16_t*)(rte_pktmbuf_mtod(cur_mbuf, char*) + 44)};
    }

    /**
     * @brief Checks the given packet against common errors.
     *
     * Errors include:
     * - IP Check sum failure
     * - Expected packet size
     *
     * Should be called by every handler.
     *
     * @param cur_mbuf The rte_mbuf containing the packet
     * @return True if the packet doesn't have errors and false otherwise.
     */
    inline bool check_packet(struct rte_mbuf* cur_mbuf) {
        if (unlikely((cur_mbuf->ol_flags & RTE_MBUF_F_RX_IP_CKSUM_MASK)
                     == RTE_MBUF_F_RX_IP_CKSUM_BAD)
            || unlikely((cur_mbuf->ol_flags & RTE_MBUF_F_RX_L4_CKSUM_MASK)
                        == RTE_MBUF_F_RX_L4_CKSUM_BAD)) {
            WARN("Port: {:d}; Got bad packet IP or UDP checksum", port);
            rx_ip_cksum_errors_total += 1;
            rx_errors_total += 1;
            return false;
        }

        if (unlikely(fpga_packet_size != cur_mbuf->pkt_len)) {
            // When using a switch we expect to pick up packets since the NIC is basically in
            // promiscuous mode.
            // TODO: Add a filter for F-engine packets to remove the risk of some equal
            // sized packet on the network getting misread as an F-engine packet.
            if (!using_switch) {
                // Checks the packet size matches the expected FPGA packet size.
                ERROR("Got packet with incorrect length: {:d}, expected {:d}", cur_mbuf->pkt_len,
                      fpga_packet_size);


                rx_packet_len_errors_total += 1;
                rx_errors_total += 1;
            }
            return false;
        }

        return true;
    }

    /**
     * @brief Checks the packet seq number hasn't gone backwards
     *
     * This check is done by looking at the @c diff value given
     * which should be the difference betwene the current FPGA seq being processed
     * and the last one seen before that.
     * @param diff The seq diff as explained above
     * @return true If the packet seq isn't older than expected, false otherwise
     */
    inline bool check_order(int64_t diff) {
        if (unlikely(diff < 0)) {
            WARN("Port: {:d}; Diff {:d} less than zero, duplicate, bad, or out-of-order packet; "
                 "last {:d}; cur: {:d}",
                 port, diff, last_seq, cur_seq);
            rx_out_of_order_errors_total += 1;
            rx_errors_total += 1;
            return false;
        }
        return true;
    }

    /**
     * @brief Checks if the seq number seems like it was reset
     *
     * This would likely be the result of an FPGA reset.
     *
     * This check is done by looking at the @c diff value given
     * which should be the difference betwene the current FPGA seq being processed
     * and the last one seen before that.
     * @param diff The seq diff as explained above
     * @return true If the packet seq isn't older than expected, false otherwise
     */
    inline bool check_for_reset(int64_t diff) {
        if (unlikely(diff < -1000)) {
            FATAL_ERROR(
                "The FPGAs likely reset, kotekan stopping... (FPGA seq number was less than 1000 "
                "of highest number seen.)");
            return false;
        }
        return true;
    }

    /**
     * @brief Get the difference between the current FPGA seq number and the last one seen.
     *
     * Requires the internal variables cur_seq and last_seq be set.
     *
     * @return int64_t The difference between the current FPGA seq number and the last one seen
     */
    inline int64_t get_packet_diff() {
        // Since the seq number is actually an unsigned 48-bit number, this cast will always be
        // safe.
        return (int64_t)cur_seq - (int64_t)last_seq;
    }

    /**
     * @brief Gets the target FPGA seq number all handlers will align their first frame to.
     *
     * The first handler to call this function with the seq number of a received packet
     * sets the shared target alignment edge: at least @c alignment_startup_delay seconds
     * past the given seq number, rounded up to the next @c alignment boundary. All later
     * calls simply return that shared target.
     *
     * This function must be called at least once with
     * @c get_alignment_target(std::numeric_limits<uint64_t>::max());
     * in order to reset the shared target before packets are processed.
     *
     * @param seq_num The FPGA seq number of the packet being processed,
     *                or uint64_t max to reset the shared target.
     * @return The shared target alignment seq number.
     */
    inline uint64_t get_alignment_target(uint64_t seq_num) {

        /// The target seq number all handlers align to.
        static std::atomic<uint64_t> alignment_target_seq{std::numeric_limits<uint64_t>::max()};

        // This provides a way to reset the alignment_target_seq to a fixed constant before
        // we start getting packets.
        if (seq_num == std::numeric_limits<uint64_t>::max()) {
            alignment_target_seq.store(std::numeric_limits<uint64_t>::max());
            DEBUG("Setting alignment target to MAX={:d}", std::numeric_limits<uint64_t>::max());
            return std::numeric_limits<uint64_t>::max();
        }

        // Fast path: the shared target has already been set by the first handler.
        uint64_t target = alignment_target_seq.load();
        if (target != std::numeric_limits<uint64_t>::max())
            return target;

        // The shared target isn't set yet, so compute a candidate and try to publish it.
        // Only the first handler to succeed at the compare-exchange sets the target for
        // everyone; any others discard their candidate and use the published value.
        const uint64_t delay_samples =
            (uint64_t)(alignment_startup_delay * 1e9 / Telescope::instance().seq_length_nsec());
        const uint64_t candidate = ((seq_num + delay_samples) / alignment + 1) * alignment;

        uint64_t expected = std::numeric_limits<uint64_t>::max();
        if (alignment_target_seq.compare_exchange_strong(expected, candidate)) {
            INFO("Port {:d}: Got first packet with seq num {:d}, set the target alignment "
                 "seq num for all ports to {:d} ({:.3f} s startup delay)",
                 port, seq_num, candidate, alignment_startup_delay);
            return candidate;
        }

        // Another handler won the race; expected now holds the published target.
        return expected;
    }

    /**
     * @brief Builds and returns a json object with all the port info
     *
     * @return The json object containing port info
     */
    nlohmann::json get_json_port_info();

    /// The FPAG seq number of the current packet being processed
    uint64_t cur_seq = 0;

    /// The FPGA seq number of the last packet seen (before the current one)
    uint64_t last_seq = 0;

    /// The streamID seen by this port handler
    /// Values of 255 = unset
    ice_stream_id_t port_stream_id = {255, 255, 255, 255};

    /// Set to true after the first packet is alligned.
    bool got_first_packet = false;

    /// Expected size of a time sample
    uint32_t sample_size;

    /// Expected size of an FPGA packet
    uint32_t fpga_packet_size;

    /// Expected number of time samples in each packet.
    uint32_t samples_per_packet;

    /// This is the value that we will align the first frame too.
    uint64_t alignment;

    /// The minimum time (in seconds) between the first packet seen and the target
    /// alignment edge all ports will start processing at.
    double alignment_startup_delay;

    /// Offset into the first byte of data after the Ethernet/UP/UDP/FPGA packet headers
    /// this shouldn't change, so we don't expose this to the config.
    const int32_t header_offset = 58;

    /// *** Stats (move into struct?) ***
    uint64_t rx_errors_total = 0;
    uint64_t rx_ip_cksum_errors_total = 0;
    uint64_t rx_packet_len_errors_total = 0;
    uint64_t rx_packets_total = 0;
    uint64_t rx_bytes_total = 0;
    uint64_t rx_out_of_order_errors_total = 0;
    uint64_t rx_lost_samples_total = 0;

    /// The number of frequences in the output stream
    int32_t num_local_freq;

    /// True if the NIC is connected to a switch, false if directly to the F-engine
    bool using_switch;

    /// Prometheus metrics
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_samples_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_lost_packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& lost_samples_total_metric;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_bytes_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_errors_total_metric;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_ip_cksum_errors_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        rx_packet_len_errors_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        rx_out_of_order_errors_total_metric;

private:
    // Last time we've printed a status message
    double last_status_message_time;

    // Timing between status messages
    uint32_t status_cadence;
};

inline iceBoardHandler::iceBoardHandler(kotekan::Config& config, const std::string& unique_name,
                                        kotekan::bufferContainer& buffer_container, int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port),
    rx_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_packets_total", unique_name, {"port"})),
    rx_samples_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_samples_total", unique_name, {"port"})),
    rx_lost_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_lost_packets_total", unique_name, {"port"})),
    lost_samples_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_lost_samples_total", unique_name, {"port"})),
    rx_bytes_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_bytes_total", unique_name, {"port"})),
    rx_errors_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_errors_total", unique_name, {"port"})),

    rx_ip_cksum_errors_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_ip_cksum_errors_total", unique_name, {"port"})),
    rx_packet_len_errors_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_packet_len_errors_total", unique_name, {"port"})),
    rx_out_of_order_errors_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_out_of_order_errors_total", unique_name, {"port"})) {

    sample_size = config.get_default<uint32_t>(unique_name, "sample_size", 2048);
    fpga_packet_size = config.get_default<uint32_t>(unique_name, "fpga_packet_size", 4928);
    samples_per_packet = config.get_default<uint32_t>(unique_name, "samples_per_packet", 2);

    num_local_freq = config.get_default<int32_t>(unique_name, "num_local_freq", 1);
    alignment = config.get<uint64_t>(unique_name, "alignment");
    alignment_startup_delay =
        config.get_default<double>(unique_name, "alignment_startup_delay", 1.0);
    using_switch = config.get_default<bool>(unique_name, "using_switch", false);

    get_alignment_target(std::numeric_limits<uint64_t>::max());

    // Don't print anything for the first 30 seconds
    last_status_message_time = e_time() + 30;
    status_cadence = config.get_default<uint32_t>(unique_name, "status_cadence", 0);
}

nlohmann::json iceBoardHandler::get_json_port_info() {
    nlohmann::json info;

    info["fpga_stream_id"] = {{"crate", port_stream_id.crate_id},
                              {"slot", port_stream_id.slot_id},
                              {"link", port_stream_id.link_id}};
    info["lost_packets"] = rx_lost_samples_total / samples_per_packet;
    info["lost_samples"] = rx_lost_samples_total;

    info["rx_packets_total"] = rx_packets_total;
    info["rx_samples_total"] = rx_packets_total;
    info["rx_bytes_total"] = rx_bytes_total;

    info["ip_cksum_errors"] = rx_ip_cksum_errors_total;
    info["out_of_order_errors"] = rx_out_of_order_errors_total;

    // This is the total number of errors from all sources other than missed packets
    // i.e. natural packet loss.
    info["errors_total"] = rx_errors_total;

    info["nic_port"] = this->port;

    std::vector<uint32_t> freq_bins;
    std::vector<float> freq_mhz;
    ice_stream_id_t temp_stream_id = port_stream_id;
    temp_stream_id.crate_id = port_stream_id.crate_id % 2;

    auto& tel = Telescope::instance();

    // TODO: this could probably be refactored now we have the Telescope object...
    const int num_shuffle_freq = (num_local_freq == 1 ? 4 : 1);

    for (int32_t i = 0; i < num_local_freq; ++i) {
        for (int j = 0; j < num_shuffle_freq; ++j) {
            if (port_stream_id.crate_id == 255) {
                freq_bins.push_back(std::numeric_limits<uint32_t>::max());
                freq_mhz.push_back(0);
            } else {
                temp_stream_id.unused = j;
                stream_t encoded_id = ice_encode_stream_id(temp_stream_id);

                freq_bins.push_back(tel.to_freq_id(encoded_id, i));
                freq_mhz.push_back(tel.to_freq_MHz(encoded_id, i));
            }
        }
    }

    info["freq_bins"] = freq_bins;
    info["freq_mhz"] = freq_mhz;

    return info;
}

inline void iceBoardHandler::update_stats() {

    std::vector<std::string> port_label = {std::to_string(port)};

    rx_packets_total_metric.labels(port_label).set(rx_packets_total);
    rx_samples_total_metric.labels(port_label).set(rx_packets_total * samples_per_packet);
    rx_lost_packets_total_metric.labels(port_label)
        .set((int)(rx_lost_samples_total / samples_per_packet));
    lost_samples_total_metric.labels(port_label).set(rx_lost_samples_total);

    rx_bytes_total_metric.labels(port_label).set(rx_bytes_total);
    rx_errors_total_metric.labels(port_label).set(rx_errors_total);

    rx_ip_cksum_errors_total_metric.labels(port_label).set(rx_ip_cksum_errors_total);
    rx_packet_len_errors_total_metric.labels(port_label).set(rx_packet_len_errors_total);
    rx_out_of_order_errors_total_metric.labels(port_label).set(rx_out_of_order_errors_total);

    double time_now = e_time();
    if (status_cadence != 0 && (time_now - last_status_message_time) > (double)status_cadence) {
        INFO("DPDK port {:d}, connected to (crate = {:d}, slot = {:d}, link = {:d}), total "
             "packets {:d}, total lost packets {:d}",
             port, port_stream_id.crate_id, port_stream_id.slot_id, port_stream_id.link_id,
             rx_packets_total, rx_lost_samples_total / samples_per_packet);
        last_status_message_time = time_now;
    }
}

#endif
