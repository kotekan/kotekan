/**
 * @file
 * @brief A distributor for CRS board packets
 */

#ifndef CRS_16BOARD_DISTRIBUTOR_HPP
#define CRS_16BOARD_DISTRIBUTOR_HPP

#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "crsUtils.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"

#include "json.hpp"


class crs16BoardDistributor : public dpdkRXhandler {
public:
    /// Default constructor
    crs16BoardDistributor(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container, int port,
                          const std::vector<rte_ring*>& worker_rings);

    /// Processes the incoming packets
    int handle_packet(struct rte_mbuf* mbuf) override;

    /// Update stats, not used by this handler yet.
    virtual void update_stats() override {};

    /// This is a distributor: it takes ownership of the mbuf.
    bool is_distributor() const override {
        return true;
    }

protected:
    std::vector<uint32_t> worker_ring_ids;

    const std::vector<rte_ring*>& worker_rings;

    uint32_t num_crs_boards;

    /// The expected CRS packet size; anything else is dropped as non-CRS traffic.
    uint32_t packet_size;

    uint64_t total_packets = 0;
    /// Dropped packets with a bad IP checksum (NIC offload flag).
    uint64_t bad_checksum_packets = 0;
    /// Dropped packets that failed the size or cookie check (LLDP, ARP, etc.)
    uint64_t non_crs_packets = 0;
    /// Dropped CRS packets whose stream_id does not map to a worker ring.
    uint64_t invalid_stream_id_packets = 0;
};

inline crs16BoardDistributor::crs16BoardDistributor(kotekan::Config& config,
                                                    const std::string& unique_name,
                                                    kotekan::bufferContainer& buffer_container,
                                                    int port,
                                                    const std::vector<rte_ring*>& worker_rings) :
    dpdkRXhandler(config, unique_name, buffer_container, port), worker_rings(worker_rings) {

    worker_ring_ids = config.get<std::vector<uint32_t>>(unique_name, "worker_rings");

    if (worker_ring_ids.size() == 0) {
        throw std::runtime_error(fmt::format(
            fmt("The crs16BoardDistributor handler must have at least one worker ring.")));
    }

    if (worker_ring_ids.size() % 2 != 0) {
        throw std::runtime_error(fmt::format(
            fmt("The crs16BoardDistributor handler must have an even number of worker rings.")));
    }

    num_crs_boards = config.get<uint32_t>(unique_name, "num_crs_boards");

    if (num_crs_boards != 16) {
        throw std::runtime_error(fmt::format(
            fmt("num_crs_boards' parameter must be 16. Other configurations are not supported.")));
    }

    packet_size = config.get<uint32_t>(unique_name, "packet_size");
}

inline int crs16BoardDistributor::handle_packet(struct rte_mbuf* mbuf) {

    // Check the packet checksum flag from the NIC.
    if (unlikely((mbuf->ol_flags & RTE_MBUF_F_RX_IP_CKSUM_MASK) == RTE_MBUF_F_RX_IP_CKSUM_BAD)) {
        WARN("Port: {:d}; Got bad packet IP checksum", port);
        rte_pktmbuf_free(mbuf);
        bad_checksum_packets++;
        return 0;
    }

    // Non-CRS traffic (LLDP, ARP, switch chatter) also lands here; drop it
    // before interpreting CRS header fields. No WARN: junk frames are routine
    // on a switch port and would flood the log.
    if (unlikely(mbuf->pkt_len != packet_size
                 || get_crs_packet_cookie(mbuf) != CRS_PACKET_COOKIE)) {
        rte_pktmbuf_free(mbuf);
        non_crs_packets++;
        return 0;
    }

    uint16_t stream_id = get_crs_packet_stream_id(mbuf);

    // Divide the streams between the worker rings
    uint32_t ring_index = stream_id / (128 / worker_ring_ids.size());

    // A corrupt or misconfigured CRS packet can carry a stream_id >= 128,
    // which would index past worker_ring_ids.
    if (unlikely(ring_index >= worker_ring_ids.size())) {
        rte_pktmbuf_free(mbuf);
        WARN("Port: {:d}; Got CRS packet with invalid stream_id {:d}, dropping", port, stream_id);
        invalid_stream_id_packets++;
        return 0;
    }

    // Put the packet into the worker ring
    int err = rte_ring_enqueue(worker_rings[worker_ring_ids[ring_index]], mbuf);
    if (err != 0) {
        // TODO Make this into a metric
        // Failed to enqueue packet into worker ring, dropping packet
        rte_pktmbuf_free(mbuf);
    } else {
        total_packets++;
    }

    return 0;
}

#endif
