/**
 * @file
 * @brief A distributor for CRS board packets
 */

#ifndef CRS_DISTRIBUTOR_HPP
#define CRS_DISTRIBUTOR_HPP

#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "crsUtils.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"

#include "json.hpp"


class crs1BoardDistributor : public dpdkRXhandler {
public:
    /// Default constructor
    crs1BoardDistributor(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container, int port,
                         const std::vector<rte_ring*>& worker_rings);

    /// Processes the incoming packets
    int handle_packet(struct rte_mbuf* mbuf) override;

    /// Update stats, not used by this handler yet.
    virtual void update_stats() override {};

protected:
    std::vector<uint32_t> worker_ring_ids;

    const std::vector<rte_ring*>& worker_rings;

    uint32_t num_crs_boards;

    uint64_t total_packets = 0;
};

inline crs1BoardDistributor::crs1BoardDistributor(kotekan::Config& config,
                                                  const std::string& unique_name,
                                                  kotekan::bufferContainer& buffer_container,
                                                  int port,
                                                  const std::vector<rte_ring*>& worker_rings) :
    dpdkRXhandler(config, unique_name, buffer_container, port), worker_rings(worker_rings) {

    worker_ring_ids = config.get<std::vector<uint32_t>>(unique_name, "worker_rings");

    if (worker_ring_ids.size() == 0) {
        throw std::runtime_error(fmt::format(
            fmt("The crs1BoardDistributor handler must have at least one worker ring.")));
    }

    if (worker_ring_ids.size() % 2 != 0) {
        throw std::runtime_error(fmt::format(
            fmt("The crs1BoardDistributor handler must have an even number of worker rings.")));
    }

    num_crs_boards = config.get<uint32_t>(unique_name, "num_crs_boards");

    if (num_crs_boards != 1) {
        throw std::runtime_error(fmt::format(
            fmt("num_crs_boards' parameter must be 1. Other configurations are not supported.")));
    }
}

inline int crs1BoardDistributor::handle_packet(struct rte_mbuf* mbuf) {

    uint16_t stream_id = get_crs_packet_stream_id(mbuf);

    // Divide the streams between the worker rings
    uint32_t ring_index = stream_id / (128 / worker_ring_ids.size());

    // Put the packet into the worker ring
    int err = rte_ring_enqueue(worker_rings[worker_ring_ids[ring_index]], mbuf);
    if (err != 0) {
        // TODO Make this into a metric
        // WARN_NON_OO("Failed to enqueue packet into worker ring {:d}, dropping packet",
        // worker_ring_ids[ring_index]);
        rte_pktmbuf_free(mbuf);
    } else {
        total_packets++;
    }

    if (total_packets % 1000000 == 0) {
        INFO_NON_OO("Distributed {:d} packets so far", total_packets);
    }

    return 0;
}

#endif
