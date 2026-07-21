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

    /// Export the distributed/dropped packet counters.
    virtual void update_stats() override {
        std::vector<std::string> port_label = {std::to_string(port)};
        packets_total_metric.labels(port_label).set(total_packets);
        non_crs_packets_total_metric.labels(port_label).set(non_crs_packets);
        invalid_stream_id_packets_total_metric.labels(port_label).set(invalid_stream_id_packets);
        ring_full_dropped_packets_total_metric.labels(port_label).set(ring_full_dropped_packets);
    }

    /// This is a distributor: it takes ownership of the mbuf.
    bool is_distributor() const override {
        return true;
    }

protected:
    std::vector<uint32_t> worker_ring_ids;

    const std::vector<rte_ring*>& worker_rings;

    uint32_t num_crs_boards;

    /// Expected size of a CRS packet; anything else is dropped before its
    /// header fields are trusted.
    uint32_t packet_size;

    uint64_t total_packets = 0;
    uint64_t non_crs_packets = 0;
    uint64_t invalid_stream_id_packets = 0;
    uint64_t ring_full_dropped_packets = 0;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& non_crs_packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        invalid_stream_id_packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        ring_full_dropped_packets_total_metric;
};

inline crs1BoardDistributor::crs1BoardDistributor(kotekan::Config& config,
                                                  const std::string& unique_name,
                                                  kotekan::bufferContainer& buffer_container,
                                                  int port,
                                                  const std::vector<rte_ring*>& worker_rings) :
    dpdkRXhandler(config, unique_name, buffer_container, port), worker_rings(worker_rings),
    packet_size(config.get<uint32_t>(unique_name, "packet_size")),
    packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_distributor_packets_total", unique_name, {"port"})),
    non_crs_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_distributor_non_crs_packets_total", unique_name, {"port"})),
    invalid_stream_id_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_distributor_invalid_stream_id_packets_total", unique_name, {"port"})),
    ring_full_dropped_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_distributor_ring_full_dropped_packets_total", unique_name, {"port"})) {

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

    // The port runs in promiscuous mode, so stray traffic (LLDP, ARP, ...)
    // lands here too, and its bytes at the CRS header offsets are arbitrary.
    // Drop anything that isn't a CRS packet before trusting header fields.
    if (unlikely(mbuf->pkt_len != packet_size
                 || get_crs_packet_cookie(mbuf) != CRS_PACKET_COOKIE)) {
        if (non_crs_packets == 0) {
            WARN("Port {:d}: dropping non-CRS packet (len {:d}, expected {:d}); counting further "
                 "drops in kotekan_dpdk_distributor_non_crs_packets_total",
                 port, mbuf->pkt_len, packet_size);
        }
        non_crs_packets++;
        rte_pktmbuf_free(mbuf);
        return 0;
    }

    uint16_t stream_id = get_crs_packet_stream_id(mbuf);

    // Divide the streams between the worker rings
    uint32_t ring_index = stream_id / (128 / worker_ring_ids.size());

    // A CRS packet with a stream_id outside [0, 128) must not index past the
    // ring table.
    if (unlikely(ring_index >= worker_ring_ids.size())) {
        if (invalid_stream_id_packets == 0) {
            WARN("Port {:d}: dropping CRS packet with out-of-range stream_id {:d}", port,
                 stream_id);
        }
        invalid_stream_id_packets++;
        rte_pktmbuf_free(mbuf);
        return 0;
    }

    // Put the packet into the worker ring
    int err = rte_ring_enqueue(worker_rings[worker_ring_ids[ring_index]], mbuf);
    if (err != 0) {
        // Failed to enqueue packet into worker ring, dropping packet
        ring_full_dropped_packets++;
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
