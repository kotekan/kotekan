/**
 * @file
 * @brief A distributor for CRS board packets
 */

#ifndef CRS_DISTRIBUTOR_HPP
#define CRS_DISTRIBUTOR_HPP

#include "Config.hpp"
#include "crsUtils.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"

#include "json.hpp"


class crsDistributor : public dpdkRXhandler {
public:
    /// Default constructor
    crsDistributor(kotekan::Config& config, const std::string& unique_name,
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

inline crsDistributor::crsDistributor(kotekan::Config& config, const std::string& unique_name,
                                      kotekan::bufferContainer& buffer_container, int port,
                                      const std::vector<rte_ring*>& worker_rings) :
    dpdkRXhandler(config, unique_name, buffer_container, port), worker_rings(worker_rings) {

    worker_ring_ids = config.get<std::vector<uint32_t>>(unique_name, "worker_rings");

    if(worker_ring_ids.size() == 0) {
        throw std::runtime_error(
            fmt::format(fmt("The crsDistributor handler must have at least one worker ring.")));
    }

    if (worker_ring_ids.size() % 2 != 0) {
        throw std::runtime_error(
            fmt::format(fmt("The crsDistributor handler must have an even number of worker rings.")));
    }

    num_crs_boards = config.get<uint32_t>(unique_name, "num_crs_boards");

    if (num_crs_boards != 1 && num_crs_boards != 16 && num_crs_boards != 128) {
        throw std::runtime_error(
            fmt::format(fmt("num_crs_boards' parameter must be 1, 16, or 128.  Other configurations are not supported.")));
    }
}

inline int crsDistributor::handle_packet(struct rte_mbuf* mbuf) {

    /*
    uint64_t seq_number = get_crs_packet_seq_num(mbuf);
    uint16_t source_id = get_crs_packet_source_id(mbuf);
    INFO("Stream ID: {:d}, Seq Number: {:d}, Source ID: {:d}", stream_id, seq_number, source_id);

    // Hex dump of the first 128 bytes of the packet for debugging
    uint8_t* data_ptr = rte_pktmbuf_mtod(mbuf, uint8_t*);
    std::string hex_dump;
    for (size_t i = 0; i < 128; ++i) {
        hex_dump += fmt::format(fmt("{:02x} "), data_ptr[i]);
    }
    INFO("Packet Hex Dump: {:s}", hex_dump);

    // Print out the destination IP address and port for debugging
    struct ip_header* ip_hdr = (struct ip_header*)(data_ptr + sizeof(ethernet_header));
    struct udp_header* udp_hdr = (struct udp_header*)(data_ptr + sizeof(ethernet_header) + sizeof(ip_header));
    uint32_t dest_ip = ntohl(ip_hdr->dest_addr);
    uint16_t dest_port = ntohs(udp_hdr->dest_port);
    INFO("Destination IP: {:d}.{:d}.{:d}.{:d}, Destination Port: {:d}",
         (dest_ip >> 24) & 0xFF, (dest_ip >> 16) & 0xFF,
         (dest_ip >> 8) & 0xFF, dest_ip & 0xFF, dest_port);
    */
   
    uint16_t stream_id = get_crs_packet_stream_id(mbuf);

    // Divide the streams between the worker rings
    uint32_t ring_index = stream_id / (128 / worker_ring_ids.size());

    // Put the packet into the worker ring
    int err = rte_ring_enqueue(worker_rings[worker_ring_ids[ring_index]], mbuf);
    if (err != 0) {
        // TODO Make this into a metric
        //WARN_NON_OO("Failed to enqueue packet into worker ring {:d}, dropping packet", worker_ring_ids[ring_index]);
        rte_pktmbuf_free(mbuf);
    } else {
        total_packets++;
    }

    if (total_packets % 1000000 == 0) {
        INFO_NON_OO("Distributed {:d} packets so far", total_packets);
    }

    //DEBUG_NON_OO("Distributed packet with Stream ID {:d} to worker ring {:d}", stream_id, ring_index);

    return 0;
}

#endif
