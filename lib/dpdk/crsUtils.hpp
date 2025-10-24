#ifndef CRS_UTILS_HPP
#define CRS_UTILS_HPP

#include <rte_mbuf.h>

#pragma pack(push, 1)
struct ethernet_header {
    uint8_t  dest_mac[6];
    uint8_t  src_mac[6];
    uint16_t ether_type;
};

struct ip_header {
    uint8_t  version_ihl;
    uint8_t  type_of_service;
    uint16_t total_length;
    uint16_t identification;
    uint16_t flags_fragment_offset;
    uint8_t  ttl;
    uint8_t  protocol;
    uint16_t header_checksum;
    uint32_t src_addr;
    uint32_t dest_addr;
};

struct udp_header {
    uint16_t src_port;
    uint16_t dest_port;
    uint16_t length;
    uint16_t checksum;
};

struct crs_packet_header {
    uint8_t cookie;
    uint8_t hrdInfo;
    uint16_t source_id;
    uint16_t stream_id;
    uint16_t reserved0;
    uint64_t seq_number;
    uint32_t reserved1;
    uint16_t reserved2;
};

struct packet_headers {
    struct ethernet_header eth_hdr;
    struct ip_header ip_hdr;
    struct udp_header udp_hdr;
    struct crs_packet_header crs_hdr;
};
#pragma pack(pop)

inline uint64_t get_crs_packet_seq_num(struct rte_mbuf* cur_mbuf) {
    return (uint64_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.seq_number;
}

inline uint16_t get_crs_packet_stream_id(struct rte_mbuf* cur_mbuf) {
    return (uint16_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.stream_id;
}

inline uint16_t get_crs_packet_source_id(struct rte_mbuf* cur_mbuf) {
    return (uint16_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.source_id;
}


#endif /* CRS_UTILS_HPP */