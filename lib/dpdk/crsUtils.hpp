#ifndef CRS_UTILS_HPP
#define CRS_UTILS_HPP

#include "udpPacket.hpp" // for ethernet_header, ip_header, udp_header

#include <rte_mbuf.h>

#define CRS_PACKET_COOKIE 0xCF

#pragma pack(push, 1)
struct source_id_t {
    uint16_t link_id : 4;
    uint16_t slot_id : 4;
    uint16_t crate_id : 4;
    uint16_t reserved : 4;
};

struct crs_packet_header {
    uint8_t cookie;
    uint8_t hrdInfo;
    source_id_t source_id;
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

// Packet headers are 64 bytes
static_assert(sizeof(struct packet_headers) == 64);
static_assert(sizeof(struct source_id_t) == 2);

inline uint64_t get_crs_packet_seq_num(struct rte_mbuf* cur_mbuf) {
    return (uint64_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.seq_number;
}

inline uint16_t get_crs_packet_stream_id(struct rte_mbuf* cur_mbuf) {
    return (uint16_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.stream_id;
}

inline source_id_t get_crs_packet_source_id(struct rte_mbuf* cur_mbuf) {
    return (source_id_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.source_id;
}

inline uint8_t get_crs_packet_cookie(struct rte_mbuf* cur_mbuf) {
    return (uint8_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.cookie;
}


#endif /* CRS_UTILS_HPP */