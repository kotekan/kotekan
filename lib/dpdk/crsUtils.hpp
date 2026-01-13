#ifndef CRS_UTILS_HPP
#define CRS_UTILS_HPP

#include <rte_mbuf.h>

#define CRS_PACKET_COOKIE 0xCF
#define NUM_FREQUENCY_BINS 48
#define TIME_SHORT_LEN 16
#define ELEMENT_SHORT_LEN 8

#pragma pack(push, 1)
struct ethernet_header {
    uint8_t dest_mac[6];
    uint8_t src_mac[6];
    uint16_t ether_type;
};

struct ip_header {
    uint8_t version_ihl;
    uint8_t type_of_service;
    uint16_t total_length;
    uint16_t identification;
    uint16_t flags_fragment_offset;
    uint8_t ttl;
    uint8_t protocol;
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

struct crs_packet_payload {
    uint8_t data[NUM_FREQUENCY_BINS * TIME_SHORT_LEN * ELEMENT_SHORT_LEN];
};

struct crs_packet {
    struct packet_headers headers;
    struct crs_packet_payload payload;
};
#pragma pack(pop)

// Packet headers are 64 bytes and payload is 48*16*8 = 6144 bytes
static_assert(sizeof(struct packet_headers) == 64);
static_assert(sizeof(struct crs_packet) == 64 + 48 * 16 * 8);

inline uint64_t get_crs_packet_seq_num(struct rte_mbuf* cur_mbuf) {
    return (uint64_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.seq_number;
}

inline uint16_t get_crs_packet_stream_id(struct rte_mbuf* cur_mbuf) {
    return (uint16_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.stream_id;
}

inline uint16_t get_crs_packet_source_id(struct rte_mbuf* cur_mbuf) {
    return (uint16_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.source_id;
}

inline uint8_t get_crs_packet_cookie(struct rte_mbuf* cur_mbuf) {
    return (uint8_t)rte_pktmbuf_mtod(cur_mbuf, struct packet_headers*)->crs_hdr.cookie;
}


#endif /* CRS_UTILS_HPP */