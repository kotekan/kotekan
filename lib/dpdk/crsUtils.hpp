#ifndef CRS_UTILS_HPP
#define CRS_UTILS_HPP

#include "crsPacket.hpp" // for crs_packet_header, packet_headers, source_id_t, CRS_PACKET_COOKIE

#include <rte_mbuf.h>

#define NUM_FREQUENCY_BINS 48
#define TIME_SHORT_LEN 16
#define ELEMENT_SHORT_LEN 8

#pragma pack(push, 1)
struct crs_packet_payload {
    uint8_t data[NUM_FREQUENCY_BINS * TIME_SHORT_LEN * ELEMENT_SHORT_LEN];
};

struct crs_packet {
    struct packet_headers headers;
    struct crs_packet_payload payload;
};
#pragma pack(pop)

// Packet headers are 64 bytes and payload is 48*16*8 = 6144 bytes
static_assert(sizeof(struct crs_packet) == 64 + 48 * 16 * 8);

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
