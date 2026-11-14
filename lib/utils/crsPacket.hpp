/**
 * @file
 * @brief On-wire CRS packet header and 12-bit sample packing, shared by the DPDK capture
 *        handlers (through crsUtils.hpp) and the ibverbs F-engine stages.
 */
#ifndef CRS_PACKET_HPP
#define CRS_PACKET_HPP

#include "udpPacket.hpp" // for udp_packet_headers

#include <stddef.h> // for size_t
#include <stdint.h> // for uint8_t, uint16_t, uint64_t

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

/// The headers of a CRS packet as they appear on the wire: Ethernet/IPv4/UDP then CRS.
struct packet_headers {
    struct ethernet_header eth_hdr;
    struct ip_header ip_hdr;
    struct udp_header udp_hdr;
    struct crs_packet_header crs_hdr;
};
#pragma pack(pop)

static_assert(sizeof(struct source_id_t) == 2);
static_assert(sizeof(struct crs_packet_header) == 22);
static_assert(sizeof(struct packet_headers) == 64);

/// The CRS header of the packet whose on-wire headers start at @p slot
/// (a header slot of the packet stages, or any contiguous packet buffer).
inline crs_packet_header* crs_header_of(uint8_t* slot) {
    return &reinterpret_cast<packet_headers*>(slot)->crs_hdr;
}

inline const crs_packet_header* crs_header_of(const uint8_t* slot) {
    return &reinterpret_cast<const packet_headers*>(slot)->crs_hdr;
}

/**
 * 12-bit unsigned samples packed two per three bytes, LSB first: sample 2i is
 * byte 3i plus the low nibble of byte 3i+1, sample 2i+1 is the high nibble of
 * byte 3i+1 plus byte 3i+2.
 */
inline uint16_t unpack12(const uint8_t* src, size_t i) {
    const uint8_t* p = src + 3 * (i / 2);
    if (i % 2 == 0)
        return p[0] | (uint16_t)(p[1] & 0x0f) << 8;
    return p[1] >> 4 | (uint16_t)p[2] << 4;
}

/// Store the low 12 bits of @p v as sample @p i of @p dst. See unpack12 for the layout.
inline void pack12(uint8_t* dst, size_t i, uint16_t v) {
    uint8_t* p = dst + 3 * (i / 2);
    if (i % 2 == 0) {
        p[0] = v & 0xff;
        p[1] = (p[1] & 0xf0) | ((v >> 8) & 0x0f);
    } else {
        p[1] = (p[1] & 0x0f) | ((v & 0x0f) << 4);
        p[2] = (v >> 4) & 0xff;
    }
}

#endif /* CRS_PACKET_HPP */
