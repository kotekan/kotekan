/**
 * @file
 * @brief On-wire Ethernet/IPv4/UDP headers and the header/payload slot layout shared by the
 *        packet stages (ibvUdpSend, ibvUdpRecv, udpPacketGen, udpPacketCheck).
 */
#ifndef UDP_PACKET_HPP
#define UDP_PACKET_HPP

#include "Config.hpp"            // for Config
#include "buffer.hpp"            // for Buffer
#include "network_functions.hpp" // for in_cksum

#include "fmt.hpp" // for format

#include <arpa/inet.h> // for htons, inet_pton
#include <stddef.h>    // for size_t
#include <stdexcept>   // for invalid_argument
#include <stdint.h>    // for uint8_t, uint16_t, uint32_t, uint64_t
#include <stdio.h>     // for sscanf
#include <string.h>    // for memcpy
#include <string>      // for string

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

/// The headers of an untagged IPv4/UDP packet as they appear on the wire.
struct udp_packet_headers {
    ethernet_header eth;
    ip_header ip;
    udp_header udp;
};

/// Application header written by udpPacketGen right after @c udp_packet_headers.
struct udp_app_header {
    uint64_t seq;
};

/// Status of a received packet, stored in the last bytes of its header slot by ibvUdpRecv.
struct udp_slot_tail {
    uint32_t wire_len; ///< Bytes received on the wire, Ethernet header included, FCS excluded.
    uint32_t flags;    ///< UDP_SLOT_CSUM_OK when the NIC verified the IP and UDP checksums.
};
#pragma pack(pop)

static_assert(sizeof(ethernet_header) == 14);
static_assert(sizeof(ip_header) == 20);
static_assert(sizeof(udp_header) == 8);
static_assert(sizeof(udp_packet_headers) == 42);
static_assert(sizeof(udp_slot_tail) == 8);

constexpr uint32_t UDP_SLOT_CSUM_OK = 1;
constexpr uint16_t ETHER_TYPE_IPV4 = 0x0800;
constexpr uint8_t IP_PROTOCOL_UDP = 17;

/// The tail of the header slot that starts at @p slot.
inline udp_slot_tail* udp_slot_tail_of(uint8_t* slot, size_t header_slot) {
    return reinterpret_cast<udp_slot_tail*>(slot + header_slot - sizeof(udp_slot_tail));
}

/**
 * @brief Fill the Ethernet, IPv4 and UDP headers of a packet.
 *
 * IP addresses are in network byte order, as @c inet_pton returns them; ports are in host order.
 * The IP header checksum is computed. The UDP checksum is left zero, which IPv4 allows and which
 * the NIC overwrites when ibvUdpSend sends with checksum offload.
 *
 * @param udp_payload_len Bytes following the UDP header.
 */
inline void fill_udp_headers(udp_packet_headers* h, const uint8_t src_mac[6],
                             const uint8_t dst_mac[6], uint32_t src_ip, uint32_t dst_ip,
                             uint16_t src_port, uint16_t dst_port, uint16_t udp_payload_len) {
    memcpy(h->eth.dest_mac, dst_mac, 6);
    memcpy(h->eth.src_mac, src_mac, 6);
    h->eth.ether_type = htons(ETHER_TYPE_IPV4);
    h->ip.version_ihl = 0x45;
    h->ip.type_of_service = 0;
    h->ip.total_length = htons(sizeof(ip_header) + sizeof(udp_header) + udp_payload_len);
    h->ip.identification = 0;
    h->ip.flags_fragment_offset = htons(0x4000); // don't fragment
    h->ip.ttl = 64;
    h->ip.protocol = IP_PROTOCOL_UDP;
    h->ip.header_checksum = 0;
    h->ip.src_addr = src_ip;
    h->ip.dest_addr = dst_ip;
    h->ip.header_checksum = (uint16_t)in_cksum((uint16_t*)&h->ip, sizeof(ip_header));
    h->udp.src_port = htons(src_port);
    h->udp.dest_port = htons(dst_port);
    h->udp.length = htons(sizeof(udp_header) + udp_payload_len);
    h->udp.checksum = 0;
}

/// Parse "aa:bb:cc:dd:ee:ff" into @p mac. Returns false on a malformed string.
inline bool parse_mac(const std::string& s, uint8_t mac[6]) {
    unsigned v[6];
    char extra;
    if (sscanf(s.c_str(), "%x:%x:%x:%x:%x:%x%c", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &extra)
        != 6)
        return false;
    for (int i = 0; i < 6; i++) {
        if (v[i] > 255)
            return false;
        mac[i] = (uint8_t)v[i];
    }
    return true;
}

/// Parse a dotted-quad IPv4 address into network byte order. Returns false on a malformed string.
inline bool parse_ipv4(const std::string& s, uint32_t& addr) {
    struct in_addr a;
    if (inet_pton(AF_INET, s.c_str(), &a) != 1)
        return false;
    addr = a.s_addr;
    return true;
}

/**
 * @brief The layout of packets across a header buffer and a payload buffer.
 *
 * Frame i of the header buffer holds @c packets_per_frame slots of @c header_slot bytes, and frame
 * i of the payload buffer the same number of slots of @c payload_slot bytes. Packet j of the frame
 * is the first @c header_size bytes of header slot j followed by the first @c payload_size bytes of
 * payload slot j. The last 8 bytes of every header slot are its @c udp_slot_tail.
 *
 * @conf header_size   Int. On-wire header bytes per packet (Ethernet + IP + UDP + application).
 * @conf header_slot   Int. Header slot stride; default the first multiple of 64 that holds
 *                     header_size plus the tail.
 * @conf payload_size  Int. Payload bytes per packet.
 * @conf payload_slot  Int. Payload slot stride; default payload_size.
 */
struct udp_slot_geometry {
    size_t header_size;
    size_t header_slot;
    size_t payload_size;
    size_t payload_slot;
    size_t packets_per_frame;

    /// Bytes of one packet on the wire, FCS excluded.
    size_t wire_len() const {
        return header_size + payload_size;
    }

    static size_t default_header_slot(size_t header_size) {
        return (header_size + sizeof(udp_slot_tail) + 63) / 64 * 64;
    }

    /// Validate the sizes against the two frame sizes and derive @c packets_per_frame.
    /// @throw std::invalid_argument on an inconsistent layout.
    static udp_slot_geometry make(size_t header_size, size_t header_slot, size_t payload_size,
                                  size_t payload_slot, size_t header_frame_size,
                                  size_t payload_frame_size) {
        if (header_size < sizeof(udp_packet_headers))
            throw std::invalid_argument(fmt::format(
                fmt("header_size {:d} is smaller than the Ethernet/IP/UDP headers ({:d})"),
                header_size, sizeof(udp_packet_headers)));
        if (header_size + sizeof(udp_slot_tail) > header_slot)
            throw std::invalid_argument(
                fmt::format(fmt("header_slot {:d} cannot hold header_size {:d} plus the {:d} "
                                "byte tail"),
                            header_slot, header_size, sizeof(udp_slot_tail)));
        if (payload_size == 0 || payload_size > payload_slot)
            throw std::invalid_argument(
                fmt::format(fmt("payload_size {:d} must be between 1 and payload_slot {:d}"),
                            payload_size, payload_slot));
        if (header_frame_size % header_slot != 0 || payload_frame_size % payload_slot != 0)
            throw std::invalid_argument(
                fmt::format(fmt("frame sizes ({:d}, {:d}) must be multiples of the slot sizes "
                                "({:d}, {:d})"),
                            header_frame_size, payload_frame_size, header_slot, payload_slot));
        udp_slot_geometry g{header_size, header_slot, payload_size, payload_slot,
                            header_frame_size / header_slot};
        if (g.packets_per_frame == 0 || g.packets_per_frame != payload_frame_size / payload_slot)
            throw std::invalid_argument(
                fmt::format(fmt("header frames hold {:d} packets but payload frames hold {:d}"),
                            g.packets_per_frame, payload_frame_size / payload_slot));
        return g;
    }

    /// Read the sizes from the stage config and validate them against the two buffers.
    static udp_slot_geometry from_config(const kotekan::Config& config,
                                         const std::string& unique_name, const Buffer* header_buf,
                                         const Buffer* payload_buf) {
        if (header_buf->num_frames != payload_buf->num_frames)
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: header buffer {:s} has {:d} frames but payload buffer "
                                "{:s} has {:d}"),
                            unique_name, header_buf->buffer_name, header_buf->num_frames,
                            payload_buf->buffer_name, payload_buf->num_frames));
        size_t header_size = config.get<size_t>(unique_name, "header_size");
        size_t payload_size = config.get<size_t>(unique_name, "payload_size");
        try {
            return make(header_size,
                        config.get_default<size_t>(unique_name, "header_slot",
                                                   default_header_slot(header_size)),
                        payload_size,
                        config.get_default<size_t>(unique_name, "payload_slot", payload_size),
                        header_buf->frame_size, payload_buf->frame_size);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(fmt::format(fmt("{:s}: {:s}"), unique_name, e.what()));
        }
    }
};

#endif /* UDP_PACKET_HPP */
