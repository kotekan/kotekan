#define BOOST_TEST_MODULE "test_udp_packet"

#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "network_functions.hpp"     // for in_cksum
#include "udpPacket.hpp"             // for udp_packet_headers, fill_udp_headers, udp_slot_geometry

#include <arpa/inet.h> // for htons
#include <boost/test/included/unit_test.hpp>
#include <stdexcept> // for invalid_argument
#include <stdint.h>  // for uint8_t, uint16_t, uint32_t
#include <string.h>  // for memcpy, memcmp

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

BOOST_AUTO_TEST_CASE(struct_sizes) {
    BOOST_CHECK_EQUAL(sizeof(ethernet_header), 14u);
    BOOST_CHECK_EQUAL(sizeof(ip_header), 20u);
    BOOST_CHECK_EQUAL(sizeof(udp_header), 8u);
    BOOST_CHECK_EQUAL(sizeof(udp_packet_headers), 42u);
    BOOST_CHECK_EQUAL(sizeof(udp_slot_tail), 8u);
}

BOOST_AUTO_TEST_CASE(ip_checksum_known_vector) {
    // The IPv4 header checksum example from RFC 1071 / Wikipedia: checksum field 0xb861.
    const uint8_t bytes[20] = {0x45, 0x00, 0x00, 0x73, 0x00, 0x00, 0x40, 0x00, 0x40, 0x11,
                               0x00, 0x00, 0xc0, 0xa8, 0x00, 0x01, 0xc0, 0xa8, 0x00, 0xc7};
    uint16_t words[10];
    memcpy(words, bytes, 20);
    uint16_t sum = (uint16_t)in_cksum(words, 20);
    const uint8_t expected[2] = {0xb8, 0x61};
    BOOST_CHECK_EQUAL(memcmp(&sum, expected, 2), 0);
}

BOOST_AUTO_TEST_CASE(fill_headers) {
    const uint8_t src_mac[6] = {0x30, 0xc5, 0x99, 0x3f, 0x09, 0x1e};
    const uint8_t dst_mac[6] = {0x10, 0x70, 0xfd, 0x53, 0x05, 0x06};
    uint32_t src_ip, dst_ip;
    BOOST_REQUIRE(parse_ipv4("10.0.0.1", src_ip));
    BOOST_REQUIRE(parse_ipv4("10.0.0.2", dst_ip));
    const uint8_t src_ip_bytes[4] = {10, 0, 0, 1};
    BOOST_CHECK_EQUAL(memcmp(&src_ip, src_ip_bytes, 4), 0);

    udp_packet_headers h;
    fill_udp_headers(&h, src_mac, dst_mac, src_ip, dst_ip, 4000, 4001, 6166);

    BOOST_CHECK_EQUAL(memcmp(h.eth.dest_mac, dst_mac, 6), 0);
    BOOST_CHECK_EQUAL(memcmp(h.eth.src_mac, src_mac, 6), 0);
    BOOST_CHECK_EQUAL(h.eth.ether_type, htons(0x0800));
    BOOST_CHECK_EQUAL(h.ip.version_ihl, 0x45);
    BOOST_CHECK_EQUAL(h.ip.total_length, htons(20 + 8 + 6166));
    BOOST_CHECK_EQUAL(h.ip.flags_fragment_offset, htons(0x4000));
    BOOST_CHECK_EQUAL(h.ip.ttl, 64);
    BOOST_CHECK_EQUAL(h.ip.protocol, 17);
    BOOST_CHECK_EQUAL(h.ip.src_addr, src_ip);
    BOOST_CHECK_EQUAL(h.ip.dest_addr, dst_ip);
    BOOST_CHECK_EQUAL(h.udp.src_port, htons(4000));
    BOOST_CHECK_EQUAL(h.udp.dest_port, htons(4001));
    BOOST_CHECK_EQUAL(h.udp.length, htons(8 + 6166));
    BOOST_CHECK_EQUAL(h.udp.checksum, 0);
    // A valid IP header sums to zero, checksum field included.
    uint16_t words[10];
    memcpy(words, &h.ip, 20);
    BOOST_CHECK_EQUAL((uint16_t)in_cksum(words, 20), 0);
}

BOOST_AUTO_TEST_CASE(mac_parsing) {
    uint8_t mac[6];
    BOOST_CHECK(parse_mac("30:c5:99:3F:09:1e", mac));
    const uint8_t expected[6] = {0x30, 0xc5, 0x99, 0x3f, 0x09, 0x1e};
    BOOST_CHECK_EQUAL(memcmp(mac, expected, 6), 0);
    BOOST_CHECK(!parse_mac("30:c5:99:3f:09", mac));
    BOOST_CHECK(!parse_mac("30:c5:99:3f:09:1e:ff", mac));
    BOOST_CHECK(!parse_mac("30:c5:99:3f:09:1ex", mac));
    BOOST_CHECK(!parse_mac("300:c5:99:3f:09:1e", mac));
    uint32_t ip;
    BOOST_CHECK(!parse_ipv4("10.0.0", ip));
    BOOST_CHECK(!parse_ipv4("cobalt", ip));
}

BOOST_AUTO_TEST_CASE(slot_tail_placement) {
    uint8_t slot[128];
    BOOST_CHECK_EQUAL((void*)udp_slot_tail_of(slot, 128), (void*)(slot + 120));
    BOOST_CHECK_EQUAL((void*)udp_slot_tail_of(slot, 64), (void*)(slot + 56));
}

BOOST_AUTO_TEST_CASE(geometry_defaults) {
    BOOST_CHECK_EQUAL(udp_slot_geometry::default_header_slot(42), 64u);
    BOOST_CHECK_EQUAL(udp_slot_geometry::default_header_slot(56), 64u);
    BOOST_CHECK_EQUAL(udp_slot_geometry::default_header_slot(57), 128u);
    BOOST_CHECK_EQUAL(udp_slot_geometry::default_header_slot(64), 128u);
}

BOOST_AUTO_TEST_CASE(geometry_accepts_consistent_layout) {
    auto g = udp_slot_geometry::make(64, 128, 6144, 6144, 128 * 1024, 6144 * 1024);
    BOOST_CHECK_EQUAL(g.packets_per_frame, 1024u);
    BOOST_CHECK_EQUAL(g.wire_len(), 6208u);
    // A payload slot larger than the payload is allowed.
    g = udp_slot_geometry::make(50, 64, 4000, 4096, 64 * 16, 4096 * 16);
    BOOST_CHECK_EQUAL(g.packets_per_frame, 16u);
}

BOOST_AUTO_TEST_CASE(geometry_rejects_inconsistent_layout) {
    // Header smaller than Ethernet + IP + UDP.
    BOOST_CHECK_THROW(udp_slot_geometry::make(40, 64, 6144, 6144, 64 * 8, 6144 * 8),
                      std::invalid_argument);
    // No room for the tail.
    BOOST_CHECK_THROW(udp_slot_geometry::make(60, 64, 6144, 6144, 64 * 8, 6144 * 8),
                      std::invalid_argument);
    // Payload larger than its slot, or empty.
    BOOST_CHECK_THROW(udp_slot_geometry::make(64, 128, 6145, 6144, 128 * 8, 6144 * 8),
                      std::invalid_argument);
    BOOST_CHECK_THROW(udp_slot_geometry::make(64, 128, 0, 6144, 128 * 8, 6144 * 8),
                      std::invalid_argument);
    // Frame not a multiple of the slot.
    BOOST_CHECK_THROW(udp_slot_geometry::make(64, 128, 6144, 6144, 128 * 8 + 1, 6144 * 8),
                      std::invalid_argument);
    // Different packet counts per frame.
    BOOST_CHECK_THROW(udp_slot_geometry::make(64, 128, 6144, 6144, 128 * 8, 6144 * 9),
                      std::invalid_argument);
    // Zero packets per frame.
    BOOST_CHECK_THROW(udp_slot_geometry::make(64, 128, 6144, 6144, 0, 0), std::invalid_argument);
}
