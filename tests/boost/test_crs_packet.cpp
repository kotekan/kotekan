#define BOOST_TEST_MODULE "test_crs_packet"

#include "crsPacket.hpp"             // for crs_packet_header, packet_headers, pack12, unpack12
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture

#include <boost/test/included/unit_test.hpp>
#include <random>   // for mt19937, uniform_int_distribution
#include <stddef.h> // for offsetof, size_t
#include <stdint.h> // for uint8_t, uint16_t
#include <string.h> // for memset
#include <vector>   // for vector

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

BOOST_AUTO_TEST_CASE(struct_layout) {
    BOOST_CHECK_EQUAL(sizeof(source_id_t), 2u);
    BOOST_CHECK_EQUAL(sizeof(crs_packet_header), 22u);
    BOOST_CHECK_EQUAL(sizeof(packet_headers), 64u);
    // The offsets the ibverbs stages and the aligner rely on within a header slot.
    BOOST_CHECK_EQUAL(offsetof(packet_headers, crs_hdr), 42u);
    BOOST_CHECK_EQUAL(offsetof(packet_headers, crs_hdr.stream_id), 46u);
    BOOST_CHECK_EQUAL(offsetof(packet_headers, crs_hdr.seq_number), 50u);
}

BOOST_AUTO_TEST_CASE(header_accessor) {
    uint8_t slot[128];
    memset(slot, 0, sizeof(slot));
    crs_packet_header* crs = crs_header_of(slot);
    crs->cookie = CRS_PACKET_COOKIE;
    crs->stream_id = 0x1234;
    crs->seq_number = 0x0102030405060708ull;
    BOOST_CHECK_EQUAL(slot[42], CRS_PACKET_COOKIE);
    BOOST_CHECK_EQUAL(slot[46], 0x34);
    BOOST_CHECK_EQUAL(slot[47], 0x12);
    BOOST_CHECK_EQUAL(slot[50], 0x08); // seq_number is little-endian in the slot
    BOOST_CHECK_EQUAL(slot[57], 0x01);
    const uint8_t* cslot = slot;
    BOOST_CHECK_EQUAL(crs_header_of(cslot)->seq_number, 0x0102030405060708ull);
}

BOOST_AUTO_TEST_CASE(pack12_known_bytes) {
    // LSB-first pairs: samples (0x123, 0x456) pack to bytes 23 61 45.
    uint8_t bytes[3] = {0, 0, 0};
    pack12(bytes, 0, 0x123);
    pack12(bytes, 1, 0x456);
    BOOST_CHECK_EQUAL(bytes[0], 0x23);
    BOOST_CHECK_EQUAL(bytes[1], 0x61);
    BOOST_CHECK_EQUAL(bytes[2], 0x45);
    BOOST_CHECK_EQUAL(unpack12(bytes, 0), 0x123);
    BOOST_CHECK_EQUAL(unpack12(bytes, 1), 0x456);
}

BOOST_AUTO_TEST_CASE(pack12_round_trip_all_values) {
    uint8_t bytes[3];
    for (uint16_t v = 0; v < 4096; v++) {
        memset(bytes, 0xa5, sizeof(bytes));
        pack12(bytes, 0, v);
        BOOST_REQUIRE_EQUAL(unpack12(bytes, 0), v);
        BOOST_REQUIRE_EQUAL(unpack12(bytes, 1), 0xa5a); // neighbor unchanged
        memset(bytes, 0xa5, sizeof(bytes));
        pack12(bytes, 1, v);
        BOOST_REQUIRE_EQUAL(unpack12(bytes, 1), v);
        BOOST_REQUIRE_EQUAL(unpack12(bytes, 0), 0x5a5);
    }
}

BOOST_AUTO_TEST_CASE(pack12_payload_round_trip) {
    // A full 4096-sample / 6144-byte payload survives pack/unpack in random write order.
    const size_t n = 4096;
    std::vector<uint8_t> payload(n * 3 / 2, 0);
    std::vector<uint16_t> samples(n);
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint16_t> dist(0, 4095);
    for (size_t i = 0; i < n; i++)
        samples[i] = dist(rng);
    for (size_t i = 0; i < n; i += 2) // even then odd of each pair
        pack12(payload.data(), i, samples[i]);
    for (size_t i = 1; i < n; i += 2)
        pack12(payload.data(), i, samples[i]);
    for (size_t i = 0; i < n; i++)
        BOOST_REQUIRE_EQUAL(unpack12(payload.data(), i), samples[i]);
}
