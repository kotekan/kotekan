#define BOOST_TEST_MODULE "test_beidou_b3i_code"

#include "beidouB3ICode.hpp" // for generate_b3i_code, B3I_NH20, B3I_CODE_LENGTH

#include <array>
#include <boost/test/included/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>

using beidou::B3I_NH20;
using beidou::generate_b3i_code;

// Every pin below comes from two independent implementations that agree with each other on all
// 63 codes and read BDS-SIS-ICD-B3I 4.3 the same way: Peter Monta's GNSS-DSP-tools
// (gnsstools/beidou/b3i.py) and PocketSDR's sdr_code.py (G1 = the first 8190 chips of the
// free-running register, repeated). Neither is ours and neither descends from gnss-sdr.

namespace {

// Chips [a, b) packed MSB-first at the ICD logic level: -1 -> 1, +1 -> 0.
template<size_t N>
uint32_t bits(const std::array<int8_t, N>& code, int a, int b) {
    uint32_t v = 0;
    for (int i = a; i < b; ++i)
        v = (v << 1) | ((code[(size_t)i] == -1) ? 1u : 0u);
    return v;
}

template<size_t N>
int balance(const std::array<int8_t, N>& code) {
    return std::accumulate(code.begin(), code.end(), 0);
}

// FNV-1a over the +-1 chips as bytes (+1 -> 0x01, -1 -> 0xFF): a whole-code pin that fits here.
template<size_t N>
uint64_t fnv64(const std::array<int8_t, N>& code) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int8_t c : code) {
        h ^= (uint8_t)c;
        h *= 0x100000001b3ull;
    }
    return h;
}

} // namespace

// The first 24 chips pin the register conventions (tap order, output end, the G2 table).
BOOST_AUTO_TEST_CASE(first24_and_balance) {
    BOOST_CHECK_EQUAL(bits(generate_b3i_code(1), 0, 24), 0x0052E0u);
    BOOST_CHECK_EQUAL(balance(generate_b3i_code(1)), 16);
    BOOST_CHECK_EQUAL(bits(generate_b3i_code(30), 0, 24), 0x000F09u);
    BOOST_CHECK_EQUAL(balance(generate_b3i_code(30)), -116);
    BOOST_CHECK_EQUAL(bits(generate_b3i_code(63), 0, 24), 0x8DDD69u);
    BOOST_CHECK_EQUAL(balance(generate_b3i_code(63)), 0);
}

// G1's truncated cycle is 8190 chips: the phase 1111111111100 is output as chip 8189 and the
// register goes back to all-ones instead of shifting (ICD 4.3). The 20 chips straddling that
// point are the discriminating pin -- an 8189-chip cycle (the generator before #178, and
// gnss-sdr's) passes first24_and_balance and fails here from chip 8189 on.
BOOST_AUTO_TEST_CASE(g1_cycle_is_8190_chips) {
    BOOST_CHECK_EQUAL(bits(generate_b3i_code(1), 8180, 8200), 0x45C00u);
    BOOST_CHECK_EQUAL(bits(generate_b3i_code(30), 8180, 8200), 0x00800u);
    BOOST_CHECK_EQUAL(bits(generate_b3i_code(63), 8180, 8200), 0x5151Bu);
}

// All 10230 chips of five codes, including both halves of the G2 table (PRN 38 is BDS-3).
BOOST_AUTO_TEST_CASE(whole_code_fingerprints) {
    BOOST_CHECK_EQUAL(fnv64(generate_b3i_code(1)), 0x9E9AB084AEBF34A1ull);
    BOOST_CHECK_EQUAL(fnv64(generate_b3i_code(7)), 0xD660CC2335158CEDull);
    BOOST_CHECK_EQUAL(fnv64(generate_b3i_code(19)), 0x33B5E3B2157B1251ull);
    BOOST_CHECK_EQUAL(fnv64(generate_b3i_code(38)), 0xC1DED289A615CC41ull);
    BOOST_CHECK_EQUAL(fnv64(generate_b3i_code(63)), 0x7D322EABEB531E15ull);
}

// The D1 NH secondary 00000100110101001110, shared with B1I.
BOOST_AUTO_TEST_CASE(nh20) {
    BOOST_CHECK_EQUAL(bits(B3I_NH20, 0, 20), 0x04D4Eu);
}

// ChannelizedReplicaBank::set_prn relies on the throw to refuse a PRN with no code.
BOOST_AUTO_TEST_CASE(invalid_prn_throws) {
    BOOST_CHECK_THROW(generate_b3i_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_b3i_code(64), std::out_of_range);
}
