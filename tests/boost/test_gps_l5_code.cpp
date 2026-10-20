#define BOOST_TEST_MODULE "test_gps_l5_code"

#include "gpsL5Code.hpp" // for generate_l5i_code, generate_l5q_code, L5_NH10/NH20

#include <array>
#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <numeric>
#include <stdexcept>

using gps::generate_l5i_code;
using gps::generate_l5q_code;
using gps::L5_CODE_LENGTH;

namespace {

// First n chips packed MSB-first: chip -1 -> bit 1, +1 -> bit 0, as in the PocketSDR
// reference values below.
uint32_t first_bits(const std::array<int8_t, L5_CODE_LENGTH>& code, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i)
        v = (v << 1) | ((code[i] == -1) ? 1u : 0u);
    return v;
}

int balance(const std::array<int8_t, L5_CODE_LENGTH>& code) {
    return std::accumulate(code.begin(), code.end(), 0);
}

} // namespace

// First chips from PocketSDR sdr_code.py (IS-GPS-705); balances from this generator.
BOOST_AUTO_TEST_CASE(l5i_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_l5i_code(1), 24), 066124275u);
    BOOST_CHECK_EQUAL(balance(generate_l5i_code(1)), -2);
    BOOST_CHECK_EQUAL(first_bits(generate_l5i_code(5), 24), 005041072u);
    BOOST_CHECK_EQUAL(balance(generate_l5i_code(5)), -2);
    BOOST_CHECK_EQUAL(first_bits(generate_l5i_code(10), 24), 041006103u);
    BOOST_CHECK_EQUAL(balance(generate_l5i_code(10)), 0);
}

BOOST_AUTO_TEST_CASE(l5q_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_l5q_code(1), 24), 063131310u);
    BOOST_CHECK_EQUAL(balance(generate_l5q_code(1)), 2);
    BOOST_CHECK_EQUAL(first_bits(generate_l5q_code(5), 24), 054417330u);
    BOOST_CHECK_EQUAL(balance(generate_l5q_code(5)), 0);
    BOOST_CHECK_EQUAL(first_bits(generate_l5q_code(10), 24), 047557674u);
    BOOST_CHECK_EQUAL(balance(generate_l5q_code(10)), 2);
}

// XA cancels in a PRN's I5 * Q5, leaving the product of two phases of the same XB sequence.
// XB is not short-cycled, so that product repeats every 8191 chips within the code period. The
// first-chip pins above cannot see the wrap, which only affects the code's last chips.
BOOST_AUTO_TEST_CASE(xb_wraps_at_8191) {
    for (int prn = 1; prn <= 32; ++prn) {
        const auto i5 = generate_l5i_code(prn);
        const auto q5 = generate_l5q_code(prn);
        int mismatches = 0;
        for (int i = 0; i + 8191 < L5_CODE_LENGTH; ++i)
            mismatches += (i5[i] * q5[i] != i5[i + 8191] * q5[i + 8191]);
        BOOST_CHECK_MESSAGE(mismatches == 0, "PRN " << prn << ": " << mismatches << " mismatches");
    }
}

BOOST_AUTO_TEST_CASE(codes_are_bipolar_full_length) {
    auto i5 = generate_l5i_code(7);
    auto q5 = generate_l5q_code(7);
    BOOST_REQUIRE_EQUAL(i5.size(), (size_t)L5_CODE_LENGTH);
    for (int8_t c : i5)
        BOOST_CHECK(c == 1 || c == -1);
    for (int8_t c : q5)
        BOOST_CHECK(c == 1 || c == -1);
    BOOST_CHECK(i5 != q5); // I5 and Q5 differ (independent XB advances)
}

BOOST_AUTO_TEST_CASE(neuman_hofman_overlays) {
    BOOST_REQUIRE_EQUAL(gps::L5_NH10.size(), (size_t)10);
    BOOST_REQUIRE_EQUAL(gps::L5_NH20.size(), (size_t)20);
    const std::array<int8_t, 10> nh10 = {1, 1, 1, 1, -1, -1, 1, -1, 1, -1};
    BOOST_CHECK(gps::L5_NH10 == nh10);
    BOOST_CHECK_EQUAL(std::accumulate(gps::L5_NH20.begin(), gps::L5_NH20.end(), 0), 4);
}

BOOST_AUTO_TEST_CASE(invalid_prn_throws) {
    BOOST_CHECK_THROW(generate_l5i_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_l5i_code(211), std::out_of_range);
    BOOST_CHECK_THROW(generate_l5q_code(0), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(sbas_prn_in_the_same_family) {
    // WAAS PRN 131: a valid entry of the 1..210 table, distinct from the GPS codes.
    auto i131 = generate_l5i_code(131);
    auto q131 = generate_l5q_code(131);
    BOOST_REQUIRE_EQUAL(i131.size(), (size_t)10230);
    BOOST_CHECK(i131 != q131);
    BOOST_CHECK(i131 != generate_l5i_code(1));
    for (int8_t c : i131)
        BOOST_CHECK(c == 1 || c == -1);
}
