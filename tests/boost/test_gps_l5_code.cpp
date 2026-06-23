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

// First n chips packed MSB-first: chip -1 -> bit 1, +1 -> bit 0 (matches the
// PocketSDR reference packing used to derive the expected octal values).
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

// Expected values cross-validated against PocketSDR sdr_code.py (IS-GPS-705).
BOOST_AUTO_TEST_CASE(l5i_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_l5i_code(1), 24), 066124275u);
    BOOST_CHECK_EQUAL(balance(generate_l5i_code(1)), 0);
    BOOST_CHECK_EQUAL(first_bits(generate_l5i_code(5), 24), 005041072u);
    BOOST_CHECK_EQUAL(balance(generate_l5i_code(5)), 40);
    BOOST_CHECK_EQUAL(first_bits(generate_l5i_code(10), 24), 041006103u);
    BOOST_CHECK_EQUAL(balance(generate_l5i_code(10)), 12);
}

BOOST_AUTO_TEST_CASE(l5q_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_l5q_code(1), 24), 063131310u);
    BOOST_CHECK_EQUAL(balance(generate_l5q_code(1)), -8);
    BOOST_CHECK_EQUAL(first_bits(generate_l5q_code(5), 24), 054417330u);
    BOOST_CHECK_EQUAL(balance(generate_l5q_code(5)), 44);
    BOOST_CHECK_EQUAL(first_bits(generate_l5q_code(10), 24), 047557674u);
    BOOST_CHECK_EQUAL(balance(generate_l5q_code(10)), -8);
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
    BOOST_CHECK_THROW(generate_l5i_code(33), std::out_of_range);
    BOOST_CHECK_THROW(generate_l5q_code(0), std::out_of_range);
}
