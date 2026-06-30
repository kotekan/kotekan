#define BOOST_TEST_MODULE "test_gps_l1c_code"

#include "gpsL1CCode.hpp" // for generate_l1cp_code, L1C_CODE_LENGTH

#include <array>
#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <numeric>
#include <stdexcept>

using gps::generate_l1cp_code;
using gps::L1C_CODE_LENGTH;

namespace {

// First n chips packed MSB-first: chip -1 -> bit 1, +1 -> bit 0.
uint32_t first_bits(const std::array<int8_t, L1C_CODE_LENGTH>& code, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i)
        v = (v << 1) | ((code[i] == -1) ? 1u : 0u);
    return v;
}

int balance(const std::array<int8_t, L1C_CODE_LENGTH>& code) {
    return std::accumulate(code.begin(), code.end(), 0);
}

} // namespace

// balance == 0 is a hard cross-check of the construction, independent of the (w,p) tables:
// for the Legendre length 10223 (== 3 mod 4) the quadratic-residue autocorrelation is -1, so the
// 10223-chip Weil code sums to -1, and the +1-sum 7-chip insertion brings it to exactly 0. A
// wrong Legendre/Weil would miss this. The first-24-chip octals are regression pins computed from
// the IS-GPS-800/PocketSDR algorithm + index tables (run PocketSDR sdr_code for a byte-level check).
BOOST_AUTO_TEST_CASE(l1cp_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_l1cp_code(1), 24), 05752067u);
    BOOST_CHECK_EQUAL(first_bits(generate_l1cp_code(5), 24), 042323273u);
    BOOST_CHECK_EQUAL(first_bits(generate_l1cp_code(10), 24), 052631623u);
    for (int prn : {1, 5, 10, 20, 32})
        BOOST_CHECK_EQUAL(balance(generate_l1cp_code(prn)), 0); // Legendre/insertion identity
}

BOOST_AUTO_TEST_CASE(codes_are_bipolar_full_length_and_distinct) {
    auto a = generate_l1cp_code(7);
    auto b = generate_l1cp_code(8);
    BOOST_REQUIRE_EQUAL(a.size(), (size_t)L1C_CODE_LENGTH);
    for (int8_t c : a)
        BOOST_CHECK(c == 1 || c == -1);
    BOOST_CHECK(a != b); // distinct PRNs (different Weil/insertion indices)
}

BOOST_AUTO_TEST_CASE(invalid_prn_throws) {
    BOOST_CHECK_THROW(generate_l1cp_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_l1cp_code(33), std::out_of_range);
}
