#define BOOST_TEST_MODULE "test_galileo_codes"

#include "galileoE1Code.hpp" // for generate_e1b_code, generate_e1c_code
#include "galileoE6Code.hpp" // for generate_e6b_code, generate_e6c_code, generate_e6c_secondary

#include <array>
#include <boost/test/included/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>

using galileo::generate_e1b_code;
using galileo::generate_e1c_code;
using galileo::generate_e6b_code;
using galileo::generate_e6c_code;
using galileo::generate_e6c_secondary;

namespace {

// First n chips packed MSB-first: chip -1 -> bit 1, +1 -> bit 0. Under the ICD's logic-level
// convention (0 -> +1, 1 -> -1) this is the leading hex of the published table entry.
template<size_t N>
uint32_t first_bits(const std::array<int8_t, N>& code, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i)
        v = (v << 1) | ((code[(size_t)i] == -1) ? 1u : 0u);
    return v;
}

template<size_t N>
int balance(const std::array<int8_t, N>& code) {
    return std::accumulate(code.begin(), code.end(), 0);
}

} // namespace

// First 24 chips are the first six hex chars of the OS SIS ICD Annex C.7/C.8 entries; every
// E1 code has 2046 ones in 4092 chips.
BOOST_AUTO_TEST_CASE(e1b_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_e1b_code(1), 24), 0xF5D710u);
    BOOST_CHECK_EQUAL(balance(generate_e1b_code(1)), 0);
    BOOST_CHECK_EQUAL(first_bits(generate_e1b_code(50), 24), 0x97051Fu);
    BOOST_CHECK_EQUAL(balance(generate_e1b_code(50)), 0);
}

BOOST_AUTO_TEST_CASE(e1c_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_e1c_code(1), 24), 0xB39340u);
    BOOST_CHECK_EQUAL(balance(generate_e1c_code(1)), 0);
    BOOST_CHECK_EQUAL(first_bits(generate_e1c_code(50), 24), 0xADDCEDu);
    BOOST_CHECK_EQUAL(balance(generate_e1c_code(50)), 0);
}

// First 24 chips are the first six hex chars of the E6-B/C Codes Technical Note entries; every
// E6 code has 2557 ones in 5115 chips, which only holds when the padding bit is left out.
BOOST_AUTO_TEST_CASE(e6b_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_e6b_code(1), 24), 0xE6648Au);
    BOOST_CHECK_EQUAL(balance(generate_e6b_code(1)), 1);
    BOOST_CHECK_EQUAL(first_bits(generate_e6b_code(50), 24), 0xFE62E6u);
    BOOST_CHECK_EQUAL(balance(generate_e6b_code(50)), 1);
}

BOOST_AUTO_TEST_CASE(e6c_first24_and_balance) {
    BOOST_CHECK_EQUAL(first_bits(generate_e6c_code(1), 24), 0xF5A3D6u);
    BOOST_CHECK_EQUAL(balance(generate_e6c_code(1)), 1);
    BOOST_CHECK_EQUAL(first_bits(generate_e6c_code(50), 24), 0xCE11D9u);
    BOOST_CHECK_EQUAL(balance(generate_e6c_code(50)), 1);
}

// ChannelizedReplicaBank::set_prn relies on the throw to refuse a PRN with no code.
BOOST_AUTO_TEST_CASE(invalid_prn_throws) {
    BOOST_CHECK_THROW(generate_e1b_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_e1b_code(51), std::out_of_range);
    BOOST_CHECK_THROW(generate_e1c_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_e1c_code(51), std::out_of_range);
    BOOST_CHECK_THROW(generate_e6b_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_e6b_code(51), std::out_of_range);
    BOOST_CHECK_THROW(generate_e6c_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_e6c_code(51), std::out_of_range);
    BOOST_CHECK_THROW(generate_e6c_secondary(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_e6c_secondary(51), std::out_of_range);
}
