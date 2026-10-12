#define BOOST_TEST_MODULE "test_gps_l2c_code"

#include "gpsL2CCode.hpp" // for generate_l2cm_code, generate_l2cl_code

#include <boost/test/included/unit_test.hpp>
#include <cstdint>   // for int8_t, uint32_t
#include <stdexcept> // for out_of_range
#include <vector>    // for vector

using gps::generate_l2cl_code;
using gps::generate_l2cm_code;

namespace {

// First n chips packed MSB-first: chip -1 -> bit 1, +1 -> bit 0 (the register LSB), as in
// PocketSDR's first-24-bit listing.
uint32_t first_bits(const std::vector<int8_t>& code, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i)
        v = (v << 1) | ((code[i] == -1) ? 1u : 0u);
    return v;
}

int count_minus(const std::vector<int8_t>& code) {
    int c = 0;
    for (int8_t x : code)
        c += (x == -1) ? 1 : 0;
    return c;
}

} // namespace

// Reference first-24-chip octals and balance counts, from PocketSDR's L2C generator.
BOOST_AUTO_TEST_CASE(l2cm_matches_reference) {
    const auto cm1 = generate_l2cm_code(1);
    BOOST_CHECK_EQUAL(cm1.size(), (size_t)10230);
    BOOST_CHECK_EQUAL(first_bits(cm1, 24), 012757036u);
    BOOST_CHECK_EQUAL(count_minus(cm1), 5115); // exactly balanced (N/2)

    const auto cm7 = generate_l2cm_code(7);
    BOOST_CHECK_EQUAL(first_bits(cm7, 24), 07533713u);
    BOOST_CHECK_EQUAL(count_minus(cm7), 5115);
}

BOOST_AUTO_TEST_CASE(l2cl_matches_reference) {
    const auto cl1 = generate_l2cl_code(1);
    BOOST_CHECK_EQUAL(cl1.size(), (size_t)767250);
    BOOST_CHECK_EQUAL(first_bits(cl1, 24), 024676104u);
    BOOST_CHECK_EQUAL(count_minus(cl1), 383625); // exactly balanced (N/2)
}

BOOST_AUTO_TEST_CASE(l2c_prn_range_checked) {
    BOOST_CHECK_THROW(generate_l2cm_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_l2cl_code(33), std::out_of_range);
    BOOST_CHECK_NO_THROW(generate_l2cm_code(32));
    BOOST_CHECK_NO_THROW(generate_l2cl_code(1));
}
