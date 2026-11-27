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

// The register's state at the last chip of @p code, stepping a Galois register with feedback
// @p mask from @p init (output = LSB, LSB 0 -> +1); 0, which the register never holds, if a chip
// is not the register's output.
uint32_t final_state(const std::vector<int8_t>& code, uint32_t init, uint32_t mask) {
    uint32_t r = init;
    for (size_t i = 0; i < code.size(); ++i) {
        if (code[i] != ((r & 1u) ? -1 : +1))
            return 0;
        if (i + 1 < code.size())
            r = (r >> 1u) ^ ((r & 1u) ? mask : 0u);
    }
    return r;
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
    for (int prn : {33, 192, 203}) {
        BOOST_CHECK_THROW(generate_l2cm_code(prn), std::out_of_range);
        BOOST_CHECK_THROW(generate_l2cl_code(prn), std::out_of_range);
    }
    BOOST_CHECK_NO_THROW(generate_l2cm_code(193));
    BOOST_CHECK_NO_THROW(generate_l2cl_code(202));
}

// QZSS L2C, IS-QZSS-PNT-006 Table 3.2.4-1: every PRN's CM and CL chips are the output of the
// register of Figure 3.2.4-2 (1 + X^3 + X^4 + X^5 + X^6 + X^9 + X^11 + X^13 + X^16 + X^19 + X^21
// + X^24 + X^27; the X^k tap is mask bit k - 1) run from the table's initial state, and the
// register holds the table's final state at the last chip.
BOOST_AUTO_TEST_CASE(qzss_l2c_matches_the_published_final_states) {
    struct Row {
        int prn;
        uint32_t cm_init, cl_init, cm_final, cl_final;
    };
    const Row rows[] = {
        {193, 0204244652, 0235525312, 0415505547, 0722043377},
        {194, 0202133131, 0507056307, 0705146647, 0240751052},
        {195, 0714351204, 0221720061, 0006215430, 0375674043},
        {196, 0657127260, 0520470122, 0371216176, 0166677056},
        {197, 0130567507, 0603764120, 0645502771, 0123055362},
        {198, 0670517677, 0145604016, 0455175106, 0707017665},
        {199, 0607275514, 0051237167, 0127161032, 0437503241},
        {200, 0045413633, 0033326347, 0470332401, 0275605155},
        {201, 0212645405, 0534627074, 0252026355, 0376333266},
        {202, 0613700455, 0645230164, 0113771472, 0467523556},
    };
    uint32_t mask = 0;
    for (int k : {3, 4, 5, 6, 9, 11, 13, 16, 19, 21, 24, 27})
        mask |= 1u << (k - 1);
    for (const Row& r : rows) {
        const auto cm = generate_l2cm_code(r.prn);
        const auto cl = generate_l2cl_code(r.prn);
        BOOST_CHECK_EQUAL(cm.size(), (size_t)10230);
        BOOST_CHECK_EQUAL(cl.size(), (size_t)767250);
        BOOST_CHECK_MESSAGE(final_state(cm, r.cm_init, mask) == r.cm_final, "CM " << r.prn);
        BOOST_CHECK_MESSAGE(final_state(cl, r.cl_init, mask) == r.cl_final, "CL " << r.prn);
    }
}
