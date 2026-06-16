#define BOOST_TEST_MODULE "test_gps_ca_code"

#include "gpsCACode.hpp" // for generate_ca_code, g2_phase_taps, CA_CODE_LENGTH

#include <array>                            // for array
#include <boost/test/included/unit_test.hpp>
#include <cstdint>   // for int8_t
#include <stdexcept> // for out_of_range
#include <vector>    // for vector

using gps::CA_CODE_LENGTH;
using gps::generate_ca_code;

namespace {

// Periodic correlation of a bipolar code with a cyclically-shifted copy of
// another, summed over the full 1023-chip period.
int periodic_correlation(const std::array<int8_t, CA_CODE_LENGTH>& a,
                         const std::array<int8_t, CA_CODE_LENGTH>& b, int shift) {
    int sum = 0;
    for (int i = 0; i < CA_CODE_LENGTH; ++i)
        sum += a[i] * b[(i + shift) % CA_CODE_LENGTH];
    return sum;
}

// Reconstruct the first n logical chips (1/0) from the bipolar code, using the
// IS-GPS-200 mapping logical 1 -> -1, logical 0 -> +1.
int first_chips_octal_bits_to_int(const std::array<int8_t, CA_CODE_LENGTH>& code, int n) {
    int value = 0;
    for (int i = 0; i < n; ++i) {
        const int logical = (code[i] == -1) ? 1 : 0;
        value = (value << 1) | logical;
    }
    return value;
}

} // namespace

// The strongest convention-independent check: a length-1023 Gold code is
// balanced (number of +1 and -1 chips differ by exactly one).
BOOST_AUTO_TEST_CASE(ca_codes_are_balanced) {
    for (int prn = 1; prn <= 32; ++prn) {
        const auto code = generate_ca_code(prn);
        int ones = 0;
        for (int8_t c : code)
            ones += (c == -1) ? 1 : 0;
        const int zeros = CA_CODE_LENGTH - ones;
        BOOST_CHECK_MESSAGE(std::abs(ones - zeros) == 1,
                            "PRN " << prn << " not balanced: " << ones << " vs " << zeros);
    }
}

// Auto-correlation: peak at zero lag is 1023; every off-peak periodic
// auto-correlation value lies in the Gold-code three-valued set {-1,-65,63}
// (for n=10, t(n)=65).
BOOST_AUTO_TEST_CASE(ca_autocorrelation_is_three_valued) {
    for (int prn = 1; prn <= 32; ++prn) {
        const auto code = generate_ca_code(prn);
        BOOST_CHECK_EQUAL(periodic_correlation(code, code, 0), CA_CODE_LENGTH);
        for (int shift = 1; shift < CA_CODE_LENGTH; ++shift) {
            const int x = periodic_correlation(code, code, shift);
            BOOST_CHECK_MESSAGE(x == -1 || x == -65 || x == 63,
                                "PRN " << prn << " autocorr at shift " << shift << " = " << x);
        }
    }
}

// Cross-correlation between distinct PRNs is also three-valued {-1,-65,63}.
BOOST_AUTO_TEST_CASE(ca_crosscorrelation_is_three_valued) {
    const std::vector<std::pair<int, int>> pairs = {{1, 2}, {1, 19}, {5, 25}, {17, 32}};
    for (const auto& [p, q] : pairs) {
        const auto cp = generate_ca_code(p);
        const auto cq = generate_ca_code(q);
        for (int shift = 0; shift < CA_CODE_LENGTH; ++shift) {
            const int x = periodic_correlation(cp, cq, shift);
            BOOST_CHECK_MESSAGE(x == -1 || x == -65 || x == 63,
                                "PRN " << p << "x" << q << " crosscorr at shift " << shift << " = "
                                       << x);
        }
    }
}

// Pin the exact code phase against published IS-GPS-200 "first 10 chips"
// octal values (Table 3-Ia), confirming both the tap table and the bipolar
// mapping. The leading-chip-first logical bits packed MSB-first equal the
// listed octal.
BOOST_AUTO_TEST_CASE(ca_first_chips_match_is_gps_200) {
    struct Expected {
        int prn;
        int octal; // value of the published octal literal, read as base-8
    };
    // Octal literals from IS-GPS-200 Table 3-Ia.
    const std::vector<Expected> expected = {
        {1, 01440},  {2, 01620},  {3, 01710},  {4, 01744},
        {5, 01133},  {6, 01455},  {7, 01131},  {19, 01633},
    };
    for (const auto& e : expected) {
        const auto code = generate_ca_code(e.prn);
        BOOST_CHECK_MESSAGE(first_chips_octal_bits_to_int(code, 10) == e.octal,
                            "PRN " << e.prn << " first-10 chips mismatch");
    }
}

BOOST_AUTO_TEST_CASE(ca_prn_range_is_checked) {
    BOOST_CHECK_THROW(generate_ca_code(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_ca_code(33), std::out_of_range);
    BOOST_CHECK_NO_THROW(generate_ca_code(1));
    BOOST_CHECK_NO_THROW(generate_ca_code(32));
}
