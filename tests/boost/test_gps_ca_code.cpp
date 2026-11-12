#define BOOST_TEST_MODULE "test_gps_ca_code"

#include "gpsCACode.hpp" // for generate_ca_code, CA_CODE_LENGTH

#include <array> // for array
#include <boost/test/included/unit_test.hpp>
#include <cstdint>   // for int8_t
#include <stdexcept> // for out_of_range
#include <vector>    // for vector

using gps::CA_CODE_LENGTH;
using gps::generate_ca_code;

namespace {

// Periodic correlation: sum over i of a[i] * b[(i + shift) mod 1023].
int periodic_correlation(const std::array<int8_t, CA_CODE_LENGTH>& a,
                         const std::array<int8_t, CA_CODE_LENGTH>& b, int shift) {
    int sum = 0;
    for (int i = 0; i < CA_CODE_LENGTH; ++i)
        sum += a[i] * b[(i + shift) % CA_CODE_LENGTH];
    return sum;
}

// The first n chips as logical bits (-1 -> 1, +1 -> 0), packed MSB-first.
int first_chips_octal_bits_to_int(const std::array<int8_t, CA_CODE_LENGTH>& code, int n) {
    int value = 0;
    for (int i = 0; i < n; ++i) {
        const int logical = (code[i] == -1) ? 1 : 0;
        value = (value << 1) | logical;
    }
    return value;
}

// The C/A code as G1 XOR the G2 sequence delayed by `delay` chips, built from the two LFSRs
// independently of generate_ca_code: the form the SBAS PRNs are specified in.
std::array<int8_t, CA_CODE_LENGTH> ca_by_delay(int delay) {
    std::array<uint8_t, 10> G1, G2;
    G1.fill(1);
    G2.fill(1);
    std::array<uint8_t, CA_CODE_LENGTH> g1, g2;
    for (int i = 0; i < CA_CODE_LENGTH; ++i) {
        g1[i] = G1[9];
        g2[i] = G2[9];
        const uint8_t fb1 = G1[2] ^ G1[9];
        const uint8_t fb2 = G2[1] ^ G2[2] ^ G2[5] ^ G2[7] ^ G2[8] ^ G2[9];
        for (int k = 9; k > 0; --k) {
            G1[k] = G1[k - 1];
            G2[k] = G2[k - 1];
        }
        G1[0] = fb1;
        G2[0] = fb2;
    }
    std::array<int8_t, CA_CODE_LENGTH> code;
    for (int i = 0; i < CA_CODE_LENGTH; ++i)
        code[i] = (g1[i] ^ g2[(i - delay + CA_CODE_LENGTH) % CA_CODE_LENGTH]) ? -1 : +1;
    return code;
}

// IS-GPS-200 Table 3-Ia's code delays (chips) for PRN 1..32, and DO-229's for SBAS PRN 120..158.
const std::vector<int> GPS_DELAY = {5,   6,   7,   8,   17,  18,  139, 140, 141, 251, 252,
                                    254, 255, 256, 257, 258, 469, 470, 471, 472, 473, 474,
                                    509, 512, 513, 514, 515, 516, 859, 860, 861, 862};
const std::vector<int> SBAS_DELAY = {145, 175,  52,  21,  237, 235,  886, 657, 634, 762,
                                     355, 1012, 176, 603, 130, 359,  595, 68,  386, 797,
                                     456, 499,  883, 307, 127, 211,  121, 118, 163, 628,
                                     853, 484,  289, 811, 202, 1021, 463, 568, 904};

} // namespace

// Every C/A code is balanced: its +1 and -1 counts differ by exactly one, whatever the sign
// convention.
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

// Autocorrelation is 1023 at zero lag, and every off-peak value lies in the Gold three-valued
// set {-1,-65,63} (n = 10, t(n) = 65).
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
            BOOST_CHECK_MESSAGE(x == -1 || x == -65 || x == 63, "PRN " << p << "x" << q
                                                                       << " crosscorr at shift "
                                                                       << shift << " = " << x);
        }
    }
}

// The first 10 chips, as logical bits packed MSB-first, equal the octal values listed in
// IS-GPS-200 Table 3-Ia; this pins both the tap table and the bipolar mapping.
BOOST_AUTO_TEST_CASE(ca_first_chips_match_is_gps_200) {
    struct Expected {
        int prn;
        int octal; // published octal value
    };
    const std::vector<Expected> expected = {
        {1, 01440}, {2, 01620}, {3, 01710}, {4, 01744},
        {5, 01133}, {6, 01455}, {7, 01131}, {19, 01633},
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


// The delay form is the tap table's: IS-GPS-200's delays reproduce PRN 1..32 chip for chip. That
// pins the delay convention generate_ca_code applies to the SBAS PRNs.
BOOST_AUTO_TEST_CASE(ca_delay_form_matches_the_taps) {
    for (int prn = 1; prn <= 32; ++prn)
        BOOST_CHECK_MESSAGE(ca_by_delay(GPS_DELAY[(size_t)prn - 1]) == generate_ca_code(prn),
                            "PRN " << prn);
}

// SBAS PRN 120..158: the delayed-G2 codes, balanced and three-valued against themselves, each
// other and a GPS code.
BOOST_AUTO_TEST_CASE(sbas_codes_are_delayed_gold_codes) {
    for (int prn = 120; prn <= 158; ++prn) {
        const auto code = generate_ca_code(prn);
        BOOST_CHECK_MESSAGE(code == ca_by_delay(SBAS_DELAY[(size_t)prn - 120]), "PRN " << prn);
        int ones = 0;
        for (int8_t c : code)
            ones += (c == -1) ? 1 : 0;
        BOOST_CHECK_MESSAGE(std::abs(2 * ones - CA_CODE_LENGTH) == 1, "PRN " << prn);
    }
    const std::vector<std::pair<int, int>> pairs = {{131, 131}, {131, 133}, {133, 135}, {135, 1}};
    for (const auto& [p, q] : pairs) {
        const auto cp = generate_ca_code(p);
        const auto cq = generate_ca_code(q);
        for (int shift = (p == q) ? 1 : 0; shift < CA_CODE_LENGTH; ++shift) {
            const int x = periodic_correlation(cp, cq, shift);
            BOOST_CHECK_MESSAGE(x == -1 || x == -65 || x == 63,
                                "PRN " << p << "x" << q << " at shift " << shift << " = " << x);
        }
    }
}

BOOST_AUTO_TEST_CASE(sbas_prn_range_is_checked) {
    BOOST_CHECK_THROW(generate_ca_code(119), std::out_of_range);
    BOOST_CHECK_THROW(generate_ca_code(159), std::out_of_range);
    BOOST_CHECK_NO_THROW(generate_ca_code(120));
    BOOST_CHECK_NO_THROW(generate_ca_code(158));
}
