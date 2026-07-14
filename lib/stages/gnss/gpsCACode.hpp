/**
 * @file
 * @brief GPS L1 C/A Gold-code generation (IS-GPS-200).
 *
 * Pure, dependency-free helpers used by the GPS replica correlator stage.
 * Generates the per-PRN 1023-chip C/A code from the two 10-bit maximal-length
 * LFSRs (G1, G2) defined in IS-GPS-200, using the full per-PRN G2
 * phase-select tap table. No kotekan dependencies, so this unit is
 * independently testable (see tests/boost/test_gps_ca_code.cpp).
 */

#ifndef GPS_CA_CODE_HPP
#define GPS_CA_CODE_HPP

#include <array>   // for array
#include <cstdint> // for int8_t
#include <utility> // for pair

namespace gps {

/// Length of one GPS L1 C/A code period, in chips (= 1 ms at 1.023 Mchip/s).
constexpr int CA_CODE_LENGTH = 1023;

/// Lowest / highest valid PRN signal number for the legacy C/A table.
constexpr int CA_PRN_MIN = 1;
constexpr int CA_PRN_MAX = 32;

/**
 * @brief G2 phase-select tap pair for a PRN.
 *
 * Returns the 1-indexed G2 register cells (tap_a, tap_b) whose XOR forms the
 * G2 output for the given PRN, per IS-GPS-200 Table 3-Ia. The code phase of a
 * PRN is set entirely by this tap selection (equivalently a fixed delay on a
 * single G2 sequence).
 *
 * @param prn  PRN signal number, 1..32.
 * @returns    (tap_a, tap_b), each in 1..10.
 * @throws std::out_of_range if @p prn is outside 1..32.
 */
std::pair<int, int> g2_phase_taps(int prn);

/**
 * @brief Generate one 1023-chip C/A code as bipolar +1 / -1.
 *
 * Both LFSRs start all-ones and clock 1023 times. The logical chip output
 * (G1 last cell XOR the selected G2 phase) is mapped to bipolar with the
 * IS-GPS-200 convention logical 1 -> -1, logical 0 -> +1.
 *
 * @param prn  PRN signal number, 1..32.
 * @returns    length-1023 array of int8_t, each +1 or -1.
 * @throws std::out_of_range if @p prn is outside 1..32.
 */
std::array<int8_t, CA_CODE_LENGTH> generate_ca_code(int prn);

} // namespace gps

#endif // GPS_CA_CODE_HPP
