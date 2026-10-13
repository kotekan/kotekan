/**
 * @file
 * @brief GPS L1 C/A Gold-code generation (IS-GPS-200).
 *
 * Generates the 1023-chip C/A code for PRN 1..32 from the two 10-stage maximal-length LFSRs
 * (G1, G2) of IS-GPS-200 and the per-PRN G2 phase-select taps. No kotekan dependencies.
 */

#ifndef GPS_CA_CODE_HPP
#define GPS_CA_CODE_HPP

#include <array>   // for array
#include <cstdint> // for int8_t

namespace gps {

/// Length of one GPS L1 C/A code period, in chips (= 1 ms at 1.023 Mchip/s).
constexpr int CA_CODE_LENGTH = 1023;

/// Valid PRN range of the C/A tap table.
constexpr int CA_PRN_MIN = 1;
constexpr int CA_PRN_MAX = 32;

/**
 * @brief Generate one 1023-chip C/A code as bipolar +1 / -1.
 *
 * Both LFSRs start all-ones and clock 1023 times. The logical chip (G1 stage 10 XOR the two
 * tapped G2 cells) maps to bipolar with the IS-GPS-200 convention
 * logical 1 -> -1, logical 0 -> +1.
 *
 * @param prn  PRN signal number, 1..32.
 * @returns    length-1023 array of int8_t, each +1 or -1.
 * @throws std::out_of_range if @p prn is outside 1..32.
 */
std::array<int8_t, CA_CODE_LENGTH> generate_ca_code(int prn);

} // namespace gps

#endif // GPS_CA_CODE_HPP
