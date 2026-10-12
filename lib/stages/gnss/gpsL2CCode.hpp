/**
 * @file
 * @brief GPS L2C (L2 CM / L2 CL) spreading-code generation (IS-GPS-200).
 *
 * L2C carries two codes, time-multiplexed chip by chip into a 1.023 Mcps stream (CM on even
 * chips, CL on odd):
 *  - CM (civil moderate): 10230 chips at 511.5 kcps, 20 ms period, carries CNAV data.
 *  - CL (civil long): 767250 chips at 511.5 kcps, 1.5 s period, dataless pilot.
 *
 * Both come from the same 27-stage modular (Galois) shift register (IS-GPS-200 polynomial
 * 1112225171 octal) and differ only in the per-PRN initial state and the reset length. The
 * generator follows GNSS-SDR and PocketSDR (feedback mask 0445112474 octal, output = register
 * LSB). No kotekan dependencies.
 */

#ifndef GPS_L2C_CODE_HPP
#define GPS_L2C_CODE_HPP

#include <cstdint> // for int8_t
#include <vector>  // for vector

namespace gps {

constexpr int L2CM_CODE_LENGTH = 10230;  ///< CM chips (20 ms @ 511.5 kcps)
constexpr int L2CL_CODE_LENGTH = 767250; ///< CL chips (1.5 s @ 511.5 kcps)
constexpr int L2C_PRN_MIN = 1;
constexpr int L2C_PRN_MAX = 32; ///< initial-state tables cover PRN 1..32

/**
 * @brief Generate the L2 CM code for a PRN as bipolar +1 / -1.
 * @param prn  PRN signal number, 1..32.
 * @returns    length-10230 vector of int8_t, each +1 or -1 (LSB 0 -> +1, 1 -> -1).
 * @throws std::out_of_range if @p prn is outside 1..32.
 */
std::vector<int8_t> generate_l2cm_code(int prn);

/**
 * @brief Generate the L2 CL code (dataless pilot) for a PRN as bipolar +1 / -1.
 * @param prn  PRN signal number, 1..32.
 * @returns    length-767250 vector of int8_t.
 * @throws std::out_of_range if @p prn is outside 1..32.
 */
std::vector<int8_t> generate_l2cl_code(int prn);

} // namespace gps

#endif // GPS_L2C_CODE_HPP
