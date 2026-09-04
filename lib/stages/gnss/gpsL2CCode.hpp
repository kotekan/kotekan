/**
 * @file
 * @brief GPS L2C (L2 CM / L2 CL) spreading-code generation (IS-GPS-200).
 *
 * The L2C civil signal carries two codes, time-division-multiplexed chip-by-chip
 * into a 1.023 Mcps stream (CM on even chips, CL on odd):
 *  - CM ("civil moderate"): 10230 chips @ 511.5 kcps, 20 ms period, carries the
 *    CNAV data.
 *  - CL ("civil long"): 767250 chips @ 511.5 kcps, 1.5 s period, dataless pilot.
 *
 * Both are produced by the same 27-stage modular (Galois) shift register
 * (IS-GPS-200 polynomial 1112225171 octal), differing only in the per-PRN
 * initial state and the reset length. This implementation is a direct port of
 * the generator used by GNSS-SDR and PocketSDR (Galois feedback mask 0445112474
 * octal, output = register LSB); the feedback mask and the CM initial states
 * agree between those two independent implementations, and the generated codes
 * are cross-checked against PocketSDR reference chips in the unit test.
 *
 * Pure / dependency-free, like gpsCACode -- see tests/boost/test_gps_l2c_code.cpp.
 */

#ifndef GPS_L2C_CODE_HPP
#define GPS_L2C_CODE_HPP

#include <cstdint> // for int8_t
#include <vector>  // for vector

namespace gps {

constexpr int L2CM_CODE_LENGTH = 10230;     ///< CM chips (20 ms @ 511.5 kcps)
constexpr int L2CL_CODE_LENGTH = 767250;    ///< CL chips (1.5 s @ 511.5 kcps)
constexpr int L2C_PRN_MIN = 1;
constexpr int L2C_PRN_MAX = 32;             ///< table covers the GPS constellation

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
