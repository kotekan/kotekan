/**
 * @file
 * @brief GPS L5 ranging-code generation (IS-GPS-705).
 *
 * L5 (1176.45 MHz) has two components in phase quadrature: I5 (data, carries
 * CNAV) and Q5 (dataless pilot). Each is a 10230-chip primary code at 10.23
 * Mcps (1 ms period), formed as the XOR of a common XA sequence with a
 * per-PRN-shifted XB sequence -- XA is a 13-stage LFSR truncated/wrapped at 8190
 * chips, XB a 13-stage LFSR run to 10230, and the per-PRN code phase is the XB
 * advance (IS-GPS-705 Table 3-I). On top of the primary code each component
 * carries a Neuman-Hofman overlay: 10 chips (NH10, 10 ms) on I5, 20 chips
 * (NH20, 20 ms) on Q5.
 *
 * Tables transcribed from IS-GPS-705 and cross-validated against PocketSDR
 * (sdr_code.py). Pure, no kotekan deps; tested in
 * tests/boost/test_gps_l5_code.cpp.
 */

#ifndef GPS_L5_CODE_HPP
#define GPS_L5_CODE_HPP

#include <array>   // for array
#include <cstdint> // for int8_t

namespace gps {

/// Chips in one L5 primary-code period (1 ms at 10.23 Mcps).
constexpr int L5_CODE_LENGTH = 10230;
constexpr int L5_NH10_LENGTH = 10; ///< I5 Neuman-Hofman overlay length
constexpr int L5_NH20_LENGTH = 20; ///< Q5 Neuman-Hofman overlay length

/**
 * I5 (data) primary code for @p prn (1..32), bipolar +1/-1.
 * @throws std::out_of_range if prn is outside 1..32.
 */
std::array<int8_t, L5_CODE_LENGTH> generate_l5i_code(int prn);

/**
 * Q5 (pilot) primary code for @p prn (1..32), bipolar +1/-1.
 * @throws std::out_of_range if prn is outside 1..32.
 */
std::array<int8_t, L5_CODE_LENGTH> generate_l5q_code(int prn);

/// Neuman-Hofman secondary overlays, bipolar +1/-1 (NH10 on I5, NH20 on Q5).
extern const std::array<int8_t, L5_NH10_LENGTH> L5_NH10;
extern const std::array<int8_t, L5_NH20_LENGTH> L5_NH20;

} // namespace gps

#endif // GPS_L5_CODE_HPP
