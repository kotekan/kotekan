/**
 * @file
 * @brief GPS L5 ranging-code generation (IS-GPS-705).
 *
 * L5 (1176.45 MHz) has two components in phase quadrature: I5 (data, CNAV) and Q5 (dataless
 * pilot). Each primary code is 10230 chips at 10.23 Mcps (1 ms), the XOR of a common XA
 * sequence (13-stage LFSR short-cycled at 8190 chips) and an XB sequence (13-stage
 * maximal-length LFSR, period 8191) started a per-PRN number of chips in (IS-GPS-705 Table 3-I).
 * XB is not short-cycled, so it wraps at 8191 within the code. Each component also carries a
 * Neuman-Hofman overlay: NH10 (10 ms) on I5, NH20 (20 ms) on Q5.
 *
 * Tables transcribed from IS-GPS-705 and checked against PocketSDR (sdr_code.py). No kotekan
 * dependencies.
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
