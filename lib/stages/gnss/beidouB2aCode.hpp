#ifndef BEIDOU_B2A_CODE_HPP
#define BEIDOU_B2A_CODE_HPP
/**
 * @file beidouB2aCode.hpp
 * @brief BeiDou-3 B2a primary and secondary codes (BDS-SIS-ICD-B2a).
 *
 * B2a-pilot (dataless, 1176.45 MHz): 10230-chip primary at 10.23 Mcps (1 ms) from two
 * 13-stage LFSRs (G1 from all-ones, truncated at 8190 chips and restarted; G2 from per-PRN
 * start values), plus a per-PRN 100-chip secondary (truncated Weil-1021), one chip per primary
 * period, 100 ms cycle. B2a-data has the same structure with its own polynomials and a shared
 * 5-chip secondary. Transmitted by BDS-3 MEO/IGSO satellites only. Tables from PocketSDR.
 */
#include <array>
#include <cstdint>

namespace beidou {

constexpr int B2A_CODE_LENGTH = 10230;

/// B2a-pilot primary code, PRN 1..63, as +-1 chips.
std::array<int8_t, B2A_CODE_LENGTH> generate_b2ap_code(int prn);

/// B2a-pilot 100-chip secondary overlay (one chip per primary period), PRN 1..63.
std::array<int8_t, 100> b2ap_secondary(int prn);

/// B2a-data primary code, PRN 1..63, as +-1 chips. Same G1/G2 structure as the pilot with its
/// own generator polynomials; its G2 start values match the pilot's for PRN 1..60 only.
/// Carries B-CNAV2.
std::array<int8_t, B2A_CODE_LENGTH> generate_b2ad_code(int prn);

/// B2a-data 5-chip secondary overlay (5 ms = one B-CNAV2 symbol), the same for every satellite.
std::array<int8_t, 5> b2ad_secondary();

} // namespace beidou
#endif // BEIDOU_B2A_CODE_HPP
