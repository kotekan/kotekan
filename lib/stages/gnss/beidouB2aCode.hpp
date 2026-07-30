#ifndef BEIDOU_B2A_CODE_HPP
#define BEIDOU_B2A_CODE_HPP
/**
 * @file beidouB2aCode.hpp
 * @brief BeiDou-3 B2a primary + secondary code generation (BDS-SIS-ICD-B2a).
 *
 * B2a-pilot (dataless, 1176.45 MHz): 10230-chip primary at 10.23 Mcps (1 ms) from two
 * 13-stage LFSRs (G1 all-ones truncated at 8190 + restart, G2 with per-PRN start
 * values), plus the per-PRN 100-chip secondary (Weil-1021, truncated) -- one chip per
 * primary period, 100 ms cycle. BDS-3 only (like B1C); B2a is transmitted by MEO/IGSO
 * BDS-3 satellites.
 *
 * Tables sourced from PocketSDR and SKY-VALIDATED 2026-07-15: unseeded acquisition of
 * C21/C22/C26/C36 at 40-45 dB-Hz from a raw 1176.45 MHz capture
 * (python/scripts/gnss/l5_band_decode.py, the bit-exactness reference).
 */
#include <array>
#include <cstdint>

namespace beidou {

constexpr int B2A_CODE_LENGTH = 10230;

/// B2a-pilot primary code, PRN 1..63, as +-1 chips.
std::array<int8_t, B2A_CODE_LENGTH> generate_b2ap_code(int prn);

/// B2a-pilot 100-chip secondary overlay (one per primary period), PRN 1..63.
std::array<int8_t, 100> b2ap_secondary(int prn);

/// B2a-DATA primary code, PRN 1..63, as +-1 chips. Same G1/G2 structure as the pilot, its own
/// generator polynomials (shares the G2 start table). Carries B-CNAV2 (S5 D-component #3).
std::array<int8_t, B2A_CODE_LENGTH> generate_b2ad_code(int prn);

/// B2a-DATA 5-chip secondary overlay (5 ms = one B-CNAV2 symbol), SAME for every satellite.
std::array<int8_t, 5> b2ad_secondary();

} // namespace beidou
#endif // BEIDOU_B2A_CODE_HPP
