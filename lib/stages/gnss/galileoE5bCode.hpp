#ifndef GALILEO_E5B_CODE_HPP
#define GALILEO_E5B_CODE_HPP
/**
 * @file galileoE5bCode.hpp
 * @brief Galileo E5b primary and secondary codes (OS SIS ICD).
 *
 * E5b (1207.14 MHz, the BeiDou B2b carrier) has the E5a structure: a 10230-chip, 10.23 Mcps
 * (1 ms) primary formed as the product of two 14-stage LFSRs, X1 starting all-ones and X2 at a
 * per-PRN start value. The polynomials differ from E5a's: X1 0o64021, X2 0o51445 (E5b-I) or
 * 0o43143 (E5b-Q).
 *
 *   E5b-Q  dataless pilot; per-PRN 100-chip CS100 secondary (100 ms).
 *   E5b-I  data channel carrying I/NAV; 4-chip CS4 secondary ('E', the same for every
 *          satellite), one period per 4 ms I/NAV symbol.
 *
 * Taps and start values are from PocketSDR's transcription of the ICD, converted to the LFSR
 * convention of galileoE5aCode.cpp. python/scripts/gnss/e5b_code_check.py keeps a copy of them
 * that must match, and reads CS100_HEX and CS100_E5BQ_HEX from the .cpp files by name.
 */
#include <array>
#include <cstdint>

namespace galileo {

constexpr int E5B_CODE_LENGTH = 10230;

/// E5b-Q (pilot) primary code, PRN 1..50, as +-1 chips.
std::array<int8_t, E5B_CODE_LENGTH> generate_e5bq_code(int prn);

/// E5b-Q CS100 secondary overlay (100 chips, one per primary period), PRN 1..50.
std::array<int8_t, 100> e5bq_secondary(int prn);

/// E5b-I (data) primary code, PRN 1..50, as +-1 chips. Same X1 as E5b-Q, with its own X2 tap
/// and start values. Carries I/NAV, the same message as E1-B, which
/// python/scripts/gnss/inav_predictor.py decodes for both.
std::array<int8_t, E5B_CODE_LENGTH> generate_e5bi_code(int prn);

/// E5b-I CS4 secondary overlay (4 chips = 4 ms = one I/NAV symbol), the same for every
/// satellite. The I/NAV data symbol multiplies it.
std::array<int8_t, 4> e5bi_secondary();

} // namespace galileo
#endif // GALILEO_E5B_CODE_HPP
