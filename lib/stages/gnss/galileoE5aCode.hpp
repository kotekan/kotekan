#ifndef GALILEO_E5A_CODE_HPP
#define GALILEO_E5A_CODE_HPP
/**
 * @file galileoE5aCode.hpp
 * @brief Galileo E5a primary + secondary code generation (OS SIS ICD).
 *
 * E5a-Q (the dataless pilot, 1176.45 MHz): 10230-chip primary at 10.23 Mcps (1 ms),
 * generated as the product of two 14-stage LFSRs (base register 1 all-ones, base
 * register 2 with a per-PRN start value), plus the per-PRN 100-chip CS100 secondary
 * (one chip per primary period, 100 ms cycle).
 *
 * Tables sourced from PocketSDR (the gpsL5Code.cpp reference) and SKY-VALIDATED
 * 2026-07-15: unseeded acquisition of E6/E9/E10/E11/E31/E36 at 40-46 dB-Hz from a raw
 * 1176.45 MHz capture (python/scripts/gnss/l5_band_decode.py, the bit-exactness
 * reference for this implementation).
 */
#include <array>
#include <cstdint>

namespace galileo {

constexpr int E5A_CODE_LENGTH = 10230;

/// E5a-Q (pilot) primary code, PRN 1..50, as +-1 chips.
std::array<int8_t, E5A_CODE_LENGTH> generate_e5aq_code(int prn);

/// E5a-Q CS100 secondary overlay (100 chips, one per primary period), PRN 1..50.
std::array<int8_t, 100> e5aq_secondary(int prn);

} // namespace galileo
#endif // GALILEO_E5A_CODE_HPP
