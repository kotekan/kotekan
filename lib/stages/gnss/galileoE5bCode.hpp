#ifndef GALILEO_E5B_CODE_HPP
#define GALILEO_E5B_CODE_HPP
/**
 * @file galileoE5bCode.hpp
 * @brief Galileo E5b primary + secondary code generation (OS SIS ICD).
 *
 * E5b sits at 1207.14 MHz -- the SAME sky frequency as BeiDou B2b, so the mid-band dongle
 * capture serves both. Like E5a it is a 10230-chip / 10.23 Mcps (1 ms) primary formed as the
 * product of two 14-stage LFSRs, but the E5b register polynomials differ from E5a: X1 tap
 * 0o64021 (vs E5a 0o40503), and per-component X2 taps 0o51445 (E5bI) / 0o43143 (E5bQ) with
 * per-PRN X2 start values.
 *
 *   E5b-Q  the dataless PILOT; per-PRN 100-chip CS100 secondary (100 ms cycle) -- best for
 *          acquisition + long coherent integration.
 *   E5b-I  the DATA channel carrying Galileo I/NAV; 4-chip CS4 secondary ('E', SAME for every
 *          satellite) = one 4 ms I/NAV symbol. Reuses the existing I/NAV decoder.
 *
 * Constants sourced from PocketSDR (sdr_code.py gen_code_E5BI/E5BQ) and pre-resolved into this
 * LFSR convention (taps/starts bit-reversed), BIT-EXACT-verified for all 50 PRNs on both
 * components against PocketSDR by python/scripts/gnss/e5b_code_check.py -- the same shared
 * lfsr14 / all-ones-X1 / per-PRN-X2-start machinery as galileoE5aCode.cpp.
 */
#include <array>
#include <cstdint>

namespace galileo {

constexpr int E5B_CODE_LENGTH = 10230;

/// E5b-Q (pilot) primary code, PRN 1..50, as +-1 chips.
std::array<int8_t, E5B_CODE_LENGTH> generate_e5bq_code(int prn);

/// E5b-Q CS100 secondary overlay (100 chips, one per primary period), PRN 1..50.
std::array<int8_t, 100> e5bq_secondary(int prn);

/// E5b-I (DATA) primary code, PRN 1..50, as +-1 chips. Same X1 as E5b-Q, its own X2 tap +
/// per-PRN start. Carries Galileo I/NAV (same message as E1-B), so its D-component reuses the
/// existing inav_predictor.
std::array<int8_t, E5B_CODE_LENGTH> generate_e5bi_code(int prn);

/// E5b-I CS4 secondary overlay (4 chips = 4 ms = ONE I/NAV symbol), SAME sequence for every
/// satellite (unlike the per-PRN E5b-Q CS100). The I/NAV data symbol rides on top.
std::array<int8_t, 4> e5bi_secondary();

} // namespace galileo
#endif // GALILEO_E5B_CODE_HPP
