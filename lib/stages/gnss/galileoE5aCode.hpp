#ifndef GALILEO_E5A_CODE_HPP
#define GALILEO_E5A_CODE_HPP
/**
 * @file galileoE5aCode.hpp
 * @brief Galileo E5a primary and secondary codes (OS SIS ICD).
 *
 * E5a (1176.45 MHz) primaries are 10230 chips at 10.23 Mcps (1 ms), the product of two
 * 14-stage LFSRs: base register 1 starts all-ones, base register 2 at a per-PRN start value.
 * E5a-Q is the dataless pilot, with a per-PRN 100-chip CS100 secondary; E5a-I carries F/NAV,
 * with a 20-chip CS20 secondary shared by every satellite. Each secondary chip spans one
 * primary period. Start values and secondary tables are from PocketSDR's transcription of the
 * ICD.
 */
#include <array>
#include <cstdint>

namespace galileo {

constexpr int E5A_CODE_LENGTH = 10230;

/// E5a-Q (pilot) primary code, PRN 1..50, as +-1 chips.
std::array<int8_t, E5A_CODE_LENGTH> generate_e5aq_code(int prn);

/// E5a-Q CS100 secondary overlay (100 chips, one per primary period), PRN 1..50.
std::array<int8_t, 100> e5aq_secondary(int prn);

/// E5a-I (data) primary code, PRN 1..50, as +-1 chips. Same generator as E5a-Q; only the
/// base-register-2 start values differ.
std::array<int8_t, E5A_CODE_LENGTH> generate_e5ai_code(int prn);

/// E5a-I CS20 secondary overlay (20 chips = 20 ms = one F/NAV symbol), the same for every
/// satellite. The F/NAV data symbol multiplies it.
std::array<int8_t, 20> e5ai_secondary();

} // namespace galileo
#endif // GALILEO_E5A_CODE_HPP
