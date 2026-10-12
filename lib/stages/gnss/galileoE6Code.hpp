#ifndef GALILEO_E6_CODE_HPP
#define GALILEO_E6_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * @file
 * @brief Galileo E6 primary codes (E6-B data, E6-C pilot) and the per-PRN E6-C secondary.
 *
 * Memory codes: 5115-chip sequences at 5.115 Mcps (1 ms period), BPSK(5), published as hex
 * tables in the EU "Galileo E6-B/C Codes Technical Note", Issue 1 (January 2019) and
 * transcribed via gnss-sdr's Galileo_E6.h, as for the E1 tables. Not LFSR-generated.
 *
 * Each hex entry is 1279 chars = 5116 bits for a 5115-chip code: the low bit of the last char
 * is padding, not a chip, and every entry ends in 0, 4, 8 or C.
 *
 * E6-B carries the HAS (High Accuracy Service) message at 1000 sps. E6-C is the dataless
 * pilot, with a per-PRN 100-chip secondary (one chip per 1 ms primary period, 100 ms).
 * PRN range 1..50.
 */
namespace galileo {

constexpr int E6_CODE_LENGTH = 5115;
constexpr int E6C_SECONDARY_LENGTH = 100;

/// E6-B (data channel, HAS 1000 sps) primary code, +-1 chips.
std::array<int8_t, E6_CODE_LENGTH> generate_e6b_code(int prn);

/// E6-C (dataless pilot) primary code, +-1 chips.
std::array<int8_t, E6_CODE_LENGTH> generate_e6c_code(int prn);

/// E6-C secondary code, +-1, per PRN (100 chips = 100 ms).
std::array<int8_t, E6C_SECONDARY_LENGTH> generate_e6c_secondary(int prn);

} // namespace galileo

#endif
