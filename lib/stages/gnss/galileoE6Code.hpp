#ifndef GALILEO_E6_CODE_HPP
#define GALILEO_E6_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * Galileo E6 primary spreading codes (E6-B data / E6-C pilot) + the per-PRN E6-C secondary.
 *
 * These are MEMORY codes -- 5115-chip pseudorandom sequences at 5.115 Mcps (1 ms period),
 * BPSK(5), published as hex tables in the EU "Galileo E6-B/C Codes Technical Note", Issue 1,
 * January 2019 (released to enable the High Accuracy Service). Not LFSR-generable. Same
 * provenance route as our E1B/E1C tables (transcribed via gnss-sdr's Galileo_E6.h).
 *
 * ⚠️ The hex is 1279 chars = 5116 bits for a 5115-chip code: chars 0..1277 contribute all
 * four bits (5112) and char 1278 contributes only its TOP THREE. The last bit is PADDING,
 * not a chip -- gnss-sdr's own reader says so ("last bit is filled up with a zero"), and
 * every table entry ends in {0,4,8,C}. Getting this wrong shifts nothing but silently drops
 * or invents a chip at the wrap, which costs lock, not correctness of the first 5112.
 *
 * E6-B carries the HAS (High Accuracy Service) message at 1000 sps -- a DATA channel with a
 * live message (HAS has been operational since Jan 2023), unlike GPS L1C-D. E6-C is the
 * dataless pilot and carries a PER-PRN 100-chip secondary (one chip per 1 ms primary period,
 * 100 ms total) -- note per-PRN, unlike E1-C's shared CS25.
 * PRN range 1..50.
 */
namespace galileo {

constexpr int E6_CODE_LENGTH = 5115;
constexpr int E6C_SECONDARY_LENGTH = 100;

/// E6-B (data channel, HAS 1000 sps) primary code, +-1 chips.
std::array<int8_t, E6_CODE_LENGTH> generate_e6b_code(int prn);

/// E6-C (dataless pilot) primary code, +-1 chips.
std::array<int8_t, E6_CODE_LENGTH> generate_e6c_code(int prn);

/// E6-C secondary code, +-1, PER-PRN (100 chips = 100 ms).
std::array<int8_t, E6C_SECONDARY_LENGTH> generate_e6c_secondary(int prn);

} // namespace galileo

#endif
