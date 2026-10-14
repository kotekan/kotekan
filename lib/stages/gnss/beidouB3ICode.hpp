#ifndef BEIDOU_B3I_CODE_HPP
#define BEIDOU_B3I_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * BeiDou B3I ranging code (1268.52 MHz) and the shared 20-chip NH secondary.
 *
 * Broadcast by BDS-2 and BDS-3 alike: 10230 chips at 10.23 Mcps (1 ms), BPSK(10), from two
 * 13-stage LFSRs per BDS-SIS-ICD-B3I v1.0. Data-only like B1I: acquired directly and nav-wiped,
 * with the same 50 bps D1 message and the same 20-chip NH secondary (one chip per 1 ms period,
 * 20 ms per symbol). PRN 1..63.
 *
 * G1 is not free-running: its 13-stage m-sequence (period 8191) is shorter than the code, so
 * the ICD truncates it to 8190 chips -- the register goes back to all-ones instead of shifting
 * out of the phase 1111111111100 (stages 1..13), which is the last chip of each cycle (see
 * generate_b3i_code). Without the reset the first 8190 chips are unchanged and only the tail
 * diverges, so the code correlates weakly instead of not at all; resetting one chip early (an
 * 8189 cycle, #178) costs a uniform 1.9 dB the same way. The per-PRN diversity is in G2's
 * initial state, not in an output tap as on B1I.
 *
 * Sign convention: ICD logic 0 -> +1, 1 -> -1, as in beidouB1ICode.hpp.
 */
namespace beidou {

constexpr int B3I_CODE_LENGTH = 10230;
constexpr int B3I_NH_LENGTH = 20;

/// B3I ranging code (PRN 1..63) as +-1 chips.
std::array<int8_t, B3I_CODE_LENGTH> generate_b3i_code(int prn);

/// The 20-chip D1 NH secondary, identical to B1I_NH20 (00000100110101001110); a separate symbol
/// so each signal's overlay registry row names its own sequence.
extern const std::array<int8_t, B3I_NH_LENGTH> B3I_NH20;

} // namespace beidou

#endif
