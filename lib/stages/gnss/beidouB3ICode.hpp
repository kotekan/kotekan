#ifndef BEIDOU_B3I_CODE_HPP
#define BEIDOU_B3I_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * BeiDou B3I ranging code (1268.52 MHz) + the shared 20-chip NH secondary.
 *
 * LEGACY BeiDou, the B1I sibling one band down: 10230 chips at 10.23 Mcps (1 ms period),
 * BPSK(10), from two 13-stage LFSRs, per BDS-SIS-ICD-B3I v1.0. DATA-ONLY (no pilot) -> acquired
 * directly and nav-wiped, with the same 50 bps D1 message and the SAME 20-chip NH secondary as
 * B1I (one chip per 1 ms period => 20 ms/symbol). PRN 1..63. Same 1 ms record geometry as
 * B2b/E5b/L5, and the same 10.23 Mcps chip rate as B2b, so the front-end and record shapes are
 * already proven here -- only the code generator is new.
 *
 * ⚠️ THE TRAP (different from B1I's): G1 is NOT a free-running m-sequence. Its period would
 * exceed the 10230-chip code, so the ICD RESETS G1 to all-ones the moment it reaches the state
 * 0011111111111. Omit that reset and the first ~8190 chips still look right while the tail
 * silently diverges -- a code that correlates weakly rather than not at all, which is the
 * hardest failure to attribute. The per-PRN diversity is in G2's INITIAL STATE (63 entries),
 * not in an output tap as B1I does it.
 *
 * Sign convention: ICD logic level 0 -> +1, 1 -> -1, matching beidouB1ICode (a global inversion
 * vs gnss-sdr; immaterial, see that header).
 */
namespace beidou {

constexpr int B3I_CODE_LENGTH = 10230;
constexpr int B3I_NH_LENGTH = 20;

/// B3I ranging code (PRN 1..63) as +-1 chips.
std::array<int8_t, B3I_CODE_LENGTH> generate_b3i_code(int prn);

/// The 20-chip D1 NH secondary -- IDENTICAL to B1I_NH20 (00000100110101001110), kept as its own
/// symbol so the two chains' overlay registry rows stay independent.
extern const std::array<int8_t, B3I_NH_LENGTH> B3I_NH20;

} // namespace beidou

#endif
