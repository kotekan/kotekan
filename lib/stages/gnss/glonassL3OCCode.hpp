#ifndef GLONASS_L3OC_CODE_HPP
#define GLONASS_L3OC_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * GLONASS L3OC ranging codes (1202.025 MHz) -- the data (L3OCd) and pilot (L3OCp) components,
 * plus their secondaries.
 *
 * GLONASS's CDMA signal, and the FIRST GLONASS signal we track. Geometrically it is the
 * signal we already know best: 10230 chips at 10.23 Mcps, 1 ms period, BPSK(10) -- identical
 * record geometry to BeiDou B2b and B3I, so the front-end, record shapes and channelizer plan
 * are already proven. Only the code generator and the constellation plumbing are new.
 * Broadcast by GLONASS-K satellites ONLY (Celestrak marks them with a trailing 'K' in the
 * Uragan number, e.g. "COSMOS 2501 (702K)"); 7 of 28 operational satellites as of 2026-08.
 *
 * Construction (both components): a SHARED 14-stage LFSR DC1, multiplied by a PER-SATELLITE
 * 7-stage LFSR that differs only in its INITIAL STATE -- DC2 = state `prn` for the data
 * component, DC3 = state `prn + 64` for the pilot. The 7-stage polynomial is primitive, so its
 * 127 non-zero states are all cyclic shifts of ONE 127-chip m-sequence: per-satellite diversity
 * here is a code-phase offset of the short factor, not a different sequence. Data uses shifts
 * 1..63 and pilot 65..127, so the two families are disjoint and no data code aliases a pilot code.
 *
 * Secondaries (one chip per 1 ms primary period):
 *   - L3OCd: the 5-chip Barker code (11101) -> 5 ms symbol = 200 sym/s = 100 bps after rate-1/2.
 *   - L3OCp: the 10-chip Neuman-Hoffman NH10 -> 10 ms.
 *
 * ⚠️ THE TRAPS, both unlike B1I/B3I's:
 *  1. **Code index 0 is DEGENERATE, not merely unused.** The per-satellite factor is seeded with
 *     the index itself, so index 0 seeds the LFSR with all-zeros -- an absorbing state that emits
 *     a CONSTANT sequence forever. The result is not a bad code, it is no code at all: it would
 *     correlate with the DC1 component of *every* satellite. Hence the 1..63 range check.
 *  2. **The code index may not equal the orbital slot number.** Every other constellation we
 *     track keys its codes by the same PRN the almanac/TLE registry reports, so the broker's
 *     slot number can be handed straight to the generator. GLONASS's slot is an FDMA orbital
 *     position, and whether the L3OC ICD indexes ranging codes by slot is NOT something to
 *     assume. Resolve it by MEASUREMENT: search the full 1..63 index range and see which
 *     indices lock, then compare against the K-satellite slots that are actually overhead.
 *     A wrong index is a clean failure (another valid code -> cross-correlation floor), not a
 *     weak-correlation one, so this is safe to settle on the sky.
 *
 * Sign convention: logic 0 -> +1, 1 -> -1 (our house convention, and here it also matches
 * PocketSDR's CHIP=(1,-1) exactly, so these codes are bit-identical to that reference rather
 * than globally inverted as beidouB1ICode is versus gnss-sdr).
 *
 * Reference: PocketSDR sdr_code.py (gen_code_G3OCD / gen_code_G3OCP / gen_code_G3OC_DC1),
 * which follows the GLONASS L3OC interface document. Cross-checked structurally by
 * python/scripts/gnss/l3oc_code_check.py (independent reimplementation).
 */
namespace glonass {

constexpr int L3OC_CODE_LENGTH = 10230;
constexpr int L3OC_BC_LENGTH = 5;
constexpr int L3OC_NH_LENGTH = 10;

/// L3OCd (data) ranging code for code index 1..63, as +-1 chips.
std::array<int8_t, L3OC_CODE_LENGTH> generate_l3ocd_code(int prn);

/// L3OCp (pilot) ranging code for code index 1..63, as +-1 chips.
std::array<int8_t, L3OC_CODE_LENGTH> generate_l3ocp_code(int prn);

/// The 5-chip Barker secondary on the DATA component (11101 -> +1 +1 +1 -1 +1).
extern const std::array<int8_t, L3OC_BC_LENGTH> L3OC_BC5;

/// The 10-chip Neuman-Hoffman secondary on the PILOT component.
extern const std::array<int8_t, L3OC_NH_LENGTH> L3OC_NH10;

} // namespace glonass

#endif
