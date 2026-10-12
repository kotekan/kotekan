#ifndef GLONASS_L3OC_CODE_HPP
#define GLONASS_L3OC_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * GLONASS L3OC ranging codes (1202.025 MHz): the data (L3OCd) and pilot (L3OCp) components and
 * their secondaries.
 *
 * GLONASS's CDMA signal: 10230 chips at 10.23 Mcps, 1 ms period, BPSK(10), the same record
 * geometry as BeiDou B2b and B3I. Broadcast only by GLONASS-K satellites.
 *
 * Both components are a 14-stage LFSR sequence DC1, shared by all satellites, times a 7-stage
 * per-satellite LFSR sequence that differs only in its initial state: DC2 starts at `prn` for the
 * data component, DC3 at `prn + 64` for the pilot. The 7-stage polynomial is primitive, so its 127
 * non-zero states are all cyclic shifts of one 127-chip m-sequence: each satellite's short factor
 * is a phase offset of the same sequence. Data uses states 1..63 and pilot 65..127, so no data
 * code equals a pilot code.
 *
 * Secondaries (one chip per 1 ms primary period):
 *   - L3OCd: the 5-chip Barker code, a 5 ms symbol (200 symbols/s, 100 bps after rate-1/2
 *     coding).
 *   - L3OCp: the 10-chip Neuman-Hoffman code NH10, 10 ms.
 *
 * Both generators accept code index 1..63. Data index 0 would seed DC2 with zeros, a fixed point
 * of the LFSR, giving DC1 alone. The code index is not assumed to equal the orbital slot number,
 * which is why the L3OC signal descriptors admit all of 1..63. A wrong index gives another valid
 * code, at the cross-correlation floor, not a weak match.
 *
 * Sign convention: logic 0 -> +1, 1 -> -1, matching PocketSDR's CHIP=(1,-1), so these codes are
 * bit-identical to that reference.
 *
 * Reference: PocketSDR sdr_code.py (gen_code_G3OCD, gen_code_G3OCP, gen_code_G3OC_DC1), which
 * follows the GLONASS L3OC ICD. python/scripts/gnss/l3oc_code_check.py reimplements the
 * construction independently and compares chip for chip.
 */
namespace glonass {

constexpr int L3OC_CODE_LENGTH = 10230;
constexpr int L3OC_BC_LENGTH = 5;
constexpr int L3OC_NH_LENGTH = 10;

/// L3OCd (data) ranging code for code index 1..63, as +-1 chips.
std::array<int8_t, L3OC_CODE_LENGTH> generate_l3ocd_code(int prn);

/// L3OCp (pilot) ranging code for code index 1..63, as +-1 chips.
std::array<int8_t, L3OC_CODE_LENGTH> generate_l3ocp_code(int prn);

/// The 5-chip Barker secondary on the data component: +1 +1 +1 -1 +1 (logic 00010).
extern const std::array<int8_t, L3OC_BC_LENGTH> L3OC_BC5;

/// The 10-chip Neuman-Hoffman secondary on the pilot component.
extern const std::array<int8_t, L3OC_NH_LENGTH> L3OC_NH10;

} // namespace glonass

#endif
