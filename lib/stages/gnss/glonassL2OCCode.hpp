#ifndef GLONASS_L2OC_CODE_HPP
#define GLONASS_L2OC_CODE_HPP

#include <array>
#include <stdint.h>
#include <vector>

/**
 * GLONASS L2OC-p ranging code (1248.06 MHz) -- the modernised CDMA PILOT that rides the SAME
 * ~1246 MHz tune as L2OF, +2.06 MHz off the k=0 nominal.
 *
 * ★★ THE ELEGANT PART: its base ranging code is BIT-IDENTICAL to L3OC-p's. Both are the shared
 * 14-stage DC1 m-sequence times the per-satellite 7-stage sequence seeded with `prn + 64`
 * (glonassL3OCCode's pilot factor). GLONASS reuses one ranging sequence across the L2 and L3 CDMA
 * pilots; the ONLY thing that differs is the modulation applied on top:
 *   - L3OC-p transmits that 10230-chip sequence RAW at 10.23 Mcps (1 ms period).
 *   - L2OC-p transmits it at 511.5 kcps and applies a per-chip [0, 0, +1, -1] sub-carrier --
 *     the ICD's BOC(1,1) + data/pilot TDM in one 4-slot pattern (PocketSDR mod_code). Each base
 *     chip becomes [0, 0, +chip, -chip]: the two zeros are the DATA component's TDM slots (a
 *     PILOT-only replica is off there), and +chip/-chip are the BOC(1,1) half-cycles of the
 *     pilot. So the transmitted code is 40920 chips at 2.046 Mcps, 20 ms period, with HALF the
 *     chips zero.
 *
 * We build this by expanding glonass::generate_l3ocp_code() rather than re-deriving the LFSRs,
 * so the reuse is explicit in the code and any change to the shared sequence stays in one place.
 * ★ The pipeline needs no TDM/BOC MECHANISM for this: the expansion is baked into the code table
 * (a plain code that happens to contain zeros), fed as ordinary BPSK at 2.046 Mcps. That side-
 * steps the "TDM+BOC not supported" constructor guard -- the zeros carry the whole structure, and
 * the matched filter, which normalises by |replica|, correctly ignores the off slots.
 *
 * DATALESS pilot. Secondary: a SHARED 50-chip OC2 overlay, one chip per 20 ms primary period ->
 * 1 s cycle (the E5a-Q / E6-C segmented-pilot shape, just longer period). GLONASS-K ONLY (the
 * same 7 satellites as L3OC), so gated by the K capability marker.
 *
 * ⚠️ CARRIER 1248.06 MHz is the ICD value; PocketSDR rounds it to 1248.0. The 60 kHz difference
 * is far outside a ±5 kHz Doppler window, so if first light fails at 1248.06, widen the search
 * and check 1248.0 -- settle it on the sky, do not assume.
 *
 * Sign convention: inherited from generate_l3ocp_code (logic 0 -> +1, = PocketSDR CHIP). The
 * [0,0,1,-1] multipliers are literal (0 stays 0), so the expansion introduces no inversion.
 * Reference: PocketSDR sdr_code.py gen_code_G2OCP + G2OCP_OC2; l2oc_code_check.py validates.
 */
namespace glonass {

constexpr int L2OC_BASE_LENGTH = 10230;    ///< the shared L3OC-p base, before expansion
constexpr int L2OC_SUBCARRIER = 4;         ///< [0,0,1,-1] -> 4 sub-chips per base chip
constexpr int L2OC_CODE_LENGTH = L2OC_BASE_LENGTH * L2OC_SUBCARRIER; // 40920
constexpr int L2OC_OC2_LENGTH = 50;
constexpr double L2OC_CARRIER_HZ = 1248.06e6;

/// L2OC-p ranging code for code index 1..63, as chips in {-1, 0, +1} (the zeros are the TDM'd
/// data slots). Length L2OC_CODE_LENGTH at 2.046 Mcps.
std::vector<int8_t> generate_l2ocp_code(int prn);

/// The shared 50-chip OC2 secondary overlay (one chip per 20 ms primary period, 1 s cycle).
extern const std::array<int8_t, L2OC_OC2_LENGTH> L2OC_OC2;

} // namespace glonass

#endif
