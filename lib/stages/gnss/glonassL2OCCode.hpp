#ifndef GLONASS_L2OC_CODE_HPP
#define GLONASS_L2OC_CODE_HPP

#include <array>
#include <stdint.h>
#include <vector>

/**
 * GLONASS L2OC-p ranging code (1248.06 MHz): the CDMA pilot, 2.06 MHz above the L2OF k=0 carrier,
 * so one ~1246 MHz tuning can cover both.
 *
 * Its base ranging code is bit-identical to L3OC-p's: the shared 14-stage DC1 sequence times the
 * 7-stage per-satellite sequence seeded with `prn + 64`. Only the modulation differs:
 *   - L3OC-p transmits the 10230-chip sequence as is, at 10.23 Mcps (1 ms period).
 *   - L2OC-p transmits it at 511.5 kcps with each base chip c expanded to [0, 0, +c, -c], the
 *     ICD's BOC(1,1) and data/pilot TDM in one 4-slot pattern (PocketSDR mod_code). The zeros are
 *     the data component's TDM slots, where a pilot replica is off; +c/-c are the BOC(1,1)
 *     half-cycles. The transmitted code is 40920 chips at 2.046 Mcps, 20 ms period, half of them
 *     zero.
 *
 * generate_l2ocp_code() expands generate_l3ocp_code() rather than re-deriving the LFSRs. The
 * expansion lives in the code table, so the signal is fed as plain BPSK at 2.046 Mcps and needs
 * no TDM or BOC handling (ChannelizedReplicaBank rejects the two combined). The matched filter
 * normalises by |replica|, so the zero slots get no weight.
 *
 * Dataless pilot. Secondary: a 50-chip OC2 overlay shared by all satellites, one chip per 20 ms
 * primary period (1 s cycle). Broadcast only by GLONASS-K satellites, as L3OC is.
 *
 * Sign convention: inherited from generate_l3ocp_code() (logic 0 -> +1, PocketSDR's CHIP); the
 * [0, 0, 1, -1] multipliers introduce no inversion. Reference: PocketSDR sdr_code.py
 * gen_code_G2OCP and G2OCP_OC2; l2oc_code_check.py checks the code and overlay chip for chip.
 */
namespace glonass {

constexpr int L2OC_BASE_LENGTH = 10230; ///< the shared L3OC-p base, before expansion
constexpr int L2OC_SUBCARRIER = 4;      ///< [0,0,1,-1] -> 4 sub-chips per base chip
constexpr int L2OC_CODE_LENGTH = L2OC_BASE_LENGTH * L2OC_SUBCARRIER; // 40920
constexpr int L2OC_OC2_LENGTH = 50;

/// L2OC-p ranging code for code index 1..63, as chips in {-1, 0, +1} (the zeros are the TDM'd
/// data slots). Length L2OC_CODE_LENGTH at 2.046 Mcps. Throws std::out_of_range outside 1..63.
std::vector<int8_t> generate_l2ocp_code(int prn);

/// The shared 50-chip OC2 secondary overlay (one chip per 20 ms primary period, 1 s cycle).
extern const std::array<int8_t, L2OC_OC2_LENGTH> L2OC_OC2;

} // namespace glonass

#endif
