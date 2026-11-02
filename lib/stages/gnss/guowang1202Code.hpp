#ifndef GUOWANG_1202_CODE_HPP
#define GUOWANG_1202_CODE_HPP

#include <array>
#include <stdint.h>
#include <vector>

/**
 * GuoWang (China's LEO broadband constellation) 1202.025 MHz ranging code, found in CHORD's
 * beam dump on 2026-11-02 (docs/CHORD_BUGLIST.md #227); there is no public ICD.
 *
 * BPSK at 2.046 Mcps. Each 2 ms frame is 4092 chips: a fixed 2046-chip code, then 2046 data
 * chips on the same chip grid; a data sign spanning two frames (4 ms) multiplies the whole
 * frame. The code is a Gold code like BeiDou B1I's: B1I's G1 register (taps and initial state
 * 01010101010 as beidouB1ICode) entered 1206 chips in, XOR a degree-11 m-sequence G2'
 * (1 + x + x^3 + x^8 + x^9 + x^10 + x^11, from the all-ones state) entered at a per-satellite
 * phase. The phase identifies the satellite; satellites are numbered here by it, PRN = phase + 1
 * (1..2047), and the broker maps TLE satellites to phases from the beam-dump database.
 *
 * Checked chip for chip against codes read off the sky: python/scripts/gnss/guowang/
 * guowang_codes.py guowang_1202() and its test vectors, which test_guowang_1202_code repeats.
 * Sign convention as beidouB1ICode: logic 0 -> +1, 1 -> -1.
 */
namespace guowang {

constexpr int GW1202_CODE_LENGTH = 2046;          ///< chips of the fixed code (1 ms)
constexpr int GW1202_FRAME_LENGTH = 4092;         ///< chips in one frame (2 ms): code, then data
constexpr int GW1202_G1_PHASE = 1206;             ///< chips of B1I's G1 skipped before the code
constexpr int GW1202_G2P_PERIOD = 2047;           ///< G2' phases, and so satellites
constexpr int GW1202_MAX_PRN = GW1202_G2P_PERIOD; ///< PRN = G2' phase + 1

/// The 2046-chip code of the satellite at G2' phase @p g2_phase (0..2046), as +-1 chips.
std::array<int8_t, GW1202_CODE_LENGTH> generate_gw1202_code(int g2_phase);

/// One frame's replica for PRN @p prn (1..2047): the code, then 2046 zeros where the data
/// millisecond is (a despread over the data half adds nothing coherent, as GLO_L2OC_P's zero
/// chips).
std::vector<int8_t> generate_gw1202_frame(int prn);

} // namespace guowang

#endif // GUOWANG_1202_CODE_HPP
