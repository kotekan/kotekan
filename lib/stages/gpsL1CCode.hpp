/**
 * @file
 * @brief GPS L1C ranging-code generation (IS-GPS-800).
 *
 * L1C (1575.42 MHz, modernized civil, Block III) has a data component L1C-D and a
 * dataless pilot L1C-P, both BOC(1,1)-modulated 10230-chip codes at 1.023 Mcps (10 ms).
 * The codes are Weil codes: from a length-10223 Legendre sequence L, the per-PRN code is
 * W(t) = L(t)*L((t+w) mod 10223) with a fixed 7-chip pattern ("0110100") inserted at a
 * per-PRN index p -- so each PRN is keyed by a (Weil index w, insertion index p) pair from
 * IS-GPS-800 Tables 3.2-2 / 6.3-1 (transcribed from PocketSDR sdr_code). The pilot also
 * carries a 1800-symbol L1CO overlay (18 s) -- a separate refinement, not generated here yet.
 *
 * This implements the L1C-P PILOT primary code (the better track/peel target: dataless,
 * fully known modulation). L1C-D would add its own (w,p) table.
 */

#ifndef GPS_L1C_CODE_HPP
#define GPS_L1C_CODE_HPP

#include <array>
#include <cstdint>

namespace gps {

constexpr int L1C_CODE_LENGTH = 10230; ///< chips in one L1C primary period (10 ms @ 1.023 Mcps)
constexpr int L1CO_LENGTH = 1800;      ///< L1C-P overlay (secondary) length, 18 s (generator TODO)

/// L1C-P (pilot) primary spreading code for @p prn (1..32), bipolar +1/-1.
std::array<int8_t, L1C_CODE_LENGTH> generate_l1cp_code(int prn);

} // namespace gps

#endif // GPS_L1C_CODE_HPP
