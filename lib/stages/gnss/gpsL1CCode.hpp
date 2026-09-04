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
 * carries a 1800-symbol L1CO overlay (18 s) -- @ref generate_l1co_code, an 11-stage LFSR.
 *
 * This implements the L1C-P PILOT primary code (the better track/peel target: dataless,
 * fully known modulation) and its overlay. L1C-D would add its own (w,p) table.
 */

#ifndef GPS_L1C_CODE_HPP
#define GPS_L1C_CODE_HPP

#include <array>
#include <cstdint>

namespace gps {

constexpr int L1C_CODE_LENGTH = 10230; ///< chips in one L1C primary period (10 ms @ 1.023 Mcps)
constexpr int L1CO_LENGTH = 1800;      ///< L1C-P overlay (secondary) length, 18 s (one symbol/10 ms)

/// L1C-P (pilot) primary spreading code for @p prn (1..32), bipolar +1/-1.
std::array<int8_t, L1C_CODE_LENGTH> generate_l1cp_code(int prn);

/// L1C-D (DATA) primary spreading code for @p prn (1..32), bipolar +1/-1. Same Weil-code
/// construction as L1C-P (shared Legendre + 7-chip insertion), only the per-PRN Weil/insertion
/// indices differ. Carries CNAV-2; the data sibling of the L1C-P pilot (no overlay of its own).
std::array<int8_t, L1C_CODE_LENGTH> generate_l1cd_code(int prn);

/// L1C-P overlay (secondary) code for @p prn (1..32), bipolar +1/-1, length @ref L1CO_LENGTH.
/// One overlay symbol multiplies each 10 ms primary period; coherently integrating L1C-P past
/// one period requires wiping this (cf. the L5 NH overlay, but per-PRN and 1800 long).
std::array<int8_t, L1CO_LENGTH> generate_l1co_code(int prn);

} // namespace gps

#endif // GPS_L1C_CODE_HPP
