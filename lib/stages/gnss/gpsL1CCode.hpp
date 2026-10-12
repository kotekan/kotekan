/**
 * @file
 * @brief GPS L1C ranging-code generation (IS-GPS-800).
 *
 * L1C (1575.42 MHz) has a data component L1C-D and a dataless pilot L1C-P, each a 10230-chip
 * code at 1.023 Mcps (10 ms). They are Weil codes: from the length-10223 Legendre sequence L,
 * W(t) = L(t)*L((t+w) mod 10223), with the 7-chip pattern "0110100" inserted at index p.
 * Each PRN is keyed by its (Weil index w, insertion index p) pair from IS-GPS-800
 * Tables 3.2-2 / 6.3-1 (transcribed from PocketSDR sdr_code). The pilot also carries the
 * 1800-symbol L1CO overlay (18 s), @ref generate_l1co_code. Only the spreading codes are
 * generated here, not the BOC modulation.
 */

#ifndef GPS_L1C_CODE_HPP
#define GPS_L1C_CODE_HPP

#include <array>
#include <cstdint>

namespace gps {

constexpr int L1C_CODE_LENGTH = 10230; ///< chips in one L1C primary period (10 ms @ 1.023 Mcps)
constexpr int L1CO_LENGTH = 1800; ///< L1C-P overlay (secondary) length, 18 s (one symbol/10 ms)

/// L1C-P (pilot) primary spreading code for @p prn (1..32), bipolar +1/-1.
std::array<int8_t, L1C_CODE_LENGTH> generate_l1cp_code(int prn);

/// L1C-D (data) primary spreading code for @p prn (1..32), bipolar +1/-1: the L1C-P
/// construction with its own (w, p) indices. Carries CNAV-2 and has no overlay.
std::array<int8_t, L1C_CODE_LENGTH> generate_l1cd_code(int prn);

/// L1C-P overlay (secondary) code for @p prn (1..32), bipolar +1/-1, length @ref L1CO_LENGTH.
/// One overlay symbol multiplies each 10 ms primary period, so a coherent L1C-P integration
/// longer than one period has to wipe it.
std::array<int8_t, L1CO_LENGTH> generate_l1co_code(int prn);

} // namespace gps

#endif // GPS_L1C_CODE_HPP
