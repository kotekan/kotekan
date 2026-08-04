#ifndef BEIDOU_B1I_CODE_HPP
#define BEIDOU_B1I_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * BeiDou B1I ranging code (1561.098 MHz) + the shared 20-chip NH secondary.
 *
 * LEGACY BeiDou (BDS-2 and BDS-3 both broadcast it) -- unlike B1C/B2a/B2b this is an
 * ALGORITHMIC Gold code, not a memory table: 2046 chips at 2.046 Mcps (1 ms period), BPSK(2),
 * from two 11-stage LFSRs, per the BDS-SIS-ICD-B1I v3.0. DATA-ONLY: there is no pilot
 * component, so it is acquired directly and nav-wiped (the GPS L1 C/A discipline), and the
 * D1 NAV message is 50 bps spread by a 20-chip NH secondary (one chip per 1 ms code period
 * => 20 ms/symbol, the GPS L5-I / B2a-D shape). PRN 1..63.
 *
 * ⚠️ THE TRAP: the G2 output tap is TWO taps XORed for PRN 1..37 but **THREE** for PRN 38..63
 * (the BDS-3 extension). Most published tap tables only list the first 37 rows, so a
 * transcription that stops there silently produces garbage codes for exactly the BDS-3
 * satellites -- which are the ones actually overhead. phase3 is 0 for PRN 1..37 and non-zero
 * beyond; a 0 entry means "two taps only".
 *
 * D2 (the GEO satellites' 500 bps message) uses NO NH secondary; only D1 is modelled here.
 *
 * Sign convention: ICD logic level 0 -> chip +1, 1 -> chip -1, matching our E1/L1C modules.
 * That is a GLOBAL inversion relative to gnss-sdr's B1I replica (which maps 1 -> +1); a global
 * sign is immaterial to |correlation| and is absorbed by the navwipe, but keep primary and
 * secondary on the SAME convention (they are, here) so their product is not accidentally
 * inverted per-record.
 */
namespace beidou {

constexpr int B1I_CODE_LENGTH = 2046;
constexpr int B1I_NH_LENGTH = 20;

/// B1I ranging code (PRN 1..63) as +-1 chips.
std::array<int8_t, B1I_CODE_LENGTH> generate_b1i_code(int prn);

/// The 20-chip NH secondary carried by the D1 NAV message, SHARED by every satellite
/// (00000100110101001110). One chip per 1 ms primary period -> 20 ms per NAV symbol.
/// ★ B3I carries the SAME sequence -- see beidouB3ICode.hpp.
extern const std::array<int8_t, B1I_NH_LENGTH> B1I_NH20;

} // namespace beidou

#endif
