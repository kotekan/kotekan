#ifndef BEIDOU_B1I_CODE_HPP
#define BEIDOU_B1I_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * BeiDou B1I ranging code (1561.098 MHz) and the shared 20-chip NH secondary.
 *
 * Broadcast by BDS-2 and BDS-3 alike. A Gold code from two 11-stage LFSRs per
 * BDS-SIS-ICD-B1I v3.0: 2046 chips at 2.046 Mcps (1 ms), BPSK(2). There is no pilot, so B1I is
 * acquired directly and nav-wiped as GPS L1 C/A is. The 50 bps D1 message is spread by a
 * 20-chip NH secondary, one chip per 1 ms code period, so one NH period is one 20 ms symbol.
 * D2 (the GEO satellites' 500 bps message) has no NH secondary; only D1 is modelled here.
 * PRN 1..63.
 *
 * The G2 output is two register taps XORed for PRN 1..37 but three for PRN 38..63 (the BDS-3
 * extension); dropping the third tap gives valid-looking but wrong codes for exactly the BDS-3
 * satellites. A 0 third tap means two taps only.
 *
 * Sign convention: ICD logic 0 -> chip +1, 1 -> chip -1, as in galileoE1Code and gpsL1CCode.
 * This is a global inversion relative to gnss-sdr's B1I replica (1 -> +1), which does not
 * change |correlation| and is absorbed by the nav wipe. The primary and secondary must use the
 * same convention so their product is not inverted.
 */
namespace beidou {

constexpr int B1I_CODE_LENGTH = 2046;
constexpr int B1I_NH_LENGTH = 20;

/// B1I ranging code (PRN 1..63) as +-1 chips.
std::array<int8_t, B1I_CODE_LENGTH> generate_b1i_code(int prn);

/// The 20-chip NH secondary of the D1 message (00000100110101001110), shared by every
/// satellite; one chip per 1 ms primary period. B3I carries the same sequence (B3I_NH20).
extern const std::array<int8_t, B1I_NH_LENGTH> B1I_NH20;

} // namespace beidou

#endif
