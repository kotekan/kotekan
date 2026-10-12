#ifndef GLONASS_CA_CODE_HPP
#define GLONASS_CA_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * GLONASS C/A ranging code: the spreading code of the FDMA open signals L1OF (~1602 MHz) and
 * L2OF (~1246 MHz).
 *
 * Every satellite transmits this one code; satellites are separated by carrier. The satellite on
 * frequency channel k transmits at f0 + k*df (L2: 1246.0 MHz + k*0.4375 MHz, k = -7..+6). So there
 * is no PRN argument here: the per-satellite carrier is set with
 * ChannelizedReplicaBank::set_prn_freq_offset().
 *
 * 511 chips (2^9 - 1) at 511 kcps, 1 ms period, from a single 9-stage LFSR: an m-sequence, not a
 * Gold code. Its periodic autocorrelation is exactly -1 at every non-zero shift (-54.2 dB below
 * the 511 peak) and its chips sum to -1 (255 of +1, 256 of -1). glonass_ca_code_check.py asserts
 * both, but they hold for any primitive tap set and any output stage (a phase shift), so the
 * script also checks the LFSR constants and a chip fingerprint.
 *
 * Data only (no pilot). The 50 bps navigation message is Manchester encoded to 100 symbols/s: a
 * symbol is 10 ms (ten code periods) and each 20 ms bit is a +-/-+ symbol pair. There is no
 * secondary code; wiping at the 10 ms symbol (navwipe_bit_records 10) handles the Manchester
 * structure, since each half-bit is a constant +-1.
 *
 * Sign convention: logic 0 -> +1 (also PocketSDR's CHIP=(1,-1)). Reference: PocketSDR sdr_code.py
 * gen_code_GLO_CA.
 */
namespace glonass {

constexpr int CA_CODE_LENGTH = 511;

/// The C/A code (L1OF and L2OF) as +-1 chips, shared by every satellite.
const std::array<int8_t, CA_CODE_LENGTH>& ca_code();

/// L2OF frequency plan: carrier for frequency-channel number k (valid k = -7..+6),
/// 1246.0 MHz + k * 0.4375 MHz. The comb spans 1242.94-1248.63 MHz (5.69 MHz), so one 10 MHz
/// tuning covers every satellite and L2OC (1248.06 MHz).
constexpr double L2OF_BASE_HZ = 1246.0e6;
constexpr double L2OF_CHANNEL_SPACING_HZ = 437.5e3;
constexpr int L2OF_K_MIN = -7;
constexpr int L2OF_K_MAX = 6;
constexpr double l2of_carrier_hz(int k) {
    return L2OF_BASE_HZ + k * L2OF_CHANNEL_SPACING_HZ;
}

} // namespace glonass

#endif
