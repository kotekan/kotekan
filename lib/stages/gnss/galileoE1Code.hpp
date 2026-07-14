#ifndef GALILEO_E1_CODE_HPP
#define GALILEO_E1_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * Galileo E1 OS primary spreading codes (E1-B data / E1-C pilot) + the E1-C secondary code.
 *
 * These are MEMORY codes -- 4092-chip pseudorandom sequences published as hex tables in the
 * Galileo OS SIS ICD (issue 2.1), Annex C.7/C.8 -- not LFSR-generable. Tables transcribed
 * from the published ICD values (obtained via the gnss-sdr project's transcription and
 * verified here: every code exactly balanced at 2046/4092 ones, all 50 distinct,
 * cross-correlation at the sqrt(N) scale). Transmitted with a CBOC(6,1,1/11) subcarrier;
 * the replica models the dominant BOC(1,1) component (~-0.4 dB, standard practice).
 * PRN range 1..50.
 */
namespace galileo {

constexpr int E1_CODE_LENGTH = 4092;

/// E1-B (data channel, I/NAV 250 sps) primary code, +-1 chips.
std::array<int8_t, E1_CODE_LENGTH> generate_e1b_code(int prn);

/// E1-C (dataless pilot) primary code, +-1 chips. The pilot carries the 25-chip CS25
/// secondary code (one chip per 4 ms primary period, 100 ms total).
std::array<int8_t, E1_CODE_LENGTH> generate_e1c_code(int prn);

/// E1-C secondary code CS25 (ICD 4.3.4), +-1, same for every satellite.
extern const std::array<int8_t, 25> E1C_CS25;

} // namespace galileo

#endif
