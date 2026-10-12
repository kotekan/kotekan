#ifndef GALILEO_E1_CODE_HPP
#define GALILEO_E1_CODE_HPP

#include <array>
#include <stdint.h>

/**
 * @file
 * @brief Galileo E1 OS primary codes (E1-B data, E1-C pilot) and the E1-C CS25 secondary.
 *
 * Memory codes: 4092-chip sequences published as hex tables in the Galileo OS SIS ICD
 * (issue 2.1), Annex C.7/C.8, transcribed via gnss-sdr; not LFSR-generated. Every code has
 * 2046 ones in 4092 chips and all 50 are distinct. The signal is CBOC(6,1,1/11); the replica
 * models only its BOC(1,1) component, which carries 10/11 of the power (~-0.4 dB).
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
