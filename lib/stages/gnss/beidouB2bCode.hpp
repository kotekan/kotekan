#ifndef BEIDOU_B2B_CODE_HPP
#define BEIDOU_B2B_CODE_HPP
/**
 * @file beidouB2bCode.hpp
 * @brief BeiDou-3 B2b-I primary code generation (BDS-SIS-ICD-B2b-1.0).
 *
 * B2b-I (data, 1207.14 MHz): 10230-chip primary at 10.23 Mcps (1 ms) from two 13-stage
 * LFSRs: G1 from all-ones, truncated at 8190 chips and restarted (tap 0b1000000011001); G2
 * from per-PRN start values (tap 0b0011010010011). The construction is B2a's with different
 * taps; the G2 start values match B2a's for PRN 1..60.
 *
 * Data-only: there is no pilot in the open service, so coherent integration needs the B-CNAV3
 * symbols wiped. B-CNAV3 is 1000 sps, one symbol per 1 ms code period, so there is no
 * secondary code. BDS-3 only.
 *
 * Tables from PocketSDR's sdr_code.py. python/scripts/gnss/b2b_code_check.py mirrors this
 * generator and checks it against PocketSDR fingerprints for PRN 1/19/20/30/45/63.
 */
#include <array>
#include <cstdint>

namespace beidou {

constexpr int B2B_CODE_LENGTH = 10230;

/// B2b-I primary code, PRN 1..63, as +-1 chips.
std::array<int8_t, B2B_CODE_LENGTH> generate_b2bi_code(int prn);

} // namespace beidou
#endif // BEIDOU_B2B_CODE_HPP
