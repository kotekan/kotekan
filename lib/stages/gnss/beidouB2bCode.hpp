#ifndef BEIDOU_B2B_CODE_HPP
#define BEIDOU_B2B_CODE_HPP
/**
 * @file beidouB2bCode.hpp
 * @brief BeiDou-3 B2b-I primary code generation (BDS-SIS-ICD-B2b / PPP-B2b, v1.0 2020).
 *
 * B2b-I (DATA, 1207.14 MHz): 10230-chip primary at 10.23 Mcps (1 ms) from two 13-stage
 * LFSRs -- G1 all-ones start truncated at 8190 then restarted (tap 0b1000000011001), G2
 * with per-PRN start values (tap 0b0011010010011). Same LFSR family as B2a; only the taps
 * and the G2 start table differ.
 *
 * DATA-only: unlike B2a/B1C there is no dataless pilot component in the open service, so
 * deep integration needs nav-bit wipe (B-CNAV3) rather than a pilot peel. B-CNAV3 runs at
 * 1000 sps = exactly one symbol per 1 ms code period, so there is NO secondary overlay.
 * BDS-3 only (PPP-B2b is a BDS-3 service).
 *
 * Tables sourced from PocketSDR (T.Takasu sdr_code.py, transcribed from the BDS B2b ICD)
 * and TRANSCRIPTION-VERIFIED bit-exact for PRN 1/19/20/30/45/63
 * (python/scripts/gnss/b2b_code_check.py). Sky-validation (live acquisition off the retuned
 * 1207.14 MHz dongle) is the remaining step, mirroring B2a's 2026-07-15 first light.
 */
#include <array>
#include <cstdint>

namespace beidou {

constexpr int B2B_CODE_LENGTH = 10230;

/// B2b-I primary code, PRN 1..63, as +-1 chips.
std::array<int8_t, B2B_CODE_LENGTH> generate_b2bi_code(int prn);

} // namespace beidou
#endif // BEIDOU_B2B_CODE_HPP
