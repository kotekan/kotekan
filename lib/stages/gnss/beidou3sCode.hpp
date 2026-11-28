#ifndef BEIDOU_3S_CODE_HPP
#define BEIDOU_3S_CODE_HPP
/**
 * @file beidou3sCode.hpp
 * @brief B2a and B2b primary codes of the BeiDou-3S experimental satellite M2-S (C58).
 *
 * Not in any ICD: measured from the sky on 2026-11-28. Each band carries a data and a pilot
 * component, chip-interleaved into one 10230-chip, 1 ms code at 10.23 Mcps: data on the even
 * chips, pilot on the odd, counted from the B3I code epoch, both on the same carrier phase.
 * Each component is a 5115-chip code, the product of two 13-stage LFSRs with taps 0x1805
 * (1 + x + x^2 + x^11 + x^13) and 0x0B15 (1 + x^2 + x^4 + x^5 + x^9 + x^11 + x^13) from
 * per-component start states, restarted every 1 ms. The overlays are the ICD's: B2a data
 * carries the 5-chip secondary and B2a pilot the 100-chip pilot secondary of PRN 58.
 *
 * The codes here are zero-stuffed at the other component's chips, so a 10230-chip replica holds
 * one component with the real signal's spectrum, as ChannelizedReplicaBank does for TDM
 * signals. M2-S's B3I is generate_b3i_code(58, false).
 *
 * tests/boost/testdata/bds3s_m2s holds the chips recovered from the sky and the spec.
 */
#include <array>
#include <cstdint>

namespace beidou {

constexpr int B3S_CODE_LENGTH = 10230;

/// B2a pilot (odd chips; zeros on the even). PRN 58 only; other PRNs throw std::out_of_range.
std::array<int8_t, B3S_CODE_LENGTH> generate_b2ap_3s_code(int prn);
/// B2a data (even chips; zeros on the odd). PRN 58 only.
std::array<int8_t, B3S_CODE_LENGTH> generate_b2ad_3s_code(int prn);
/// B2b pilot (odd chips; zeros on the even). PRN 58 only.
std::array<int8_t, B3S_CODE_LENGTH> generate_b2bp_3s_code(int prn);

} // namespace beidou
#endif // BEIDOU_3S_CODE_HPP
