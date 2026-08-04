#include "beidouB3ICode.hpp"

#include <stdexcept>
#include <string>

namespace beidou {
namespace {

// Per-PRN G2 INITIAL STATE (BDS-SIS-ICD-B3I v1.0), transcribed via gnss-sdr's
// beidou_b3i_signal_replica.cc. B3I puts its per-satellite diversity here, in G2's starting
// state -- unlike B1I, which uses a per-PRN output TAP on a fixed initial state.
// Leftmost character is register index 12, rightmost is index 0 (the output end).
constexpr const char* B3I_G2_INIT[63] = {
    "1010111111111", "1111000101011", "1011110001010", "1111111111011",
    "1100100011111", "1001001100100", "1111111010010", "1110111111101",
    "1010000000010", "0010000011011", "1110101110000", "0010110011110",
    "0110010010101", "0111000100110", "1000110001001", "1110001111100",
    "0010011000101", "0000011101100", "1000101010111", "0001011011110",
    "0010000101101", "0010110001010", "0001011001111", "0011001100010",
    "0011101001000", "0100100101001", "1011011010011", "1010111100010",
    "0001011110101", "0111111111111", "0110110001111", "1010110001001",
    "1001010101011", "1100110100101", "1101001011101", "1111101110100",
    "0010101100111", "1110100010000", "1101110010000", "1101011001110",
    "1000000110100", "0101111011001", "0110110111100", "1101001110001",
    "0011100100010", "0101011000101", "1001111100110", "1111101001000",
    "0000101001001", "1000010101100", "1111001001100", "0100110001111",
    "0000000011000", "1000000000100", "0011010100110", "1011001000110",
    "0111001111000", "0010111001010", "1100111110110", "1001001000101",
    "0111000100000", "0011001000010", "0010001001110",
};

} // namespace

std::array<int8_t, B3I_CODE_LENGTH> generate_b3i_code(int prn) {
    if (prn < 1 || prn > 63)
        throw std::out_of_range("generate_b3i_code: PRN must be 1..63, got " + std::to_string(prn));
    int g1[13], g2[13];
    const char* s = B3I_G2_INIT[prn - 1];
    for (int i = 0; i < 13; ++i) {
        g1[i] = 1;                       // G1 starts all-ones
        g2[12 - i] = (s[i] == '1') ? 1 : 0;
    }
    std::array<int8_t, B3I_CODE_LENGTH> out{};
    for (int k = 0; k < B3I_CODE_LENGTH; ++k) {
        // ICD logic level -> BPSK: 0 -> +1, 1 -> -1 (see the header's convention note).
        out[(size_t)k] = (g1[0] ^ g2[0]) ? -1 : 1;

        const int fb1 = g1[0] ^ g1[9] ^ g1[10] ^ g1[12];
        const int fb2 = g2[0] ^ g2[1] ^ g2[3] ^ g2[4] ^ g2[6] ^ g2[7] ^ g2[8] ^ g2[12];
        for (int i = 0; i < 12; ++i) {
            g1[i] = g1[i + 1];
            g2[i] = g2[i + 1];
        }
        g1[12] = fb1;
        g2[12] = fb2;

        // ★ THE RESET (see the header): G1's natural period exceeds the 10230-chip code, so the
        // ICD forces it back to all-ones on reaching 0011111111111 (indices 0 and 1 low, the
        // rest high). Without this the tail of every code silently diverges.
        bool at_reset = (g1[0] == 0 && g1[1] == 0);
        for (int i = 2; i < 13 && at_reset; ++i)
            at_reset = (g1[i] == 1);
        if (at_reset)
            for (int i = 0; i < 13; ++i)
                g1[i] = 1;
    }
    return out;
}

// D1 NH secondary "00000100110101001110", 0 -> +1 -- the same sequence B1I carries.
const std::array<int8_t, B3I_NH_LENGTH> B3I_NH20 = {
    1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1,
};

} // namespace beidou
