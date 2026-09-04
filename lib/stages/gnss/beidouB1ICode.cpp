#include "beidouB1ICode.hpp"

#include <stdexcept>
#include <string>

namespace beidou {
namespace {

// G2 output-tap selection per PRN (BDS-SIS-ICD-B1I v3.0), transcribed via gnss-sdr's
// beidou_b1i_signal_replica.cc. ⚠️ phase3 is ZERO for PRN 1..37 (two taps) and NON-ZERO for
// PRN 38..63 (three taps) -- see the header note; dropping the third tap yields valid-looking
// but WRONG codes for precisely the BDS-3 satellites.
constexpr int B1I_PHASE1[63] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 8, 8, 8, 9, 9, 10, 2, 3, 3, 3, 3,
                                3, 4, 4, 5, 5, 5, 5, 6, 8, 9, 9, 3, 5, 7, 4, 4, 5, 5, 5, 5, 6};
constexpr int B1I_PHASE2[63] = {3, 4, 5, 6, 8, 9, 10, 11, 7, 4, 5, 6, 8, 9, 10, 11, 5, 6, 8, 9,
                                10, 11, 6, 8, 9, 10, 11, 8, 9, 10, 11, 9, 10, 11, 10, 11, 11, 7,
                                4, 6, 8, 10, 11, 5, 9, 6, 8, 10, 11, 9, 9, 10, 11, 7, 7, 9, 5, 9,
                                6, 8, 10, 11, 9};
constexpr int B1I_PHASE3[63] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3};

} // namespace

std::array<int8_t, B1I_CODE_LENGTH> generate_b1i_code(int prn) {
    if (prn < 1 || prn > 63)
        throw std::out_of_range("generate_b1i_code: PRN must be 1..63, got " + std::to_string(prn));
    const int idx = prn - 1;
    // Both registers start at 01010101010 (ICD). Index 0 is the OUTPUT end; the string's
    // leftmost character is the HIGHEST index (bitset convention), so 01010101010 loads as
    // reg[10]=0, reg[9]=1, ... reg[0]=0.
    int g1[11], g2[11];
    for (int i = 0; i < 11; ++i) {
        const int bit = ((10 - i) % 2 == 1) ? 1 : 0; // "01010101010" -> reg[10-i] = s[i]
        g1[10 - i] = bit;
        g2[10 - i] = bit;
    }
    std::array<int8_t, B1I_CODE_LENGTH> out{};
    for (int k = 0; k < B1I_CODE_LENGTH; ++k) {
        const int g1_out = g1[0];
        // G2's output is a per-PRN XOR of two (PRN<=37) or three (PRN>=38) register taps.
        // Tap index = 11 - phase (the ICD numbers stages 1..11 from the far end).
        int g2_out = g2[11 - B1I_PHASE1[idx]] ^ g2[11 - B1I_PHASE2[idx]];
        if (B1I_PHASE3[idx] != 0)
            g2_out ^= g2[11 - B1I_PHASE3[idx]];
        // ICD logic level -> BPSK: 0 -> +1, 1 -> -1 (see the header's convention note).
        out[(size_t)k] = (g1_out ^ g2_out) ? -1 : 1;

        const int fb1 = g1[0] ^ g1[1] ^ g1[2] ^ g1[3] ^ g1[4] ^ g1[10];
        const int fb2 = g2[0] ^ g2[2] ^ g2[3] ^ g2[6] ^ g2[7] ^ g2[8] ^ g2[9] ^ g2[10];
        for (int i = 0; i < 10; ++i) {
            g1[i] = g1[i + 1];
            g2[i] = g2[i + 1];
        }
        g1[10] = fb1;
        g2[10] = fb2;
    }
    return out;
}

// D1 NH secondary "00000100110101001110", 0 -> +1 (same convention as the primary above).
const std::array<int8_t, B1I_NH_LENGTH> B1I_NH20 = {
    1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1,
};

} // namespace beidou
