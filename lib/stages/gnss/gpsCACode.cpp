#include "gpsCACode.hpp"

#include <stdexcept> // for out_of_range
#include <string>    // for to_string

namespace gps {

namespace {

/**
 * G2 phase-select tap pairs, 1-indexed register cells, PRN 1..32, from
 * IS-GPS-200 Table 3-Ia (Code Phase Assignments). Index 0 is unused so the
 * table is addressed directly by PRN. The XOR of cells (a, b) forms the G2
 * output; this selection is what makes each PRN's code unique.
 */
constexpr int G2_TAPS[CA_PRN_MAX + 1][2] = {
    {0, 0},   // PRN 0 - unused
    {2, 6},   // PRN 1
    {3, 7},   // PRN 2
    {4, 8},   // PRN 3
    {5, 9},   // PRN 4
    {1, 9},   // PRN 5
    {2, 10},  // PRN 6
    {1, 8},   // PRN 7
    {2, 9},   // PRN 8
    {3, 10},  // PRN 9
    {2, 3},   // PRN 10
    {3, 4},   // PRN 11
    {5, 6},   // PRN 12
    {6, 7},   // PRN 13
    {7, 8},   // PRN 14
    {8, 9},   // PRN 15
    {9, 10},  // PRN 16
    {1, 4},   // PRN 17
    {2, 5},   // PRN 18
    {3, 6},   // PRN 19
    {4, 7},   // PRN 20
    {5, 8},   // PRN 21
    {6, 9},   // PRN 22
    {1, 3},   // PRN 23
    {4, 6},   // PRN 24
    {5, 7},   // PRN 25
    {6, 8},   // PRN 26
    {7, 9},   // PRN 27
    {8, 10},  // PRN 28
    {1, 6},   // PRN 29
    {2, 7},   // PRN 30
    {3, 8},   // PRN 31
    {4, 9},   // PRN 32
};

void check_prn(int prn) {
    if (prn < CA_PRN_MIN || prn > CA_PRN_MAX)
        throw std::out_of_range("GPS C/A PRN out of range (1..32): " + std::to_string(prn));
}

} // namespace

std::pair<int, int> g2_phase_taps(int prn) {
    check_prn(prn);
    return {G2_TAPS[prn][0], G2_TAPS[prn][1]};
}

std::array<int8_t, CA_CODE_LENGTH> generate_ca_code(int prn) {
    check_prn(prn);
    const auto [a, b] = g2_phase_taps(prn);

    // Both 10-bit LFSRs start all-ones. Cells are stored 0-indexed: G[k] is
    // register stage (k+1) in the IS-GPS-200 numbering, so stage 10 is G[9].
    std::array<uint8_t, 10> G1;
    std::array<uint8_t, 10> G2;
    G1.fill(1);
    G2.fill(1);

    std::array<int8_t, CA_CODE_LENGTH> code;

    for (int i = 0; i < CA_CODE_LENGTH; ++i) {
        const uint8_t g2_phase = G2[a - 1] ^ G2[b - 1];
        const uint8_t chip = G1[9] ^ g2_phase; // output = stage-10 of G1 XOR G2 phase
        code[i] = chip ? -1 : +1;              // IS-GPS-200: logical 1 -> -1, 0 -> +1

        // Feedback: G1 taps 3,10; G2 taps 2,3,6,8,9,10 (1-indexed) -> shift in.
        const uint8_t fb1 = G1[2] ^ G1[9];
        const uint8_t fb2 = G2[1] ^ G2[2] ^ G2[5] ^ G2[7] ^ G2[8] ^ G2[9];
        for (int k = 9; k > 0; --k) {
            G1[k] = G1[k - 1];
            G2[k] = G2[k - 1];
        }
        G1[0] = fb1;
        G2[0] = fb2;
    }
    return code;
}

} // namespace gps
