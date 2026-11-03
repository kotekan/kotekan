#include "guowang1202Code.hpp"

#include <stdexcept>
#include <string>

namespace guowang {
namespace {

// One period of G2' from the all-ones state: s[k] = s[k-1] ^ s[k-3] ^ s[k-8] ^ s[k-9] ^ s[k-10]
// ^ s[k-11], the recurrence of 1 + x + x^3 + x^8 + x^9 + x^10 + x^11.
std::array<uint8_t, GW1202_G2P_PERIOD> g2prime() {
    std::array<uint8_t, GW1202_G2P_PERIOD + 11> s{};
    for (int k = 0; k < 11; ++k)
        s[(size_t)k] = 1;
    for (int k = 11; k < GW1202_G2P_PERIOD + 11; ++k)
        s[(size_t)k] = s[(size_t)k - 1] ^ s[(size_t)k - 3] ^ s[(size_t)k - 8] ^ s[(size_t)k - 9]
                       ^ s[(size_t)k - 10] ^ s[(size_t)k - 11];
    std::array<uint8_t, GW1202_G2P_PERIOD> out{};
    for (int k = 0; k < GW1202_G2P_PERIOD; ++k)
        out[(size_t)k] = s[(size_t)k];
    return out;
}

// B1I's G1 (beidouB1ICode: index 0 is the output end, initial state 01010101010, taps
// 0,1,2,3,4,10), GW1202_CODE_LENGTH chips from GW1202_G1_PHASE on.
std::array<uint8_t, GW1202_CODE_LENGTH> g1() {
    int r[11];
    for (int i = 0; i < 11; ++i)
        r[i] = (i % 2 == 1) ? 1 : 0;
    std::array<uint8_t, GW1202_CODE_LENGTH> out{};
    for (int k = 0; k < GW1202_G1_PHASE + GW1202_CODE_LENGTH; ++k) {
        if (k >= GW1202_G1_PHASE)
            out[(size_t)(k - GW1202_G1_PHASE)] = (uint8_t)r[0];
        const int fb = r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[10];
        for (int i = 0; i < 10; ++i)
            r[i] = r[i + 1];
        r[10] = fb;
    }
    return out;
}

} // namespace

std::array<int8_t, GW1202_CODE_LENGTH> generate_gw1202_code(int g2_phase) {
    if (g2_phase < 0 || g2_phase >= GW1202_G2P_PERIOD)
        throw std::out_of_range("generate_gw1202_code: G2' phase must be 0..2046, got "
                                + std::to_string(g2_phase));
    static const std::array<uint8_t, GW1202_CODE_LENGTH> a = g1();
    static const std::array<uint8_t, GW1202_G2P_PERIOD> b = g2prime();
    std::array<int8_t, GW1202_CODE_LENGTH> out{};
    for (int k = 0; k < GW1202_CODE_LENGTH; ++k)
        out[(size_t)k] = (a[(size_t)k] ^ b[(size_t)((g2_phase + k) % GW1202_G2P_PERIOD)]) ? -1 : 1;
    return out;
}

std::vector<int8_t> generate_gw1202_frame(int prn) {
    if (prn < 1 || prn > GW1202_MAX_PRN)
        throw std::out_of_range("generate_gw1202_frame: PRN must be 1..2047 (G2' phase + 1), got "
                                + std::to_string(prn));
    const auto code = generate_gw1202_code(prn - 1);
    std::vector<int8_t> out((size_t)GW1202_FRAME_LENGTH, 0);
    for (int k = 0; k < GW1202_CODE_LENGTH; ++k)
        out[(size_t)k] = code[(size_t)k];
    return out;
}

} // namespace guowang
