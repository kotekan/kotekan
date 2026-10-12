#include "beidouB2bCode.hpp"

#include <stdexcept> // for out_of_range

namespace beidou {

namespace {

// 13-stage Fibonacci LFSR, the same as lfsr13 in beidouB2aCode.cpp: the chip is the register
// LSB (0 -> +1, 1 -> -1), then shift right with parity(reg & tap) into bit 12. An ICD
// polynomial term x^k is tap bit 13 - k.
void lfsr13(int8_t* out, int count, uint32_t reg, uint32_t tap) {
    for (int i = 0; i < count; ++i) {
        out[i] = (reg & 1u) ? -1 : 1;
        const uint32_t fb = __builtin_parity(reg & tap) & 1u;
        reg = (fb << 12) | (reg >> 1);
    }
}

// B2b-I G1: all-ones start, truncated at 8190 chips then restarted (B2b ICD).
const std::array<int8_t, B2B_CODE_LENGTH>& g1_code_b2b() {
    static const std::array<int8_t, B2B_CODE_LENGTH> g1 = [] {
        std::array<int8_t, B2B_CODE_LENGTH> a{};
        int8_t base[8190];
        lfsr13(base, 8190, 0x1FFFu, 0b1000000011001u);
        for (int i = 0; i < 8190; ++i)
            a[i] = base[i];
        for (int i = 8190; i < B2B_CODE_LENGTH; ++i)
            a[i] = base[i - 8190];
        return a;
    }();
    return g1;
}

// Per-PRN G2 start values, PRN 1..63 (B2b ICD, via PocketSDR's B2BI_G2_init).
constexpr uint32_t B2BI_G2_START[63] = {
    0b1000000100101, 0b1000000110100, 0b1000010101101, 0b1000101001111, 0b1000101010101,
    0b1000110101110, 0b1000111101110, 0b1000111111011, 0b1001100101001, 0b1001111011010,
    0b1010000110101, 0b1010001000100, 0b1010001010101, 0b1010001011011, 0b1010001011100,
    0b1010010100011, 0b1010011110111, 0b1010100000001, 0b1010100111110, 0b1010110101011,
    0b1010110110001, 0b1011001010011, 0b1011001100010, 0b1011010011000, 0b1011010110110,
    0b1011011110010, 0b1011011111111, 0b1011100010010, 0b1011100111100, 0b1011110100001,
    0b1011111001000, 0b1011111010100, 0b1011111101011, 0b1011111110011, 0b1100001010001,
    0b1100010010100, 0b1100010110111, 0b1100100010001, 0b1100100011001, 0b1100110101011,
    0b1100110110001, 0b1100111010010, 0b1101001010101, 0b1101001110100, 0b1101011001011,
    0b1101101010111, 0b1110000110100, 0b1110010000011, 0b1110010001011, 0b1110010100011,
    0b1110010101000, 0b1110100111011, 0b1110110010111, 0b1111001001000, 0b1111010010100,
    0b1111010011001, 0b1111011011010, 0b1111011111000, 0b1111011111111, 0b1111110110101,
    0b1111110111101, 0b0101110000101, 0b0101100111011,
};

} // namespace

std::array<int8_t, B2B_CODE_LENGTH> generate_b2bi_code(int prn) {
    if (prn < 1 || prn > 63)
        throw std::out_of_range("BeiDou B2b PRN must be 1..63");
    std::array<int8_t, B2B_CODE_LENGTH> a{}, g2{};
    lfsr13(g2.data(), B2B_CODE_LENGTH, B2BI_G2_START[prn - 1], 0b0011010010011u);
    const auto& g1 = g1_code_b2b();
    for (int i = 0; i < B2B_CODE_LENGTH; ++i)
        a[i] = (int8_t)(g1[i] * g2[i]);
    return a;
}

} // namespace beidou
