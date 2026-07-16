#include "beidouB2aCode.hpp"

#include <stdexcept> // for out_of_range

namespace beidou {

namespace {

// 13-stage Fibonacci LFSR, same convention as gps::lfsr / galileo::lfsr14 (verified
// bit-identical against the sky-validated python bench, l5_band_decode.py / PocketSDR).
void lfsr13(int8_t* out, int count, uint32_t reg, uint32_t tap) {
    for (int i = 0; i < count; ++i) {
        out[i] = (reg & 1u) ? -1 : 1;
        const uint32_t fb = __builtin_parity(reg & tap) & 1u;
        reg = (fb << 12) | (reg >> 1);
    }
}

// B2a-pilot G1: all-ones start, truncated at 8190 then restarted (BDS-3 B2a ICD 5.2.2).
const std::array<int8_t, B2A_CODE_LENGTH>& g1_code() {
    static const std::array<int8_t, B2A_CODE_LENGTH> g1 = [] {
        std::array<int8_t, B2A_CODE_LENGTH> a{};
        int8_t base[8190];
        lfsr13(base, 8190, 0x1FFFu, 0b0010011000001u);
        for (int i = 0; i < 8190; ++i)
            a[i] = base[i];
        for (int i = 8190; i < B2A_CODE_LENGTH; ++i)
            a[i] = base[i - 8190];
        return a;
    }();
    return g1;
}

// Per-PRN G2 register START VALUES (BDS-3 B2a ICD Table 5-4 via PocketSDR), PRN 1..63.
constexpr uint32_t B2AP_G2_START[63] = {
    0x1025, 0x1034, 0x10AD, 0x114F, 0x1155, 0x11AE, 0x11EE, 0x11FB,
    0x1329, 0x13DA, 0x1435, 0x1444, 0x1455, 0x145B, 0x145C, 0x14A3,
    0x14F7, 0x1501, 0x153E, 0x15AB, 0x15B1, 0x1653, 0x1662, 0x1698,
    0x16B6, 0x16F2, 0x16FF, 0x1712, 0x173C, 0x17A1, 0x17C8, 0x17D4,
    0x17EB, 0x17F3, 0x1851, 0x1894, 0x18B7, 0x1911, 0x1919, 0x19AB,
    0x19B1, 0x19D2, 0x1A55, 0x1A74, 0x1ACB, 0x1B57, 0x1C34, 0x1C83,
    0x1C8B, 0x1CA3, 0x1CA8, 0x1D3B, 0x1D97, 0x1E48, 0x1E94, 0x1E99,
    0x1EDA, 0x1EF8, 0x1EFF, 0x1FB5, 0x1486, 0x05F8, 0x0355,
};

// B2a-pilot secondary: 100 chips truncated from a Weil-1021 sequence, per-PRN
// (phase difference w, truncation point tp) -- ICD Table 5-5 via PocketSDR.
constexpr int16_t B2AS_PH_DIFF[63] = {
    123, 55, 40, 139, 31, 175, 350, 450, 478, 8,
    73, 97, 213, 407, 476, 4, 15, 47, 163, 280,
    322, 353, 375, 510, 332, 7, 13, 16, 18, 25,
    50, 81, 118, 127, 132, 134, 164, 177, 208, 249,
    276, 349, 439, 477, 498, 88, 155, 330, 3, 21,
    84, 111, 128, 153, 197, 199, 214, 256, 265, 291,
    324, 326, 340,
};
constexpr int16_t B2AS_TRUNC[63] = {
    138, 570, 351, 77, 885, 247, 413, 180, 3, 26,
    17, 172, 30, 1008, 646, 158, 170, 99, 53, 179,
    925, 114, 10, 584, 60, 3, 684, 263, 545, 22,
    546, 190, 303, 234, 38, 822, 57, 668, 697, 93,
    18, 66, 318, 133, 98, 70, 132, 26, 354, 58,
    41, 182, 944, 205, 23, 1, 792, 641, 83, 7,
    111, 96, 92,
};

// Legendre sequence over 1021 (the B1C machinery's twin at a smaller prime).
const std::array<int8_t, 1021>& legendre1021() {
    static const std::array<int8_t, 1021> L = [] {
        std::array<int8_t, 1021> a{};
        // a[k] = +1 if k is a nonzero quadratic residue mod 1021, else -1; a[0] = -1
        // (chip 0 -> logic 0 -> +1? convention below matches the python bench: CHIP[b],
        //  b = 1 for residue -> -1). Verified bit-exact vs pocketsdr gen_legendre_seq.
        bool isres[1021] = {};
        for (int x = 1; x < 1021; ++x)
            isres[(x * x) % 1021] = true;
        for (int k = 0; k < 1021; ++k)
            a[k] = (k != 0 && isres[k]) ? -1 : 1;
        return a;
    }();
    return L;
}

} // namespace

std::array<int8_t, B2A_CODE_LENGTH> generate_b2ap_code(int prn) {
    if (prn < 1 || prn > 63)
        throw std::out_of_range("BeiDou B2a PRN must be 1..63");
    std::array<int8_t, B2A_CODE_LENGTH> a{}, g2{};
    lfsr13(g2.data(), B2A_CODE_LENGTH, B2AP_G2_START[prn - 1], 0b1000101100011u);
    const auto& g1 = g1_code();
    for (int i = 0; i < B2A_CODE_LENGTH; ++i)
        a[i] = (int8_t)(g1[i] * g2[i]);
    return a;
}

std::array<int8_t, 100> b2ap_secondary(int prn) {
    if (prn < 1 || prn > 63)
        throw std::out_of_range("BeiDou B2a PRN must be 1..63");
    const auto& L = legendre1021();
    const int w = B2AS_PH_DIFF[prn - 1];
    const int tp = B2AS_TRUNC[prn - 1];
    std::array<int8_t, 100> s{};
    for (int i = 0; i < 100; ++i) {
        const int j = (i + tp - 1) % 1021;
        s[i] = (int8_t)(L[j] * L[(j + w) % 1021]);
    }
    return s;
}

} // namespace beidou
