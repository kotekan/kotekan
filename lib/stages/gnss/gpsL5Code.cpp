#include "gpsL5Code.hpp"

#include <stdexcept> // for out_of_range

namespace gps {

namespace {

// Fibonacci LFSR (PocketSDR convention): output chip = (R & 1) ? -1 : +1, then
// shift right and feed parity(R & tap) into bit n-1. R holds the n-bit state.
void lfsr(int8_t* out, int count, uint32_t reg, uint32_t tap, int n) {
    for (int i = 0; i < count; ++i) {
        out[i] = (reg & 1u) ? -1 : 1;
        const uint32_t fb = __builtin_parity(reg & tap) & 1u;
        reg = (fb << (n - 1)) | (reg >> 1);
    }
}

// XA: 13-stage LFSR (taps for x^9+x^10+x^12+x^13) run for 8190 chips, then its
// first 2040 chips repeated -> 10230 (XA "short cycle" wrap).
const std::array<int8_t, L5_CODE_LENGTH>& xa_code() {
    static const std::array<int8_t, L5_CODE_LENGTH> xa = [] {
        std::array<int8_t, L5_CODE_LENGTH> a{};
        int8_t base[8190];
        lfsr(base, 8190, 0x1FFFu, 0b0000000011011u, 13);
        for (int i = 0; i < 8190; ++i)
            a[i] = base[i];
        for (int i = 8190; i < L5_CODE_LENGTH; ++i)
            a[i] = base[i - 8190];
        return a;
    }();
    return xa;
}

// XB: 13-stage LFSR run for the full 10230 chips.
const std::array<int8_t, L5_CODE_LENGTH>& xb_code() {
    static const std::array<int8_t, L5_CODE_LENGTH> xb = [] {
        std::array<int8_t, L5_CODE_LENGTH> a{};
        lfsr(a.data(), L5_CODE_LENGTH, 0x1FFFu, 0b1011011100011u, 13);
        return a;
    }();
    return xb;
}

// Per-PRN XB advance (chips), PRN 1..32 (IS-GPS-705 Table 3-I / PocketSDR).
constexpr int L5I_XB_ADV[32] = {266,  365,  804,  1138, 1509, 1559, 1756, 2084,
                                2170, 2303, 2527, 2687, 2930, 3471, 3940, 4132,
                                4332, 4924, 5343, 5443, 5641, 5816, 5898, 5918,
                                5955, 6243, 6345, 6477, 6518, 6875, 7168, 7187};
constexpr int L5Q_XB_ADV[32] = {1701, 323,  5292, 2020, 5429, 7136, 1041, 5947,
                                4315, 148,  535,  1939, 5206, 5910, 3595, 5135,
                                6082, 6990, 3546, 1523, 4548, 4484, 1893, 3961,
                                7106, 5299, 4660, 276,  4389, 3783, 1591, 1601};

std::array<int8_t, L5_CODE_LENGTH> l5_code(int prn, const int* adv) {
    if (prn < 1 || prn > 32)
        throw std::out_of_range("GPS L5 PRN must be 1..32");
    const auto& xa = xa_code();
    const auto& xb = xb_code();
    const int shift = adv[prn - 1];
    std::array<int8_t, L5_CODE_LENGTH> code{};
    for (int i = 0; i < L5_CODE_LENGTH; ++i) {
        const int j = (i + shift) % L5_CODE_LENGTH; // XB advanced (left-rolled) by `shift`
        code[i] = static_cast<int8_t>(xa[i] * xb[j]); // +-1 product == XOR
    }
    return code;
}

} // namespace

std::array<int8_t, L5_CODE_LENGTH> generate_l5i_code(int prn) {
    return l5_code(prn, L5I_XB_ADV);
}

std::array<int8_t, L5_CODE_LENGTH> generate_l5q_code(int prn) {
    return l5_code(prn, L5Q_XB_ADV);
}

const std::array<int8_t, L5_NH10_LENGTH> L5_NH10 = {1, 1, 1, 1, -1, -1, 1, -1, 1, -1};
const std::array<int8_t, L5_NH20_LENGTH> L5_NH20 = {1,  1,  1, 1,  1,  -1, 1,  1,  -1, -1,
                                                    1, -1, 1, -1, 1,  1,  -1, -1, -1, 1};

} // namespace gps
