#include "gpsL5Code.hpp"

#include <stdexcept> // for out_of_range

namespace gps {

namespace {

// n-stage Fibonacci LFSR (PocketSDR convention): output chip = (reg & 1) ? -1 : +1, then
// shift right and feed parity(reg & tap) into bit n-1. An ICD polynomial term x^k is tap bit n-k.
void lfsr(int8_t* out, int count, uint32_t reg, uint32_t tap, int n) {
    for (int i = 0; i < count; ++i) {
        out[i] = (reg & 1u) ? -1 : 1;
        const uint32_t fb = __builtin_parity(reg & tap) & 1u;
        reg = (fb << (n - 1)) | (reg >> 1);
    }
}

// XA (1 + x^9 + x^10 + x^12 + x^13), short-cycled: reset to all-ones after 8190 chips, so
// chips 8190..10229 repeat its first 2040.
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

// XB (1 + x + x^3 + x^4 + x^6 + x^7 + x^8 + x^12 + x^13): one full 8191-chip period from
// all-ones. XB is not short-cycled, so within a code period it wraps at 8191, not 10230.
constexpr int L5_XB_PERIOD = 8191;
const std::array<int8_t, L5_XB_PERIOD>& xb_code() {
    static const std::array<int8_t, L5_XB_PERIOD> xb = [] {
        std::array<int8_t, L5_XB_PERIOD> a{};
        lfsr(a.data(), L5_XB_PERIOD, 0x1FFFu, 0b1011011100011u, 13);
        return a;
    }();
    return xb;
}

// Per-PRN XB advance (chips), PRN 1..32 (IS-GPS-705 Table 3-I / PocketSDR).
constexpr int L5I_XB_ADV[32] = {266,  365,  804,  1138, 1509, 1559, 1756, 2084, 2170, 2303, 2527,
                                2687, 2930, 3471, 3940, 4132, 4332, 4924, 5343, 5443, 5641, 5816,
                                5898, 5918, 5955, 6243, 6345, 6477, 6518, 6875, 7168, 7187};
constexpr int L5Q_XB_ADV[32] = {1701, 323,  5292, 2020, 5429, 7136, 1041, 5947, 4315, 148,  535,
                                1939, 5206, 5910, 3595, 5135, 6082, 6990, 3546, 1523, 4548, 4484,
                                1893, 3961, 7106, 5299, 4660, 276,  4389, 3783, 1591, 1601};

std::array<int8_t, L5_CODE_LENGTH> l5_code(int prn, const int* adv) {
    if (prn < 1 || prn > 32)
        throw std::out_of_range("GPS L5 PRN must be 1..32");
    const auto& xa = xa_code();
    const auto& xb = xb_code();
    const int shift = adv[prn - 1];
    std::array<int8_t, L5_CODE_LENGTH> code{};
    for (int i = 0; i < L5_CODE_LENGTH; ++i) {
        const int j = (i + shift) % L5_XB_PERIOD;     // XB, `shift` chips past all-ones
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
const std::array<int8_t, L5_NH20_LENGTH> L5_NH20 = {1, 1,  1, 1,  1, -1, 1,  1,  -1, -1,
                                                    1, -1, 1, -1, 1, 1,  -1, -1, -1, 1};

} // namespace gps
