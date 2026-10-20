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

// Per-PRN XB advance (chips), PRN 1..210 (IS-GPS-705 Table 3-I via PocketSDR's
// L5I_XB_adv / L5Q_XB_adv; PRN 1..32 unchanged from the first version). 120..158 are the
// SBAS GEOs (WAAS 131/133/135/138, EGNOS, GAGAN, MSAS, SDCM), the same L5 code family and
// PRN space (#190 coverage: the WAAS GEOs sit at 1176 permanently from DRAO).
constexpr int L5I_XB_ADV[210] = {
     266,  365,  804, 1138, 1509, 1559, 1756, 2084, 2170, 2303,
    2527, 2687, 2930, 3471, 3940, 4132, 4332, 4924, 5343, 5443,
    5641, 5816, 5898, 5918, 5955, 6243, 6345, 6477, 6518, 6875,
    7168, 7187, 7329, 7577, 7720, 7777, 8057, 5358, 3550, 3412,
     819, 4608, 3698,  962, 3001, 4441, 4937, 3717, 4730, 7291,
    2279, 7613, 5723, 7030, 1475, 2593, 2904, 2056, 2757, 3756,
    6205, 5053, 6437, 7789, 2311, 7432, 5155, 1593, 5841, 5014,
    1545, 3016, 4875, 2119,  229, 7634, 1406, 4506, 1819, 7580,
    5446, 6053, 7958, 5267, 2956, 3544, 1277, 2996, 1758, 3360,
    2718, 3754, 7440, 2781, 6756, 7314,  208, 5252,  696,  527,
    1399, 5879, 6868,  217, 7681, 3788, 1337, 2424, 4243, 5686,
    1955, 4791,  492, 1518, 6566, 5349,  506,  113, 1953, 2797,
     934, 3023, 3632, 1330, 4909, 4867, 1183, 3990, 6217, 1224,
    1733, 2319, 3928, 2380,  841, 5049, 7027, 1197, 7208, 8000,
     152, 6762, 3745, 4723, 5502, 4796,  123, 8142, 5091, 7875,
     330, 5272, 4912,  374, 2045, 6616, 6321, 7605, 2570, 2419,
    1234, 1922, 4317, 5110,  825,  958, 1089, 7813, 6058, 7703,
    6702, 1714, 6371, 2281, 1986, 6282, 3201, 3760, 1056, 6233,
    1150, 2823, 6250,  645, 2401, 1639, 2946, 7091,  923, 7045,
    6493, 1706, 5836,  926, 6086,  950, 5905, 3240, 6675, 3197,
    1555, 3589, 4555, 5671, 6948, 4664, 2086, 5950, 5521, 1515,
};
constexpr int L5Q_XB_ADV[210] = {
    1701,  323, 5292, 2020, 5429, 7136, 1041, 5947, 4315,  148,
     535, 1939, 5206, 5910, 3595, 5135, 6082, 6990, 3546, 1523,
    4548, 4484, 1893, 3961, 7106, 5299, 4660,  276, 4389, 3783,
    1591, 1601,  749, 1387, 1661, 3210,  708, 4226, 5604, 6375,
    3056, 1772, 3662, 4401, 5218, 2838, 6913, 1685, 1194, 6963,
    5001, 6694,  991, 7489, 2441,  639, 2097, 2498, 6470, 2399,
     242, 3768, 1186, 5246, 4259, 5907, 3870, 3262, 7387, 3069,
    2999, 7993, 7849, 4157, 5031, 5986, 4833, 5739, 7846,  898,
    2022, 7446, 6404,  155, 7862, 7795, 6121, 4840, 6585,  429,
    6020,  200, 1664, 1499, 7298, 1305, 7323, 7544, 4438, 2485,
    3387, 7319, 1853, 5781, 1874, 7555, 2132, 6441, 6722, 1192,
    2588, 2188,  297, 1540, 4138, 5231, 4789,  659,  871, 6837,
    1393, 7383,  611, 4920, 5416, 1611, 2474,  118, 1382, 1092,
    7950, 7223, 1769, 4721, 1252, 5147, 2165, 7897, 4054, 3498,
    6571, 2858, 8126, 7017, 1901,  181, 1114, 5195, 7479, 4186,
    3904, 7128, 1396, 4513, 5967, 2580, 2575, 7961, 2598, 4508,
    2090, 3685, 7748,  684,  913, 5558, 2894, 5858, 6432, 3813,
    3573, 7523, 5280, 3376, 7424, 2918, 5793, 1747, 7079, 2921,
    2490, 4119, 3373,  977,  681, 4273, 5419, 5626, 1266, 5804,
    2414, 6444, 4757,  427, 5452, 5182, 6606, 6531, 4268, 3115,
    6835,  862, 4856, 2765,   37, 1943, 7977, 2512, 4451, 4071,
};

std::array<int8_t, L5_CODE_LENGTH> l5_code(int prn, const int* adv) {
    if (prn < 1 || prn > 210)
        throw std::out_of_range("GPS L5 PRN must be 1..210");
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
