#include "beidouB1CCode.hpp"

#include <stdexcept>
#include <string>

namespace beidou {

static const int B1CP_PH_DIFF[63] = {
    796, 156, 4198, 3941, 1374, 1338, 1833, 2521, 3175, 168, 2715, 4408,
    3160, 2796, 459, 3594, 4813, 586, 1428, 2371, 2285, 3377, 4965, 3779,
    4547, 1646, 1430, 607, 2118, 4709, 1149, 3283, 2473, 1006, 3670, 1817,
    771, 2173, 740, 1433, 2458, 3459, 2155, 1205, 413, 874, 2463, 1106,
    1590, 3873, 4026, 4272, 3556, 128, 1200, 130, 4494, 1871, 3073, 4386,
    4098, 1923, 1176,
};

static const int B1CP_TRUNC[63] = {
    7575, 2369, 5688, 539, 2270, 7306, 6457, 6254, 5644, 7119, 1402, 5557,
    5764, 1073, 7001, 5910, 10060, 2710, 1546, 6887, 1883, 5613, 5062, 1038,
    10170, 6484, 1718, 2535, 1158, 526, 7331, 5844, 6423, 6968, 1280, 1838,
    1989, 6468, 2091, 1581, 1453, 6252, 7122, 7711, 7216, 2113, 1095, 1628,
    1713, 6102, 6123, 6070, 1115, 8047, 6795, 2575, 53, 1729, 6388, 682,
    5565, 7160, 2277,
};

// B1C-DATA Weil-code parameters (BDS-SIS-ICD-B1C Table via PocketSDR B1CD_ph_diff/trunc_pnt,
// used directly). Same Legendre-10243 Weil construction as the pilot, different phase-diff /
// truncation tables -- verified generate_b1cd_code bit-exact vs PocketSDR gen_code_B1CD
// (de-BOC) for PRN 19/20/30/45/63 (2026-07-30).
static const int B1CD_PH_DIFF[63] = {
    2678, 4802, 958, 859, 3843, 2232, 124, 4352, 1816, 1126, 1860, 4800,
    2267, 424, 4192, 4333, 2656, 4148, 243, 1330, 1593, 1470, 882, 3202,
    5095, 2546, 1733, 4795, 4577, 1627, 3638, 2553, 3646, 1087, 1843, 216,
    2245, 726, 1966, 670, 4130, 53, 4830, 182, 2181, 2006, 1080, 2288,
    2027, 271, 915, 497, 139, 3693, 2054, 4342, 3342, 2592, 1007, 310,
    4203, 455, 4318,
};
static const int B1CD_TRUNC[63] = {
    699, 694, 7318, 2127, 715, 6682, 7850, 5495, 1162, 7682, 6792, 9973,
    6596, 2092, 19, 10151, 6297, 5766, 2359, 7136, 1706, 2128, 6827, 693,
    9729, 1620, 6805, 534, 712, 1929, 5355, 6139, 6339, 1470, 6867, 7851,
    1162, 7659, 1156, 2672, 6043, 2862, 180, 2663, 6940, 1645, 1582, 951,
    6878, 7701, 1823, 2391, 2606, 822, 6403, 239, 442, 6769, 2560, 2502,
    5072, 7268, 341,
};

static const int B1CS_PH_DIFF[63] = {
    269, 1448, 1028, 1324, 822, 5, 155, 458, 310, 959, 1238, 1180,
    1288, 334, 885, 1362, 181, 1648, 838, 313, 750, 225, 1477, 309,
    108, 1457, 149, 322, 271, 576, 1103, 450, 399, 241, 1045, 164,
    513, 687, 422, 303, 324, 495, 725, 780, 367, 882, 631, 37,
    647, 1043, 24, 120, 134, 136, 158, 214, 335, 340, 661, 889,
    929, 1002, 1149,
};

static const int B1CS_TRUNC[63] = {
    1889, 1268, 1593, 1186, 1239, 1930, 176, 1696, 26, 1344, 1271, 1182,
    1381, 1604, 1333, 1185, 31, 704, 1190, 1646, 1385, 113, 860, 1656,
    1921, 1173, 1928, 57, 150, 1214, 1148, 1458, 1519, 1635, 1257, 1687,
    1382, 1514, 1, 1583, 1806, 1664, 1338, 1111, 1706, 1543, 1813, 228,
    2871, 2884, 1823, 75, 11, 63, 1937, 22, 1768, 1526, 1402, 1445,
    1680, 1290, 1245,
};

namespace {
// Legendre sequence: L[(i*i) mod N] = -1 for i in 1..N-1, +1 elsewhere (incl. L[0]).
std::vector<int8_t> legendre(int N) {
    std::vector<int8_t> L((size_t)N, 1);
    for (long long i = 1; i < N; ++i)
        L[(size_t)((i * i) % N)] = -1;
    return L;
}
void check_prn(int prn) {
    if (prn < 1 || prn > 63)
        throw std::out_of_range("beidou B1C PRN " + std::to_string(prn) + " (valid 1..63)");
}
} // namespace

std::array<int8_t, B1C_CODE_LENGTH> generate_b1cp_code(int prn) {
    check_prn(prn);
    static const std::vector<int8_t> L = legendre(10243);
    const int w = B1CP_PH_DIFF[prn - 1], p = B1CP_TRUNC[prn - 1];
    std::array<int8_t, B1C_CODE_LENGTH> out{};
    for (int i = 0; i < B1C_CODE_LENGTH; ++i) {
        const int j = (i + p - 1) % 10243;
        out[(size_t)i] = (int8_t)(L[(size_t)j] * L[(size_t)((j + w) % 10243)]);
    }
    return out;
}

std::array<int8_t, B1C_CODE_LENGTH> generate_b1cd_code(int prn) {
    check_prn(prn);
    static const std::vector<int8_t> L = legendre(10243);
    const int w = B1CD_PH_DIFF[prn - 1], p = B1CD_TRUNC[prn - 1];
    std::array<int8_t, B1C_CODE_LENGTH> out{};
    for (int i = 0; i < B1C_CODE_LENGTH; ++i) {
        const int j = (i + p - 1) % 10243;
        out[(size_t)i] = (int8_t)(L[(size_t)j] * L[(size_t)((j + w) % 10243)]);
    }
    return out;
}

std::vector<int8_t> generate_b1cp_secondary(int prn) {
    check_prn(prn);
    static const std::vector<int8_t> LS = legendre(3607);
    const int w = B1CS_PH_DIFF[prn - 1], p = B1CS_TRUNC[prn - 1];
    std::vector<int8_t> out((size_t)B1C_SECONDARY_LENGTH);
    for (int i = 0; i < B1C_SECONDARY_LENGTH; ++i) {
        const int j = (i + p - 1) % 3607;
        out[(size_t)i] = (int8_t)(LS[(size_t)j] * LS[(size_t)((j + w) % 3607)]);
    }
    return out;
}

} // namespace beidou
