#include "galileoE5aCode.hpp"

#include <stdexcept> // for out_of_range

namespace galileo {

namespace {

// Shared 14-stage Fibonacci LFSR (output = reg LSB as +1/-1 for bit 0/1; feedback = parity of
// reg & tap into the top bit) -- identical convention to gps::lfsr in gpsL5Code.cpp, verified
// bit-identical against the sky-validated python bench (l5_band_decode.py / PocketSDR).
void lfsr14(int8_t* out, int count, uint32_t reg, uint32_t tap) {
    for (int i = 0; i < count; ++i) {
        out[i] = (reg & 1u) ? -1 : 1;
        const uint32_t fb = __builtin_parity(reg & tap) & 1u;
        reg = (fb << 13) | (reg >> 1);
    }
}

// E5a-Q base register 1: all-ones start, free-runs the full 10230 chips.
// Tap = Galileo OS SIS ICD polynomial 0o40503, bit-reversed into this LFSR convention
// (the octal constants below are PRE-RESOLVED; see python/scripts/gnss/l5_band_decode.py).
const std::array<int8_t, E5A_CODE_LENGTH>& x1_code() {
    static const std::array<int8_t, E5A_CODE_LENGTH> x1 = [] {
        std::array<int8_t, E5A_CODE_LENGTH> a{};
        lfsr14(a.data(), E5A_CODE_LENGTH, 0x3FFFu, 0x2141u);
        return a;
    }();
    return x1;
}

// Per-PRN base-register-2 START VALUES (ICD Annex C via PocketSDR sdr_code.py, already
// bit-reversed into this LFSR convention), PRN 1..50. Tap polynomial 0o50661, pre-resolved.
constexpr uint32_t E5AQ_X2_START[50] = {
    0x1575, 0x1194, 0x32E5, 0x25F3, 0x1BDD, 0x0365, 0x2D7B, 0x053D,
    0x1A7C, 0x28F0, 0x3CEA, 0x128B, 0x2F0E, 0x096E, 0x1DBF, 0x3F28,
    0x2B48, 0x062C, 0x1915, 0x2EEA, 0x1344, 0x0652, 0x3E00, 0x3E8C,
    0x14C4, 0x1861, 0x1392, 0x1549, 0x36A3, 0x0C65, 0x3BFC, 0x28EB,
    0x1514, 0x35BD, 0x24F8, 0x2108, 0x2FE7, 0x0F54, 0x1DDF, 0x284E,
    0x074F, 0x006C, 0x37ED, 0x2FDE, 0x3B57, 0x2D4F, 0x090A, 0x112D,
    0x2271, 0x2435,
};

} // namespace

std::array<int8_t, E5A_CODE_LENGTH> generate_e5aq_code(int prn) {
    if (prn < 1 || prn > 50)
        throw std::out_of_range("Galileo E5a PRN must be 1..50");
    std::array<int8_t, E5A_CODE_LENGTH> a{}, x2{};
    lfsr14(x2.data(), E5A_CODE_LENGTH, E5AQ_X2_START[prn - 1], 0x6C5u);
    const auto& x1 = x1_code();
    for (int i = 0; i < E5A_CODE_LENGTH; ++i)
        a[i] = (int8_t)(x1[i] * x2[i]);
    return a;
}

// Per-PRN base-register-2 START VALUES for E5a-I (the DATA channel), same source and
// convention as E5AQ_X2_START (ICD Annex C via PocketSDR sdr_code.py, bit-reversed into this
// LFSR convention). Derived by rev_reg(E5AI_X2_init, 14) -- VALIDATED by reproducing all 50
// E5a-Q values from PocketSDR's E5AQ_X2_init through the identical transform (2026-07-30).
namespace {
constexpr uint32_t E5AI_X2_START[50] = {
    0x28C3, 0x0E46, 0x345D, 0x3FA1, 0x14D9, 0x333B, 0x0C76, 0x3EAA,
    0x3AB0, 0x1E43, 0x09DD, 0x175C, 0x3FCF, 0x191E, 0x0E2C, 0x2836,
    0x1545, 0x2672, 0x1FE5, 0x0660, 0x03B2, 0x175E, 0x293D, 0x10F3,
    0x141A, 0x2026, 0x3AE7, 0x3A69, 0x3263, 0x1D5D, 0x02B0, 0x2446,
    0x2B33, 0x0B89, 0x1CBB, 0x3EE8, 0x1CD1, 0x2A37, 0x3677, 0x2D65,
    0x0D06, 0x3A1D, 0x086C, 0x2D33, 0x2B27, 0x3755, 0x33E1, 0x22F3,
    0x14DE, 0x3F5A,
};
} // namespace

std::array<int8_t, E5A_CODE_LENGTH> generate_e5ai_code(int prn) {
    if (prn < 1 || prn > 50)
        throw std::out_of_range("Galileo E5a PRN must be 1..50");
    std::array<int8_t, E5A_CODE_LENGTH> a{}, x2{};
    lfsr14(x2.data(), E5A_CODE_LENGTH, E5AI_X2_START[prn - 1], 0x6C5u);
    const auto& x1 = x1_code(); // shared base register 1 (all-ones), identical to E5a-Q
    for (int i = 0; i < E5A_CODE_LENGTH; ++i)
        a[i] = (int8_t)(x1[i] * x2[i]);
    return a;
}

// E5a-Q CS100 secondary (ICD Annex C, hex MSB-first, 100 chips / PRN), PRN 1..50.
namespace {
const char* const CS100_HEX[50] = {
    "83F6F69D8F6E15411FB8C9B1C",
    "66558BD3CE0C7792E83350525",
    "59A025A9C1AF0651B779A8381",
    "D3A32640782F7B18E4DF754B7",
    "B91FCAD7760C218FA59348A93",
    "BAC77E933A779140F094FBF98",
    "537785DE280927C6B58BA6776",
    "EFCAB4B65F38531ECA22257E2",
    "79F8CAE838475EA5584BEFC9B",
    "CA5170FEA3A810EC606B66494",
    "1FC32410652A2C49BD845E567",
    "FE0A9A7AFDAC44E42CB95D261",
    "B03062DC2B71995D5AD8B7DBE",
    "F6C398993F598E2DF4235D3D5",
    "1BB2FB8B5BF24395C2EF3C5A1",
    "2F920687D238CC7046EF6AFC9",
    "34163886FC4ED7F2A92EFDBB8",
    "66A872CE47833FB2DFD5625AD",
    "99D5A70162C920A4BB9DE1CA8",
    "81D71BD6E069A7ACCBEDC66CA",
    "A654524074A9E6780DB9D3EC6",
    "C3396A101BEDAF623CFC5BB37",
    "C3D4AB211DF36F2111F2141CD",
    "3DFF25EAE761739265AF145C1",
    "994909E0757D70CDE389102B5",
    "B938535522D119F40C25FDAEC",
    "C71AB549C0491537026B390B7",
    "0CDB8C9E7B53F55F5B0A0597B",
    "61C5FA252F1AF81144766494F",
    "626027778FD3C6BB4BAA7A59D",
    "E745412FF53DEBD03F1C9A633",
    "3592AC083F3175FA724639098",
    "52284D941C3DCAF2721DDB1FD",
    "73B3D8F0AD55DF4FE814ED890",
    "94BF16C83BD7462F6498E0282",
    "A8C3DE1AC668089B0B45B3579",
    "E23FFC2DD2C14388AD8D6BEC8",
    "F2AC871CDF89DDC06B5960D2B",
    "06191EC1F622A77A526868BA1",
    "22D6E2A768E5F35FFC8E01796",
    "25310A06675EB271F2A09EA1D",
    "9F7993C621D4BEC81A0535703",
    "D62999EACF1C99083C0B4A417",
    "F665A7EA441BAA4EA0D01078C",
    "46F3D3043F24CDEABD6F79543",
    "E2E3E8254616BD96CEFCA651A",
    "E548231A82F9A01A19DB5E1B2",
    "265C7F90A16F49EDE2AA706C8",
    "364A3A9EB0F0481DA0199D7EA",
    "9810A7A898961263A0F749F56",
};
} // namespace

std::array<int8_t, 100> e5aq_secondary(int prn) {
    if (prn < 1 || prn > 50)
        throw std::out_of_range("Galileo E5a PRN must be 1..50");
    std::array<int8_t, 100> s{};
    const char* hex = CS100_HEX[prn - 1];
    for (int i = 0; i < 100; ++i) {
        const char c = hex[i / 4];
        const int v = (c <= '9') ? c - '0' : (c | 0x20) - 'a' + 10;
        s[i] = ((v >> (3 - i % 4)) & 1) ? -1 : 1;
    }
    return s;
}

// E5a-I CS20 secondary (ICD Annex C, hex MSB-first, 20 chips, SAME for every satellite --
// verified identical source/convention to CS100 above, which matches live E5a-Q). '842E9'.
std::array<int8_t, 20> e5ai_secondary() {
    static const char* const hex = "842E9";
    std::array<int8_t, 20> s{};
    for (int i = 0; i < 20; ++i) {
        const char c = hex[i / 4];
        const int v = (c <= '9') ? c - '0' : (c | 0x20) - 'a' + 10;
        s[i] = ((v >> (3 - i % 4)) & 1) ? -1 : 1;
    }
    return s;
}

} // namespace galileo
