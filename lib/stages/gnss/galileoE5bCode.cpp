#include "galileoE5bCode.hpp"

#include <stdexcept> // for out_of_range

namespace galileo {

namespace {

// Shared 14-stage Fibonacci LFSR -- identical convention to galileoE5aCode.cpp's lfsr14 (output
// = reg LSB as +1/-1 for bit 0/1; feedback = parity of reg & tap into the top bit), verified
// bit-identical against PocketSDR (python/scripts/gnss/e5b_code_check.py).
void lfsr14(int8_t* out, int count, uint32_t reg, uint32_t tap) {
    for (int i = 0; i < count; ++i) {
        out[i] = (reg & 1u) ? -1 : 1;
        const uint32_t fb = __builtin_parity(reg & tap) & 1u;
        reg = (fb << 13) | (reg >> 1);
    }
}

// E5b base register 1 (X1): all-ones start, free-runs the full 10230 chips. Tap = Galileo OS
// SIS ICD E5b polynomial 0o64021, bit-reversed into this LFSR convention (pre-resolved; see
// e5b_code_check.py). NOTE: distinct from E5a's X1 tap -- E5b is a different code family.
const std::array<int8_t, E5B_CODE_LENGTH>& x1_code() {
    static const std::array<int8_t, E5B_CODE_LENGTH> x1 = [] {
        std::array<int8_t, E5B_CODE_LENGTH> a{};
        lfsr14(a.data(), E5B_CODE_LENGTH, 0x3FFFu, 0x040Bu);
        return a;
    }();
    return x1;
}

// Per-PRN base-register-2 START VALUES for E5b-I (DATA channel), ICD Annex C via PocketSDR
// E5BI_X2_init, bit-reversed into this LFSR convention. X2 tap 0o51445 -> 0x1265 (pre-resolved).
constexpr uint32_t E5BI_X2_START[50] = {
    0x025C, 0x390D, 0x1540, 0x1B9E, 0x2386, 0x01A8, 0x3EB0, 0x320D,
    0x2AF0, 0x3E61, 0x0BC8, 0x26FD, 0x0163, 0x0F87, 0x0B2C, 0x1F7C,
    0x264B, 0x35C2, 0x2DE0, 0x3F0D, 0x0972, 0x3EB2, 0x2C97, 0x1521,
    0x2967, 0x0B75, 0x3031, 0x354B, 0x3EC8, 0x3FE3, 0x129D, 0x3FC9,
    0x2F8F, 0x2F1B, 0x259B, 0x0E9A, 0x363C, 0x1C42, 0x1B37, 0x36A8,
    0x25D4, 0x2283, 0x0406, 0x1FAB, 0x1AC3, 0x363F, 0x120D, 0x07F7,
    0x076C, 0x3C0A,
};

// Per-PRN base-register-2 START VALUES for E5b-Q (PILOT channel), ICD Annex C via PocketSDR
// E5BQ_X2_init, bit-reversed. X2 tap 0o43143 -> 0x2331 (pre-resolved).
constexpr uint32_t E5BQ_X2_START[50] = {
    0x26D8, 0x318C, 0x12D5, 0x27D9, 0x3420, 0x2E4F, 0x05FE, 0x29E4,
    0x281A, 0x019F, 0x2D83, 0x3E41, 0x2338, 0x1373, 0x199D, 0x39B4,
    0x1D28, 0x3403, 0x12C0, 0x23E2, 0x3BED, 0x080F, 0x34E3, 0x137C,
    0x0471, 0x0E81, 0x1352, 0x18E2, 0x3187, 0x2512, 0x0335, 0x2B7E,
    0x0DCB, 0x0651, 0x1981, 0x13D2, 0x195F, 0x09CE, 0x0E16, 0x2735,
    0x1975, 0x3D89, 0x1EC4, 0x1DCE, 0x2F04, 0x1A68, 0x18F2, 0x241F,
    0x211D, 0x0A13,
};

constexpr uint32_t E5BI_X2_TAP = 0x1265u;
constexpr uint32_t E5BQ_X2_TAP = 0x2331u;

std::array<int8_t, E5B_CODE_LENGTH> e5b_primary(uint32_t x2_start, uint32_t x2_tap) {
    std::array<int8_t, E5B_CODE_LENGTH> a{}, x2{};
    lfsr14(x2.data(), E5B_CODE_LENGTH, x2_start, x2_tap);
    const auto& x1 = x1_code(); // shared all-ones X1 for both E5b components
    for (int i = 0; i < E5B_CODE_LENGTH; ++i)
        a[i] = (int8_t)(x1[i] * x2[i]);
    return a;
}

// Decode a hex secondary string (MSB-first, 4 bits/char) into +-1 chips.
template <int N> std::array<int8_t, N> hex_secondary(const char* hex) {
    std::array<int8_t, N> s{};
    for (int i = 0; i < N; ++i) {
        const char c = hex[i / 4];
        const int v = (c <= '9') ? c - '0' : (c | 0x20) - 'a' + 10;
        s[i] = ((v >> (3 - i % 4)) & 1) ? -1 : 1;
    }
    return s;
}

// E5b-Q CS100 secondary (ICD Annex C, hex MSB-first, 100 chips / PRN), PRN 1..50 -- PocketSDR
// CS100[50:100] (the E5bQ half; the E5aQ half is in galileoE5aCode.cpp).
const char* const CS100_E5BQ_HEX[50] = {
    "CFF914EE3C6126A49FD5E5C94", "FC317C9A9BF8C6038B5CADAB3", "A2EAD74B6F9866E414393F239",
    "72F2B1180FA6B802CB84DF997", "13E3AE93BC52391D09E84A982", "77C04202B91B22C6D3469768E",
    "FEBC592DD7C69AB103D0BB29C", "0B494077E7C66FB6C51942A77", "DD0E321837A3D52169B7B577C",
    "43DEA90EA6C483E7990C3223F", "0366AB33F0167B6FA979DAE18", "99CCBBFAB1242CBE31E1BD52D",
    "A3466923CEFDF451EC0FCED22", "1A5271F22A6F9A8D76E79B7F0", "3204A6BB91B49D1A2D3857960",
    "32F83ADD43B599CBFB8628E5B", "3871FB0D89DB77553EB613CC1", "6A3CBDFF2D64D17E02773C645",
    "2BCD09889A1D7FC219F2EDE3B", "3E49467F4D4280B9942CD6F8C", "658E336DCFD9809F86D54A501",
    "ED4284F345170CF77268C8584", "29ECCE910D832CAF15E3DF5D1", "456CCF7FE9353D50E87A708FA",
    "FB757CC9E18CBC02BF1B84B9A", "5686229A8D98224BC426BC7FC", "700A2D325EA14C4B7B7AA8338",
    "1210A330B4D3B507D854CBA3F", "438EE410BD2F7DBCDD85565BA", "4B9764CC455AE1F61F7DA432B",
    "BF1F45FDDA3594ACF3C4CC806", "DA425440FE8F6E2C11B8EC1A4", "EE2C8057A7C16999AFA33FED1",
    "2C8BD7D8395C61DFA96243491", "391E4BB6BC43E98150CDDCADA", "399F72A9EADB42C90C3ECF7F0",
    "93031FDEA588F88E83951270C", "BA8061462D873705E95D5CB37", "D24188F88544EB121E963FD34",
    "D5F6A8BB081D8F383825A4DCA", "0FA4A205F0D76088D08EAF267", "272E909FAEBC65215E263E258",
    "3370F35A674922828465FC816", "54EF96116D4A0C8DB0E07101F", "DE347C7B27FADC48EF1826A2B",
    "01B16ECA6FC343AE08C5B8944", "1854DB743500EE94D8FC768ED", "28E40C684C87370CD0597FAB4",
    "5E42C19717093353BCAAF4033", "64310BAD8EB5B36E38646AF01",
};

} // namespace

std::array<int8_t, E5B_CODE_LENGTH> generate_e5bi_code(int prn) {
    if (prn < 1 || prn > 50)
        throw std::out_of_range("Galileo E5b PRN must be 1..50");
    return e5b_primary(E5BI_X2_START[prn - 1], E5BI_X2_TAP);
}

std::array<int8_t, E5B_CODE_LENGTH> generate_e5bq_code(int prn) {
    if (prn < 1 || prn > 50)
        throw std::out_of_range("Galileo E5b PRN must be 1..50");
    return e5b_primary(E5BQ_X2_START[prn - 1], E5BQ_X2_TAP);
}

std::array<int8_t, 100> e5bq_secondary(int prn) {
    if (prn < 1 || prn > 50)
        throw std::out_of_range("Galileo E5b PRN must be 1..50");
    return hex_secondary<100>(CS100_E5BQ_HEX[prn - 1]);
}

// E5b-I CS4 secondary (ICD Annex C, 'E' = 1110b, 4 chips, SAME for every satellite). One CS4
// period = 4 ms = one Galileo I/NAV symbol.
std::array<int8_t, 4> e5bi_secondary() {
    return hex_secondary<4>("E");
}

} // namespace galileo
