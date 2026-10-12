#include "glonassL3OCCode.hpp"

#include <stdexcept>
#include <string>

namespace glonass {
namespace {

/// One Fibonacci LFSR step, in the reference's convention: emit the low bit, shift right, and
/// load the parity of the tapped bits into the vacated top bit (n-1). Emitting before shifting
/// makes the initial state the source of the first chip.
struct Lfsr {
    uint32_t r;
    uint32_t tap;
    int n;

    int8_t next() {
        const int8_t chip = (r & 1u) ? -1 : +1; // logic 0 -> +1
        uint32_t fb = r & tap;
        // parity of the tapped bits
        fb ^= fb >> 16;
        fb ^= fb >> 8;
        fb ^= fb >> 4;
        fb ^= fb >> 2;
        fb ^= fb >> 1;
        r = ((fb & 1u) << (n - 1)) | (r >> 1);
        return chip;
    }
};

/// DC1, shared by both components and all satellites. Its 16383-chip period exceeds the code, so
/// the code is a truncation; unlike BeiDou B3I there is no mid-sequence reset.
const std::array<int8_t, L3OC_CODE_LENGTH>& dc1() {
    static const std::array<int8_t, L3OC_CODE_LENGTH> cached = [] {
        std::array<int8_t, L3OC_CODE_LENGTH> c{};
        Lfsr lfsr{0b00110100111000u, 0b00010001000011u, 14};
        for (int i = 0; i < L3OC_CODE_LENGTH; ++i)
            c[i] = lfsr.next();
        return c;
    }();
    return cached;
}

/// DC1 times the 7-stage per-satellite factor seeded with `init`. That factor has period 127 and
/// 10230 = 127*80 + 70, so it is truncated mid-period; the truncation is part of the code
/// definition.
std::array<int8_t, L3OC_CODE_LENGTH> compose(uint32_t init) {
    const auto& d1 = dc1();
    std::array<int8_t, L3OC_CODE_LENGTH> code{};
    Lfsr shrt{init, 0b0000011u, 7};
    for (int i = 0; i < L3OC_CODE_LENGTH; ++i)
        code[i] = static_cast<int8_t>(d1[i] * shrt.next());
    return code;
}

void check_prn(int prn, const char* who) {
    // For the data code, 0 is the all-zero LFSR state, a fixed point (see the header).
    if (prn < 1 || prn > 63)
        throw std::out_of_range(std::string(who) + ": code index must be 1..63, got "
                                + std::to_string(prn));
}

} // namespace

std::array<int8_t, L3OC_CODE_LENGTH> generate_l3ocd_code(int prn) {
    check_prn(prn, "generate_l3ocd_code");
    return compose(static_cast<uint32_t>(prn));
}

std::array<int8_t, L3OC_CODE_LENGTH> generate_l3ocp_code(int prn) {
    check_prn(prn, "generate_l3ocp_code");
    return compose(static_cast<uint32_t>(prn) + 64u);
}

// Barker-5 on the data component, logic 00010 in the house convention: the 5 ms symbol.
const std::array<int8_t, L3OC_BC_LENGTH> L3OC_BC5 = {+1, +1, +1, -1, +1};

// NH10 on the pilot, logic 0000110101 (the same sequence as gps::L5_NH10): 10 ms.
const std::array<int8_t, L3OC_NH_LENGTH> L3OC_NH10 = {+1, +1, +1, +1, -1, -1, +1, -1, +1, -1};

} // namespace glonass
