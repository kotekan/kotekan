#include "glonassL3OCCode.hpp"

#include <stdexcept>
#include <string>

namespace glonass {
namespace {

/// One Fibonacci LFSR step-and-emit, in the reference's exact convention: the OUTPUT is the
/// low bit, the register shifts RIGHT, and the vacated top bit (index n-1) takes the parity of
/// the tapped bits. Emitting before shifting matters -- it is what makes the initial state the
/// first chip's source.
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

/// The 14-stage sequence shared by BOTH components and ALL satellites. Its natural period
/// (16383) exceeds the 10230-chip code, so the code is a TRUNCATION -- but unlike BeiDou B3I
/// there is no mid-sequence reset to get wrong.
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

/// DC1 times the 7-stage per-satellite factor seeded with `init`. The 7-stage polynomial is
/// primitive (period 127) and 10230 = 127*80 + 70, so the short factor is truncated mid-period;
/// that truncation is part of the code definition, not an implementation shortcut.
std::array<int8_t, L3OC_CODE_LENGTH> compose(uint32_t init) {
    const auto& d1 = dc1();
    std::array<int8_t, L3OC_CODE_LENGTH> code{};
    Lfsr shrt{init, 0b0000011u, 7};
    for (int i = 0; i < L3OC_CODE_LENGTH; ++i)
        code[i] = static_cast<int8_t>(d1[i] * shrt.next());
    return code;
}

void check_prn(int prn, const char* who) {
    // 0 is excluded because it is an ABSORBING LFSR state, not merely an unassigned index --
    // see the trap note in the header.
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

// Barker 11101 in the house convention (logic 0 -> +1): the data component's 5 ms symbol.
const std::array<int8_t, L3OC_BC_LENGTH> L3OC_BC5 = {+1, +1, +1, -1, +1};

// NH10 on the pilot: 10 ms.
const std::array<int8_t, L3OC_NH_LENGTH> L3OC_NH10 = {+1, +1, +1, +1, -1, -1, +1, -1, +1, -1};

} // namespace glonass
