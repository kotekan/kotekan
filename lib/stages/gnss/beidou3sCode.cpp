#include "beidou3sCode.hpp"

#include <stdexcept> // for out_of_range
#include <string>    // for to_string

namespace beidou {

namespace {

constexpr uint32_t TAP_X = 0x1805u;
constexpr uint32_t TAP_Y = 0x0B15u;

// Start states (register bit i is component chip i): B2a data, B2a pilot, B2b pilot. The B2b
// data component (x 0x01FE, y 0x0C6D) has no replica: the B2b slot tracks the pilot.
struct Start3s {
    int prn;
    uint32_t b2ad_x, b2ad_y, b2ap_x, b2ap_y, b2bp_x, b2bp_y;
};
constexpr Start3s START_3S[] = {
    {58, 0x15A9u, 0x19A5u, 0x067Au, 0x1E3Au, 0x1793u, 0x1B18u},
};

const Start3s& start_3s(int prn) {
    for (const Start3s& s : START_3S)
        if (s.prn == prn)
            return s;
    throw std::out_of_range("beidou3sCode: no BeiDou-3S code for PRN " + std::to_string(prn));
}

// 13-stage Fibonacci LFSR as in beidouB2bCode.cpp: the chip is the register LSB (0 -> +1,
// 1 -> -1), then shift right with parity(reg & tap) into bit 12.
int8_t lfsr13_step(uint32_t& reg, uint32_t tap) {
    const int8_t chip = (reg & 1u) ? -1 : 1;
    reg = ((uint32_t)(__builtin_parity(reg & tap) & 1) << 12) | (reg >> 1);
    return chip;
}

// One 5115-chip component on chip parity @p phase of the 10230-chip code, zeros elsewhere.
std::array<int8_t, B3S_CODE_LENGTH> component(uint32_t x, uint32_t y, int phase) {
    std::array<int8_t, B3S_CODE_LENGTH> out{};
    for (int k = 0; k < B3S_CODE_LENGTH / 2; ++k)
        out[(size_t)(2 * k + phase)] = (int8_t)(lfsr13_step(x, TAP_X) * lfsr13_step(y, TAP_Y));
    return out;
}

} // namespace

std::array<int8_t, B3S_CODE_LENGTH> generate_b2ap_3s_code(int prn) {
    const Start3s& s = start_3s(prn);
    return component(s.b2ap_x, s.b2ap_y, 1);
}

std::array<int8_t, B3S_CODE_LENGTH> generate_b2ad_3s_code(int prn) {
    const Start3s& s = start_3s(prn);
    return component(s.b2ad_x, s.b2ad_y, 0);
}

std::array<int8_t, B3S_CODE_LENGTH> generate_b2bp_3s_code(int prn) {
    const Start3s& s = start_3s(prn);
    return component(s.b2bp_x, s.b2bp_y, 1);
}

} // namespace beidou
