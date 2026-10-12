#include "glonassCACode.hpp"

namespace glonass {

const std::array<int8_t, CA_CODE_LENGTH>& ca_code() {
    static const std::array<int8_t, CA_CODE_LENGTH> cached = [] {
        std::array<int8_t, CA_CODE_LENGTH> c{};
        // 9-stage LFSR, all-ones seed, polynomial 1 + x^5 + x^9: the register shifts right, bit 8
        // takes the parity of bits 0 and 4, and the output is bit 2 (not the low bit, unlike the
        // L3OC generators). Emitting before shifting makes the seed the source of chip 0.
        uint32_t r = 0b111111111u;
        for (int i = 0; i < CA_CODE_LENGTH; ++i) {
            c[i] = ((r >> 2) & 1u) ? -1 : +1; // logic 0 -> +1
            const uint32_t fb = ((r & 1u) ^ ((r >> 4) & 1u));
            r = (fb << 8) | (r >> 1);
        }
        return c;
    }();
    return cached;
}

} // namespace glonass
