#include "glonassL2OCCode.hpp"

#include "glonassL3OCCode.hpp" // the base ranging code is L3OC-p's, verbatim

#include <stdexcept>
#include <string>

namespace glonass {

std::vector<int8_t> generate_l2ocp_code(int prn) {
    if (prn < 1 || prn > 63)
        throw std::out_of_range("generate_l2ocp_code: code index must be 1..63, got "
                                + std::to_string(prn));
    // ★ The base IS the L3OC-p code -- GLONASS reuses one sequence across the L2/L3 CDMA pilots.
    const std::array<int8_t, L3OC_CODE_LENGTH> base = generate_l3ocp_code(prn);

    // Per-chip [0, 0, +1, -1] expansion = the ICD's BOC(1,1) + data/pilot TDM. Each base chip c
    // becomes [0, 0, c, -c]: zeros where the DATA component's TDM slots are (this is the pilot),
    // and +c/-c the BOC(1,1) half-cycles.
    static constexpr int8_t SUB[L2OC_SUBCARRIER] = {0, 0, 1, -1};
    std::vector<int8_t> code;
    code.reserve(L2OC_CODE_LENGTH);
    for (int i = 0; i < L2OC_BASE_LENGTH; ++i)
        for (int s = 0; s < L2OC_SUBCARRIER; ++s)
            code.push_back(static_cast<int8_t>(base[(size_t)i] * SUB[s]));
    return code;
}

// PocketSDR G2OCP_OC2 (50 chips), the shared L2OC-p secondary overlay. One chip per 20 ms
// primary period -> 1 s cycle.
const std::array<int8_t, L2OC_OC2_LENGTH> L2OC_OC2 = {
    1,  1,  -1, 1,  -1, -1, 1,  -1, 1,  -1, -1, -1, -1, -1, 1,  1,  1,
    -1, -1, 1,  1,  1,  1,  -1, 1,  1,  1,  -1, -1, 1,  -1, 1,  1,  1,
    1,  1,  1,  -1, 1,  -1, -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,
};

} // namespace glonass
