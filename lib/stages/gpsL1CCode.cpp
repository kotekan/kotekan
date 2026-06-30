#include "gpsL1CCode.hpp"

#include <stdexcept> // for out_of_range

namespace gps {
namespace {

constexpr int N_L = 10223; // Legendre sequence length (prime)

// IS-GPS-800 L1C-P Weil index (w) and insertion index (p, 1-indexed), PRN 1..32.
// Transcribed from PocketSDR sdr_code (L1CP_weil_idx / L1CP_ins_idx).
constexpr int L1CP_WEIL[32] = {
    5111, 5109, 5108, 5106, 5103, 5101, 5100, 5098, 5095, 5094, 5093, 5091,
    5090, 5081, 5080, 5069, 5068, 5054, 5044, 5027, 5026, 5014, 5004, 4980,
    4915, 4909, 4893, 4885, 4832, 4824, 4591, 3706};
constexpr int L1CP_INS[32] = {
    412,  161,  1,    303,  207,  4971, 4496, 5,    4557, 485,  253,  4676,
    1,    66,   4485, 282,  193,  5211, 729,  4848, 982,  5955, 9805, 670,
    464,  29,   429,  394,  616,  9457, 4429, 4771};

// 7-chip insertion pattern "0110100" in bipolar (+1 for 0, -1 for 1).
constexpr int8_t INS[7] = {1, -1, -1, 1, -1, 1, 1};

// Legendre sequence L (length N_L), built once: +1 everywhere, -1 at quadratic residues
// mod N_L (PocketSDR convention; L(0)=+1). Static so it's computed on first use only.
const std::array<int8_t, N_L>& legendre() {
    static const std::array<int8_t, N_L> L = [] {
        std::array<int8_t, N_L> a;
        a.fill(1);
        for (long i = 1; i < N_L; ++i)
            a[(size_t)((i * i) % N_L)] = -1;
        return a;
    }();
    return L;
}

} // namespace

std::array<int8_t, L1C_CODE_LENGTH> generate_l1cp_code(int prn) {
    if (prn < 1 || prn > 32)
        throw std::out_of_range("generate_l1cp_code: PRN must be 1..32");
    const auto& L = legendre();
    const int w = L1CP_WEIL[prn - 1];
    const int p = L1CP_INS[prn - 1]; // 1-indexed insertion position

    std::array<int8_t, L1C_CODE_LENGTH> code;
    auto weil = [&](int t) -> int8_t { return (int8_t)(L[(size_t)t] * L[(size_t)((t + w) % N_L)]); };
    for (int t = 0; t < p - 1; ++t) // Weil chips before the insertion
        code[(size_t)t] = weil(t);
    for (int t = p - 1; t < p + 6; ++t) // the 7-chip insertion
        code[(size_t)t] = INS[t - (p - 1)];
    for (int t = p + 6; t < L1C_CODE_LENGTH; ++t) // Weil chips after, shifted by the 7 inserted
        code[(size_t)t] = weil(t - 7);
    return code;
}

} // namespace gps
