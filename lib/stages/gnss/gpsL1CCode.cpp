#include "gpsL1CCode.hpp"

#include <stdexcept> // for out_of_range

namespace gps {
namespace {

constexpr int N_L = 10223; // Legendre sequence length (prime)

// IS-GPS-800 L1C-P Weil index (w) and insertion index (p, 1-indexed), PRN 1..32,
// transcribed from PocketSDR sdr_code (L1CP_weil_idx / L1CP_ins_idx).
constexpr int L1CP_WEIL[32] = {5111, 5109, 5108, 5106, 5103, 5101, 5100, 5098, 5095, 5094, 5093,
                               5091, 5090, 5081, 5080, 5069, 5068, 5054, 5044, 5027, 5026, 5014,
                               5004, 4980, 4915, 4909, 4893, 4885, 4832, 4824, 4591, 3706};
constexpr int L1CP_INS[32] = {412,  161, 1,   303,  207, 4971, 4496, 5,    4557, 485, 253,
                              4676, 1,   66,  4485, 282, 193,  5211, 729,  4848, 982, 5955,
                              9805, 670, 464, 29,   429, 394,  616,  9457, 4429, 4771};

// L1C-D indices, same layout and source (PocketSDR L1CD_weil_idx / L1CD_ins_idx).
constexpr int L1CD_WEIL[32] = {5097, 5110, 5079, 4403, 4121, 5043, 5042, 5104, 4940, 5035, 4372,
                               5064, 5084, 5048, 4950, 5019, 5076, 3736, 4993, 5060, 5061, 5096,
                               4983, 4783, 4991, 4815, 4443, 4769, 4879, 4894, 4985, 5056};
constexpr int L1CD_INS[32] = {181,  359, 72,  1110, 1480, 5034, 4622, 1,    4547, 826, 6284,
                              4195, 368, 1,   4796, 523,  151,  713,  9850, 5734, 34,  6142,
                              190,  644, 467, 5384, 801,  594,  4450, 9437, 4307, 5906};

// 7-chip insertion pattern "0110100" in bipolar (+1 for 0, -1 for 1).
constexpr int8_t INS[7] = {1, -1, -1, 1, -1, 1, 1};

// L1CO overlay: per-PRN feedback polynomial (S1_poly) and initial state (S1_init), octal, of
// the 11-stage S1 LFSR, PRN 1..32 (IS-GPS-800, from PocketSDR sec_code_L1CP). As in PocketSDR,
// the tap mask (poly >> 1) and the initial state are bit-reversed over 11 bits, and the register
// shifts right, outputs bit 0 and feeds parity(R & tap) into bit 10. GPS PRNs 1..63 use S1
// alone (the S2 product applies from PRN 64), so this is exact for every GPS satellite.
constexpr int L1CO_S1_POLY[32] = {05111, 05421, 05501, 05403, 06417, 06141, 06351, 06501,
                                  06205, 06235, 07751, 06623, 06733, 07627, 05667, 05051,
                                  07665, 06325, 04365, 04745, 07633, 06747, 04475, 04225,
                                  07063, 04423, 06651, 04161, 07237, 04473, 05477, 06163};
constexpr int L1CO_S1_INIT[32] = {03266, 02040, 01527, 03307, 03756, 03026, 00562, 00420,
                                  03415, 00337, 00265, 01230, 02204, 01440, 02412, 03516,
                                  02761, 03750, 02701, 01206, 01544, 01774, 00546, 02213,
                                  03707, 02051, 03650, 01777, 03203, 01762, 02100, 00571};

// Reverse the low 11 bits of r (PocketSDR rev_reg with N=11).
inline unsigned rev11(unsigned r) {
    unsigned rr = 0;
    for (int i = 0; i < 11; ++i)
        rr = (rr << 1) | ((r >> (unsigned)i) & 1u);
    return rr;
}

// XOR of all bits of x: the LFSR feedback (portable, unlike __builtin_parity).
inline unsigned parity(unsigned x) {
    unsigned p = 0;
    while (x) {
        p ^= 1u;
        x &= x - 1u;
    }
    return p;
}

// Legendre sequence L of length N_L, built on first use: -1 at the non-zero quadratic residues
// mod N_L and +1 elsewhere, including L(0) (PocketSDR convention).
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

// L1C Weil code (IS-GPS-800): L(t)*L((t+w) mod N_L) with "0110100" inserted at 1-indexed
// position p. L1C-P and L1C-D differ only in (w, p).
std::array<int8_t, L1C_CODE_LENGTH> weil_code(int w, int p) {
    const auto& L = legendre();
    std::array<int8_t, L1C_CODE_LENGTH> code;
    auto weil = [&](int t) -> int8_t {
        return (int8_t)(L[(size_t)t] * L[(size_t)((t + w) % N_L)]);
    };
    for (int t = 0; t < p - 1; ++t) // Weil chips before the insertion
        code[(size_t)t] = weil(t);
    for (int t = p - 1; t < p + 6; ++t) // the 7-chip insertion
        code[(size_t)t] = INS[t - (p - 1)];
    for (int t = p + 6; t < L1C_CODE_LENGTH; ++t) // Weil chips after, shifted by the 7 inserted
        code[(size_t)t] = weil(t - 7);
    return code;
}

} // namespace

std::array<int8_t, L1C_CODE_LENGTH> generate_l1cp_code(int prn) {
    if (prn < 1 || prn > 32)
        throw std::out_of_range("generate_l1cp_code: PRN must be 1..32");
    return weil_code(L1CP_WEIL[prn - 1], L1CP_INS[prn - 1]);
}

std::array<int8_t, L1C_CODE_LENGTH> generate_l1cd_code(int prn) {
    if (prn < 1 || prn > 32)
        throw std::out_of_range("generate_l1cd_code: PRN must be 1..32");
    return weil_code(L1CD_WEIL[prn - 1], L1CD_INS[prn - 1]);
}

std::array<int8_t, L1CO_LENGTH> generate_l1co_code(int prn) {
    if (prn < 1 || prn > 32)
        throw std::out_of_range("generate_l1co_code: PRN must be 1..32");
    const unsigned tap = rev11((unsigned)L1CO_S1_POLY[prn - 1] >> 1);
    unsigned R = rev11((unsigned)L1CO_S1_INIT[prn - 1]);
    std::array<int8_t, L1CO_LENGTH> code;
    for (int i = 0; i < L1CO_LENGTH; ++i) {
        code[(size_t)i] = (R & 1u) ? (int8_t)-1 : (int8_t)1; // LSB 0 -> +1, 1 -> -1
        R = (parity(R & tap) << 10) | (R >> 1);              // shift right, feedback into bit 10
    }
    return code;
}

} // namespace gps
