#ifndef BEIDOU_B1C_CODE_HPP
#define BEIDOU_B1C_CODE_HPP

#include <array>
#include <stdint.h>
#include <vector>

/**
 * BeiDou-3 B1C spreading codes (BDS-SIS-ICD-B1C-1.0).
 *
 * B1C primary: 10230 chips at 1.023 Mcps (10 ms). Each PRN's code is a truncated Weil code,
 * chip n = W((n + p - 1) mod 10243; w) with W(k; w) = L(k) * L((k+w) mod 10243) over the
 * length-10243 Legendre sequence L. The phase difference w and truncation point p per PRN come
 * from the ICD tables (as transcribed in PocketSDR); data and pilot have separate tables.
 * Transmitted as QMBOC(6,1,4/33); the replica models the dominant BOC(1,1) component.
 * The pilot's 1800-chip per-PRN secondary (Weil over Legendre-3607) is wiped by the combiner
 * (gnssOverlay registry row "B1C"). PRN 1..63 (BDS-3 MEO/IGSO).
 */
namespace beidou {

constexpr int B1C_CODE_LENGTH = 10230;
constexpr int B1C_SECONDARY_LENGTH = 1800;

/// B1C pilot primary code, +-1 chips.
std::array<int8_t, B1C_CODE_LENGTH> generate_b1cp_code(int prn);

/// B1C data primary code, +-1 chips. Same Weil construction as the pilot with its own phase
/// difference and truncation tables. Carries B-CNAV1.
std::array<int8_t, B1C_CODE_LENGTH> generate_b1cd_code(int prn);

/// B1C pilot secondary (overlay) code, +-1 chips, per PRN.
std::vector<int8_t> generate_b1cp_secondary(int prn);

} // namespace beidou

#endif
