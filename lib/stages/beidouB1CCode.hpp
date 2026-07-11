#ifndef BEIDOU_B1C_CODE_HPP
#define BEIDOU_B1C_CODE_HPP

#include <array>
#include <stdint.h>
#include <vector>

/**
 * BeiDou-3 B1C pilot spreading codes (BDS-SIS-ICD-B1C-1.0, 2017).
 *
 * B1C-pilot primary: 10230 chips at 1.023 Mcps (10 ms), each PRN a truncated WEIL code --
 * W(k; w) = L(k) * L((k+w) mod 10243) over the length-10243 Legendre sequence, phase
 * difference w and truncation point p per PRN from the ICD tables (transcribed via
 * PocketSDR's validated tables; cross-checked against its generator chip-for-chip).
 * Transmitted as QMBOC(6,1,4/33); the replica models the dominant BOC(1,1) component.
 * The 1800-chip PER-PRN secondary (Weil over Legendre-3607) is exposed for the combiner's
 * overlay handling; per-record sign wipe absorbs it until an alignment search lands.
 * PRN range 1..63 (BDS-3 MEO/IGSO).
 */
namespace beidou {

constexpr int B1C_CODE_LENGTH = 10230;
constexpr int B1C_SECONDARY_LENGTH = 1800;

/// B1C pilot primary code, +-1 chips.
std::array<int8_t, B1C_CODE_LENGTH> generate_b1cp_code(int prn);

/// B1C pilot secondary (overlay) code, +-1 chips, PER PRN.
std::vector<int8_t> generate_b1cp_secondary(int prn);

} // namespace beidou

#endif
