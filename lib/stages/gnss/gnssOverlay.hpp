/**
 * @file
 * @brief Secondary-overlay registry, the table behind the `secondary_overlay` config key.
 *
 * One @ref OverlayDescriptor row per overlay name maps the config string to the overlay's
 * length and its chip generator. The rows are the shared overlays under a data component, which
 * GnssCoherentCombiner wipes ahead of its nav-bit wipe. A new overlay is a new row.
 *
 * @c SignalDescriptor::secondary_length in gnssSignal.hpp states the same lengths, and
 * @ref overlay_registry_check reports any disagreement.
 */

#ifndef GNSS_OVERLAY_HPP
#define GNSS_OVERLAY_HPP

#include <cstdint> // for int8_t
#include <string>  // for string (name lookup)
#include <vector>  // for vector

namespace gnss {

/// One secondary overlay, shared by every PRN: what a consumer needs to build its +-1 chips for
/// a `secondary_overlay` config value.
struct OverlayDescriptor {
    const char* name;                  ///< config string (`secondary_overlay`), e.g. "L5_NH10"
    int length;                        ///< overlay chips = primary-code periods per overlay period
    std::vector<int8_t> (*generate)(); ///< +-1 chips
    const char* signal; ///< matching SignalDescriptor::name, for overlay_registry_check
};

/// Look up an overlay by its config string. Returns nullptr for an unknown or empty name,
/// which callers treat as no overlay.
const OverlayDescriptor* overlay_by_name(const std::string& name);

/// Check each registry row that names a signal against its SignalDescriptor's
/// @c secondary_length. Returns one message per disagreement or missing descriptor
/// (empty = consistent); diagnostic only.
std::vector<std::string> overlay_registry_check();

} // namespace gnss

#endif // GNSS_OVERLAY_HPP
