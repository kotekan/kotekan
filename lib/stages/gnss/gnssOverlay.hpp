/**
 * @file
 * @brief Secondary-overlay registry, the table behind the `secondary_overlay` config key.
 *
 * One @ref OverlayDescriptor row per overlay name maps the config string to the overlay's
 * shape (shared or per-PRN, PRN count, length) and its chip generator, so consumers
 * (GnssCoherentCombiner) dispatch by table lookup. A new overlay is a new row.
 *
 * The combiner's overlay chips come from here. @c SignalDescriptor::secondary_length in
 * gnssSignal.hpp states the same lengths, and @ref overlay_registry_check reports any
 * disagreement.
 */

#ifndef GNSS_OVERLAY_HPP
#define GNSS_OVERLAY_HPP

#include <cstdint> // for int8_t
#include <string>  // for string (name lookup)
#include <vector>  // for vector

namespace gnss {

/// One secondary overlay: what a consumer needs to build the +-1 chip sequence(s) for a
/// `secondary_overlay` config value.
struct OverlayDescriptor {
    const char* name; ///< config string (`secondary_overlay`), e.g. "L5_NH20", "B1C"
    bool per_prn;     ///< true: a distinct sequence per PRN; false: one shared sequence
    int n_prn;        ///< PRN slots to cache when per_prn (PRN 1..n_prn); 1 otherwise
    int length;       ///< overlay chips = primary-code periods per overlay period
    std::vector<int8_t> (*generate)(int prn); ///< +-1 chips; prn is ignored unless per_prn
    const char* signal; ///< matching SignalDescriptor::name, for overlay_registry_check;
                        ///< nullptr for a synthetic overlay with no spec secondary (COHERENT)
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
