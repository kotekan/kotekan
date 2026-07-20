/**
 * @file
 * @brief Secondary-overlay registry -- the table behind the `secondary_overlay` config key.
 *
 * One @ref OverlayDescriptor row per known overlay name maps the config string to the
 * overlay's shape (shared vs per-PRN, PRN count, length) and its chip generator, so the
 * consumers (GnssCoherentCombiner today) dispatch by a single table lookup instead of a
 * hand-coded string ladder. Adding a signal's overlay means adding a ROW here, not a branch.
 *
 * The registry is the runtime authority; the @ref SignalDescriptor rows in gnssSignal.hpp
 * carry @c secondary_length as spec documentation only. @ref overlay_registry_check
 * cross-checks the two so a transcription drift is logged instead of silently diverging.
 */

#ifndef GNSS_OVERLAY_HPP
#define GNSS_OVERLAY_HPP

#include <cstdint> // for int8_t
#include <string>  // for string (name lookup)
#include <vector>  // for vector

namespace gnss {

/// One secondary-overlay entry: everything a consumer needs to build the +-1 chip
/// sequence(s) for a `secondary_overlay` config value.
struct OverlayDescriptor {
    const char* name; ///< config string (`secondary_overlay`), e.g. "L5_NH20", "B1C"
    bool per_prn;     ///< true: a distinct sequence per PRN (cache all, pick at wipe time);
                      ///< false: one shared sequence for every satellite
    int n_prn;        ///< PRN slots to cache when per_prn (PRN 1..n_prn); 1 otherwise
    int length;       ///< overlay chips = primary-code periods per overlay period
    std::vector<int8_t> (*generate)(int prn); ///< +-1 chips; prn is ignored unless per_prn
    const char* signal; ///< matching SignalDescriptor::name (documentation cross-check);
                        ///< nullptr for synthetic overlays with no spec secondary (COHERENT)
};

/// Look up an overlay by its config string. Returns nullptr if unknown (or empty) --
/// the caller treats that exactly as "no overlay configured", matching the historical
/// string-ladder fall-through.
const OverlayDescriptor* overlay_by_name(const std::string& name);

/// Cross-check every registry row against its SignalDescriptor's documentary
/// @c secondary_length. Returns one human-readable message per disagreement
/// (empty = consistent). Purely diagnostic: callers log, nothing changes behaviour.
std::vector<std::string> overlay_registry_check();

} // namespace gnss

#endif // GNSS_OVERLAY_HPP
