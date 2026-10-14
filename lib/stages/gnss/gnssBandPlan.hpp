/**
 * @file
 * @brief Band plan: map a GNSS signal onto the channelized (PFB) frequency grid.
 *
 * A signal's main lobe spans many PFB channels (L1 C/A's ~2 MHz covers ~10 channels of
 * ~195 kHz) and can straddle a node boundary. Given a @ref SignalDescriptor and a maximum
 * |Doppler|, @ref covering_channels returns the channels whose passband overlaps the signal's
 * occupied band.
 *
 * @ref covering_channels takes a plain list of @ref ChannelBand rather than the kotekan
 * @c Telescope, so it can describe any node's channels and is testable without one.
 * ChannelizedReplicaBank::covering_bins builds the list from its own r2c grid.
 * config/chord_band_plan.py mirrors these functions for the config generator's per-node channel
 * choice; keep the two in step.
 */

#ifndef GNSS_BAND_PLAN_HPP
#define GNSS_BAND_PLAN_HPP

#include "gnssSignal.hpp" // for SignalDescriptor

#include <cstdint> // for uint32_t
#include <vector>  // for vector

namespace gnss {

/// Logical channel ID on the PFB grid (matches kotekan's freq_id_t).
using freq_id_t = std::uint32_t;

/// One channel of the PFB grid: its passband is center_hz +/- width_hz/2.
struct ChannelBand {
    freq_id_t id;
    double center_hz;
    double width_hz;
};

/// Sky frequency span [lo_hz, hi_hz] a signal occupies.
struct BandSpan {
    double lo_hz;
    double hi_hz;
};

/// One-sided half-width of the signal's main spectral lobe(s), in Hz. BPSK: the first null,
/// one chip rate from the carrier; a time-multiplexed component (L2C CM/CL) spans the combined
/// stream, twice its own rate. BOC(m,n): the outer null of the upper subcarrier lobe,
/// (m+n)*1.023 MHz from the carrier.
double half_bandwidth_hz(const SignalDescriptor& sig);

/// Sky band the signal occupies, widened by the worst-case |Doppler| (Hz).
BandSpan occupied_band(const SignalDescriptor& sig, double max_doppler_hz);

/// Channel IDs whose passband overlaps the signal's occupied band (a shared edge does not
/// count), ordered by center frequency. `chans` need not be sorted.
std::vector<freq_id_t> covering_channels(const std::vector<ChannelBand>& chans,
                                         const SignalDescriptor& sig, double max_doppler_hz);

} // namespace gnss

#endif // GNSS_BAND_PLAN_HPP
