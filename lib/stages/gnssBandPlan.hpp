/**
 * @file
 * @brief Band plan: map a GNSS signal onto the channelized (PFB) frequency grid.
 *
 * In the distributed-band pipeline a carrier does not live in a single PFB
 * channel -- L1 C/A's main lobe is ~2 MHz, which at ~195 kHz/channel spans ~10
 * channels and can straddle a node boundary. This header is the pure-logic seam
 * that answers "which channels carry PRN k's signal?": given a @ref
 * SignalDescriptor and a maximum |Doppler|, it returns the set of channel IDs
 * whose passband overlaps the signal's occupied band.
 *
 * The core works on a plain list of @ref ChannelBand (id, center, width) so it
 * is unit-testable without the kotekan @c Telescope singleton and can reason
 * about *other* nodes' channels too. @ref gnssBandPlanTelescope.hpp builds that
 * list from a live Telescope. Correlate-at-data, ship-records is the transport
 * model: a node runs the correlators for the carriers whose covering channels
 * it holds, so this map drives per-node work assignment in the broker.
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

/// One-sided half-occupied bandwidth of the signal's main spectral lobe(s), in
/// Hz. BPSK: the first spectral null at the chip rate. BOC(m,n): the outer null
/// of the upper subcarrier lobe, ~(m+n)*1.023 MHz from the carrier.
double half_bandwidth_hz(const SignalDescriptor& sig);

/// Sky band the signal occupies, widened by the worst-case |Doppler| (Hz).
BandSpan occupied_band(const SignalDescriptor& sig, double max_doppler_hz);

/// Channel IDs whose passband overlaps the signal's occupied band, ordered by
/// center frequency. `chans` need not be sorted.
std::vector<freq_id_t> covering_channels(const std::vector<ChannelBand>& chans,
                                         const SignalDescriptor& sig, double max_doppler_hz);

} // namespace gnss

#endif // GNSS_BAND_PLAN_HPP
