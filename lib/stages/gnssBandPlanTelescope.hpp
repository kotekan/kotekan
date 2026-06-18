/**
 * @file
 * @brief Build a @ref gnss::ChannelBand list from a live kotekan Telescope.
 *
 * Kept out of @ref gnssBandPlan.hpp so the band-plan core stays free of the
 * Telescope singleton (and its dataset/EOP dependencies) and remains
 * unit-testable on a hand-built channel list. Stages that already have a
 * Telescope include this to feed the real PFB grid into @ref
 * gnss::covering_channels.
 */

#ifndef GNSS_BAND_PLAN_TELESCOPE_HPP
#define GNSS_BAND_PLAN_TELESCOPE_HPP

#include "Telescope.hpp"   // for Telescope, freq_id_t
#include "gnssBandPlan.hpp" // for ChannelBand

#include <vector> // for vector

namespace gnss {

/// Snapshot every channel of the telescope's PFB grid as a ChannelBand list.
inline std::vector<ChannelBand> channels_from_telescope(const Telescope& tel) {
    std::vector<ChannelBand> chans;
    const size_t n = tel.num_freq();
    chans.reserve(n);
    for (::freq_id_t id = 0; id < n; ++id) {
        chans.push_back({static_cast<freq_id_t>(id), tel.to_freq_MHz(id) * 1e6,
                         tel.freq_width_MHz(id) * 1e6});
    }
    return chans;
}

} // namespace gnss

#endif // GNSS_BAND_PLAN_TELESCOPE_HPP
