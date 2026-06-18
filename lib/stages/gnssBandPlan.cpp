#include "gnssBandPlan.hpp"

#include <algorithm> // for sort

namespace gnss {

// 1.023 MHz: the GNSS base chipping/subcarrier unit that BOC orders multiply.
static constexpr double kBaseRate = 1.023e6;

double half_bandwidth_hz(const SignalDescriptor& sig) {
    if (sig.mod == Modulation::BOC) {
        // BOC(m,n): main lobes sit at +/- m*1.023 MHz; the upper one's outer
        // null is a further n*1.023 MHz out. Use that outer edge.
        return (sig.boc_m + sig.boc_n) * kBaseRate;
    }
    // BPSK: first spectral null is one chip rate from the carrier.
    return sig.chip_rate_hz;
}

BandSpan occupied_band(const SignalDescriptor& sig, double max_doppler_hz) {
    const double half = half_bandwidth_hz(sig) + max_doppler_hz;
    return {sig.carrier_hz - half, sig.carrier_hz + half};
}

std::vector<freq_id_t> covering_channels(const std::vector<ChannelBand>& chans,
                                         const SignalDescriptor& sig, double max_doppler_hz) {
    const BandSpan band = occupied_band(sig, max_doppler_hz);

    // Collect (center, id) for overlapping channels, then order by center.
    std::vector<std::pair<double, freq_id_t>> hits;
    for (const ChannelBand& c : chans) {
        const double pass_lo = c.center_hz - 0.5 * c.width_hz;
        const double pass_hi = c.center_hz + 0.5 * c.width_hz;
        if (pass_lo < band.hi_hz && pass_hi > band.lo_hz) // open-interval overlap
            hits.emplace_back(c.center_hz, c.id);
    }
    std::sort(hits.begin(), hits.end());

    std::vector<freq_id_t> ids;
    ids.reserve(hits.size());
    for (const auto& h : hits)
        ids.push_back(h.second);
    return ids;
}

} // namespace gnss
