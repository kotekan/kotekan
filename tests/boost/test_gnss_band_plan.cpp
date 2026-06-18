#define BOOST_TEST_MODULE "test_gnss_band_plan"

#include "gnssBandPlan.hpp" // for covering_channels, occupied_band, half_bandwidth_hz
#include "gnssSignal.hpp"   // for GPS_L1CA, SignalDescriptor, Modulation

#include <algorithm> // for reverse
#include <boost/test/included/unit_test.hpp>
#include <vector>

using gnss::ChannelBand;
using gnss::covering_channels;
using gnss::half_bandwidth_hz;
using gnss::occupied_band;

namespace {

// A uniform PFB grid of `n` channels of width `w` centered on `f0` (channel 0's
// center is f0; ids run 0..n-1 with center f0 + (i - n/2)*w).
std::vector<ChannelBand> uniform_grid(double f0, double w, int n) {
    std::vector<ChannelBand> g;
    for (int i = 0; i < n; ++i)
        g.push_back({static_cast<gnss::freq_id_t>(i), f0 + (i - n / 2) * w, w});
    return g;
}

} // namespace

BOOST_AUTO_TEST_CASE(bpsk_half_bandwidth_is_chip_rate) {
    BOOST_CHECK_CLOSE(half_bandwidth_hz(gnss::GPS_L1CA), 1.023e6, 1e-9);
}

BOOST_AUTO_TEST_CASE(boc_half_bandwidth_uses_outer_lobe) {
    gnss::SignalDescriptor boc = gnss::GPS_L1CA;
    boc.mod = gnss::Modulation::BOC;
    boc.boc_m = 1;
    boc.boc_n = 1; // BOC(1,1): outer null at (1+1)*1.023 MHz
    BOOST_CHECK_CLOSE(half_bandwidth_hz(boc), 2.046e6, 1e-9);
}

BOOST_AUTO_TEST_CASE(occupied_band_widens_by_doppler) {
    auto b = occupied_band(gnss::GPS_L1CA, 5000.0);
    BOOST_CHECK_CLOSE(b.lo_hz, 1575.42e6 - 1.023e6 - 5000.0, 1e-9);
    BOOST_CHECK_CLOSE(b.hi_hz, 1575.42e6 + 1.023e6 + 5000.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(covering_channels_selects_overlapping_block) {
    // 200 kHz channels centered on L1; carrier sits at channel id n/2.
    const double w = 0.2e6;
    const int n = 21; // ids 0..20, center id 10 == carrier
    auto grid = uniform_grid(gnss::GPS_L1CA.carrier_hz, w, n);

    // No Doppler -> half-band 1.023 MHz. Overlap math: channel offset k (in
    // units of w from carrier) is in [-5, 5] -> ids 5..15 (11 channels).
    auto ids = covering_channels(grid, gnss::GPS_L1CA, 0.0);
    BOOST_REQUIRE_EQUAL(ids.size(), 11u);
    for (size_t i = 0; i < ids.size(); ++i)
        BOOST_CHECK_EQUAL(ids[i], 5u + i); // returned ordered by center freq
}

BOOST_AUTO_TEST_CASE(covering_channels_is_order_independent) {
    const double w = 0.2e6;
    auto grid = uniform_grid(gnss::GPS_L1CA.carrier_hz, w, 21);
    std::reverse(grid.begin(), grid.end()); // shuffle input order

    auto ids = covering_channels(grid, gnss::GPS_L1CA, 0.0);
    BOOST_REQUIRE_EQUAL(ids.size(), 11u);
    for (size_t i = 0; i < ids.size(); ++i)
        BOOST_CHECK_EQUAL(ids[i], 5u + i); // still ascending by center
}

BOOST_AUTO_TEST_CASE(doppler_pulls_in_edge_channels) {
    const double w = 0.2e6;
    auto grid = uniform_grid(gnss::GPS_L1CA.carrier_hz, w, 21);

    // Push half-band past the k=+-5.5 boundary so k=+-6 (ids 4 and 16) join.
    auto ids = covering_channels(grid, gnss::GPS_L1CA, 90e3); // 1.023+0.09 = 1.113 MHz
    BOOST_REQUIRE_EQUAL(ids.size(), 13u);
    BOOST_CHECK_EQUAL(ids.front(), 4u);
    BOOST_CHECK_EQUAL(ids.back(), 16u);
}

BOOST_AUTO_TEST_CASE(out_of_band_grid_returns_empty) {
    // A grid sitting at L2 (1227.6 MHz) does not carry an L1 signal.
    auto grid = uniform_grid(1227.6e6, 0.2e6, 21);
    BOOST_CHECK(covering_channels(grid, gnss::GPS_L1CA, 5000.0).empty());
}
