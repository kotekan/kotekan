#define BOOST_TEST_MODULE "test_gnss_broker"

#include "gnssBandPlan.hpp" // for ChannelBand
#include "gnssBroker.hpp"   // for assign_targets, Worker, Target

#include <boost/test/included/unit_test.hpp>
#include <vector>

using gnss::assign_targets;
using gnss::ChannelBand;
using gnss::Target;
using gnss::Worker;

namespace {

// 21 channels of 0.2 MHz centred on L1; with no Doppler, L1 C/A (half-band
// 1.023 MHz) covers ids 5..15 (per the band-plan tests).
std::vector<ChannelBand> l1_grid() {
    std::vector<ChannelBand> g;
    for (int i = 0; i < 21; ++i)
        g.push_back({(gnss::freq_id_t)i, 1575.42e6 + (i - 10) * 0.2e6, 0.2e6});
    return g;
}

Worker l1_worker(const char* name, int clo, int chi, int plo = 1, int phi = 32) {
    return {name, "GPS_L1CA", (gnss::freq_id_t)clo, (gnss::freq_id_t)chi, plo, phi};
}

} // namespace

BOOST_AUTO_TEST_CASE(places_target_on_covering_worker) {
    std::vector<Worker> ws = {l1_worker("a", 0, 20)};
    std::vector<Target> ts = {{"GPS_L1CA", 5, true}, {"GPS_L1CA", 12, true}};
    auto r = assign_targets(ws, ts, l1_grid(), 0.0);
    BOOST_CHECK(r.unplaced.empty());
    BOOST_REQUIRE_EQUAL(r.worker_prns["a"].size(), 2u);
    BOOST_CHECK_EQUAL(r.worker_prns["a"][0], 5);
    BOOST_CHECK_EQUAL(r.worker_prns["a"][1], 12);
}

BOOST_AUTO_TEST_CASE(straddling_target_is_unplaced) {
    // Worker owns only 0..10, but L1 covers 5..15 -> would straddle.
    std::vector<Worker> ws = {l1_worker("a", 0, 10)};
    std::vector<Target> ts = {{"GPS_L1CA", 5, true}};
    auto r = assign_targets(ws, ts, l1_grid(), 0.0);
    BOOST_CHECK(r.worker_prns["a"].empty());
    BOOST_REQUIRE_EQUAL(r.unplaced.size(), 1u);
    BOOST_CHECK_EQUAL(r.unplaced[0].prn, 5);
}

BOOST_AUTO_TEST_CASE(exact_covering_range_holds) {
    std::vector<Worker> ws = {l1_worker("a", 5, 15)}; // exactly the covering set
    std::vector<Target> ts = {{"GPS_L1CA", 7, true}};
    auto r = assign_targets(ws, ts, l1_grid(), 0.0);
    BOOST_CHECK(r.unplaced.empty());
    BOOST_REQUIRE_EQUAL(r.worker_prns["a"].size(), 1u);
}

BOOST_AUTO_TEST_CASE(load_balances_across_equal_workers) {
    std::vector<Worker> ws = {l1_worker("a", 0, 20), l1_worker("b", 0, 20)};
    std::vector<Target> ts = {{"GPS_L1CA", 1, true},
                              {"GPS_L1CA", 2, true},
                              {"GPS_L1CA", 3, true},
                              {"GPS_L1CA", 4, true}};
    auto r = assign_targets(ws, ts, l1_grid(), 0.0);
    BOOST_CHECK(r.unplaced.empty());
    BOOST_CHECK_EQUAL(r.worker_prns["a"].size(), 2u); // evenly split
    BOOST_CHECK_EQUAL(r.worker_prns["b"].size(), 2u);
}

BOOST_AUTO_TEST_CASE(respects_prn_subset) {
    std::vector<Worker> ws = {l1_worker("a", 0, 20, 1, 16), l1_worker("b", 0, 20, 17, 32)};
    std::vector<Target> ts = {{"GPS_L1CA", 5, true}, {"GPS_L1CA", 20, true}};
    auto r = assign_targets(ws, ts, l1_grid(), 0.0);
    BOOST_REQUIRE_EQUAL(r.worker_prns["a"].size(), 1u);
    BOOST_CHECK_EQUAL(r.worker_prns["a"][0], 5);
    BOOST_REQUIRE_EQUAL(r.worker_prns["b"].size(), 1u);
    BOOST_CHECK_EQUAL(r.worker_prns["b"][0], 20);
}

BOOST_AUTO_TEST_CASE(skips_invisible_and_mismatched_signal) {
    std::vector<Worker> ws = {l1_worker("a", 0, 20)};
    std::vector<Target> ts = {{"GPS_L1CA", 9, false},   // not visible -> skipped
                              {"GPS_L2C_CL", 9, true}}; // no L1 worker takes L2C -> unplaced
    auto r = assign_targets(ws, ts, l1_grid(), 0.0);
    BOOST_CHECK(r.worker_prns["a"].empty());
    BOOST_REQUIRE_EQUAL(r.unplaced.size(), 1u);
    BOOST_CHECK_EQUAL(r.unplaced[0].signal, "GPS_L2C_CL");
}

BOOST_AUTO_TEST_CASE(unknown_signal_is_unplaced) {
    std::vector<Worker> ws = {l1_worker("a", 0, 20)};
    std::vector<Target> ts = {{"GPS_FOO", 1, true}};
    auto r = assign_targets(ws, ts, l1_grid(), 0.0);
    BOOST_REQUIRE_EQUAL(r.unplaced.size(), 1u);
}
