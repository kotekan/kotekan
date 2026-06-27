#define BOOST_TEST_MODULE "test_gnss_channelized_replica"

#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank
#include "gnssSignal.hpp"              // for signal_by_name
#include "pfbPrototype.hpp"            // for Window

#include <boost/test/included/unit_test.hpp>
#include <complex>
#include <vector>

using cf = std::complex<float>;
using gnss::ChannelizedReplicaBank;

namespace {

// L5/L1 both have a 1 ms primary period, so at this Fs both give a clean replica
// period; 2 samples/chip on L5, 20 on L1.
constexpr double FS = 20.46e6;
constexpr double FOFF = 5.115e6; // carrier a few channels up from DC
constexpr int N = 10;            // 1.023 MHz channels
constexpr int P = 4;

// Matched-filter amplitude of replica `data` against replica `repl`, over all
// channels (the bank's replica energy normalizes it, so self -> 1).
double despread_mag(const std::vector<std::vector<cf>>& data,
                    const std::vector<std::vector<cf>>& repl) {
    return std::abs(gnss::channelized_despread(data, repl).amplitude);
}

ChannelizedReplicaBank make_bank(const char* signal, const std::vector<int>& prns) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    BOOST_REQUIRE_MESSAGE(sig != nullptr, "unknown signal");
    return ChannelizedReplicaBank(*sig, FS, FOFF, N, P, dsp::Window::Hamming, prns);
}

// Relative rms error of a hop series against a reference series.
double rel_err(const std::vector<cf>& a, const std::vector<cf>& ref) {
    double num = 0.0, den = 0.0;
    for (size_t m = 0; m < a.size(); ++m) {
        num += std::norm(a[m] - ref[m]);
        den += std::norm(ref[m]);
    }
    return std::sqrt(num / den);
}

} // namespace

// The bank dispatches the L5 Q5 code: self-despread is a perfect matched filter (1),
// and a different PRN's replica decorrelates -- so it really uses PRN-distinct L5
// codes (a broken dispatch would give all-PRN-identical or empty codes -> no rejection).
BOOST_AUTO_TEST_CASE(l5q_self_and_cross_prn) {
    std::vector<int> prns = {1, 2};
    auto bank = make_bank("GPS_L5_Q", prns);
    const int H = bank.repl_period_hops();
    auto rA = bank.channels(0, 0, 0.0, 0.0, H); // PRN 1
    auto rB = bank.channels(1, 0, 0.0, 0.0, H); // PRN 2

    BOOST_CHECK_CLOSE(despread_mag(rA, rA), 1.0, 1e-2); // self == matched filter
    BOOST_CHECK_LT(despread_mag(rA, rB), 0.3);          // different PRN decorrelates
}

// I5 and Q5 are distinct codes on the same carrier -> they decorrelate.
BOOST_AUTO_TEST_CASE(l5i_distinct_from_l5q) {
    std::vector<int> prns = {1};
    auto bq = make_bank("GPS_L5_Q", prns);
    auto bi = make_bank("GPS_L5_I", prns);
    const int H = bq.repl_period_hops();
    auto rQ = bq.channels(0, 0, 0.0, 0.0, H);
    auto rI = bi.channels(0, 0, 0.0, 0.0, H);

    BOOST_CHECK_CLOSE(despread_mag(rQ, rQ), 1.0, 1e-2);
    BOOST_CHECK_LT(despread_mag(rQ, rI), 0.3); // I vs Q decorrelate
}

// The dispatch picks a genuinely different code per signal: an L1 C/A replica does
// not correlate with an L5 Q5 replica (same Fs/channelization, different code).
BOOST_AUTO_TEST_CASE(l1ca_distinct_from_l5q) {
    std::vector<int> prns = {1};
    auto bl1 = make_bank("GPS_L1CA", prns);
    auto bl5 = make_bank("GPS_L5_Q", prns);
    const int H = bl5.repl_period_hops();
    auto r1 = bl1.channels(0, 0, 0.0, 0.0, H);
    auto r5 = bl5.channels(0, 0, 0.0, 0.0, H);

    BOOST_CHECK_CLOSE(despread_mag(r1, r1), 1.0, 1e-2);
    BOOST_CHECK_LT(despread_mag(r1, r5), 0.3); // L1 code vs L5 code decorrelate
}

// The hop-rate prefix-sum generator must reproduce the exact full-PFB channels() to
// ~machine precision on the covering channels (the chip-collapse is an identity; the
// chip-edge-in-the-filter is exact via Phi[k_hi]-Phi[k_lo-1] over integer taps). The
// C++ analog of python/scripts/gps_hoprate_validate.py. Airspy-scale (5 MSPS, N=20,
// ~4.9 samples/chip).
BOOST_AUTO_TEST_CASE(hoprate_matches_exact_pfb) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L1CA");
    BOOST_REQUIRE(sig != nullptr);
    gnss::ChannelizedReplicaBank bank(*sig, 5.0e6, 1.25e6, 20, 4, dsp::Window::Hamming, {1});
    const long long ws = 1000000;
    const int n_hops = 200;
    const std::vector<int> want = {8, 10, 12}; // covering channels near the 1.25 MHz carrier
    for (double dop : {0.0, 2000.0, -3500.0}) {
        auto exact = bank.channels(0, ws, 300.0, dop, n_hops);
        auto hop = bank.channels_hoprate(0, ws, 300.0, dop, n_hops, want);
        // Constant nav bit -1 must just negate the result (per-chip wipe plumbing). The
        // edge-exactness (bit edge on a code-period boundary) is covered in the python.
        auto wiped = bank.channels_hoprate(0, ws, 300.0, dop, n_hops, want,
                                           [](long long) { return -1.0f; });
        for (size_t ci = 0; ci < want.size(); ++ci) {
            BOOST_CHECK_MESSAGE(rel_err(hop[ci], exact[want[ci]]) < 1e-5,
                                "dop " << dop << " ch " << want[ci] << " hop-rate err "
                                       << rel_err(hop[ci], exact[want[ci]]));
            std::vector<cf> neg(n_hops);
            for (int m = 0; m < n_hops; ++m)
                neg[m] = -exact[want[ci]][m];
            BOOST_CHECK_LT(rel_err(wiped[ci], neg), 1e-5);
        }
    }
}

// The streaming wrapper builds the prefix-sum filter once and reuses it across
// contiguous generate() calls, rebuilding only on a large Doppler step -- so steady
// generation is O(n_chips)/hop. Checks: exact match to channels(), the amortization
// (one rebuild over many chunks), and the refresh policy.
BOOST_AUTO_TEST_CASE(hoprate_stream_amortizes_and_matches) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L1CA");
    BOOST_REQUIRE(sig != nullptr);
    gnss::ChannelizedReplicaBank bank(*sig, 5.0e6, 1.25e6, 20, 4, dsp::Window::Hamming, {1});
    const std::vector<int> want = {8, 10, 12};
    gnss::HopRateReplicaStream stream(bank, 0, want, 40.0); // refresh on >40 Hz drift
    const int n_hops = 50;
    const double cp = 300.0, dop = 1500.0;

    long long ws = 1000000;
    for (int chunk = 0; chunk < 5; ++chunk) { // contiguous chunks, stable Doppler
        const auto& out = stream.generate(ws, cp, dop, n_hops);
        auto exact = bank.channels(0, ws, cp, dop, n_hops);
        for (size_t ci = 0; ci < want.size(); ++ci)
            BOOST_CHECK_LT(rel_err(out[ci], exact[want[ci]]), 1e-5);
        ws += (long long)n_hops * bank.fft_len();
    }
    BOOST_CHECK_EQUAL(stream.rebuilds(), 1); // filter built once, reused across 5 chunks

    // Small wobble (< refresh): no rebuild; the 20 Hz-stale filter error is bounded (~-54 dB
    // here -- it scales ~linearly with the drift, so refresh_hz sets the peel depth).
    const auto& wob = stream.generate(ws, cp, dop + 20.0, n_hops);
    BOOST_CHECK_EQUAL(stream.rebuilds(), 1);
    BOOST_CHECK_LT(rel_err(wob[1], bank.channels(0, ws, cp, dop + 20.0, n_hops)[10]), 3e-3);

    // Large jump (> refresh): rebuild.
    stream.generate(ws, cp, dop + 500.0, n_hops);
    BOOST_CHECK_EQUAL(stream.rebuilds(), 2);
}
