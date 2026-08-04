#define BOOST_TEST_MODULE "test_gnss_channelized_replica"

#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank
#include "gnssSignal.hpp"              // for signal_by_name
#include "gpsL5Code.hpp"               // for L5_NH20 (overlay reference)
#include "pfbPrototype.hpp"            // for Window

#include <algorithm>
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

// The L5 Q5 NH20 secondary overlay is applied per primary period: it matches IS-GPS-705,
// and a multi-period coherent despread stays matched only when the overlay alignment
// (nh_phase) agrees -- which is what unlocks coherent integration of the dataless pilot
// past the 1 ms primary period (vs the overlay-blind sum, which the balanced NH20 averages
// to ~0). Single-period despread is unaffected (the overlay is a constant sign there).
BOOST_AUTO_TEST_CASE(l5q_nh20_overlay) {
    std::vector<int> prns = {1};
    auto bank = make_bank("GPS_L5_Q", prns);
    BOOST_REQUIRE_EQUAL(bank.secondary_length(), 20);
    // the overlay sequence the bank applies is exactly the IS-GPS-705 NH20
    int s = 0;
    for (int k = 0; k < 20; ++k) {
        BOOST_CHECK_EQUAL(bank.overlay_sign(k, 0), (int)gps::L5_NH20[k]);
        s += gps::L5_NH20[k];
    }
    BOOST_CHECK_LE(std::abs(s), 4); // NH20 is (near-)balanced -> overlay-blind sum decorrelates

    const int K = 20, H = K * bank.repl_period_hops(); // span a full NH period (20 ms)
    auto data = bank.channels(0, 0, 0.0, 0.0, H, 0);      // overlaid signal, alignment 0
    auto aligned = bank.channels(0, 0, 0.0, 0.0, H, 0);   // matching replica
    auto shifted = bank.channels(0, 0, 0.0, 0.0, H, 1);   // overlay off by one NH chip

    BOOST_TEST_MESSAGE("overlay: aligned=" << despread_mag(data, aligned)
                                           << " shifted=" << despread_mag(data, shifted));
    BOOST_CHECK_CLOSE(despread_mag(data, aligned), 1.0, 1e-2); // aligned -> coherent over 20 periods
    BOOST_CHECK_LT(despread_mag(data, shifted), 0.2);          // one-chip NH misalignment decorrelates

    // sanity: a single primary period is overlay-phase-INSENSITIVE (one constant chip)
    const int H1 = bank.repl_period_hops();
    auto p0 = bank.channels(0, 0, 0.0, 0.0, H1, 0);
    auto p1 = bank.channels(0, 0, 0.0, 0.0, H1, 1);
    BOOST_CHECK_CLOSE(despread_mag(p0, p1), 1.0, 1e-2); // single period: still a matched filter
}

// BOC(1,1) sub-carrier (the first non-BPSK modulation, for L1C / Galileo): applied here to the
// C/A code as a synthetic BOC signal. The BOC replica self-despreads (matched filter), but
// DECORRELATES against the plain BPSK replica of the SAME code -- because the sub-carrier
// averages to zero over a chip, so it has genuinely split the spectrum off the carrier.
BOOST_AUTO_TEST_CASE(boc_subcarrier_applied) {
    gnss::SignalDescriptor boc = *gnss::signal_by_name("GPS_L1CA"); // C/A chips...
    boc.mod = gnss::Modulation::BOC;                                // ...+ a BOC(1,1) sub-carrier
    boc.boc_m = 1;
    boc.boc_n = 1;
    ChannelizedReplicaBank bank_boc(boc, FS, FOFF, N, P, dsp::Window::Hamming, {1});
    auto bank_bpsk = make_bank("GPS_L1CA", {1});
    const int H = bank_boc.repl_period_hops();
    auto rboc = bank_boc.channels(0, 0, 0.0, 0.0, H);
    auto rbpsk = bank_bpsk.channels(0, 0, 0.0, 0.0, H);

    BOOST_CHECK_CLOSE(despread_mag(rboc, rboc), 1.0, 1e-2); // self == matched filter
    BOOST_CHECK_LT(despread_mag(rboc, rbpsk), 0.3);         // BOC vs BPSK (same code) decorrelate
}

// End-to-end L1C-P through the bank: the descriptor dispatches the Weil code AND applies BOC(1,1),
// so a self-despread is a matched filter and a different PRN decorrelates -- the real L1C signal.
BOOST_AUTO_TEST_CASE(l1cp_self_and_cross_prn) {
    std::vector<int> prns = {1, 2};
    auto bank = make_bank("GPS_L1C_P", prns);
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

// L2C is TIME-MULTIPLEXED: CM and CL interleave chip-by-chip into the combined 1.023 Mcps
// stream (CM on the even combined chips, CL on the odd). ChannelizedReplicaBank must model
// that -- the bare 511.5 kcps component code loses ~9-12 dB vs the real interleaved signal
// (python/scripts/gnss/gps_l2c_subband_validate.py). The tell-tale of a correct model: CM and CL
// have DISJOINT combined-chip slots, so the true interleaved signal (CM+CL) despreads with the
// CM replica to ~the same amplitude as CM alone -- the CL half lands in the CM replica's zeros
// and doesn't bleed in. A bank that held the bare CM code (no zeros) would let CL cross-talk in.
BOOST_AUTO_TEST_CASE(l2c_cm_time_multiplexed) {
    std::vector<int> prns = {1, 2};
    auto bcm = make_bank("GPS_L2C_CM", prns);
    auto bcl = make_bank("GPS_L2C_CL", prns);
    // Partial window: the full 20 ms CM period is ~20k hops; a few hundred hops already give a
    // clean matched filter (self is energy-normalised to 1) without the slow full-period run.
    const int H = 400;
    auto cmA = bcm.channels(0, 0, 0.0, 0.0, H);  // PRN 1 CM  (even combined chips)
    auto cmB = bcm.channels(1, 0, 0.0, 0.0, H);  // PRN 2 CM
    auto clA = bcl.channels(0, 0, 0.0, 0.0, H);  // PRN 1 CL  (odd combined chips)

    BOOST_CHECK_CLOSE(despread_mag(cmA, cmA), 1.0, 1e-2); // self == matched filter
    BOOST_CHECK_LT(despread_mag(cmA, cmB), 0.3);          // different PRN decorrelates

    // The true L2C signal = CM (even) + CL (odd), summed channel-by-channel.
    auto inter = cmA;
    for (size_t ch = 0; ch < inter.size(); ++ch)
        for (size_t m = 0; m < inter[ch].size(); ++m)
            inter[ch][m] += clA[ch][m];
    // CM recovered from the interleaved stream ~ CM recovered alone: the CL half is in the CM
    // replica's zero slots, so it barely bleeds in. (No fix -> CL cross-talk drags this down.)
    BOOST_CHECK_CLOSE(despread_mag(inter, cmA), 1.0, 5.0);
}

// The hop-rate prefix-sum generator must reproduce the exact full-PFB channels() to
// ~machine precision on the covering channels (the chip-collapse is an identity; the
// chip-edge-in-the-filter is exact via Phi[k_hi]-Phi[k_lo-1] over integer taps). The
// C++ analog of python/scripts/gnss/gps_hoprate_validate.py. Airspy-scale (5 MSPS, N=20,
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

// Same identity at CHORD scale, which is a genuinely different regime from the airspy case
// above and is the one the banded search now depends on: Fs = 3200 MSPS (real), N = 8192,
// fft_len = 16384 so the prefix-sum filter is 65536 taps long, ~313 samples/chip, and the
// carrier sits at bin ~6023 instead of ~5. Two things could plausibly break here and not
// there -- the unit-modulus phasor recurrence over a 16384-sample hop, and the chip-to-tap
// integer range mapping when a chip spans hundreds of samples -- so gate it directly.
// GnssChannelizedSearch builds repl0 for ONLY the covering channels via this path; if it
// ever stops matching channels(), the search silently correlates against the wrong replica.
BOOST_AUTO_TEST_CASE(hoprate_matches_exact_pfb_chord_scale) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    BOOST_REQUIRE(sig != nullptr);
    // CHORD CRS F-engine: 4-tap Hamming, 16384 real samples/hop -> 8192 channels.
    gnss::ChannelizedReplicaBank bank(*sig, 3200.0e6, 1176.45e6, 8192, 4, dsp::Window::Hamming,
                                      {1});
    const long long ws = 3125LL * bank.fft_len(); // the search's anchor
    // Channels covering the L5 carrier: bin 6023.4 = 1176.45 MHz / (3200 MHz / 16384).
    const std::vector<int> want = {5990, 6023, 6024, 6060};
    // MUST BE A LONG RUN. The failure this guards against is a floating-point tie at an exact
    // chip/tap boundary, which at zero Doppler recurs on a fixed period (~every 391 hops, = 2
    // code periods) and FIRST STRIKES AT HOP 255. A dozen hops -- the natural thing to write,
    // since channels() costs O(n_hops * 16384) here -- passes at 6e-8 while a full period is
    // wrong by 1.8e-3 rms and 1.99 absolute. Do not shorten this.
    for (double dop : {0.0, 2500.0, -2500.0}) {
        const int n_hops = (dop == 0.0) ? bank.repl_period_hops() : 400;
        auto exact = bank.channels(0, ws, 137.0, dop, n_hops);
        auto hop = bank.channels_hoprate(0, ws, 137.0, dop, n_hops, want);
        for (size_t ci = 0; ci < want.size(); ++ci) {
            // 1e-6, not the 1e-5 used at airspy scale: the two paths agree to ~8e-8 here (both
            // sit ~3.5e-8 from a double-precision reference), so a decade of headroom is ample
            // and anything looser readmits the one-tap error at -53 dB.
            BOOST_CHECK_MESSAGE(rel_err(hop[ci], exact[want[ci]]) < 1e-6,
                                "dop " << dop << " ch " << want[ci] << " over " << n_hops
                                       << " hops: CHORD hop-rate err "
                                       << rel_err(hop[ci], exact[want[ci]]));
            // Per-hop, so a single bad hop cannot hide under an rms over thousands of good ones
            // (1.99 absolute out of 3125 hops is only 1.8e-3 in rms).
            double worst = 0.0, ref = 0.0;
            for (int m = 0; m < n_hops; ++m) {
                worst = std::max(worst, (double)std::abs(hop[ci][m] - exact[want[ci]][m]));
                ref += std::norm(exact[want[ci]][m]);
            }
            const double rms = std::sqrt(ref / n_hops);
            BOOST_CHECK_MESSAGE(worst < 1e-4 * rms, "dop " << dop << " ch " << want[ci]
                                                           << " worst single-hop err " << worst
                                                           << " vs rms |R| " << rms);
        }
    }
}

// ---------------------------------------------------------------------------------------
// GLONASS FDMA: the per-PRN carrier offset.
//
// Every other signal here is CDMA -- one carrier per band, satellites separated by code. GLONASS
// L2OF inverts that: EVERY satellite transmits the same 511-chip m-sequence and they are
// separated by CARRIER, satellite k sitting at 1246.0 + k*0.4375 MHz. So the per-satellite
// identity lives entirely in ChannelizedReplicaBank::set_prn_freq_offset(), and these cases pin
// that it enters the carrier EXACTLY, and nowhere else.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(fdma_offset_is_exactly_a_carrier_shift) {
    // ★ THE IDENTITY: an FDMA offset of X Hz on a PRN must produce precisely the replica that
    // PRN would have at Doppler+X with no offset -- because both enter as carrier_offset(p) +
    // doppler in the same phasor. Every GLONASS PRN shares one code, so the two replicas are
    // comparable chip-for-chip and any discrepancy is the offset leaking somewhere it should not.
    //
    // ⚠️ code_doppler_sign = 0 is what makes it an EXACT identity rather than an approximate
    // one, and it is also the physics: an FDMA offset is a STATIC carrier difference, not a
    // line-of-sight velocity, so it must NOT scale the code rate the way Doppler does (every
    // GLONASS satellite clocks its code at 511 kcps regardless of k). Leaving code-Doppler on
    // would make the two sides differ by exactly that term -- which is the bug this guards.
    auto bank = make_bank("GLO_L2OF", {1, 2});
    bank.code_doppler_sign = 0.0;

    const double X = 437.5e3; // one GLONASS L2 channel
    bank.set_prn_freq_offset(1, X);

    const long long ws = 3 * 2 * N * 7;
    const int n_hops = 200;
    for (double dop : {0.0, 1200.0, -1200.0}) {
        auto shifted = bank.channels(1, ws, 61.0, dop, n_hops);     // offset X, Doppler dop
        auto by_dop = bank.channels(0, ws, 61.0, dop + X, n_hops);  // no offset, Doppler dop+X
        for (int c = 0; c < N; ++c)
            BOOST_CHECK_MESSAGE(rel_err(shifted[c], by_dop[c]) < 1e-6,
                                "dop " << dop << " ch " << c << ": FDMA offset != carrier shift, err "
                                       << rel_err(shifted[c], by_dop[c]));
    }
}

BOOST_AUTO_TEST_CASE(fdma_offset_moves_the_occupied_channels) {
    // The offset must actually relocate the signal on the channel grid -- a no-op that silently
    // did nothing would still pass an identity test comparing two no-ops.
    auto bank = make_bank("GLO_L2OF", {1, 2});
    const double bin_w = FS / (2 * N);
    const double X = 3 * bin_w; // three whole channels, so the shift is unambiguous
    bank.set_prn_freq_offset(1, X);

    const auto base = bank.covering_bins(0.0, 0.0, 0);
    const auto moved = bank.covering_bins(0.0, 0.0, 1);
    BOOST_REQUIRE(!base.empty() && !moved.empty());
    // Centre of the covering set should move by X/bin_w channels.
    const double c0 = 0.5 * (base.front() + base.back());
    const double c1 = 0.5 * (moved.front() + moved.back());
    BOOST_CHECK_MESSAGE(std::abs((c1 - c0) - 3.0) < 0.51,
                        "covering set moved by " << (c1 - c0) << " channels, expected ~3");
    // And an unset PRN index must behave exactly as before (band offset only) -- this is what
    // keeps every CDMA caller untouched.
    BOOST_CHECK(bank.covering_bins(0.0, 0.0, -1) == base);
}

BOOST_AUTO_TEST_CASE(fdma_hoprate_matches_exact_with_offset) {
    // The hop-rate generator carries its own copy of the carrier, so the offset has to reach
    // BOTH paths. If it reached only channels(), the fast path used in production would quietly
    // despread every GLONASS satellite at the wrong frequency.
    auto bank = make_bank("GLO_L2OF", {1, 2});
    bank.set_prn_freq_offset(1, -2 * 437.5e3); // k = -2

    std::vector<int> want = bank.covering_bins(0.0, 0.0, 1);
    BOOST_REQUIRE(!want.empty());
    const long long ws = 2 * 2 * N * 11;
    const int n_hops = 150;
    auto exact = bank.channels(1, ws, 23.0, 900.0, n_hops);
    auto hop = bank.channels_hoprate(1, ws, 23.0, 900.0, n_hops, want);
    for (size_t ci = 0; ci < want.size(); ++ci)
        BOOST_CHECK_MESSAGE(rel_err(hop[ci], exact[want[ci]]) < 1e-5,
                            "ch " << want[ci] << ": hop-rate/exact disagree under an FDMA offset, err "
                                  << rel_err(hop[ci], exact[want[ci]]));
}

BOOST_AUTO_TEST_CASE(fdma_same_code_every_satellite) {
    // Sanity on the other half of FDMA: unlike every CDMA signal here, cross-PRN despread must
    // be PERFECT, because the satellites really do share one code. A cross-PRN null would mean
    // someone had wired a per-PRN code table in, defeating the whole design.
    auto bank = make_bank("GLO_L2OF", {1, 2});
    const long long ws = 0;
    auto a = bank.channels(0, ws, 0.0, 0.0, 40);
    auto b = bank.channels(1, ws, 0.0, 0.0, 40);
    BOOST_CHECK_MESSAGE(despread_mag(a, b) > 0.99,
                        "GLONASS PRNs must share one code; cross-despread " << despread_mag(a, b));
}
