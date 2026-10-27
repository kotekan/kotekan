#define BOOST_TEST_MODULE "test_gnss_channelized_replica"

#include "beidouB2aCode.hpp"           // for b2ap_secondary (baked-CS reference)
#include "dspPfbPrototype.hpp"         // for Window
#include "galileoE5aCode.hpp"          // for e5aq_secondary (baked-CS reference)
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank
#include "gnssSignal.hpp"              // for signal_by_name
#include "gpsL5Code.hpp"               // for L5_NH20 (overlay reference)

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <complex>
#include <vector>

using cf = std::complex<float>;
using gnss::ChannelizedReplicaBank;

namespace {

// L5 and L1 C/A both have a 1 ms primary period, a whole number of hops at this Fs; 2 samples
// per chip on L5, 20 on L1.
constexpr double FS = 20.46e6;
constexpr double FOFF = 5.115e6; // carrier 5 channels above DC
constexpr int N = 10;            // 1.023 MHz channels
constexpr int P = 4;

// Matched-filter amplitude of `data` against `repl` over all channels. channelized_despread
// normalizes by the replica energy, so a self-despread gives 1.
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

// L5 Q5 dispatch: a self-despread is a matched filter (1) and a different PRN's replica
// decorrelates, so the bank holds distinct per-PRN codes.
BOOST_AUTO_TEST_CASE(l5q_self_and_cross_prn) {
    std::vector<int> prns = {1, 2};
    auto bank = make_bank("GPS_L5_Q", prns);
    const int H = bank.repl_period_hops();
    auto rA = bank.channels(0, 0, 0.0, 0.0, H); // PRN 1
    auto rB = bank.channels(1, 0, 0.0, 0.0, H); // PRN 2

    BOOST_CHECK_CLOSE(despread_mag(rA, rA), 1.0, 1e-2); // self == matched filter
    BOOST_CHECK_LT(despread_mag(rA, rB), 0.3);          // different PRN decorrelates
}

// The L5 Q5 NH20 overlay (IS-GPS-705) is applied per primary period. A coherent despread over
// several periods stays matched only when the overlay alignment (nh_phase) agrees; an
// overlay-blind sum averages the near-balanced NH20 to ~0. Within one period the overlay is a
// constant sign and has no effect.
BOOST_AUTO_TEST_CASE(l5q_nh20_overlay) {
    std::vector<int> prns = {1};
    auto bank = make_bank("GPS_L5_Q", prns);
    BOOST_REQUIRE_EQUAL(bank.secondary_length(), 20);
    // the bank's overlay is the IS-GPS-705 NH20
    int s = 0;
    for (int k = 0; k < 20; ++k) {
        BOOST_CHECK_EQUAL(bank.overlay_sign(k, 0), (int)gps::L5_NH20[k]);
        s += gps::L5_NH20[k];
    }
    BOOST_CHECK_LE(std::abs(s), 4); // near-balanced: an overlay-blind sum decorrelates

    const int K = 20, H = K * bank.repl_period_hops();  // span a full NH period (20 ms)
    auto data = bank.channels(0, 0, 0.0, 0.0, H, 0);    // overlaid signal, alignment 0
    auto aligned = bank.channels(0, 0, 0.0, 0.0, H, 0); // matching replica
    auto shifted = bank.channels(0, 0, 0.0, 0.0, H, 1); // overlay off by one NH chip

    BOOST_TEST_MESSAGE("overlay: aligned=" << despread_mag(data, aligned)
                                           << " shifted=" << despread_mag(data, shifted));
    BOOST_CHECK_CLOSE(despread_mag(data, aligned), 1.0,
                      1e-2);                          // aligned -> coherent over 20 periods
    BOOST_CHECK_LT(despread_mag(data, shifted), 0.2); // one-chip NH misalignment decorrelates

    // A single primary period does not depend on the overlay alignment (one constant chip).
    const int H1 = bank.repl_period_hops();
    auto p0 = bank.channels(0, 0, 0.0, 0.0, H1, 0);
    auto p1 = bank.channels(0, 0, 0.0, 0.0, H1, 1);
    BOOST_CHECK_CLOSE(despread_mag(p0, p1), 1.0, 1e-2); // single period: still a matched filter
}

// Per-PRN baked secondaries (GAL_E5A_Q_CS, BDS_B2A_P_CS): the primary tiled by that PRN's
// 100-chip secondary into one 1023000-chip code. Seeding depends on this convention, which no
// single stage can check, so pin it to the plain-primary bank chip by chip: a baked code phase
// is primary cp + 10230 * (period index mod 100), with sign chip * secondary[period] as in
// overlay_sign().
BOOST_AUTO_TEST_CASE(e5aq_cs_baked_convention) {
    const gnss::SignalDescriptor* cs = gnss::signal_by_name("GAL_E5A_Q_CS");
    BOOST_REQUIRE(cs != nullptr);
    BOOST_CHECK_EQUAL(cs->code_length, 1023000L);
    BOOST_CHECK_EQUAL(cs->secondary_length, 0); // baked in: nothing left to overlay
    for (int prn : {1, 30}) {
        auto bank_cs = make_bank("GAL_E5A_Q_CS", {prn});
        auto bank_q = make_bank("GAL_E5A_Q", {prn});
        // The plain bank's overlay slot holds only overlays shared by every PRN, so GAL_E5A_Q
        // has none and nh_search ignores it; GAL_E5A_Q_CS trackers are seeded by dead
        // reckoning for that reason. A nonzero value means the bank gained per-PRN overlays.
        BOOST_CHECK_EQUAL(bank_q.secondary_length(), 0);
        const auto sec = galileo::e5aq_secondary(prn);
        for (int k = 0; k < 100; ++k)
            for (int i = 0; i < 10230; i += 341) // 30 probes per primary period
                BOOST_REQUIRE_EQUAL((int)bank_cs.code_chip(0, (double)k * 10230 + i),
                                    (int)bank_q.code_chip(0, (double)i) * (int)sec[(size_t)k]);
    }
}

BOOST_AUTO_TEST_CASE(b2ap_cs_baked_convention) {
    const gnss::SignalDescriptor* cs = gnss::signal_by_name("BDS_B2A_P_CS");
    BOOST_REQUIRE(cs != nullptr);
    BOOST_CHECK_EQUAL(cs->code_length, 1023000L);
    BOOST_CHECK_EQUAL(cs->secondary_length, 0);
    for (int prn : {19, 45}) { // BDS-3 range (B2a is not on BDS-2)
        auto bank_cs = make_bank("BDS_B2A_P_CS", {prn});
        auto bank_p = make_bank("BDS_B2A_P", {prn});
        BOOST_CHECK_EQUAL(bank_p.secondary_length(), 0); // per-PRN: not in the shared slot
        const auto sec = beidou::b2ap_secondary(prn);
        for (int k = 0; k < 100; ++k)
            for (int i = 0; i < 10230; i += 341)
                BOOST_REQUIRE_EQUAL((int)bank_cs.code_chip(0, (double)k * 10230 + i),
                                    (int)bank_p.code_chip(0, (double)i) * (int)sec[(size_t)k]);
    }
}

// Over a full 100 ms baked period the CS replica is a matched filter against itself, while the
// overlay-blind primary tiling decorrelates to ~|sum(CS100)|/100, which is why multi-period
// coherent integration needs the overlay (a CHORD record spans ~10 secondary chips). A 20-sample
// hop divides the 20460-sample primary period exactly (1023 hops), so the blind bound is tight.
BOOST_AUTO_TEST_CASE(e5aq_cs_multiperiod_coherence) {
    std::vector<int> prns = {1};
    auto cs = make_bank("GAL_E5A_Q_CS", prns);
    auto q = make_bank("GAL_E5A_Q", prns);
    const auto sec = galileo::e5aq_secondary(1);
    int s = 0;
    for (int8_t c : sec)
        s += c;
    BOOST_TEST_MESSAGE("CS100(PRN 1) balance sum = " << s);

    const int H = cs.repl_period_hops(); // one full baked period = 100 primary periods
    BOOST_CHECK_EQUAL(H, 100 * q.repl_period_hops());
    auto data = cs.channels(0, 0, 0.0, 0.0, H); // the real pilot: overlay signs in
    auto blind = q.channels(0, 0, 0.0, 0.0, H); // primary tiled with NO overlay signs

    const double blind_mag = despread_mag(data, blind);
    BOOST_TEST_MESSAGE("baked-vs-blind despread = " << blind_mag);
    BOOST_CHECK_CLOSE(despread_mag(data, data), 1.0, 1e-2); // self == matched filter
    BOOST_CHECK_LT(blind_mag, std::abs(s) / 100.0 + 0.1);   // ~|sum|/100, the design bound
    BOOST_CHECK_LT(blind_mag, 0.3);                         // and it decorrelates
}

// BOC(1,1) applied to the C/A code as a synthetic signal. The sub-carrier averages to zero over
// a chip, so the BOC replica is a matched filter against itself but decorrelates against the
// BPSK replica of the same code.
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

// L1C-P through the bank: the descriptor selects the Weil code and BOC(1,1), so a self-despread
// is a matched filter and a different PRN decorrelates.
BOOST_AUTO_TEST_CASE(l1cp_self_and_cross_prn) {
    std::vector<int> prns = {1, 2};
    auto bank = make_bank("GPS_L1C_P", prns);
    const int H = bank.repl_period_hops();
    auto rA = bank.channels(0, 0, 0.0, 0.0, H);         // PRN 1
    auto rB = bank.channels(1, 0, 0.0, 0.0, H);         // PRN 2
    BOOST_CHECK_CLOSE(despread_mag(rA, rA), 1.0, 1e-2); // self == matched filter
    BOOST_CHECK_LT(despread_mag(rA, rB), 0.3);          // different PRN decorrelates
}

// I5 and Q5 are distinct codes on the same carrier, so they decorrelate.
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

// The dispatch picks a different code per signal: an L1 C/A replica does not correlate with
// an L5 Q5 replica (same Fs and channelization, different code).
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

// L2C is time-multiplexed: CM and CL interleave chip by chip into the combined 1.023 Mcps
// stream (CM on the even combined chips, CL on the odd), and the bank models that. CM and CL
// then occupy disjoint combined-chip slots, so the interleaved signal (CM + CL) despreads
// against the CM replica to about the same amplitude as CM alone: CL falls in the CM replica's
// zeros. A bank holding the bare 511.5 kcps CM code (no zeros) would let CL leak in.
BOOST_AUTO_TEST_CASE(l2c_cm_time_multiplexed) {
    std::vector<int> prns = {1, 2};
    auto bcm = make_bank("GPS_L2C_CM", prns);
    auto bcl = make_bank("GPS_L2C_CL", prns);
    // 400 hops, not the full 20 ms (~20k-hop) CM period, already give a clean matched filter.
    const int H = 400;
    auto cmA = bcm.channels(0, 0, 0.0, 0.0, H); // PRN 1 CM  (even combined chips)
    auto cmB = bcm.channels(1, 0, 0.0, 0.0, H); // PRN 2 CM
    auto clA = bcl.channels(0, 0, 0.0, 0.0, H); // PRN 1 CL  (odd combined chips)

    BOOST_CHECK_CLOSE(despread_mag(cmA, cmA), 1.0, 1e-2); // self == matched filter
    BOOST_CHECK_LT(despread_mag(cmA, cmB), 0.3);          // different PRN decorrelates

    // The interleaved L2C signal: CM (even) + CL (odd), summed per channel.
    auto inter = cmA;
    for (size_t ch = 0; ch < inter.size(); ++ch)
        for (size_t m = 0; m < inter[ch].size(); ++m)
            inter[ch][m] += clA[ch][m];
    // CL sits in the CM replica's zero slots, so CM from the interleaved stream matches CM alone.
    BOOST_CHECK_CLOSE(despread_mag(inter, cmA), 1.0, 5.0);
}

// The hop-rate generator reproduces the full-PFB channels() to ~machine precision on the
// covering channels: the per-chip collapse is an identity, and a chip contributes
// Phi[k_hi+1] - Phi[k_lo] over its integer tap range. C++ counterpart of
// python/scripts/gnss/gps_hoprate_validate.py, at 5 MSPS, N = 20 (~4.9 samples/chip).
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
        // A constant nav bit of -1 must negate the result. Bit edges on code-period boundaries
        // are checked in gps_hoprate_validate.py.
        auto wiped =
            bank.channels_hoprate(0, ws, 300.0, dop, n_hops, want, [](long long) { return -1.0f; });
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

// The same identity at CHORD scale: Fs = 3200 MSPS real, N = 8192, fft_len = 16384, so the
// prefix-sum filter is 65536 taps, a chip spans ~313 samples, and the carrier sits at bin ~6023
// rather than 10. The phasor recurrence over a 16384-sample hop and the chip-to-tap mapping at
// hundreds of samples per chip can fail here and not at the small geometry.
// GnssChannelizedSearch builds repl0 for the covering channels only through this path, so a
// mismatch here means the search correlates against the wrong replica.
BOOST_AUTO_TEST_CASE(hoprate_matches_exact_pfb_chord_scale) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    BOOST_REQUIRE(sig != nullptr);
    // CHORD F-engine: 4-tap Hamming, 16384 real samples per hop, 8192 channels.
    gnss::ChannelizedReplicaBank bank(*sig, 3200.0e6, 1176.45e6, 8192, 4, dsp::Window::Hamming,
                                      {1});
    const long long ws = 3125LL * bank.fft_len(); // the search's anchor, Mp * fft_len
    // Channels covering the L5 carrier: bin 6023.4 = 1176.45 MHz / (3200 MHz / 16384).
    const std::vector<int> want = {5990, 6023, 6024, 6060};
    // Run a full replica period at zero Doppler; do not shorten it. Ties at an exact chip/tap
    // boundary, where floating point can hand a tap to the neighbouring chip, occur only ~8
    // times per replica period at this geometry (see hoprate_stream_into), so a short run can
    // miss them all.
    for (double dop : {0.0, 2500.0, -2500.0}) {
        const int n_hops = (dop == 0.0) ? bank.repl_period_hops() : 400;
        auto exact = bank.channels(0, ws, 137.0, dop, n_hops);
        auto hop = bank.channels_hoprate(0, ws, 137.0, dop, n_hops, want);
        for (size_t ci = 0; ci < want.size(); ++ci) {
            // 1e-6, tighter than the 1e-5 above: the paths agree to ~1e-7 at this scale.
            BOOST_CHECK_MESSAGE(rel_err(hop[ci], exact[want[ci]]) < 1e-6,
                                "dop " << dop << " ch " << want[ci] << " over " << n_hops
                                       << " hops: CHORD hop-rate err "
                                       << rel_err(hop[ci], exact[want[ci]]));
            // Also bound each hop, so one bad hop cannot hide in an rms over thousands.
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

// ---- GLONASS FDMA: the per-PRN carrier offset ----
// GLONASS L2OF satellites share one 511-chip code and are separated by carrier, satellite k at
// 1246.0 + k*0.4375 MHz, so a satellite's identity is only its set_prn_freq_offsets() entry. These
// cases check that the offset enters the carrier exactly and nowhere else.

BOOST_AUTO_TEST_CASE(fdma_offset_is_exactly_a_carrier_shift) {
    // An FDMA offset of X Hz must give exactly the replica at Doppler + X with no offset: both
    // enter the same carrier phasor as carrier_offset(p) + doppler, and every PRN shares the
    // code, so the replicas compare chip for chip.
    // code_doppler_sign = 0 makes the identity exact: code Doppler scales the code rate by
    // doppler_hz alone, so with it on the Doppler + X side would also get a faster code. An FDMA
    // offset is a static carrier difference, not a velocity, and leaves the 511 kcps code rate
    // alone.
    auto bank = make_bank("GLO_L2OF", {1, 2});
    bank.code_doppler_sign = 0.0;

    const double X = 437.5e3; // one GLONASS L2 channel
    bank.set_prn_freq_offsets({0.0, X});

    const long long ws = 3 * 2 * N * 7;
    const int n_hops = 200;
    for (double dop : {0.0, 1200.0, -1200.0}) {
        auto shifted = bank.channels(1, ws, 61.0, dop, n_hops);    // offset X, Doppler dop
        auto by_dop = bank.channels(0, ws, 61.0, dop + X, n_hops); // no offset, Doppler dop+X
        for (int c = 0; c < N; ++c)
            BOOST_CHECK_MESSAGE(rel_err(shifted[c], by_dop[c]) < 1e-6,
                                "dop " << dop << " ch " << c
                                       << ": FDMA offset != carrier shift, err "
                                       << rel_err(shifted[c], by_dop[c]));
    }
}

BOOST_AUTO_TEST_CASE(fdma_offset_moves_the_occupied_channels) {
    // The offset must also move the PRN's covering channels, not only its replica.
    auto bank = make_bank("GLO_L2OF", {1, 2});
    const double bin_w = FS / (2 * N);
    const double X = 3 * bin_w; // three whole channels, so the shift is unambiguous
    bank.set_prn_freq_offsets({0.0, X});

    const auto base = bank.covering_bins(0.0, 0.0, 0);
    const auto moved = bank.covering_bins(0.0, 0.0, 1);
    BOOST_REQUIRE(!base.empty() && !moved.empty());
    // Centre of the covering set should move by X/bin_w channels.
    const double c0 = 0.5 * (base.front() + base.back());
    const double c1 = 0.5 * (moved.front() + moved.back());
    BOOST_CHECK_MESSAGE(std::abs((c1 - c0) - 3.0) < 0.51,
                        "covering set moved by " << (c1 - c0) << " channels, expected ~3");
    // An unset PRN index (-1) gives the band offset alone, as CDMA callers expect.
    BOOST_CHECK(bank.covering_bins(0.0, 0.0, -1) == base);
}

BOOST_AUTO_TEST_CASE(fdma_hoprate_matches_exact_with_offset) {
    // The hop-rate generator computes its own carrier, so the offset must reach it as well as
    // channels(); GnssChannelizedSearch builds its replica on the hop-rate path.
    auto bank = make_bank("GLO_L2OF", {1, 2});
    bank.set_prn_freq_offsets({0.0, -2 * 437.5e3}); // k = -2

    std::vector<int> want = bank.covering_bins(0.0, 0.0, 1);
    BOOST_REQUIRE(!want.empty());
    const long long ws = 2 * 2 * N * 11;
    const int n_hops = 150;
    auto exact = bank.channels(1, ws, 23.0, 900.0, n_hops);
    auto hop = bank.channels_hoprate(1, ws, 23.0, 900.0, n_hops, want);
    for (size_t ci = 0; ci < want.size(); ++ci)
        BOOST_CHECK_MESSAGE(rel_err(hop[ci], exact[want[ci]]) < 1e-5,
                            "ch " << want[ci]
                                  << ": hop-rate/exact disagree under an FDMA offset, err "
                                  << rel_err(hop[ci], exact[want[ci]]));
}

BOOST_AUTO_TEST_CASE(fdma_same_code_every_satellite) {
    // Unlike the CDMA signals, a cross-PRN despread must match perfectly: GLONASS satellites
    // share one code.
    auto bank = make_bank("GLO_L2OF", {1, 2});
    const long long ws = 0;
    auto a = bank.channels(0, ws, 0.0, 0.0, 40);
    auto b = bank.channels(1, ws, 0.0, 0.0, 40);
    BOOST_CHECK_MESSAGE(despread_mag(a, b) > 0.99,
                        "GLONASS PRNs must share one code; cross-despread " << despread_mag(a, b));
}

// SBAS L5 is one data channel on the I5 code (no pilot), each 2 ms symbol a Manchester pair of
// opposite 1 ms halves. SBAS_L5_I_MC must resolve through signal_by_name (a name missing there
// FATALs the node at start) and tile the I5 code of the SBAS PRN as [c, -c]; SBAS_L5_I with its
// 2-chip secondary applied from period 0 is the same replica.
BOOST_AUTO_TEST_CASE(sbas_l5i_mc_is_the_i5_code_under_the_manchester_pair) {
    const std::vector<int> prns = {131};
    auto bmc = make_bank("SBAS_L5_I_MC", prns);
    auto bsi = make_bank("SBAS_L5_I", prns);
    auto bi = make_bank("GPS_L5_I", prns); // the raw I5 code, overlay off by default
    auto bq = make_bank("GPS_L5_Q", prns);
    const int H = bi.repl_period_hops(); // one 1 ms code period
    BOOST_REQUIRE_EQUAL(bmc.repl_period_hops(), 2 * H);

    auto rMC = bmc.channels(0, 0, 0.0, 0.0, 2 * H);
    auto rI = bi.channels(0, 0, 0.0, 0.0, 2 * H);
    auto rQ = bq.channels(0, 0, 0.0, 0.0, 2 * H);
    auto rS = bsi.channels(0, 0, 0.0, 0.0, 2 * H, 0); // the pair from period 0

    auto half = [](const std::vector<std::vector<cf>>& r, int a, int b) {
        std::vector<std::vector<cf>> out;
        for (const auto& ch : r)
            out.emplace_back(ch.begin() + a, ch.begin() + b);
        return out;
    };
    auto amp = [](const std::vector<std::vector<cf>>& d, const std::vector<std::vector<cf>>& r) {
        return gnss::channelized_despread(d, r).amplitude;
    };

    // First half is +c, second half -c (the PFB's few-hop edge at the sign flip aside).
    BOOST_CHECK_GT(std::real(amp(half(rMC, 0, H), half(rI, 0, H))), 0.95);
    BOOST_CHECK_LT(std::real(amp(half(rMC, H, 2 * H), half(rI, H, 2 * H))), -0.95);
    // A replica without the pair sums a symbol's halves to ~zero.
    BOOST_CHECK_LT(std::abs(amp(rMC, rI)), 0.05);
    // The tiled code and the per-period secondary are the same replica.
    BOOST_CHECK_GT(std::real(amp(rS, rMC)), 0.99);
    // It is the I5 code: the Q5 code of the same PRN does not correlate.
    BOOST_CHECK_LT(std::abs(amp(half(rMC, 0, H), half(rQ, 0, H))), 0.3);
}

// GLONASS L2OF's 50 bps bit is a Manchester pair of 10 ms halves. GLO_L2OF_MC must resolve
// through signal_by_name and tile the C/A code ten periods +, ten periods -; the FDMA carrier
// offset applies to it as to GLO_L2OF.
BOOST_AUTO_TEST_CASE(glo_l2of_mc_is_the_ca_code_under_the_10ms_halves) {
    const std::vector<int> prns = {1, 2};
    auto bmc = make_bank("GLO_L2OF_MC", prns);
    auto bca = make_bank("GLO_L2OF", prns);
    const std::vector<double> df = {-7 * 437.5e3, 6 * 437.5e3};
    bmc.set_prn_freq_offsets(df);
    bca.set_prn_freq_offsets(df);
    const int H = bca.repl_period_hops(); // one 1 ms code period
    BOOST_REQUIRE_EQUAL(bmc.repl_period_hops(), 20 * H);

    auto amp = [](const std::vector<std::vector<cf>>& d, const std::vector<std::vector<cf>>& r) {
        return gnss::channelized_despread(d, r).amplitude;
    };
    for (int p = 0; p < 2; ++p) {
        // the first and the last 1 ms of each 10 ms half, away from the PFB's edge at the flip
        for (int period : {1, 8, 11, 18}) {
            const long long ws = (long long)period * H * N * 2; // window start, samples
            auto rMC = bmc.channels(p, ws, 0.0, 0.0, H);
            auto rCA = bca.channels(p, ws, 0.0, 0.0, H);
            const double a = std::real(amp(rMC, rCA));
            if (period < 10)
                BOOST_CHECK_MESSAGE(a > 0.95, "slot " << p << " period " << period << ": " << a);
            else
                BOOST_CHECK_MESSAGE(a < -0.95, "slot " << p << " period " << period << ": " << a);
        }
    }
    // A replica without the halves sums the bit to ~zero.
    auto rMC = bmc.channels(0, 0, 0.0, 0.0, 20 * H);
    auto rCA = bca.channels(0, 0, 0.0, 0.0, 20 * H);
    BOOST_CHECK_LT(std::abs(amp(rMC, rCA)), 0.05);
}
