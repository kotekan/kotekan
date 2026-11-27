#define BOOST_TEST_MODULE "test_gnss_channelized_replica"

#include "beidouB2aCode.hpp"           // for b2ap_secondary (baked-CS reference)
#include "beidouB2aCode.hpp"           // for b2ad_secondary
#include "dspPfbPrototype.hpp"         // for Window
#include "galileoE1Code.hpp"           // for E1C_CS25 (baked-CS reference)
#include "galileoE5aCode.hpp"          // for e5aq_secondary (baked-CS reference)
#include "galileoE5aCode.hpp"          // for e5ai_secondary
#include "galileoE5bCode.hpp"          // for e5bi_secondary
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank
#include "gnssSignal.hpp"              // for signal_by_name
#include "gpsL2CCode.hpp"              // for generate_l2cm_code, generate_l2cl_code
#include "gpsL5Code.hpp"               // for L5_NH20, L5_NH10 (overlay references)
#include "pfbPrototype.hpp"            // for fftwEngine's pfb_prototype, pfb_push, pfb_fold

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

// ---- Channelizations other than the default ----
namespace {

// The conjugated channels `want` of a PFB that folds in time order, built from fftwEngine's own
// prototype, push and fold and a direct DFT. Hop m ends at sample ws + (m + 1) * M - 1, and
// sample(n) is input sample n as the type fftwEngine folds (float or complex<float>).
template<typename T, typename Sample>
std::vector<std::vector<cf>> time_order_pfb(Sample sample, long long ws, int M, int taps,
                                            int n_hops, const std::vector<int>& want) {
    const auto proto = pfb_prototype(M, taps, PfbWindow::Hamming);
    std::vector<T> hist((size_t)M * taps, T{}), block((size_t)M), folded((size_t)M);
    std::vector<std::vector<cf>> chan(want.size(), std::vector<cf>((size_t)n_hops));
    for (int h = -(taps - 1); h < n_hops; ++h) {
        for (int i = 0; i < M; ++i)
            block[(size_t)i] = sample(ws + (long long)h * M + i);
        pfb_push(hist.data(), block.data(), M, taps);
        if (h < 0)
            continue;
        pfb_fold(hist.data(), proto.data(), M, taps, folded.data());
        for (size_t ci = 0; ci < want.size(); ++ci) {
            std::complex<double> acc = 0.0;
            for (int i = 0; i < M; ++i)
                acc += std::complex<double>(folded[(size_t)i])
                       * std::polar(1.0, -2.0 * M_PI * want[ci] * i / M);
            chan[ci][(size_t)h] = (cf)std::conj(acc);
        }
    }
    return chan;
}

// The carrier phase and code chip of `bank`'s PRN slot 0 at input sample n, for a carrier at
// f_hz and the code phase argument cp.
struct SkySample {
    const ChannelizedReplicaBank& bank;
    double fs, f_hz, dop, cp;
    std::complex<double> carrier(long long n) const {
        const long double w = 2.0L * (long double)M_PI * (long double)(f_hz + dop) / fs;
        return std::polar(1.0, (double)std::fmod(w * (long double)n, 2.0L * (long double)M_PI));
    }
    double chip(long long n) const {
        const long double cps = bank.chip_rate_hz() / fs * (1.0 + dop / bank.carrier_hz());
        const long double L = (long double)bank.code_length();
        const long double ph = (long double)cp + (long double)n * cps;
        return bank.code_chip(0, (double)(ph - std::floor(ph / L) * L));
    }
};

} // namespace

// I/Q samples through fftwEngine's PFB, as the bladeRF chain runs it: 122.88 MSPS, 64 channels
// of 1.92 MHz, 1920 hops per millisecond. L5 lies 48.55 MHz below the tuned frequency, in the
// upper half of the DFT bins. Both replica generators must give the conjugate of that
// channelizer's output.
BOOST_AUTO_TEST_CASE(iq_time_order_fold_matches_fftwengine_fold) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    BOOST_REQUIRE(sig != nullptr);
    const double fs = 122.88e6, f_off = 1176.45e6 - 1225.0e6;
    gnss::Channelization chz{fs, 64, 4, dsp::Window::Hamming, true, true};
    BOOST_CHECK_EQUAL(chz.fft_len(), 64);
    BOOST_CHECK_CLOSE(chz.chan_freq(39), -48.0e6, 1e-9);
    BOOST_CHECK_CLOSE(chz.chan_freq(31), 59.52e6, 1e-9);
    gnss::ChannelizedReplicaBank bank(*sig, chz, f_off, {7});
    BOOST_CHECK_EQUAL(bank.repl_period_hops(), 1920);

    // The channels of the main lobe, 1176.45 +- 10.23 MHz, lowest first.
    std::vector<int> want;
    for (int k = 33; k <= 44; ++k)
        want.push_back(k);
    BOOST_TEST(bank.covering_bins(0.0, 0.0) == want, boost::test_tools::per_element());

    const long long ws = 122880LL * 100000 + 640; // 100 s in, on a hop boundary
    const int n_hops = 400;
    for (double dop : {0.0, 2000.0, -3500.0}) {
        const SkySample sky{bank, fs, f_off, dop, 4000.25};
        const auto data =
            time_order_pfb<cf>([&](long long n) { return (cf)(sky.chip(n) * sky.carrier(n)); }, ws,
                               64, 4, n_hops, want);
        const auto exact = bank.channels(0, ws, 4000.25, dop, n_hops);
        const auto hop = bank.channels_hoprate(0, ws, 4000.25, dop, n_hops, want);
        for (size_t ci = 0; ci < want.size(); ++ci) {
            BOOST_CHECK_MESSAGE(rel_err(exact[(size_t)want[ci]], data[ci]) < 1e-5,
                                "dop " << dop << " ch " << want[ci] << " channels() err "
                                       << rel_err(exact[(size_t)want[ci]], data[ci]));
            BOOST_CHECK_MESSAGE(rel_err(hop[ci], data[ci]) < 1e-5,
                                "dop " << dop << " ch " << want[ci] << " hop-rate err "
                                       << rel_err(hop[ci], data[ci]));
        }
    }
}

// hoprate_adjoint is the transpose of hoprate_stream: the despread of any data against the
// bank's own replica equals the chip outputs weighted by the code. GW_1202 at the bladeRF
// geometry, where GnssCskDespread uses it on the CSK half.
BOOST_AUTO_TEST_CASE(hoprate_adjoint_is_transpose) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GW_1202");
    BOOST_REQUIRE(sig != nullptr);
    const double fs = 122.88e6, f_off = 1202.025e6 - 1225.0e6;
    gnss::Channelization chz{fs, 64, 4, dsp::Window::Hamming, true, true};
    gnss::ChannelizedReplicaBank bank(*sig, chz, f_off, {1145});
    const std::vector<int> want = {50, 51, 52, 53, 54};
    const long long ws = 122880LL * 100000 + 640;
    const int n_hops = 2000;
    std::vector<std::vector<cf>> data(want.size(), std::vector<cf>(n_hops));
    unsigned seed = 7;
    auto rnd = [&]() {
        seed = seed * 1103515245u + 12345u;
        return (float)((seed >> 8) & 0xffff) / 32768.0f - 1.0f;
    };
    for (auto& row : data)
        for (auto& x : row)
            x = cf(rnd(), rnd());
    std::vector<const cf*> ptr;
    for (auto& row : data)
        ptr.push_back(row.data());
    for (double dop : {0.0, 21000.0, -17500.0}) {
        const double cp = 1234.5;
        const auto f = bank.hoprate_filter(want, dop, -1);
        const auto rep = bank.hoprate_stream(f, 0, ws, cp, dop, n_hops);
        std::complex<double> g_ref(0.0, 0.0);
        for (size_t ci = 0; ci < want.size(); ++ci)
            for (int m = 0; m < n_hops; ++m)
                g_ref +=
                    std::complex<double>(data[ci][m]) * std::conj(std::complex<double>(rep[ci][m]));
        const double cps =
            bank.eff_chip_rate() / fs * (1.0 + bank.code_doppler_sign * dop / sig->carrier_hz);
        const long long chip_lo = (long long)std::floor(cp + (double)(ws - 512) * cps);
        std::vector<std::complex<double>> z((size_t)((n_hops * 64 + 1024) * cps));
        bank.hoprate_adjoint(f, ws, cp, dop, n_hops, ptr, chip_lo, z);
        std::complex<double> g_adj(0.0, 0.0);
        for (size_t i = 0; i < z.size(); ++i)
            g_adj += (double)bank.code_chip(0, (double)(chip_lo + (long long)i)) * z[i];
        BOOST_CHECK_MESSAGE(std::abs(g_adj - g_ref) < 1e-5 * std::abs(g_ref),
                            "dop " << dop << ": adjoint " << g_adj << " vs replica " << g_ref);
    }
}

// Real samples through the same time-order fold, at the airspy geometry: 20 MSPS, 10 channels
// of 1 MHz, 1000 hops per millisecond.
BOOST_AUTO_TEST_CASE(real_time_order_fold_matches_fftwengine_fold) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L1CA");
    BOOST_REQUIRE(sig != nullptr);
    const double fs = 20.0e6, f_off = 4.58e6;
    gnss::Channelization chz{fs, 10, 4, dsp::Window::Hamming, false, true};
    BOOST_CHECK_EQUAL(chz.fft_len(), 20);
    gnss::ChannelizedReplicaBank bank(*sig, chz, f_off, {3});
    BOOST_CHECK_EQUAL(bank.repl_period_hops(), 1000);

    const std::vector<int> want = {3, 4, 5, 6};
    const long long ws = 20000LL * 100000 + 200;
    const int n_hops = 400;
    for (double dop : {0.0, 2000.0, -3500.0}) {
        const SkySample sky{bank, fs, f_off, dop, 300.5};
        const auto data = time_order_pfb<float>(
            [&](long long n) { return (float)(sky.chip(n) * sky.carrier(n).real()); }, ws, 20, 4,
            n_hops, want);
        const auto exact = bank.channels(0, ws, 300.5, dop, n_hops);
        const auto hop = bank.channels_hoprate(0, ws, 300.5, dop, n_hops, want);
        for (size_t ci = 0; ci < want.size(); ++ci) {
            BOOST_CHECK_MESSAGE(rel_err(exact[(size_t)want[ci]], data[ci]) < 1e-5,
                                "dop " << dop << " ch " << want[ci] << " channels() err "
                                       << rel_err(exact[(size_t)want[ci]], data[ci]));
            BOOST_CHECK_MESSAGE(rel_err(hop[ci], data[ci]) < 1e-5,
                                "dop " << dop << " ch " << want[ci] << " hop-rate err "
                                       << rel_err(hop[ci], data[ci]));
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

// GuoWang 1202 (#227): GW_1202 must resolve through signal_by_name, take PRNs above 63 (the
// G2' phase + 1), match itself, decorrelate from another satellite, and leave the frame's data
// millisecond dark (zero chips, as GLO_L2OC_P's), so data chips there cannot bias the despread.
BOOST_AUTO_TEST_CASE(gw1202_self_cross_and_the_dark_data_half) {
    const std::vector<int> prns = {1214, 2028, 2047}; // DIGUI-26, DIGUI-79, the last phase
    auto bank = make_bank("GW_1202", prns);
    const int H = bank.repl_period_hops(); // one 2 ms frame
    auto rA = bank.channels(0, 0, 0.0, 0.0, H);
    auto rB = bank.channels(1, 0, 0.0, 0.0, H);
    auto rC = bank.channels(2, 0, 0.0, 0.0, H);
    BOOST_CHECK_CLOSE(despread_mag(rA, rA), 1.0, 1e-2);
    BOOST_CHECK_LT(despread_mag(rA, rB), 0.3);
    BOOST_CHECK_LT(despread_mag(rB, rC), 0.3);
    // The data half (second millisecond) carries no replica energy, the PFB's few-hop edge
    // at the code/zero boundary aside.
    double e_code = 0.0, e_data = 0.0;
    for (const auto& ch : rA)
        for (int m = 0; m < H; ++m)
            (m < H / 2 ? e_code : e_data) += std::norm(ch[(size_t)m]);
    BOOST_TEST_MESSAGE("GW_1202 replica energy, data half / code half: " << e_data / e_code);
    BOOST_CHECK_LT(e_data / e_code, 0.02);
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

// BeiDou B1I's D1 data is spread by the shared NH20, one chip per 1 ms period. BDS_B1I_NH must
// resolve through signal_by_name and bake that overlay into one 20 ms code: the same replica as
// BDS_B1I with its secondary applied from period 0, and not the overlay-blind one.
BOOST_AUTO_TEST_CASE(bds_b1i_nh_is_the_b1i_code_under_the_nh20) {
    const gnss::SignalDescriptor* nh = gnss::signal_by_name("BDS_B1I_NH");
    BOOST_REQUIRE(nh != nullptr);
    BOOST_CHECK_EQUAL(nh->code_length, 40920L);
    BOOST_CHECK_EQUAL(nh->secondary_length, 0);
    const std::vector<int> prns = {6, 38}; // an IGSO and a BDS-3 MEO (three-tap G2)
    auto bnh = make_bank("BDS_B1I_NH", prns);
    auto b1i = make_bank("BDS_B1I", prns);
    BOOST_REQUIRE_EQUAL(b1i.secondary_length(), 20);
    const int H = b1i.repl_period_hops(); // one 1 ms code period
    BOOST_REQUIRE_EQUAL(bnh.repl_period_hops(), 20 * H);

    auto amp = [](const std::vector<std::vector<cf>>& d, const std::vector<std::vector<cf>>& r) {
        return gnss::channelized_despread(d, r).amplitude;
    };
    for (int p = 0; p < 2; ++p) {
        auto rNH = bnh.channels(p, 0, 0.0, 0.0, 20 * H);
        auto rOv = b1i.channels(p, 0, 0.0, 0.0, 20 * H, 0); // NH20 from period 0
        auto rRaw = b1i.channels(p, 0, 0.0, 0.0, 20 * H);   // overlay off
        BOOST_CHECK_GT(std::real(amp(rNH, rOv)), 0.99);
        // NH20 sums to +4 of 20: the overlay-blind replica keeps ~0.2 of the amplitude.
        BOOST_CHECK_LT(std::abs(amp(rNH, rRaw)), 0.3);
    }
    BOOST_CHECK_LT(
        std::abs(amp(bnh.channels(0, 0, 0.0, 0.0, 20 * H), bnh.channels(1, 0, 0.0, 0.0, 20 * H))),
        0.3);
}

// GPS L1 C/A has no secondary; GPS_L1CA_BIT tiles twenty identical periods into one 20 ms code,
// so its period is the LNAV bit's while every 1 ms of it is the bare C/A code.
BOOST_AUTO_TEST_CASE(gps_l1ca_bit_is_twenty_ca_periods) {
    const gnss::SignalDescriptor* bit = gnss::signal_by_name("GPS_L1CA_BIT");
    BOOST_REQUIRE(bit != nullptr);
    BOOST_CHECK_EQUAL(bit->code_length, 20460L);
    BOOST_CHECK_EQUAL(bit->code_period_s, 20e-3);
    const std::vector<int> prns = {1, 32};
    auto bbit = make_bank("GPS_L1CA_BIT", prns);
    auto bca = make_bank("GPS_L1CA", prns);
    const int H = bca.repl_period_hops(); // one 1 ms code period
    BOOST_REQUIRE_EQUAL(bbit.repl_period_hops(), 20 * H);

    auto amp = [](const std::vector<std::vector<cf>>& d, const std::vector<std::vector<cf>>& r) {
        return gnss::channelized_despread(d, r).amplitude;
    };
    for (int p = 0; p < 2; ++p) {
        for (int period : {0, 9, 19}) {
            const long long ws = (long long)period * H * N * 2; // window start, samples
            const double a =
                std::real(amp(bbit.channels(p, ws, 0.0, 0.0, H), bca.channels(p, ws, 0.0, 0.0, H)));
            BOOST_CHECK_MESSAGE(a > 0.99, "slot " << p << " period " << period << ": " << a);
        }
        BOOST_CHECK_GT(std::real(amp(bbit.channels(p, 0, 0.0, 0.0, 20 * H),
                                     bca.channels(p, 0, 0.0, 0.0, 20 * H))),
                       0.99);
    }
    BOOST_CHECK_LT(
        std::abs(amp(bbit.channels(0, 0, 0.0, 0.0, 20 * H), bbit.channels(1, 0, 0.0, 0.0, 20 * H))),
        0.3);
}

// Galileo E1-C's shared CS25 baked into one 100 ms BOC(1,1) code, as the per-PRN *_CS codes are:
// period k of the baked code is the primary times CS25[k]. code_chip indexes the BOC-expanded
// stream, two half-chips per chip, so a period there is 8184 entries.
BOOST_AUTO_TEST_CASE(e1c_cs_baked_convention) {
    const gnss::SignalDescriptor* cs = gnss::signal_by_name("GAL_E1C_CS");
    BOOST_REQUIRE(cs != nullptr);
    BOOST_CHECK_EQUAL(cs->code_length, 102300L);
    BOOST_CHECK_EQUAL(cs->secondary_length, 0);
    const int L = 2 * 4092; // one primary period, BOC-expanded
    for (int prn : {5, 26}) {
        auto bank_cs = make_bank("GAL_E1C_CS", {prn});
        auto bank_c = make_bank("GAL_E1C", {prn});
        BOOST_CHECK_EQUAL(bank_c.secondary_length(), 25); // the shared slot's CS25
        for (int k = 0; k < 25; ++k)
            for (int i = 0; i < L; i += 247) // 34 probes per primary period
                BOOST_REQUIRE_EQUAL((int)bank_cs.code_chip(0, (double)k * L + i),
                                    (int)bank_c.code_chip(0, (double)i)
                                        * (int)galileo::E1C_CS25[(size_t)k]);
    }
}

// SBAS L1 has no secondary; SBAS_L1_SYM tiles two identical C/A periods into one 2 ms code, one
// 500 sps symbol, so each 1 ms of it is the bare SBAS code.
BOOST_AUTO_TEST_CASE(sbas_l1_sym_is_two_ca_periods) {
    const gnss::SignalDescriptor* sym = gnss::signal_by_name("SBAS_L1_SYM");
    BOOST_REQUIRE(sym != nullptr);
    BOOST_CHECK_EQUAL(sym->code_length, 2046L);
    BOOST_CHECK_EQUAL(sym->code_period_s, 2e-3);
    for (int prn : {131, 135}) {
        auto bank_sym = make_bank("SBAS_L1_SYM", {prn});
        auto bank_l1 = make_bank("SBAS_L1", {prn});
        for (int k = 0; k < 2; ++k)
            for (int i = 0; i < 1023; i += 31)
                BOOST_REQUIRE_EQUAL((int)bank_sym.code_chip(0, (double)k * 1023 + i),
                                    (int)bank_l1.code_chip(0, (double)i));
    }
}

// GPS_L5_I_NH tiles NH10 into the I5 code, one 10 ms CNAV symbol: period k is the I5 primary
// times NH10[k], which the plain GPS_L5_I bank applies as its secondary.
BOOST_AUTO_TEST_CASE(l5i_nh_is_the_i5_code_under_the_nh10) {
    const gnss::SignalDescriptor* nh = gnss::signal_by_name("GPS_L5_I_NH");
    BOOST_REQUIRE(nh != nullptr);
    BOOST_CHECK_EQUAL(nh->code_length, 102300L);
    BOOST_CHECK_EQUAL(nh->secondary_length, 0);
    for (int prn : {1, 30}) {
        auto bank_nh = make_bank("GPS_L5_I_NH", {prn});
        auto bank_i = make_bank("GPS_L5_I", {prn});
        BOOST_CHECK_EQUAL(bank_i.secondary_length(), 10);
        for (int k = 0; k < 10; ++k)
            for (int i = 0; i < 10230; i += 331)
                BOOST_REQUIRE_EQUAL((int)bank_nh.code_chip(0, (double)k * 10230 + i),
                                    (int)bank_i.code_chip(0, (double)i)
                                        * (int)gps::L5_NH10[(size_t)k]);
    }
}

// The *_CS data codes tile their shared secondary as GPS_L5_I_NH tiles NH10: period k is the
// primary times secondary[k], which the plain data bank applies as its secondary.
BOOST_AUTO_TEST_CASE(data_cs_is_the_primary_under_the_secondary) {
    const auto e5a = galileo::e5ai_secondary();
    const auto e5b = galileo::e5bi_secondary();
    const auto b2a = beidou::b2ad_secondary();
    const std::vector<int8_t> cs20(e5a.begin(), e5a.end()), cs4(e5b.begin(), e5b.end()),
        b2ad(b2a.begin(), b2a.end());
    const struct {
        const char* tiled;
        const char* plain;
        const std::vector<int8_t>& sec;
    } cases[] = {{"GAL_E5A_I_CS", "GAL_E5A_I", cs20},
                 {"GAL_E5B_I_CS", "GAL_E5B_I", cs4},
                 {"BDS_B2A_D_CS", "BDS_B2A_D", b2ad}};
    for (const auto& c : cases) {
        const gnss::SignalDescriptor* d = gnss::signal_by_name(c.tiled);
        BOOST_REQUIRE(d != nullptr);
        const int n = (int)c.sec.size();
        BOOST_CHECK_EQUAL(d->code_length, 10230L * n);
        BOOST_CHECK_EQUAL(d->secondary_length, 0);
        for (int prn : {1, 30}) {
            auto bank_cs = make_bank(c.tiled, {prn});
            auto bank_d = make_bank(c.plain, {prn});
            BOOST_CHECK_EQUAL(bank_d.secondary_length(), n);
            for (int k = 0; k < n; ++k)
                for (int i = 0; i < 10230; i += 331)
                    BOOST_REQUIRE_EQUAL((int)bank_cs.code_chip(0, (double)k * 10230 + i),
                                        (int)bank_d.code_chip(0, (double)i) * (int)c.sec[k]);
        }
    }
}

namespace {
// Whether /set_prns takes @p prn for @p signal, in a one-slot map (cudaGnssChordTrack.cpp).
bool set_prns_takes(const char* signal, int prn) {
    const gnss::SignalDescriptor* s = gnss::signal_by_name(signal);
    BOOST_REQUIRE_MESSAGE(s != nullptr, "unknown signal");
    return gnss::prn_map_error(*s, {prn}).empty();
}
} // namespace

// An SBAS chain takes the SBAS PRNs, 120..158, which its code bank builds, and nothing else: the
// L5 and C/A tables also hold the GPS PRNs. The other signals keep 1..63 and GW_1202 1..2047.
BOOST_AUTO_TEST_CASE(set_prns_takes_the_sbas_prns_on_sbas_signals_only) {
    for (const char* name : {"SBAS_L5_I", "SBAS_L5_I_MC", "SBAS_L1", "SBAS_L1_SYM"}) {
        auto bank = make_bank(name, {131});
        for (int prn : {120, 138, 158}) {
            BOOST_CHECK_MESSAGE(set_prns_takes(name, prn), name << " refuses PRN " << prn);
            BOOST_CHECK_MESSAGE(bank.set_prn(0, prn), name << " has no code for PRN " << prn);
        }
        for (int prn : {0, 1, 32, 63, 119, 159, 2047})
            BOOST_CHECK_MESSAGE(!set_prns_takes(name, prn), name << " takes PRN " << prn);
    }
    for (const char* name : {"GPS_L1CA", "GPS_L1CA_BIT", "GPS_L5_I", "GPS_L5_Q_NH"}) {
        auto bank = make_bank(name, {1});
        for (int prn : {1, 32}) {
            BOOST_CHECK_MESSAGE(set_prns_takes(name, prn), name << " refuses PRN " << prn);
            BOOST_CHECK_MESSAGE(bank.set_prn(0, prn), name << " has no code for PRN " << prn);
        }
        for (int prn : {0, 64, 120, 138, 158})
            BOOST_CHECK_MESSAGE(!set_prns_takes(name, prn), name << " takes PRN " << prn);
    }
    for (const gnss::SignalDescriptor* s : gnss::ALL_SIGNALS) {
        const std::string name(s->name);
        if (name.rfind("SBAS_", 0) == 0 || name.rfind("QZSS_", 0) == 0)
            continue;
        BOOST_CHECK_MESSAGE(gnss::min_prn(*s) == 1, name);
        BOOST_CHECK_MESSAGE(gnss::max_prn(*s) == (name == "GW_1202" ? 2047 : 63), name);
    }
    // A map is refused whole: one PRN out of range, or one PRN in two slots.
    const gnss::SignalDescriptor& l5 = *gnss::signal_by_name("SBAS_L5_I_MC");
    BOOST_CHECK(gnss::prn_map_error(l5, {131, 133, 135, 121, 123, 136}).empty());
    BOOST_CHECK_EQUAL(gnss::prn_map_error(l5, {131, 133, 135, 121, 123, 32}),
                      "PRN 32 at slot 5 is outside 120..158");
    BOOST_CHECK_EQUAL(gnss::prn_map_error(l5, {131, 133, 131}), "PRN 131 appears in slots 0 and 2");
}

// The QZSS L5 pilot is GPS L5 Q5 on the QZSS PRNs: QZSS_L5_Q_NH's period k is the Q5 primary of
// the PRN times NH20[k], QZSS_L5_Q applies the NH20 as its secondary, and /set_prns takes
// 193..202 only, so a QZSS chain cannot be handed a GPS or SBAS row of the same table.
BOOST_AUTO_TEST_CASE(qzss_l5q_is_the_q5_code_of_the_qzss_prns) {
    const gnss::SignalDescriptor* nh = gnss::signal_by_name("QZSS_L5_Q_NH");
    const gnss::SignalDescriptor* gps_nh = gnss::signal_by_name("GPS_L5_Q_NH");
    BOOST_REQUIRE(nh != nullptr && gps_nh != nullptr);
    BOOST_CHECK_EQUAL(nh->code_length, gps_nh->code_length);
    BOOST_CHECK_EQUAL(nh->code_period_s, gps_nh->code_period_s);
    BOOST_CHECK_EQUAL(nh->carrier_hz, gps_nh->carrier_hz);
    BOOST_CHECK_EQUAL(nh->secondary_length, 0);
    for (int prn : {194, 201}) {
        const auto q5 = gps::generate_l5q_code(prn);
        auto bank_nh = make_bank("QZSS_L5_Q_NH", {prn});
        auto bank_q = make_bank("QZSS_L5_Q", {prn});
        BOOST_REQUIRE_EQUAL(bank_q.secondary_length(), 20);
        for (int k = 0; k < 20; ++k) {
            BOOST_CHECK_EQUAL(bank_q.overlay_sign(k, 0), (int)gps::L5_NH20[(size_t)k]);
            for (int i = 0; i < 10230; i += 331) {
                BOOST_REQUIRE_EQUAL((int)bank_nh.code_chip(0, (double)k * 10230 + i),
                                    (int)q5[(size_t)i] * (int)gps::L5_NH20[(size_t)k]);
                BOOST_REQUIRE_EQUAL((int)bank_q.code_chip(0, (double)i), (int)q5[(size_t)i]);
            }
        }
    }
    // J09's code is its own: against a GPS PRN and another QZSS PRN it decorrelates.
    auto bank = make_bank("QZSS_L5_Q", {201, 194});
    auto gps_bank = make_bank("GPS_L5_Q", {9});
    const int H = bank.repl_period_hops();
    auto r201 = bank.channels(0, 0, 0.0, 0.0, H);
    BOOST_CHECK_CLOSE(despread_mag(r201, r201), 1.0, 1e-2);
    BOOST_CHECK_LT(despread_mag(r201, bank.channels(1, 0, 0.0, 0.0, H)), 0.3);
    BOOST_CHECK_LT(despread_mag(r201, gps_bank.channels(0, 0, 0.0, 0.0, H)), 0.3);

    for (const char* name : {"QZSS_L5_Q", "QZSS_L5_Q_NH"}) {
        auto b = make_bank(name, {194});
        for (int prn : {193, 196, 201, 202}) {
            BOOST_CHECK_MESSAGE(set_prns_takes(name, prn), name << " refuses PRN " << prn);
            BOOST_CHECK_MESSAGE(b.set_prn(0, prn), name << " has no code for PRN " << prn);
        }
        for (int prn : {0, 1, 9, 32, 63, 131, 192, 203, 210})
            BOOST_CHECK_MESSAGE(!set_prns_takes(name, prn), name << " takes PRN " << prn);
    }
    BOOST_CHECK(gnss::prn_map_error(*nh, {194, 195, 196, 201, 199, 200}).empty());
    BOOST_CHECK_EQUAL(gnss::prn_map_error(*nh, {194, 195, 9}),
                      "PRN 9 at slot 2 is outside 193..202");
}

// QZSS L2C is GPS L2C on the QZSS rows of the L2C tables: the bank interleaves the PRN's CM on
// the even combined chips and its CL on the odd, a QZSS code decorrelates against another QZSS
// PRN and against GPS, and /set_prns takes 193..202 only.
BOOST_AUTO_TEST_CASE(qzss_l2c_is_the_l2c_code_of_the_qzss_prns) {
    const gnss::SignalDescriptor* cm = gnss::signal_by_name("QZSS_L2C_CM");
    const gnss::SignalDescriptor* gps_cm = gnss::signal_by_name("GPS_L2C_CM");
    BOOST_REQUIRE(cm != nullptr && gps_cm != nullptr);
    BOOST_CHECK_EQUAL(cm->carrier_hz, gps_cm->carrier_hz);
    BOOST_CHECK_EQUAL(cm->chip_rate_hz, gps_cm->chip_rate_hz);
    BOOST_CHECK_EQUAL(cm->code_length, gps_cm->code_length);
    BOOST_CHECK_EQUAL(cm->code_period_s, gps_cm->code_period_s);
    BOOST_CHECK(cm->time_multiplexed && cm->tdm_phase == 0);
    for (int prn : {194, 199}) {
        const auto cm_code = gps::generate_l2cm_code(prn);
        const auto cl_code = gps::generate_l2cl_code(prn);
        auto bank_cm = make_bank("QZSS_L2C_CM", {prn});
        auto bank_cl = make_bank("QZSS_L2C_CL", {prn});
        BOOST_REQUIRE_EQUAL(bank_cm.comb_mult(), 2);
        for (int i = 0; i < 10230; i += 331) {
            BOOST_REQUIRE_EQUAL((int)bank_cm.code_chip(0, 2.0 * i), (int)cm_code[(size_t)i]);
            BOOST_REQUIRE_EQUAL((int)bank_cm.code_chip(0, 2.0 * i + 1), 0);
            BOOST_REQUIRE_EQUAL((int)bank_cl.code_chip(0, 2.0 * i), 0);
            BOOST_REQUIRE_EQUAL((int)bank_cl.code_chip(0, 2.0 * i + 1), (int)cl_code[(size_t)i]);
        }
    }
    auto bank = make_bank("QZSS_L2C_CM", {194, 195});
    auto gps_bank = make_bank("GPS_L2C_CM", {1});
    const int H = 400; // as l2c_cm_time_multiplexed
    auto r194 = bank.channels(0, 0, 0.0, 0.0, H);
    BOOST_CHECK_CLOSE(despread_mag(r194, r194), 1.0, 1e-2);
    BOOST_CHECK_LT(despread_mag(r194, bank.channels(1, 0, 0.0, 0.0, H)), 0.3);
    BOOST_CHECK_LT(despread_mag(r194, gps_bank.channels(0, 0, 0.0, 0.0, H)), 0.3);

    for (const char* name : {"QZSS_L2C_CM", "QZSS_L2C_CL"}) {
        auto b = make_bank(name, {194});
        for (int prn : {193, 196, 199, 202}) {
            BOOST_CHECK_MESSAGE(set_prns_takes(name, prn), name << " refuses PRN " << prn);
            BOOST_CHECK_MESSAGE(b.set_prn(0, prn), name << " has no code for PRN " << prn);
        }
        for (int prn : {0, 1, 32, 63, 131, 192, 203})
            BOOST_CHECK_MESSAGE(!set_prns_takes(name, prn), name << " takes PRN " << prn);
    }
    BOOST_CHECK(gnss::prn_map_error(*cm, {194, 195, 196, 199, 200, 193}).empty());
    BOOST_CHECK_EQUAL(gnss::prn_map_error(*cm, {194, 3}), "PRN 3 at slot 1 is outside 193..202");
}
