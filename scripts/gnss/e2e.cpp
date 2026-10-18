// ============================================================================================
// End-to-end seeding harness.
// ============================================================================================
//
// Injects a known code phase into a synthetic CHORD sky (noiseless unless --noise), pushes it
// through the production code of every stage between the antenna and the despread, and prints
// the error in chips at each hand-off. It needs no sky data, runs in seconds and is
// deterministic.
//
//   truth (cp204, dop, at absolute window W0)
//     |
//     |  gnss::ChannelizedReplicaBank::channels_hoprate   <- the generator the replicas use
//     v                                                       (tracker signal, e.g. GPS_L5_Q_NH)
//   synthetic channelized data
//     |
//     |  gnss::channelized_accumulate / channelized_peak  <- GnssChannelizedSearch's acquire
//     |  gnss::refine_peak                                <- ...its refine
//     |  gnss::detection_phase                            <- ...its reporting
//     v
//   detection {cp0, cp_long, cp_at_ref, nh, dop}          == /get_detections
//     |
//     |  (built directly from the detection)
//     v
//   seed {code_phase_at_ref_chips, ref_hop, doppler_hz, code_phase_rate, ...}
//     |
//     |  gnss::propagate_seed                             <- cudaGnssInject
//     v
//   commanded argument cp per record
//     |
//     |  GnssCudaDespread::despread_batch                 <- the GPU kernel
//     v
//   E/P/L power  +  error vs truth, in chips
//
// The conventions between stages (what a number means, what it is reduced modulo, which epoch
// it refers to) are invisible to a test of either stage alone. The harness calls the stage
// arithmetic (gnssSeedTransport.hpp) instead of restating it, so it tests the code that runs.
//
// A `code_phase_chips` argument is referred to absolute sample 0, so one argument describes the
// satellite for all time: synthesizing a record at window W with argument cp204 continues the
// same signal, and a 300 s seed age costs one more 2048-hop synthesis, not 300 s of samples.
//
// Build:  scripts/gnss/site/build_tool.sh e2e
// Run:    scripts/gnss/e2e --help
//
// @author Keith Vanderlinde
// ============================================================================================
#include "GnssCudaDespread.hpp"
#include "dspPfbPrototype.hpp"
#include "gnssChannelizedAcquire.hpp"
#include "gnssChannelizedDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSeedTransport.hpp"
#include "gnssSignal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using cf = std::complex<float>;

// ---------------------------------------------------------------------------------------------
// Options: the geometry numbers the live chain uses, so a run can be pointed at a production
// configuration or at a perturbed one.
// ---------------------------------------------------------------------------------------------
struct Opt {
    int prn = 3;
    double cp204 = 137456.75;        ///< injected truth: generator argument, mod 204600
    double dop = 1893.4;             ///< injected truth Doppler, Hz, constant over the run
    long long hop0 = 114436200145LL; ///< snapshot start hop (~6.8 days of F-engine uptime)
    double age_s = 27.0;             ///< seed age at the first tracker record
    int nrec = 4;                    ///< tracker records to propagate through
    double rec_gap_s = 10.0;         ///< spacing between the records we test

    // Search side: a 27-channel stride-4 comb
    int s_chan0 = 5972, s_stride = 4, s_nchan = 27;
    double dop_half = 1000.0, dop_step = 31.25;
    int acquire_windows = 1, fine_step = 32, threads = 6;
    /// Replica periods of data the refine integrates over. The stage refines on at most the
    /// first window (16 ms at L5). The refine picks between grating lobes of nearly equal
    /// height, so its noise decides, and that falls as the square root of the integration
    /// length. hoprate_stream indexes the overlay by absolute period, so a longer window stays
    /// coherent.
    int refine_windows = 1;
    // refine_span 0 means fft_len, the stage default (one hop, the full lag ambiguity);
    // refine_step 303 is what gen_chord_gnss_config.py emits for this comb, fft_len/n_chan/2.
    // A harness defaulting to anything else would reproduce a misconfiguration and call it the
    // baseline.
    int refine_span = 0, refine_step = 303;
    /// Refine integration hops; 0 = the whole refine window. Production runs 391, two L5 code
    /// periods (gen_chord_gnss_config.py's refine_hops), so pass --refine-hops 391 to match it.
    int refine_hops = 0;

    // Tracker side: one GPU's stride-16 comb of 7 channels
    int t_chan0 = 5972, t_stride = 16, t_nchan = 7;
    int hops_per_record = 2048;
    double dll_spacing = 0.5;
    /// Commanded DLL trim, chips, passed to propagate_seed as cudaGnssInject passes the fleet
    /// controller's value. With an exact seed (--skip-search --seed-dop-rate 0) a nonzero trim
    /// is a known code error of known sign, so this is the sign test for the actuator: the
    /// loop's correction tau = -disc/4 (dll_tau at 0.5-chip spacing) must point back toward
    /// zero. A sign error here makes the loop diverge.
    double trim_chips = 0.0;

    // Shared geometry
    double sample_rate = 3.2e9, f_offset = 1176.45e6;
    int spectrum_length = 8192, num_taps = 4;
    double code_doppler_sign = 1.0;

    /// Tracker signal, overlay tiled into the code. The search signal is this name minus its
    /// _NH/_CS suffix (GPS_L5_Q_NH -> GPS_L5_Q, GAL_E5A_Q_CS -> GAL_E5A_Q). Per-PRN-secondary
    /// signals (the _CS ones) have no blind search, because the bank's single-sequence overlay
    /// slot cannot hold them, so they need --skip-search (docs/CHORD_MULTIBAND.md section 5).
    const char* signal = "GPS_L5_Q_NH";

    // Seed perturbations, applied to the seed only (never the truth), so a sweep can measure
    // the chain's sensitivity to each.
    double seed_dop_err = 0.0; ///< Hz added to the seed's Doppler
    double seed_cp_err = 0.0;  ///< chips added to the seed's code phase. +k*primary_len is a
                               ///< secondary-phase error on a _CS/_NH signal: it shows what a
                               ///< wrong overlay period looks like downstream.
    double seed_cp_rate = 0.0; ///< chips/hop residual handed to the tracker
    /// Hz/s handed to the tracker. Only the seed carries it: the synthetic sky's Doppler stays
    /// at --dop, so a non-zero rate is also a seed error, rate*dt in Doppler and, through
    /// propagate_seed's quadratic term, 0.5*(chip_rate/f_offset)*rate*dt^2 in code phase
    /// (~1.2 chips at L5 for the default at the 27 s age, ~5 chips by record 3). The default
    /// is a real satellite's rate rather than zero because with zero the commanded Doppler is
    /// the same in every record, and the per-record phase step (delta f)*t_abs that a changing
    /// Doppler causes at days of uptime cannot arise. That family of defects is a small number
    /// times a huge absolute argument: hop0 supplies the argument and this the small number.
    /// --seed-dop-rate 0 removes it.
    double seed_dop_rate = -0.36893;
    int quantize = 0;            ///< 1 = 4+4b like the F-engine, 0 = float
    int skip_search = 0;         ///< 1 = seed straight from truth (isolates the tracker leg)
    bool cuda_refine = false;    ///< also run the GPU refine and compare it with the CPU one
    int max_chips = 0;           ///< truncate the search's PFB chip gather (0 = full span)
    bool chips_centered = false; ///< center the truncation window on the prototype peak
    /// Additive complex Gaussian noise, std as a multiple of the synthetic signal's rms, on the
    /// search snapshot and on every tracker record. --quantize alone does not set the SNR
    /// regime: on sky the satellite is far below the noise, and without noise every margin the
    /// refine relies on (the true grating lobe beating its neighbour) holds at infinite SNR.
    double noise = 0.0;
    int trials = 1;         ///< independent noise realizations (>1 prints a distribution)
    unsigned nseed = 12345; ///< RNG seed, so a run is reproducible
};

static void usage() {
    printf(
        "e2e -- inject a known code phase, run the shipped search->seed->track->despread chain\n"
        "       offline, print the error in chips at each hand-off.\n\n"
        "  --prn N            PRN to inject and search for (default 3)\n"
        "  --signal NAME      tracker (baked) signal: GPS_L5_Q_NH (default), GAL_E5A_Q_CS,\n"
        "                     BDS_B2A_P_CS. The _CS signals have NO blind search: --skip-search\n"
        "  --cp204 X          INJECTED truth: generator argument, chips mod the tracker code\n"
        "                     (204600 for L5_Q_NH; 1023000 for the _CS signals)\n"
        "  --dop F            INJECTED truth Doppler, Hz\n"
        "  --hop0 H           snapshot start hop (default ~6.8 days of uptime, as on sky)\n"
        "  --age-s S          seed age at the first tracker record (default 27)\n"
        "  --nrec N           tracker records to test (default 4)\n"
        "  --rec-gap-s S      spacing between tested records (default 10)\n"
        "  --dop-half F       Doppler search half-range about truth (default 1000; 0 = exact)\n"
        "  --dop-step F       Doppler grid step (default 31.25)\n"
        "  --windows N        acquire_windows (default 1 -- see 16-vs-20 in the state doc)\n"
        "  --refine-windows N replica-periods the REFINE integrates (default 1 = 16 ms)\n"
        "  --fine-step N      acquire_fine_step (default 32)\n"
        "  --threads N        acquire threads (default 6)\n"
        "  --refine-span N    refine +-samples (default fft_len)  --refine-step N (default 303)\n"
        "  --refine-hops N    refine integration length, hops (default = the full record)\n"
        "  --seed-dop-err F   Hz of Doppler error given ONLY to the seed (the live residual)\n"
        "  --seed-cp-rate X   code_phase_rate handed to the tracker, chips/hop\n"
        "  --seed-dop-rate F  doppler_rate_hz_s handed to the tracker (default -0.36893, a\n"
        "                     REAL fleet rate; 0 removes the re-pin lever and makes this\n"
        "                     harness blind to the (delta f)*t_abs family -- see #52)\n"
        "  --dll-spacing D    Early/Late tap offset from prompt, COMPONENT chips (default 0.5,\n"
        "                     the node default). With --trim sweeps this measures q on the\n"
        "                     peak and the discriminator slope for a spacing choice.\n"
        "  --s-stride N       search comb stride (default 4 = 2 nodes; 1 = the 8-node agg)\n"
        "  --t-stride N       TRACKER comb stride (default 16 = one GPU's comb; 1 = contiguous)\n"
        "  --t-nchan N        tracker channel count (default 7)\n"
        "  --t-chan0 N        tracker comb first channel (default 5972)\n"
        "  --f-offset HZ      CARRIER, and the band under test. Recentres both combs on it\n"
        "                     unless --t-chan0/--s-chan0 pin them. Without this the harness\n"
        "                     is band-blind: it builds sky AND replica at 1176.45e6, so it is\n"
        "                     self-consistent at any carrier\n"
        "                     (be7d46e8e:docs/gnss_gpu_search.md 11.24).\n"
        "  --s-nchan N        search channel count (default 27; 106 for the 8-node agg)\n"
        "  --s-chan0 N        search comb first channel (default 5972)\n"
        "  --noise R          add complex Gaussian noise, std = R x signal rms (0 = noiseless)\n"
        "  --trials N         repeat the search on N noise realizations, print the spread\n"
        "  --nseed N          RNG seed (default 12345), so a noisy run is reproducible\n"
        "  --quantize         quantize the synthetic sky to 4+4b (default: noiseless float)\n"
        "  --skip-search      seed straight from truth -- isolates the tracker/despread leg\n"
        "  --trim X           commanded DLL trim, chips (task #51). With --skip-search this\n"
        "                     IS a known code error: the sign test for the actuator.\n"
        "  --code-doppler-sign S   (default +1)\n");
}

static bool arg_eq(const char* a, const char* b) {
    return std::strcmp(a, b) == 0;
}

// ---------------------------------------------------------------------------------------------
// Wrap a chip difference into (-half, +half]. Every error in this tool is a phase difference
// modulo a code length.
// ---------------------------------------------------------------------------------------------
static double wrap(double x, double period) {
    double r = std::fmod(x, period);
    if (r > 0.5 * period)
        r -= period;
    if (r <= -0.5 * period)
        r += period;
    return r;
}

// 4+4b round trip, as the F-engine and GnssChordDequantize apply it live, scaled so the peak
// component lands at 6.
static void quantize44(std::vector<std::vector<cf>>& ch) {
    double peak = 0.0;
    for (const auto& c : ch)
        for (const cf& v : c)
            peak = std::max({peak, (double)std::fabs(v.real()), (double)std::fabs(v.imag())});
    const double s = (peak > 0.0) ? 6.0 / peak : 1.0;
    for (auto& c : ch)
        for (cf& v : c) {
            int re = std::clamp((int)std::lround(v.real() * s), -8, 7);
            int im = std::clamp((int)std::lround(v.imag() * s), -8, 7);
            v = cf((float)re, (float)im);
        }
}

int main(int argc, char** argv) {
    Opt o;
    // Whether the caller set these explicitly. --f-offset recentres only combs left at their
    // L5 defaults, so an explicit --t-chan0 or --s-chan0 still allows a deliberate "carrier
    // here, comb there" test.
    int f_offset_set = 0, t_chan0_set = 0, s_chan0_set = 0;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next_d = [&]() { return atof(argv[++i]); };
        auto next_i = [&]() { return atoi(argv[++i]); };
        if (arg_eq(a, "--help") || arg_eq(a, "-h")) {
            usage();
            return 0;
        } else if (arg_eq(a, "--prn"))
            o.prn = next_i();
        else if (arg_eq(a, "--cp204"))
            o.cp204 = next_d();
        else if (arg_eq(a, "--dop"))
            o.dop = next_d();
        else if (arg_eq(a, "--hop0"))
            o.hop0 = atoll(argv[++i]);
        else if (arg_eq(a, "--age-s"))
            o.age_s = next_d();
        else if (arg_eq(a, "--nrec"))
            o.nrec = next_i();
        else if (arg_eq(a, "--rec-gap-s"))
            o.rec_gap_s = next_d();
        else if (arg_eq(a, "--dop-half"))
            o.dop_half = next_d();
        else if (arg_eq(a, "--dop-step"))
            o.dop_step = next_d();
        else if (arg_eq(a, "--windows"))
            o.acquire_windows = next_i();
        else if (arg_eq(a, "--refine-windows"))
            o.refine_windows = next_i();
        else if (arg_eq(a, "--fine-step"))
            o.fine_step = next_i();
        else if (arg_eq(a, "--threads"))
            o.threads = next_i();
        else if (arg_eq(a, "--refine-span"))
            o.refine_span = next_i();
        else if (arg_eq(a, "--refine-hops"))
            o.refine_hops = next_i();
        else if (arg_eq(a, "--refine-step"))
            o.refine_step = next_i();
        else if (arg_eq(a, "--seed-dop-err"))
            o.seed_dop_err = next_d();
        else if (arg_eq(a, "--seed-cp-err"))
            o.seed_cp_err = next_d();
        else if (arg_eq(a, "--seed-cp-rate"))
            o.seed_cp_rate = next_d();
        else if (arg_eq(a, "--seed-dop-rate"))
            o.seed_dop_rate = next_d();
        else if (arg_eq(a, "--trim"))
            o.trim_chips = next_d();
        else if (arg_eq(a, "--dll-spacing"))
            o.dll_spacing = next_d();
        else if (arg_eq(a, "--signal"))
            o.signal = argv[++i];
        else if (arg_eq(a, "--cuda-refine"))
            o.cuda_refine = true;
        else if (arg_eq(a, "--max-chips"))
            o.max_chips = next_i();
        else if (arg_eq(a, "--chips-centered"))
            o.chips_centered = true;
        else if (arg_eq(a, "--t-stride"))
            o.t_stride = next_i();
        else if (arg_eq(a, "--t-nchan"))
            o.t_nchan = next_i();
        else if (arg_eq(a, "--t-chan0")) {
            o.t_chan0 = next_i();
            t_chan0_set = 1;
        } else if (arg_eq(a, "--s-stride"))
            o.s_stride = next_i();
        else if (arg_eq(a, "--s-nchan"))
            o.s_nchan = next_i();
        else if (arg_eq(a, "--s-chan0")) {
            o.s_chan0 = next_i();
            s_chan0_set = 1;
        } else if (arg_eq(a, "--noise"))
            o.noise = next_d();
        else if (arg_eq(a, "--trials"))
            o.trials = next_i();
        else if (arg_eq(a, "--nseed"))
            o.nseed = (unsigned)next_i();
        else if (arg_eq(a, "--quantize"))
            o.quantize = 1;
        else if (arg_eq(a, "--skip-search"))
            o.skip_search = 1;
        else if (arg_eq(a, "--code-doppler-sign"))
            o.code_doppler_sign = next_d();
        else if (arg_eq(a, "--f-offset")) {
            o.f_offset = next_d();
            f_offset_set = 1;
        } else {
            printf("unknown option %s\n", a);
            usage();
            return 2;
        }
    }

    const int FFT = 2 * o.spectrum_length;

    // -- Band: recentre the combs on the carrier ------------------------------------------
    //
    // The synthetic sky and the replica are both built at f_offset, so the harness is
    // self-consistent at any carrier and sees a band-dependent fault only if f_offset is the
    // band under test. The combs have to move with it: L5's default comb (t_chan0 5972) under
    // another band's carrier would not overlap the replica at all, and the zero correlation
    // would say nothing about the signal. So --f-offset recentres both combs unless
    // --t-chan0/--s-chan0 set them.
    //
    // The recentred comb stays on the node grid: its members satisfy freq_id % stride ==
    // (default chan0) % stride, and its centre is the nearest such bin to the carrier. For the
    // tracker comb, L5 1176.45 MHz -> bin 6023.424 -> centre 6020 -> chan0 5972 (the default),
    // and E5b 1207.14 MHz -> bin 6180.557 -> centre 6180 -> chan0 6132.
    //
    // The two carriers sit +0.424 and -0.443 channels from their nearest bin, so a floor() in
    // place of round() in any carrier-to-bin conversion is exact for L5 and a whole channel
    // (195.3 kHz) off for E5b.
    if (f_offset_set) {
        const double bin_w = o.sample_rate / (double)FFT;
        auto recentre = [&](int chan0, int stride, int nchan) {
            const int phase = ((chan0 % stride) + stride) % stride;
            const double carrier_bin = o.f_offset / bin_w;
            // nearest bin congruent to `phase` (mod stride)
            const long k = lround((carrier_bin - (double)phase) / (double)stride);
            const long centre = (long)phase + k * (long)stride;
            return (int)(centre - (long)stride * (long)(nchan / 2));
        };
        if (!t_chan0_set)
            o.t_chan0 = recentre(o.t_chan0, o.t_stride, o.t_nchan);
        if (!s_chan0_set)
            o.s_chan0 = recentre(o.s_chan0, o.s_stride, o.s_nchan);
        printf("band: f_offset %.6f MHz -> carrier bin %.4f (%+.4f ch off centre); "
               "tracker comb %d..%d step %d, search comb %d..%d step %d\n",
               o.f_offset / 1e6, o.f_offset / bin_w,
               o.f_offset / bin_w - (double)llround(o.f_offset / bin_w), o.t_chan0,
               o.t_chan0 + o.t_stride * (o.t_nchan - 1), o.t_stride, o.s_chan0,
               o.s_chan0 + o.s_stride * (o.s_nchan - 1), o.s_stride);
    }
    if (o.refine_span <= 0)
        o.refine_span = FFT; // the stage's own default
    const long long W0 = o.hop0 * (long long)FFT;

    std::vector<int> s_chans, t_chans, s_cov, t_cov;
    for (int c = 0; c < o.s_nchan; ++c) {
        s_chans.push_back(o.s_chan0 + o.s_stride * c);
        s_cov.push_back(c);
    }
    for (int c = 0; c < o.t_nchan; ++c) {
        t_chans.push_back(o.t_chan0 + o.t_stride * c);
        t_cov.push_back(c);
    }

    // Two banks, as production runs them. The search works in the primary code (e.g.
    // GPS_L5_Q, L = 10230) with the overlay carried separately as an alignment index; the
    // tracker works in the overlaid code (e.g. GPS_L5_Q_NH, L = 204600) with the overlay tiled
    // into the table. The translation between the two is what this harness checks. The search
    // signal is the tracker name minus its _NH/_CS suffix.
    const std::string tname(o.signal);
    std::string sname = tname;
    for (const char* suf : {"_NH", "_CS"}) {
        const size_t sl = std::strlen(suf);
        if (sname.size() > sl && sname.compare(sname.size() - sl, sl, suf) == 0) {
            sname.resize(sname.size() - sl);
            break;
        }
    }
    const gnss::SignalDescriptor* sig_s = gnss::signal_by_name(sname);
    const gnss::SignalDescriptor* sig_t = gnss::signal_by_name(tname);
    if (!sig_s || !sig_t) {
        printf("unknown --signal %s (derived search signal %s)\n", tname.c_str(), sname.c_str());
        return 2;
    }
    gnss::ChannelizedReplicaBank sbank(*sig_s, o.sample_rate, o.f_offset, o.spectrum_length,
                                       o.num_taps, dsp::Window::Hamming, {o.prn});
    // Two slots holding the same PRN: slot 0 takes the tracker's commanded phase, slot 1 the
    // truth, so both ride one kernel launch against one uploaded window (despread_batch holds
    // at most n_prn jobs; see max_batch_specs).
    gnss::ChannelizedReplicaBank tbank(*sig_t, o.sample_rate, o.f_offset, o.spectrum_length,
                                       o.num_taps, dsp::Window::Hamming, {o.prn, o.prn});
    sbank.code_doppler_sign = o.code_doppler_sign;
    // Truncate the search bank only. sbank builds the search's replica; the synthetic sky comes
    // from tbank (channels_hoprate in [1]). Truncating both would make the sky and the replica
    // agree by construction and the error ~0. The sky keeps the full ~210-chip PFB span and the
    // search analyses with a shorter one, as on sky.
    if (o.max_chips > 0) {
        sbank.set_max_chips(o.max_chips);
        if (o.chips_centered) {
            sbank.set_chips_centered(true);
            printf("    [--chips-centered] the %d-chip window is CENTERED on the prototype peak\n",
                   o.max_chips);
        }
        printf("    [--max-chips %d] SEARCH replica gather truncated; the injected sky keeps the "
               "full span\n",
               o.max_chips);
    }
    tbank.code_doppler_sign = o.code_doppler_sign;

    const int Mp = sbank.repl_period_hops();
    const long long anchor = (long long)Mp * FFT; // what the search generates repl0 at
    const double L = (double)sbank.code_length();
    // Overlay period count from the code-length ratio, not the search bank's overlay slot:
    // the two agree for shared secondaries (L5: 204600/10230 = 20 = NH20), but per-PRN
    // secondaries (E5a/B2a CS100) are absent from the bank's single-sequence slot, while the
    // ratio still reads 100.
    const int n_nh = (int)(sig_t->code_length / sig_s->code_length);
    if ((long)n_nh * sig_s->code_length != sig_t->code_length) {
        printf("--signal %s: tracker code %ld is not an integer tiling of search code %ld\n",
               tname.c_str(), sig_t->code_length, sig_s->code_length);
        return 2;
    }
    if (sbank.secondary_length() > 0 && sbank.secondary_length() != n_nh) {
        printf("--signal %s: search overlay %d != code-length ratio %d -- descriptor bug\n",
               tname.c_str(), sbank.secondary_length(), n_nh);
        return 2;
    }
    if (!o.skip_search && n_nh > 1 && sbank.secondary_length() == 0) {
        printf("--signal %s: %s has a PER-PRN secondary, which the search bank's "
               "single-sequence overlay slot cannot carry -- blind acquisition does not exist "
               "for this signal. Run with --skip-search (the dead-reckon leg; "
               "docs/CHORD_MULTIBAND.md section 5).\n",
               tname.c_str(), sname.c_str());
        return 2;
    }
    const double LL = L * (double)n_nh; // == tbank.code_length()
    const double hop_s = (double)FFT / o.sample_rate;

    printf("================================================================================\n");
    printf("INJECTED TRUTH   %s (search %s)   PRN %d\n", tname.c_str(), sname.c_str(), o.prn);
    printf("  cp %.4f mod %.0f  (primary %.4f, overlay period %d)   dop %+.4f Hz\n", o.cp204, LL,
           std::fmod(o.cp204, L), (int)(o.cp204 / L), o.dop);
    printf("  snapshot hop %lld (sample %lld, %.2f days of uptime)\n", o.hop0, W0,
           (double)W0 / o.sample_rate / 86400.0);
    printf("  geometry: record %d hops = %.4f code periods | replica period %d hops = %.0f "
           "periods | overlay %d\n",
           o.hops_per_record, o.hops_per_record * FFT * sbank.chip_rate_hz() / o.sample_rate / L,
           Mp, Mp * FFT * sbank.chip_rate_hz() / o.sample_rate / L, n_nh);
    printf("  argument lever d(arg)/d(dop) here = %.1f chips/Hz\n",
           tbank.window_advance_chips(W0, 1.0) - tbank.window_advance_chips(W0, 0.0));
    printf("  search: %d chan stride %d | tracker: %d chan stride %d | data %s\n", o.s_nchan,
           o.s_stride, o.t_nchan, o.t_stride, o.quantize ? "4+4b QUANTIZED" : "float (noiseless)");
    printf("================================================================================\n\n");

    // The true physical phase at any window, from the injected argument (constant for all
    // time), in wire units: component chips, as cp_at_ref reports and seeds carry in
    // code_phase_at_ref_chips. phase_from_arg returns combined chips (comb_mult x component),
    // hence the division; without it the harness agrees with propagate_seed on a TDM signal
    // (L2C, comb_mult 2) while the sky does not.
    auto truth_phase_at = [&](long long wstart) {
        return tbank.phase_from_arg(o.cp204, wstart, o.dop) / (double)tbank.comb_mult();
    };

    // -----------------------------------------------------------------------------------------
    // Leg 1: the search. Synthesize the snapshot, run the shipped acquire + refine, and report
    // through the shipped gnss::detection_phase().
    // -----------------------------------------------------------------------------------------
    gnss::DetectionPhase dp;
    double det_dop = o.dop;
    int best_nh = -1;
    double snr = 0.0;

    if (!o.skip_search) {
        const int nwin = std::max(std::max(1, o.acquire_windows), std::max(1, o.refine_windows));
        printf("[1] SEARCH -- synthesizing %d x %d hops (%.1f ms) on %d channels...\n", nwin, Mp,
               1e3 * nwin * Mp * hop_s, o.s_nchan);
        const auto clean =
            tbank.channels_hoprate(0, W0, o.cp204, o.dop, nwin * Mp, s_chans, {}, -1);
        double sig2 = 0.0;
        size_t nsamp = 0;
        for (const auto& ch : clean)
            for (const cf& v : ch) {
                sig2 += std::norm(v);
                ++nsamp;
            }
        const double sig_rms = std::sqrt(sig2 / (double)std::max<size_t>(nsamp, 1));
        if (o.noise > 0.0)
            printf("    noise: std = %.3f x signal rms (%.4g), %d trial(s), seed %u\n", o.noise,
                   o.noise * sig_rms, o.trials, o.nseed);
        std::mt19937 rng(o.nseed);
        std::normal_distribution<double> gauss(0.0, o.noise * sig_rms / std::sqrt(2.0));
        std::vector<double> errs;
        std::vector<double> snrs;
        auto data = clean;

        // The grid sits on integer multiples of dop_step, as in GnssChannelizedSearch, not on
        // the hint. A grid anchored to the hint moves with the satellite, so every trial would
        // sample the same sub-bin position and a Doppler sweep could not see interpolation
        // error.
        std::vector<double> grid;
        if (o.dop_half <= 0.0)
            grid.push_back(-o.dop); // r2c fold: the grid runs in the flipped convention
        else {
            const double lo = std::ceil((-o.dop - o.dop_half) / o.dop_step) * o.dop_step;
            for (double f = lo; f <= -o.dop + o.dop_half + 1e-9; f += o.dop_step)
                grid.push_back(f);
        }

        // Per-stage timers.
        double T_acq = 0.0, T_ref = 0.0;
        const auto tnow = [] {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        };
        gnss::AcquireWorkspace ws;
        gnss::AcquisitionSurface adims{}; // hoisted: the summary below needs s_stored
        for (int trial = 0; trial < std::max(1, o.trials); ++trial) {
            // Fresh realization each trial; the clean signal is never overwritten.
            if (o.noise > 0.0) {
                for (size_t c = 0; c < clean.size(); ++c)
                    for (size_t m = 0; m < clean[c].size(); ++m)
                        data[c][m] = clean[c][m] + cf((float)gauss(rng), (float)gauss(rng));
            }
            if (o.quantize)
                quantize44(data);
            gnss::AcquisitionResult a{};
            const double t_acq0 = tnow();
            best_nh = -1;
            std::vector<double> surf;
            std::vector<std::vector<cf>> w((size_t)o.s_nchan, std::vector<cf>((size_t)Mp));
            for (int nh = 0; nh < n_nh; ++nh) {
                // repl0: code 0, Doppler 0, overlay alignment nh -- what the stage precomputes.
                auto repl0 = sbank.channels_hoprate(0, anchor, 0.0, 0.0, Mp, s_chans, {}, nh);
                gnss::AcquisitionSurface dims{};
                surf.assign(surf.size(), 0.0);
                // The acquire accumulates acquire_windows, not nwin, which is only how much data
                // was synthesized (the larger of acquire_windows and refine_windows), so
                // --refine-windows does not also lengthen the acquire.
                for (int wi = 0; wi < std::max(1, o.acquire_windows); ++wi) {
                    for (int c = 0; c < o.s_nchan; ++c)
                        for (int m = 0; m < Mp; ++m)
                            w[(size_t)c][(size_t)m] = data[(size_t)c][(size_t)(wi * Mp + m)];
                    dims = gnss::channelized_accumulate(w, repl0, s_cov, grid, o.sample_rate,
                                                        o.s_nchan, surf, ws, s_chans, FFT,
                                                        o.threads, o.fine_step);
                }
                auto ai = gnss::channelized_peak(surf, dims, grid, o.sample_rate,
                                                 sbank.chip_rate_hz(), sbank.code_length());
                if (best_nh < 0 || ai.snr > a.snr) {
                    a = ai;
                    best_nh = nh;
                    adims = dims;
                }
            }
            T_acq += tnow() - t_acq0;
            snr = a.snr;
            det_dop = -a.doppler_hz; // r2c fold conjugates the channel frequency axis

            // ---- the stage's refine (refine_peak: de-alias, then localize) ----
            const int rhops = Mp * std::max(1, o.refine_windows);
            std::vector<std::vector<cf>> d((size_t)o.s_nchan, std::vector<cf>((size_t)rhops));
            for (int c = 0; c < o.s_nchan; ++c)
                for (int m = 0; m < rhops; ++m)
                    d[(size_t)c][(size_t)m] = data[(size_t)c][(size_t)m];
            const double t_ref0 = tnow();
            const double best_cp = gnss::refine_peak(
                sbank, 0, d, s_chans, a.code_phase_chips, best_nh, det_dop, anchor,
                (o.refine_hops > 0 && o.refine_hops < rhops) ? o.refine_hops : rhops, o.sample_rate,
                o.refine_span, o.refine_step, o.threads);

            T_ref += tnow() - t_ref0;
            printf("    [stages] acquire %.2fs | refine %.2fs\n", T_acq, T_ref);
#ifdef GNSS_CUDA
            // The GPU refine (refine_peak_cuda) runs the same scan as refine_peak on the same
            // data, so the two should agree to float precision, except that it takes no nh_phase:
            // a search bank with a per-period overlay (GPS_L5_Q) is scanned overlay-blind there,
            // so a MISMATCH on such a bank is not by itself a GPU fault. Both run and the delta
            // is printed.
            if (o.cuda_refine) {
                const int rh = (o.refine_hops > 0 && o.refine_hops < rhops) ? o.refine_hops : rhops;
                const size_t nc = d.size();
                // refine_peak takes [chan][hop]; the GPU driver takes the stage's native
                // [hop][chan] interleave. Transpose here so the comparison is like for like.
                std::vector<std::complex<float>> win((size_t)rh * nc);
                for (size_t c = 0; c < nc; ++c)
                    for (int m = 0; m < rh; ++m)
                        win[(size_t)m * nc + c] = d[c][(size_t)m];
                std::vector<int> loc(nc);
                for (size_t i = 0; i < nc; ++i)
                    loc[i] = (int)i;
                // One engine per <=64-channel group: chan_mask is a uint64_t. The default comb
                // is one group, but this is the grouping code a wider comb needs.
                std::vector<std::unique_ptr<GnssCudaDespread>> engines;
                std::vector<gnss::CudaRefineGroup> rgroups;
                const size_t GMAX = 64;
                for (size_t i0 = 0; i0 < nc; i0 += GMAX) {
                    const size_t ng = std::min(GMAX, nc - i0);
                    std::vector<int> gg(s_chans.begin() + i0, s_chans.begin() + i0 + ng);
                    engines.emplace_back(new GnssCudaDespread(sbank, 1, gg, rh, o.sample_rate));
                    gnss::CudaRefineGroup grp;
                    grp.gpu = engines.back().get();
                    grp.local.resize(ng);
                    for (size_t i = 0; i < ng; ++i)
                        grp.local[i] = (int)(i0 + i);
                    grp.n_hops = rh;
                    rgroups.push_back(std::move(grp));
                }
                const double t_g0 = tnow();
                const double gcp = gnss::refine_peak_cuda(
                    rgroups, sbank, 0, win.data(), (int)nc, anchor, a.code_phase_chips, det_dop,
                    o.sample_rate, o.refine_span, o.refine_step);
                const double t_g = tnow() - t_g0;
                printf("    CUDA refine: cp %.6f  CPU %.6f  delta %+.3e chips  |  %.3f s vs %.3f s "
                       "= %.1fx  %s\n",
                       gcp, best_cp, gcp - best_cp, t_g, T_ref, T_ref / t_g,
                       std::fabs(gcp - best_cp) < 1e-6 ? "OK" : "MISMATCH");
            }
#endif
            // ---- the stage's reporting arithmetic ----
            // The coarse lag spans Mp hops (3125 = 16 code periods at L5), so it carries a
            // whole-period count that detection_phase folds back in.
            dp = gnss::detection_phase(sbank, best_cp, best_nh, n_nh, det_dop, o.hop0, anchor,
                                       o.sample_rate, a.peak_tau_samples);

            const double ph_true = truth_phase_at(W0);
            const double e_ref = wrap(dp.cp_at_ref - ph_true, LL);
            // The broker's clock solve, joint feed and search monitor re-advance cp0 (the
            // sample-0 argument) to the snapshot start at the code-Doppler rate (deadreckon
            // _det_offset); phase_from_arg is that arithmetic. Printed beside e_ref because
            // the two reported phases disagreed on live detections by a 16 ms Doppler lever
            // (10-18) and the harness had only ever been read at --dop 0, where a lever is
            // invisible.
            const double ph_cp0 = sbank.phase_from_arg(dp.cp0, W0, det_dop);
            const double e_cp0 = wrap(ph_cp0 - std::fmod(ph_true, (double)sbank.code_length()),
                                      (double)sbank.code_length());
            printf("    cp0 re-advanced to the snapshot start: %.3f  (truth mod L %.3f)  "
                   ">>> CP0 LEG ERROR: %+.3f chips\n",
                   ph_cp0, std::fmod(ph_true, (double)sbank.code_length()), e_cp0);

            // Period diagnostics. The coarse lag spans a full replica period (16 primary periods
            // at L5), but `code_phase_chips` is that lag mod one period. The overlay is 20
            // periods and 16 is not 0 mod 20, so the whole-period count matters; detection_phase
            // recovers it from peak_tau_samples. Printed against the truth so the relation can be
            // read off directly.
            const double tau_chips =
                (double)a.peak_tau_samples * sbank.chip_rate_hz() / o.sample_rate;
            const int tau_per = ((int)std::floor(tau_chips / L)) % n_nh;
            const int true_per = (int)std::floor(ph_true / L);
            const int got_per = (int)std::floor(dp.cp_at_ref / L);
            printf("    tau %ld samp = %.2f chips = %d periods + %.2f | nh %d | period true %d "
                   "got %d (err %+d)\n",
                   a.peak_tau_samples, tau_chips, tau_per, std::fmod(tau_chips, L), best_nh,
                   true_per, got_per, ((got_per - true_per + 30) % n_nh) - 10);
            printf("    snr %.1f   dop %+.4f (err %+.4f Hz)   nh %d   coarse %.2f refine %+.2f\n",
                   snr, det_dop, det_dop - o.dop, best_nh, a.code_phase_chips,
                   best_cp - a.code_phase_chips);
            printf("    reported cp0 %.3f | cp_long %.3f | ph@ref %.3f   (truth ph@ref %.3f)\n",
                   dp.cp0, dp.cp_long, dp.cp_at_ref, ph_true);
            printf(
                "    >>> SEARCH LEG ERROR: %+.3f chips   (= %+.3f periods %+.3f within-period)\n",
                e_ref, std::round(e_ref / L), wrap(e_ref, L));
            errs.push_back(e_ref);
            snrs.push_back(snr);
        } // trial
        if (o.trials > 1) {
            // A "wrong lobe" is a within-period error beyond half a grating spacing: the comb's
            // fine-lag response repeats every s_stored samples, so past half of that the refine
            // chose a neighbour.
            const double lobe = (double)((adims.s_stored > 0) ? adims.s_stored : FFT)
                                * sbank.chip_rate_hz() / o.sample_rate;
            int wrong = 0, badper = 0;
            double s1 = 0.0, s2 = 0.0, worst = 0.0;
            for (double e : errs) {
                const double w = wrap(e, L);
                if (std::fabs(e) > L / 2.0)
                    ++badper;
                if (std::fabs(w) > lobe / 2.0)
                    ++wrong;
                s1 += w;
                s2 += w * w;
                worst = std::max(worst, std::fabs(w));
            }
            const double n = (double)errs.size();
            double msnr = 0.0;
            for (double v : snrs)
                msnr += v;
            printf("\n    ===== %d trials at noise %.3f =====\n", (int)n, o.noise);
            printf("    acquire snr   mean %.1f\n", msnr / n);
            printf("    within-period error: mean %+.3f  rms %.3f  worst %.3f chips\n", s1 / n,
                   std::sqrt(s2 / n), worst);
            printf("    grating lobe spacing %.2f chips -> WRONG LOBE in %d/%d (%.0f%%)\n", lobe,
                   wrong, (int)n, 100.0 * wrong / n);
            printf("    wrong PERIOD in %d/%d\n", badper, (int)n);
        }
        printf("\n");
    } else {
        dp.cp_at_ref = truth_phase_at(W0);
        dp.cp_long = -1.0;
        printf("[1] SEARCH -- SKIPPED (--skip-search): seeding straight from truth, "
               "ph@ref %.3f\n\n",
               dp.cp_at_ref);
    }

    // -----------------------------------------------------------------------------------------
    // Leg 2: the seed, built directly from the detection (what the broker does when it has
    // nothing to add).
    // -----------------------------------------------------------------------------------------
    gnss::SeedState sd;
    sd.ref_hop = o.hop0;
    sd.phase_ref_chips = dp.cp_at_ref + o.seed_cp_err;
    sd.doppler_hz = det_dop + o.seed_dop_err;
    sd.cp_rate = o.seed_cp_rate;
    sd.dop_rate = o.seed_dop_rate;
    printf("[2] SEED (direct from the detection)\n");
    printf("    ref_hop %lld  ph@ref %.3f  dop %+.4f  cp_rate %.6e  dop_rate %+.4f\n\n", sd.ref_hop,
           sd.phase_ref_chips, sd.doppler_hz, sd.cp_rate, sd.dop_rate);

    // -----------------------------------------------------------------------------------------
    // Leg 3: the tracker. Propagate through the shipped gnss::propagate_seed(), synthesize the
    // record at its true window, and despread on the GPU with the shipped kernel.
    // -----------------------------------------------------------------------------------------
    printf("[3] TRACK + DESPREAD  (seed age %.1f s at record 0, records %.1f s apart)\n", o.age_s,
           o.rec_gap_s);
    GnssCudaDespread ds(tbank, 2, t_chans, o.hops_per_record, o.sample_rate);
    printf("    rec   age_s        cp cmd    phase err     per   |   P/P_true    q=2P/(E+L)      "
           "disc\n");

    const long long age_hops = (long long)std::llround(o.age_s / hop_s);
    const long long gap_hops = (long long)std::llround(o.rec_gap_s / hop_s);
    double worst = 0.0;
    std::vector<std::complex<double>> prompts; // per-record P, for the deep fold below
    // Independent of the search realization, so a tracker sweep is reproducible on its own.
    std::mt19937 trk_rng(o.nseed + 1000u);
    for (int r = 0; r < o.nrec; ++r) {
        const long long h = o.hop0 + age_hops + (long long)r * gap_hops;
        const long long W = h * (long long)FFT;

        // ---- the tracker's propagation (propagate_seed) ----
        // trim defaults to 0, the open loop. --trim commands the actuator, so the sign of the
        // response can be measured.
        const gnss::SeedPropagation pr =
            gnss::propagate_seed(tbank, sd, h, o.sample_rate, o.f_offset, o.trim_chips);

        // The error is a difference of physical phases at this window, never of arguments: two
        // arguments derived at Dopplers 1 Hz apart differ by ~5095 chips at 6.8 days of uptime
        // and describe the same signal. pr.phase_now is in combined chips; the truth is in wire
        // (component) units.
        const double err = wrap(pr.phase_now / (double)tbank.comb_mult() - truth_phase_at(W), LL);

        auto rec = tbank.channels_hoprate(0, W, o.cp204, o.dop, o.hops_per_record, t_chans, {}, -1);
        // --noise applies to every tracker record too, from its own RNG stream, so the tracker
        // and deep-fold legs see the same SNR regime as the search.
        if (o.noise > 0.0) {
            double s2 = 0.0;
            size_t ns = 0;
            for (const auto& ch : rec)
                for (const cf& v : ch) {
                    s2 += std::norm(v);
                    ++ns;
                }
            const double rms = std::sqrt(s2 / (double)std::max<size_t>(ns, 1));
            std::normal_distribution<double> g(0.0, o.noise * rms / std::sqrt(2.0));
            for (auto& ch : rec)
                for (cf& v : ch)
                    v += cf((float)g(trk_rng), (float)g(trk_rng));
        }
        if (o.quantize)
            quantize44(rec);
        std::vector<cf> win((size_t)o.hops_per_record * (size_t)o.t_nchan);
        for (int m = 0; m < o.hops_per_record; ++m)
            for (int c = 0; c < o.t_nchan; ++c)
                win[(size_t)m * (size_t)o.t_nchan + (size_t)c] = rec[(size_t)c][(size_t)m];
        ds.upload_window(win.data(), W);

        // Job 0 = the tracker's commanded phase; job 1 = perfect knowledge, the normalizer.
        const std::vector<GnssCudaDespread::Spec> jobs{
            {0, pr.cp, o.dll_spacing, pr.doppler_hz, t_cov},
            {1, o.cp204, o.dll_spacing, o.dop, t_cov}};
        const auto res = ds.despread_batch(jobs);
        const double P = std::norm(res[0][1].correlation);
        const double E = std::norm(res[0][0].correlation);
        const double La = std::norm(res[0][2].correlation);
        const double Pt = std::norm(res[1][1].correlation);
        const double disc = (E + La) > 0 ? (E - La) / (E + La) : 0.0;
        printf("    %3d %7.1f  %13.3f  %+12.3f  %+5.0f   |   %8.4f    %8.3f  %+8.4f\n", r,
               (double)(h - o.hop0) * hop_s, pr.cp, err, std::round(err / L), Pt > 0 ? P / Pt : 0.0,
               (E + La) > 0 ? 2.0 * P / (E + La) : 0.0, disc);
        worst = std::max(worst, std::fabs(err));
        prompts.push_back(res[0][1].correlation);
    }

    // ---- the deep fold, with the combiner's own coherent_sum ----
    // coherent_sum takes its noise from the records' component orthogonal to the sum, so a
    // per-record phase error sigma_phi caps its snr near sqrt(N)/sigma_phi whatever the signal
    // strength. With no noise and an exact seed (--skip-search --seed-dop-rate 0), any sigma_phi
    // left is in the arithmetic, not on the sky.
    if (prompts.size() >= 4) {
        const int N = (int)prompts.size();
        // Derotate first, as the combiner's rate search does: a fold of raw prompts measures the
        // phase ramp from a Doppler error, not the residual about it.
        std::vector<std::complex<double>> der = prompts;
        { // best-fit linear phase over the record index, removed as a frequency
            std::vector<double> p0(prompts.size());
            double acc = 0.0, pv = 0.0;
            for (size_t i = 0; i < prompts.size(); ++i) {
                const double q = std::arg(prompts[i]);
                if (i) {
                    double d = q - pv;
                    while (d > M_PI)
                        d -= 2 * M_PI;
                    while (d < -M_PI)
                        d += 2 * M_PI;
                    acc += d;
                }
                pv = q;
                p0[i] = acc;
            }
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (int i = 0; i < N; ++i) {
                sx += i;
                sy += p0[i];
                sxx += (double)i * i;
                sxy += (double)i * p0[i];
            }
            const double dn = N * sxx - sx * sx, sl = dn != 0.0 ? (N * sxy - sx * sy) / dn : 0.0;
            for (int i = 0; i < N; ++i)
                der[i] = prompts[i] * std::polar(1.0, -sl * i);
        }
        const gnss::OverlayWipeResult cs = gnss::coherent_sum(der);
        // Residual of the unwrapped per-record phase about a best-fit linear ramp: whatever is
        // left is not a ramp.
        std::vector<double> ph(prompts.size());
        double pu = 0.0, prev = 0.0;
        for (size_t i = 0; i < prompts.size(); ++i) { // unwrapped phase
            const double a = std::arg(prompts[i]);
            if (i) {
                double d = a - prev;
                while (d > M_PI)
                    d -= 2 * M_PI;
                while (d < -M_PI)
                    d += 2 * M_PI;
                pu += d;
            }
            prev = a;
            ph[i] = pu;
        }
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < N; ++i) {
            sx += i;
            sy += ph[i];
            sxx += (double)i * i;
            sxy += (double)i * ph[i];
        }
        const double den = N * sxx - sx * sx;
        const double b = den != 0.0 ? (N * sxy - sx * sy) / den : 0.0, a0 = (sy - b * sx) / N;
        double s2 = 0.0;
        for (int i = 0; i < N; ++i) {
            const double e = ph[i] - (a0 + b * i);
            s2 += e * e;
        }
        const double sig_res = std::sqrt(s2 / N);
        printf("\n[4] DEEP FOLD over %d records (the combiner's own coherent_sum)\n", N);
        printf("    coherent snr %.2f   amplitude %.6g\n", cs.snr, cs.amplitude);
        printf("    implied sigma_phi = sqrt(N)/snr = %.4f rad\n",
               cs.snr > 0 ? std::sqrt((double)N) / cs.snr : 0.0);
        // The unwrap assumes the phase step between records is under pi. With a real Doppler
        // rate at days of uptime it often is not; the unwrap then aliases and sig_res saturates
        // near the uniform-phase value whatever the arithmetic does, so a large median step is
        // reported as unresolved instead of as a number.
        double med_step;
        {
            std::vector<double> st;
            for (int i = 1; i < N; ++i) {
                double d = std::arg(prompts[i]) - std::arg(prompts[i - 1]);
                while (d > M_PI)
                    d -= 2 * M_PI;
                while (d < -M_PI)
                    d += 2 * M_PI;
                st.push_back(std::fabs(d));
            }
            std::sort(st.begin(), st.end());
            med_step = st.empty() ? 0.0 : st[st.size() / 2];
        }
        if (med_step > 1.5)
            printf("    per-record phase residual: UNRESOLVED -- median |dphi| = %.2f rad, the "
                   "unwrap has aliased\n      (records are %.4g s apart; use --rec-gap-s "
                   "%.6f, the live record cadence, to measure this)\n",
                   med_step, o.rec_gap_s, 2048.0 / 195312.5);
        else
            printf("    per-record phase residual about a linear ramp: %.4f rad rms\n", sig_res);
        printf("    LIVE reads sigma_phi ~0.745 rad, flat across a 9x amp_snr range.\n");
        printf(
            "    %s\n",
            med_step > 1.5 ? ">>> cannot tell: the per-record step aliases the unwrap (see above)"
            : sig_res > 0.2
                ? ">>> the floor REPRODUCES with no noise -- it is in our arithmetic OR in the\n"
                  "        seed model extrapolated over this record spacing. Re-run with\n"
                  "        --rec-gap-s 0.0104858 (the live cadence, which is what the deep fold\n"
                  "        actually integrates) to separate the two: arithmetic survives that,\n"
                  "        a multi-second extrapolation residual does not."
                : ">>> noiseless chain is phase-clean; the live floor comes from something this "
                  "harness does not model (overlay/nav wipe, straddle, multi-channel, or the sky)");

        // ---- [4a] the per-record phase step series ----
        // A smooth model mismatch (a quadratic phase) gives nearly constant steps, so a small
        // step sd. A per-record reference error gives random steps: an independent offset per
        // record differences to a lag-1 autocorrelation of -0.5, a random walk to ~0. Unlike the
        // residual about a ramp, this separates a quadratic from noise, so it is the statistic
        // to compare with the sky.
        auto step_stats = [&](const std::vector<std::complex<double>>& ps, const char* tag) {
            std::vector<double> stp;
            for (size_t i = 1; i < ps.size(); ++i) {
                double d = std::arg(ps[i] * std::conj(ps[i - 1])) / (2.0 * M_PI);
                stp.push_back(d);
            }
            if (stp.size() < 4)
                return;
            double m = 0.0;
            for (double v : stp)
                m += v;
            m /= (double)stp.size();
            double v2 = 0.0, c1 = 0.0;
            int big = 0;
            for (size_t i = 0; i < stp.size(); ++i) {
                v2 += (stp[i] - m) * (stp[i] - m);
                if (i + 1 < stp.size())
                    c1 += (stp[i] - m) * (stp[i + 1] - m);
                if (std::fabs(stp[i] - m) > 0.2)
                    ++big;
            }
            const double sd = std::sqrt(v2 / (double)stp.size());
            printf("    [4a] %-7s per-record step: mean %+9.5f cyc  sd %.5f cyc  lag-1 %+.3f"
                   "  |step-mean|>0.2: %d of %zu\n",
                   tag, m, sd, v2 > 0 ? c1 / v2 : 0.0, big, stp.size());
        };
        step_stats(prompts, "RAW");
    }

    printf("\n================================================================================\n");
    printf("VERDICT: worst |error| over %d records = %.3f chips", o.nrec, worst);
    if (worst < 0.5)
        printf("   -- CHAIN CLOSES (sub-chip)\n");
    else if (worst < L)
        printf("   -- within-period error, DLL-recoverable if < ~1 chip\n");
    else
        printf("   -- OFF BY %+.0f OVERLAY PERIODS (%.3f within-period)\n", std::round(worst / L),
               wrap(worst, L));
    printf("================================================================================\n");
    return 0;
}
