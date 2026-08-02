// ============================================================================================
// THE END-TO-END SEEDING HARNESS. Build this before touching anything else in the chain.
// ============================================================================================
//
// Inject a KNOWN code phase into a noiseless synthetic CHORD sky, push it through the ACTUAL
// SHIPPED code of every stage between the antenna and the despread, and print the error in
// chips at each hand-off. No sky, no GPUs full of real data, no waiting for a transit; runs in
// seconds and is fully deterministic.
//
//   truth (cp204, dop, at absolute window W0)
//     |
//     |  gnss::ChannelizedReplicaBank::channels_hoprate   <- the same generator the sky is
//     v                                                       modelled with (GPS_L5_Q_NH)
//   synthetic channelized data
//     |
//     |  gnss::channelized_accumulate / channelized_peak  <- GnssChannelizedSearch's acquire
//     |  gnss::ChannelizedReplicaBank::hoprate_stream     <- ...its refine
//     |  gnss::detection_phase                            <- ...its reporting  [SHIPPED]
//     v
//   detection {cp0, cp_long, cp_at_ref, nh, dop}          == /get_detections
//     |
//     |  (direct, or --seed-file: the REAL broker's captured /set_seeds payload)
//     v
//   seed {code_phase_at_ref_chips, ref_hop, doppler_hz, code_phase_rate, ...}
//     |
//     |  gnss::propagate_seed                             <- cudaGnssChordTrack  [SHIPPED]
//     v
//   commanded argument cp per record
//     |
//     |  GnssCudaDespread::despread_batch                 <- the GPU kernel  [SHIPPED]
//     v
//   E/P/L power  +  error vs truth, in chips
//
// WHY IT EXISTS. Every seeding bug in the week to 2026-08-02 lived in the CONVENTION BETWEEN
// two stages -- what a number means, what it is reduced modulo, which epoch it references --
// and none was visible to a test of either stage alone. Each component passed its own test and
// the chain still failed, so bugs were found one at a time, in deployment order, over three
// days. Three separately paper-derived mappings reached deployment WRONG in one week. This
// tool would have settled each of them in minutes.
//
// The one rule that makes it worth anything: it CALLS the stage arithmetic (gnssSeedTransport)
// rather than restating it. A harness that re-derives the mapping tests the harness author's
// understanding of the mapping, which is precisely the thing already known to be unreliable.
//
// THE KEY TRICK that makes arbitrary seed ages free: a `code_phase_chips` ARGUMENT is
// referenced to absolute sample 0, so ONE argument describes the satellite for all time.
// Synthesizing a record at window W with argument cp204 gives exactly the continuation of the
// same signal -- so a 300 s seed age costs one extra 2048-hop synthesis, not 300 s of samples.
//
// Build:  scripts/gnss/build_tool.sh e2e
// Run:    scripts/gnss/e2e --help
//
// @author Keith Vanderlinde
// ============================================================================================
#include "GnssCudaDespread.hpp"
#include "gnssChannelizedAcquire.hpp"
#include "gnssChannelizedDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSeedTransport.hpp"
#include "gnssSignal.hpp"
#include "pfbPrototype.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using cf = std::complex<float>;

// ---------------------------------------------------------------------------------------------
// Options -- every geometry number the live chain uses, so a run can be pointed at the exact
// production configuration (or at a deliberately perturbed one, to see what breaks).
// ---------------------------------------------------------------------------------------------
struct Opt {
    int prn = 3;
    double cp204 = 137456.75; ///< the INJECTED truth: generator argument, mod 204600
    double dop = 1893.4;      ///< the INJECTED truth Doppler, Hz
    long long hop0 = 114436200145LL; ///< snapshot start hop (~6.8 days of uptime, as on sky)
    double age_s = 27.0;             ///< seed age at the first tracker record (live: <= ~27 s)
    int nrec = 4;                    ///< tracker records to propagate through
    double rec_gap_s = 10.0;         ///< spacing between the records we test

    // Search side (chord_gnss_agg.yaml gps_search)
    int s_chan0 = 5972, s_stride = 4, s_nchan = 27;
    double dop_half = 1000.0, dop_step = 31.25;
    int acquire_windows = 1, fine_step = 32, threads = 6;
    /// How many replica-periods of data the REFINE integrates over. The stage (and this
    /// harness until now) refines on window 0 only, i.e. 16 ms. The refine's job is to pick
    /// between grating lobes that differ by only ~15%, so its noise -- not the margin -- is
    /// what decides, and that falls as sqrt of the integration length. The overlay is indexed
    /// by ABSOLUTE period inside hoprate_stream, so a longer window stays coherent for free.
    int refine_windows = 1;
    // Defaults MATCH WHAT SHIPS: refine_span 0 -> fft_len (the stage default, one whole
    // hop = the full lag ambiguity), refine_step 303 (what gen_chord_gnss_config.py
    // emits: fft_len/n_chan/2). The old 4096/75 here were the hand-added override that
    // turned out to BE the bug -- a harness defaulting to a broken config silently
    // reproduces the break and calls it baseline.
    int refine_span = 0, refine_step = 303;

    // Tracker side (chord_gnss_cx19.yaml gnss0_track)
    int t_chan0 = 5972, t_stride = 16, t_nchan = 7;
    int hops_per_record = 2048;
    double dll_spacing = 0.5;

    // Shared geometry
    double sample_rate = 3.2e9, f_offset = 1176.45e6;
    int spectrum_length = 8192, num_taps = 4;
    double code_doppler_sign = 1.0;

    // Seed source / perturbations. These are the levers the chain is actually sensitive to,
    // exposed so a sweep can measure the sensitivity instead of arguing about it.
    const char* seed_file = nullptr; ///< read a REAL broker seed (seeds.jsonl) instead of direct
    double seed_dop_err = 0.0;       ///< Hz added to the seed's Doppler only (not the truth)
    double seed_cp_rate = 0.0;       ///< chips/hop residual handed to the tracker
    double seed_dop_rate = 0.0;      ///< Hz/s handed to the tracker
    int quantize = 0;                ///< 1 = 4+4b like GnssQuantize44, 0 = float (noiseless)
    int skip_search = 0;             ///< 1 = seed straight from truth (isolates the tracker leg)
    /// Write the detection in /get_detections wire format. e2e_broker.py serves this to the
    /// REAL broker, so its seed arithmetic can be put in the loop too (--seed-file the result).
    const char* emit_det = nullptr;
    const char* dump_refine = nullptr; ///< dump the refine objective over a full hop
    int ms_split = 0;   ///< >0 = run the ms-split acquire with this many ~1 ms sub-windows
    int sub_hops = 196; ///< N: ceil(code period in hops) at CHORD
    /// Additive complex Gaussian noise, std as a MULTIPLE of the synthetic signal's rms.
    /// This is the regime knob that --quantize is not: quantizing a noiseless full-scale
    /// signal changes nothing (measured, byte-identical), because on sky the satellite sits
    /// far UNDER the noise floor and every margin the refine relies on -- notably "the true
    /// grating lobe beats its neighbour by 15%" -- is an infinite-SNR statement without it.
    double noise = 0.0;
    int trials = 1;            ///< independent noise realizations (>1 prints a distribution)
    unsigned nseed = 12345;    ///< RNG seed, so a run is reproducible
};

static void usage() {
    printf(
        "e2e -- inject a known code phase, run the shipped search->seed->track->despread chain\n"
        "       offline, print the error in chips at each hand-off.\n\n"
        "  --prn N            PRN to inject and search for (default 3)\n"
        "  --cp204 X          INJECTED truth: generator argument, chips mod 204600\n"
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
        "  --seed-dop-err F   Hz of Doppler error given ONLY to the seed (the live residual)\n"
        "  --seed-cp-rate X   code_phase_rate handed to the tracker, chips/hop\n"
        "  --seed-dop-rate F  doppler_rate_hz_s handed to the tracker\n"
        "  --seed-file P      use a REAL captured broker seed (seeds.jsonl) for this PRN\n"
        "  --emit-detection P write the detection in /get_detections wire format (for e2e_broker)\n"
        "  --ms-split K       run the ms-split acquire over K ~1 ms sub-windows instead\n"
        "  --sub-hops N       hops per sub-window (default 196 = ceil(code period))\n"
        "  --noise R          add complex Gaussian noise, std = R x signal rms (0 = noiseless)\n"
        "  --trials N         repeat the search on N noise realizations, print the spread\n"
        "  --nseed N          RNG seed (default 12345), so a noisy run is reproducible\n"
        "  --quantize         quantize the synthetic sky to 4+4b (default: noiseless float)\n"
        "  --skip-search      seed straight from truth -- isolates the tracker/despread leg\n"
        "  --code-doppler-sign S   (default +1)\n");
}

static bool arg_eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

// ---------------------------------------------------------------------------------------------
// Wrap a chip difference into (-half, +half]. Every error in this tool is a phase difference
// modulo the long code, so this is the only sane way to print one.
// ---------------------------------------------------------------------------------------------
static double wrap(double x, double period) {
    double r = std::fmod(x, period);
    if (r > 0.5 * period)
        r -= period;
    if (r <= -0.5 * period)
        r += period;
    return r;
}

// 4+4b offset-encoded round trip, as GnssQuantize44 -> GnssChordDequantize does it live.
// Scaled so the peak lands near full scale; that is what the live AGC targets.
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

// Minimal extraction of one numeric field from the last seed object mentioning "prn": N in a
// seed_sink.py capture. Deliberately dumb -- this reads a file we write ourselves, and pulling
// in a JSON dependency for six numbers is not worth it.
static bool seed_field(const std::string& blob, int prn, const char* key, double& out) {
    const std::string pk = "\"prn\":" + std::to_string(prn);
    size_t at = std::string::npos;
    for (size_t i = blob.find(pk); i != std::string::npos; i = blob.find(pk, i + 1))
        at = i;
    if (at == std::string::npos)
        return false;
    const size_t end = blob.find('}', at);
    const size_t k = blob.find(std::string("\"") + key + "\":", at);
    if (k == std::string::npos || (end != std::string::npos && k > end))
        return false;
    out = atof(blob.c_str() + k + std::strlen(key) + 3);
    return true;
}

int main(int argc, char** argv) {
    Opt o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next_d = [&]() { return atof(argv[++i]); };
        auto next_i = [&]() { return atoi(argv[++i]); };
        if (arg_eq(a, "--help") || arg_eq(a, "-h")) { usage(); return 0; }
        else if (arg_eq(a, "--prn")) o.prn = next_i();
        else if (arg_eq(a, "--cp204")) o.cp204 = next_d();
        else if (arg_eq(a, "--dop")) o.dop = next_d();
        else if (arg_eq(a, "--hop0")) o.hop0 = atoll(argv[++i]);
        else if (arg_eq(a, "--age-s")) o.age_s = next_d();
        else if (arg_eq(a, "--nrec")) o.nrec = next_i();
        else if (arg_eq(a, "--rec-gap-s")) o.rec_gap_s = next_d();
        else if (arg_eq(a, "--dop-half")) o.dop_half = next_d();
        else if (arg_eq(a, "--dop-step")) o.dop_step = next_d();
        else if (arg_eq(a, "--windows")) o.acquire_windows = next_i();
        else if (arg_eq(a, "--refine-windows")) o.refine_windows = next_i();
        else if (arg_eq(a, "--fine-step")) o.fine_step = next_i();
        else if (arg_eq(a, "--threads")) o.threads = next_i();
        else if (arg_eq(a, "--refine-span")) o.refine_span = next_i();
        else if (arg_eq(a, "--refine-step")) o.refine_step = next_i();
        else if (arg_eq(a, "--seed-dop-err")) o.seed_dop_err = next_d();
        else if (arg_eq(a, "--seed-cp-rate")) o.seed_cp_rate = next_d();
        else if (arg_eq(a, "--seed-dop-rate")) o.seed_dop_rate = next_d();
        else if (arg_eq(a, "--seed-file")) o.seed_file = argv[++i];
        else if (arg_eq(a, "--emit-detection")) o.emit_det = argv[++i];
        else if (arg_eq(a, "--dump-refine")) o.dump_refine = argv[++i];
        else if (arg_eq(a, "--ms-split")) o.ms_split = next_i();
        else if (arg_eq(a, "--sub-hops")) o.sub_hops = next_i();
        else if (arg_eq(a, "--noise")) o.noise = next_d();
        else if (arg_eq(a, "--trials")) o.trials = next_i();
        else if (arg_eq(a, "--nseed")) o.nseed = (unsigned)next_i();
        else if (arg_eq(a, "--quantize")) o.quantize = 1;
        else if (arg_eq(a, "--skip-search")) o.skip_search = 1;
        else if (arg_eq(a, "--code-doppler-sign")) o.code_doppler_sign = next_d();
        else { printf("unknown option %s\n", a); usage(); return 2; }
    }

    const int FFT = 2 * o.spectrum_length;
    if (o.refine_span <= 0)
        o.refine_span = FFT; // the stage's own default
    const long long W0 = o.hop0 * (long long)FFT;

    std::vector<int> s_chans, t_chans, s_cov, t_cov;
    for (int c = 0; c < o.s_nchan; ++c) { s_chans.push_back(o.s_chan0 + o.s_stride * c); s_cov.push_back(c); }
    for (int c = 0; c < o.t_nchan; ++c) { t_chans.push_back(o.t_chan0 + o.t_stride * c); t_cov.push_back(c); }

    // TWO banks, exactly as production runs them, and the difference between them is the whole
    // reason this harness exists: the SEARCH works in the primary code (GPS_L5_Q, L = 10230)
    // with the NH20 overlay carried separately as an alignment index, while the TRACKER works
    // in the overlaid code (GPS_L5_Q_NH, L = 204600) with the overlay baked into the table.
    // Everything that has gone wrong in seeding has gone wrong in that translation.
    const gnss::SignalDescriptor* sig_s = gnss::signal_by_name("GPS_L5_Q");
    const gnss::SignalDescriptor* sig_t = gnss::signal_by_name("GPS_L5_Q_NH");
    gnss::ChannelizedReplicaBank sbank(*sig_s, o.sample_rate, o.f_offset, o.spectrum_length,
                                       o.num_taps, dsp::Window::Hamming, {o.prn});
    // Two slots holding the SAME PRN: slot 0 takes the tracker's commanded phase, slot 1 the
    // truth, so both despreads ride one kernel launch against one uploaded window (the device
    // job arena is sized by PRN slot count).
    gnss::ChannelizedReplicaBank tbank(*sig_t, o.sample_rate, o.f_offset, o.spectrum_length,
                                       o.num_taps, dsp::Window::Hamming, {o.prn, o.prn});
    sbank.code_doppler_sign = o.code_doppler_sign;
    tbank.code_doppler_sign = o.code_doppler_sign;

    const int Mp = sbank.repl_period_hops();
    const long long anchor = (long long)Mp * FFT; // what the search generates repl0 at
    const double L = (double)sbank.code_length();
    const int n_nh = sbank.secondary_length(); // 20
    const double LL = L * (double)n_nh;        // 204600 == tbank.code_length()
    const double hop_s = (double)FFT / o.sample_rate;

    printf("================================================================================\n");
    printf("INJECTED TRUTH   PRN %d   cp204 %.4f  (primary %.4f, NH period %d)   dop %+.4f Hz\n",
           o.prn, o.cp204, std::fmod(o.cp204, L), (int)(o.cp204 / L), o.dop);
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

    // The truth, in the currency each leg has to get right. The argument is constant over all
    // time (that is what an argument IS), so the true physical phase at any window is one call.
    auto truth_phase_at = [&](long long wstart) {
        return tbank.phase_from_arg(o.cp204, wstart, o.dop);
    };

    // -----------------------------------------------------------------------------------------
    // LEG 1: the search. Synthesize the snapshot, run the shipped acquire + refine, and report
    // through the shipped gnss::detection_phase().
    // -----------------------------------------------------------------------------------------
    gnss::DetectionPhase dp;
    double det_dop = o.dop;
    int best_nh = -1;
    double snr = 0.0;

    if (!o.skip_search) {
        // ceil(K*N / Mp). NOT -(-a/b): C++ integer division truncates TOWARD ZERO, so
        // -(-784/3125) is 0, not 1 -- which synthesized ZERO hops (instant crash), and for
        // K=16 gave 1 window = 3125 hops where 16*196 = 3136 are needed, i.e. a read past the
        // end of the buffer. Every ms-split number measured before this fix came off that
        // out-of-bounds memory, which is why the lag offsets would not sit still.
        const int nwin = o.ms_split > 0
                             ? (o.ms_split * o.sub_hops + Mp - 1) / Mp
                             : std::max(std::max(1, o.acquire_windows),
                                        std::max(1, o.refine_windows));
        printf("[1] SEARCH -- synthesizing %d x %d hops (%.1f ms) on %d channels...\n", nwin, Mp,
               1e3 * nwin * Mp * hop_s, o.s_nchan);
        const auto clean = tbank.channels_hoprate(0, W0, o.cp204, o.dop, nwin * Mp, s_chans, {}, -1);
        double sig2 = 0.0;
        size_t nsamp = 0;
        for (const auto& ch : clean)
            for (const cf& v : ch) { sig2 += std::norm(v); ++nsamp; }
        const double sig_rms = std::sqrt(sig2 / (double)std::max<size_t>(nsamp, 1));
        if (o.noise > 0.0)
            printf("    noise: std = %.3f x signal rms (%.4g), %d trial(s), seed %u\n", o.noise,
                   o.noise * sig_rms, o.trials, o.nseed);
        std::mt19937 rng(o.nseed);
        std::normal_distribution<double> gauss(0.0, o.noise * sig_rms / std::sqrt(2.0));
        std::vector<double> errs;
        std::vector<double> snrs;
        auto data = clean;

        // ANCHOR THE GRID ABSOLUTELY, as GnssChannelizedSearch does (134f197dc): integer
        // multiples of doppler_step, NOT the hint. Anchoring to the hint slides the origin with
        // the satellite, so every trial samples the SAME sub-bin position -- which silently
        // turned the first Doppler-bias sweep here into seven measurements of one point (all
        // +1.6 to +1.8 Hz, "independent of Doppler", which is exactly what a fixed sub-bin
        // offset looks like). The stage got this right for a live reason; the harness must
        // match it or it cannot see interpolation error at all.
        std::vector<double> grid;
        if (o.dop_half <= 0.0)
            grid.push_back(-o.dop); // r2c fold: the grid runs in the flipped convention
        else {
            const double lo =
                std::ceil((-o.dop - o.dop_half) / o.dop_step) * o.dop_step;
            for (double f = lo; f <= -o.dop + o.dop_half + 1e-9; f += o.dop_step)
                grid.push_back(f);
        }

      gnss::AcquireWorkspace ws;
      gnss::AcquisitionSurface adims{}; // hoisted: the summary below needs s_stored
      for (int trial = 0; trial < std::max(1, o.trials); ++trial) {
        // Fresh realization each trial; the CLEAN signal is never overwritten.
        if (o.noise > 0.0) {
            for (size_t c = 0; c < clean.size(); ++c)
                for (size_t m = 0; m < clean[c].size(); ++m)
                    data[c][m] = clean[c][m] + cf((float)gauss(rng), (float)gauss(rng));
        }
        if (o.quantize)
            quantize44(data);
        gnss::AcquisitionResult a{};
        best_nh = -1;
        std::vector<double> surf;
        std::vector<std::vector<cf>> w((size_t)o.s_nchan, std::vector<cf>((size_t)Mp));
        if (o.ms_split > 0) {
            // MS-SPLIT: one pass, no NH alignment axis at all -- a ~1 ms sub-window spans
            // exactly one overlay chip, a constant +-1, which |D|^2 cannot see. That is the
            // 20x. The lag axis is one code period, not sixteen: the other 12.5x.
            surf.assign(surf.size(), 0.0);
            gnss::AcquisitionSurface dm = gnss::ms_split_accumulate(
                sbank, 0, data, s_chans, W0, o.sub_hops, o.ms_split, grid, o.sample_rate, surf,
                ws, o.fine_step, o.threads);
            const auto ai = gnss::channelized_peak(surf, dm, grid, o.sample_rate,
                                                   sbank.chip_rate_hz(), sbank.code_length());
            a = ai; best_nh = 0; adims = dm;
            printf("    ms-split: %d sub-windows x %d hops (%.1f ms each, %.0f ms total), "
                   "lag axis %d, no NH axis\n",
                   o.ms_split, o.sub_hops, o.sub_hops * hop_s * 1e3,
                   o.ms_split * o.sub_hops * hop_s * 1e3, dm.Mp);
        }
        for (int nh = 0; o.ms_split == 0 && nh < n_nh; ++nh) {
            // repl0: code 0, Doppler 0, overlay alignment nh -- what the stage precomputes.
            auto repl0 = sbank.channels_hoprate(0, anchor, 0.0, 0.0, Mp, s_chans, {}, nh);
            gnss::AcquisitionSurface dims{};
            surf.assign(surf.size(), 0.0);
            // ACQUIRE accumulates acquire_windows, NOT nwin. nwin is only how much data was
            // synthesized (max of acquire_windows and refine_windows); conflating them made
            // --refine-windows silently raise acquire_windows, which smears across NH bins --
            // acquire snr fell 185->55 and the period broke 0/12 -> 10/12, all of it artifact.
            for (int wi = 0; wi < std::max(1, o.acquire_windows); ++wi) {
                for (int c = 0; c < o.s_nchan; ++c)
                    for (int m = 0; m < Mp; ++m)
                        w[(size_t)c][(size_t)m] = data[(size_t)c][(size_t)(wi * Mp + m)];
                dims = gnss::channelized_accumulate(w, repl0, s_cov, grid, o.sample_rate, o.s_nchan,
                                                    surf, ws, s_chans, FFT, o.threads,
                                                    o.fine_step);
            }
            const auto ai = gnss::channelized_peak(surf, dims, grid, o.sample_rate,
                                                   sbank.chip_rate_hz(), sbank.code_length());
            if (best_nh < 0 || ai.snr > a.snr) { a = ai; best_nh = nh; adims = dims; }
        }
        snr = a.snr;
        det_dop = -a.doppler_hz; // r2c fold conjugates the channel frequency axis
        const double cps = sbank.chip_rate_hz() / o.sample_rate;

        // ---- THE SHIPPED REFINE (de-alias, then localize) ----
        const int rhops = Mp * std::max(1, o.refine_windows);
        std::vector<std::vector<cf>> d((size_t)o.s_nchan, std::vector<cf>((size_t)rhops));
        for (int c = 0; c < o.s_nchan; ++c)
            for (int m = 0; m < rhops; ++m)
                d[(size_t)c][(size_t)m] = data[(size_t)c][(size_t)m];
        // DIAGNOSTIC: dump the refine's own objective over a full hop, so the shape of the
        // ambiguity is MEASURED rather than inferred from a coincidence. 13.09 chips is both
        // s_stored*cps and (the old) refine_span*cps -- two completely different explanations
        // for one number, which is exactly how a wrong story gets confirmed.
        if (o.dump_refine) {
            const auto rf = sbank.hoprate_filter(s_chans, det_dop);
            FILE* fp = fopen(o.dump_refine, "w");
            const double ph_t = truth_phase_at(W0);
            fprintf(fp, "# off_samples off_chips power  (truth ph@ref %.4f)\n", ph_t);
            for (int off = -FFT / 2; off <= FFT / 2; off += 16) {
                const double cp = a.code_phase_chips + off * cps;
                const auto r = sbank.hoprate_stream(rf, 0, anchor, cp, det_dop, Mp, {}, best_nh);
                fprintf(fp, "%d %.4f %.6e\n", off, off * cps,
                        std::norm(gnss::channelized_despread(d, r).amplitude));
            }
            fclose(fp);
            printf("    dumped refine profile -> %s\n", o.dump_refine);
        }
        const double best_cp =
            gnss::refine_peak(sbank, 0, d, s_chans, a.code_phase_chips, adims, best_nh, det_dop,
                              anchor, rhops, o.sample_rate, o.refine_span, o.refine_step,
                              o.threads);

        // ---- THE SHIPPED REPORTING ARITHMETIC ----
        dp = gnss::detection_phase(sbank, best_cp, best_nh, n_nh, det_dop, o.hop0, anchor,
                                   o.sample_rate, a.peak_tau_samples);

        const double ph_true = truth_phase_at(W0);
        const double e_ref = wrap(dp.cp_at_ref - ph_true, LL);

        // PERIOD DIAGNOSTICS. The acquire's coarse lag runs over a FULL replica period,
        // Mp*sph samples = 16 primary code periods at CHORD -- but `code_phase_chips` is that
        // lag reduced mod ONE period, so the whole-period part of the lag is discarded before
        // anything downstream sees it. The overlay is 20 periods and 16 is not 0 mod 20, so
        // that discarded count is exactly the term the nh lift needs and does not have.
        // Printed here so the relationship can be READ OFF ground truth rather than derived on
        // paper for a fourth time.
        const double tau_chips = (double)a.peak_tau_samples * sbank.chip_rate_hz() / o.sample_rate;
        const int tau_per = ((int)std::floor(tau_chips / L)) % n_nh;
        const int true_per = (int)std::floor(ph_true / L);
        const int got_per = (int)std::floor(dp.cp_at_ref / L);
        printf("    tau %ld samp = %.2f chips = %d periods + %.2f | nh %d | period true %d "
               "got %d (err %+d)\n",
               a.peak_tau_samples, tau_chips, tau_per, std::fmod(tau_chips, L), best_nh, true_per,
               got_per, ((got_per - true_per + 30) % n_nh) - 10);
        printf("    snr %.1f   dop %+.4f (err %+.4f Hz)   nh %d   coarse %.2f refine %+.2f\n",
               snr, det_dop, det_dop - o.dop, best_nh, a.code_phase_chips,
               best_cp - a.code_phase_chips);
        printf("    reported cp0 %.3f | cp_long %.3f | ph@ref %.3f   (truth ph@ref %.3f)\n",
               dp.cp0, dp.cp_long, dp.cp_at_ref, ph_true);
        printf("    >>> SEARCH LEG ERROR: %+.3f chips   (= %+.3f periods %+.3f within-period)\n",
               e_ref, std::round(e_ref / L), wrap(e_ref, L));
        errs.push_back(e_ref);
        snrs.push_back(snr);
      } // trial
        if (o.trials > 1) {
            // A "wrong lobe" is a within-period error beyond half a grating spacing. The comb's
            // fine-lag response repeats every s_stored samples, so that is the natural unit:
            // anything past half of it means the refine chose the neighbour.
            const double lobe = (double)((adims.s_stored > 0) ? adims.s_stored : FFT)
                                * sbank.chip_rate_hz() / o.sample_rate;
            int wrong = 0, badper = 0;
            double s1 = 0.0, s2 = 0.0, worst = 0.0;
            for (double e : errs) {
                const double w = wrap(e, L);
                if (std::fabs(e) > L / 2.0) ++badper;
                if (std::fabs(w) > lobe / 2.0) ++wrong;
                s1 += w; s2 += w * w; worst = std::max(worst, std::fabs(w));
            }
            const double n = (double)errs.size();
            double msnr = 0.0;
            for (double v : snrs) msnr += v;
            printf("\n    ===== %d trials at noise %.3f =====\n", (int)n, o.noise);
            printf("    acquire snr   mean %.1f\n", msnr / n);
            printf("    within-period error: mean %+.3f  rms %.3f  worst %.3f chips\n",
                   s1 / n, std::sqrt(s2 / n), worst);
            printf("    grating lobe spacing %.2f chips -> WRONG LOBE in %d/%d (%.0f%%)\n",
                   lobe, wrong, (int)n, 100.0 * wrong / n);
            printf("    wrong PERIOD in %d/%d\n", badper, (int)n);
        }
        printf("\n");
    } else {
        dp.cp_at_ref = truth_phase_at(W0);
        dp.cp_long = -1.0;
        printf("[1] SEARCH -- SKIPPED (--skip-search): seeding straight from truth, "
               "ph@ref %.3f\n\n", dp.cp_at_ref);
    }

    // -----------------------------------------------------------------------------------------
    // LEG 2: the seed. Either built directly from the detection (what the broker does when it
    // has nothing to add), or read from a REAL captured broker payload.
    // -----------------------------------------------------------------------------------------
    // The detection in /get_detections wire format, so the REAL broker can be put in the loop:
    // e2e_broker.py serves this file, runs gps_distributed_broker.py against it, and captures
    // the /set_seeds payload for --seed-file. Written even with --skip-search (the fields are
    // then the truth, which is the useful control case for the broker's own arithmetic).
    if (o.emit_det) {
        FILE* fp = fopen(o.emit_det, "wb");
        if (!fp) { printf("cannot write %s\n", o.emit_det); return 3; }
        fprintf(fp,
                "[{\"prn\":%d,\"doppler_hz\":%.10g,\"code_phase_chips\":%.10g,\"ref_hop\":%lld,"
                "\"nh\":%d,\"code_phase_long_chips\":%.10g,\"code_phase_at_ref_chips\":%.10g,"
                "\"snr\":%.6g}]\n",
                o.prn, det_dop, dp.cp0, o.hop0, best_nh, dp.cp_long, dp.cp_at_ref,
                o.skip_search ? 1e4 : snr);
        fclose(fp);
        printf("    wrote detection -> %s\n", o.emit_det);
    }

    gnss::SeedState sd;
    sd.ref_hop = o.hop0;
    sd.phase_ref_chips = dp.cp_at_ref;
    sd.doppler_hz = det_dop + o.seed_dop_err;
    sd.cp_rate = o.seed_cp_rate;
    sd.dop_rate = o.seed_dop_rate;
    if (o.seed_file) {
        FILE* fp = fopen(o.seed_file, "rb");
        if (!fp) { printf("cannot open %s\n", o.seed_file); return 3; }
        std::string blob;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, fp)) > 0)
            blob.append(buf, n);
        fclose(fp);
        double v;
        if (seed_field(blob, o.prn, "code_phase_at_ref_chips", v)) sd.phase_ref_chips = v;
        if (seed_field(blob, o.prn, "doppler_hz", v)) sd.doppler_hz = v;
        if (seed_field(blob, o.prn, "code_phase_rate", v)) sd.cp_rate = v;
        if (seed_field(blob, o.prn, "doppler_rate_hz_s", v)) sd.dop_rate = v;
        if (seed_field(blob, o.prn, "ref_hop", v)) sd.ref_hop = (long long)v;
        printf("[2] SEED (from %s)\n", o.seed_file);
    } else {
        printf("[2] SEED (direct from the detection)\n");
    }
    printf("    ref_hop %lld  ph@ref %.3f  dop %+.4f  cp_rate %.6e  dop_rate %+.4f\n\n",
           sd.ref_hop, sd.phase_ref_chips, sd.doppler_hz, sd.cp_rate, sd.dop_rate);

    // -----------------------------------------------------------------------------------------
    // LEG 3: the tracker. Propagate through the shipped gnss::propagate_seed(), synthesize the
    // record at its true window, and despread on the GPU with the shipped kernel.
    // -----------------------------------------------------------------------------------------
    printf("[3] TRACK + DESPREAD  (seed age %.1f s at record 0, records %.1f s apart)\n",
           o.age_s, o.rec_gap_s);
    GnssCudaDespread ds(tbank, 2, t_chans, o.hops_per_record, o.sample_rate, o.f_offset);
    printf("    rec   age_s        cp cmd    phase err     per   |   P/P_true    q=2P/(E+L)\n");

    const long long age_hops = (long long)std::llround(o.age_s / hop_s);
    const long long gap_hops = (long long)std::llround(o.rec_gap_s / hop_s);
    double worst = 0.0;
    for (int r = 0; r < o.nrec; ++r) {
        const long long h = o.hop0 + age_hops + (long long)r * gap_hops;
        const long long W = h * (long long)FFT;

        // ---- THE SHIPPED PROPAGATION ARITHMETIC (trim 0: we are measuring the OPEN loop) ----
        const gnss::SeedPropagation pr =
            gnss::propagate_seed(tbank, sd, h, o.sample_rate, o.f_offset, 0.0);

        // The error is a difference of PHYSICAL PHASES at this window -- never of arguments.
        //
        // The first version of this harness printed wrap(pr.cp - cp204), i.e. it compared the
        // commanded ARGUMENT against the injected one, and reported 8650 chips of error on a
        // chain that was in fact ~0.4 chips off. Two arguments derived at Dopplers 1.7 Hz apart
        // differ by 1.7 * 5095 = 8650 chips BY CONSTRUCTION and describe the same signal; the
        // difference cancels when each is fed to the generator alongside its own Doppler. That
        // is the exact mistake this whole tool exists to catch, made inside the tool, caught in
        // one run by the despread power disagreeing with the arithmetic. Leave the warning here.
        const double err = wrap(pr.phase_now - truth_phase_at(W), LL);

        auto rec = tbank.channels_hoprate(0, W, o.cp204, o.dop, o.hops_per_record, t_chans, {}, -1);
        if (o.quantize)
            quantize44(rec);
        std::vector<cf> win((size_t)o.hops_per_record * (size_t)o.t_nchan);
        for (int m = 0; m < o.hops_per_record; ++m)
            for (int c = 0; c < o.t_nchan; ++c)
                win[(size_t)m * (size_t)o.t_nchan + (size_t)c] = rec[(size_t)c][(size_t)m];
        ds.upload_window(win.data(), W);

        // Job 0 = the tracker's commanded phase; job 1 = perfect knowledge, the normalizer.
        auto res = ds.despread_batch({{0, pr.cp, o.dll_spacing, pr.doppler_hz, t_cov},
                                      {1, o.cp204, o.dll_spacing, o.dop, t_cov}});
        const double P = std::norm(res[0][1].correlation);
        const double E = std::norm(res[0][0].correlation);
        const double La = std::norm(res[0][2].correlation);
        const double Pt = std::norm(res[1][1].correlation);
        printf("    %3d %7.1f  %13.3f  %+12.3f  %+5.0f   |   %8.4f    %8.3f\n", r,
               (double)(h - o.hop0) * hop_s, pr.cp, err, std::round(err / L),
               Pt > 0 ? P / Pt : 0.0, (E + La) > 0 ? 2.0 * P / (E + La) : 0.0);
        worst = std::max(worst, std::fabs(err));
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
