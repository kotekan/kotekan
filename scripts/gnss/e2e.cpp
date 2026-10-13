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
//     |  gnss::propagate_seed                             <- cudaGnssInject  [SHIPPED]
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
// Build:  scripts/gnss/site/build_tool.sh e2e
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
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
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
    double cp204 = 137456.75;        ///< the INJECTED truth: generator argument, mod 204600
    double dop = 1893.4;             ///< the INJECTED truth Doppler, Hz
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
    int refine_hops = 0; ///< refine integration hops (0 = the full record)

    // Tracker side (chord_gnss_cx19.yaml gnss0_track)
    int t_chan0 = 5972, t_stride = 16, t_nchan = 7;
    int hops_per_record = 2048;
    double dll_spacing = 0.5;
    /// [#51 F3] THE COMMANDED DLL TRIM, chips, handed to propagate_seed exactly as
    /// cudaGnssInject hands it the fleet controller's value. With a perfect seed
    /// (--skip-search) a nonzero trim IS a known code error of known sign, which makes this
    /// the sign test for the whole actuator: the loop must respond with tau = -disc/4 pointing
    /// BACK toward zero. A sign error here diverges the loop, and no amount of rate fixes it.
    double trim_chips = 0.0;

    // Shared geometry
    double sample_rate = 3.2e9, f_offset = 1176.45e6;
    int spectrum_length = 8192, num_taps = 4;
    double code_doppler_sign = 1.0;

    /// TRACKER (baked-overlay) signal. The SEARCH signal is derived by stripping the _NH/_CS
    /// suffix (GPS_L5_Q_NH -> GPS_L5_Q, GAL_E5A_Q_CS -> GAL_E5A_Q). Per-PRN-secondary signals
    /// (the _CS pair) have no blind search -- the bank's single-sequence overlay slot skips
    /// them -- so they require --skip-search (docs/CHORD_MULTIBAND.md section 5).
    const char* signal = "GPS_L5_Q_NH";

    // Seed source / perturbations. These are the levers the chain is actually sensitive to,
    // exposed so a sweep can measure the sensitivity instead of arguing about it.
    const char* seed_file = nullptr; ///< read a REAL broker seed (seeds.jsonl) instead of direct
    double seed_dop_err = 0.0;       ///< Hz added to the seed's Doppler only (not the truth)
    double seed_cp_err = 0.0;        ///< chips added to the seed's code phase only (not the
                                     ///< truth). +k*primary_len = a SECONDARY-PHASE error on a
                                     ///< baked _CS/_NH signal: calibrates what a wrong CS/NH
                                     ///< period looks like downstream (the E6 bring-up tool).
    double seed_cp_rate = 0.0;       ///< chips/hop residual handed to the tracker
    /// Hz/s handed to the tracker. ⚠️ NOT ZERO, AND THAT IS THE POINT (task #52, 2026-08-13).
    ///
    /// This defaulted to 0.0, and a satellite whose Doppler never moves has no re-pin: the
    /// replica's f_ref is identical in every record, so the (delta f)*t_abs phase step that
    /// dominates the live chain CANNOT ARISE HERE. That is why this harness reported a
    /// per-record phase residual of 0.0008 rad while the sky read 0.745, and why it printed
    /// "the live floor comes from something this harness does not model" for weeks. It was
    /// modelling it fine; it was handed a satellite that does not exist.
    ///
    /// -0.369 Hz/s is PRN 21's measured rate on 2026-08-13 -- mid-range for the E5a fleet
    /// (0.036 to 0.369 across 12 satellites). With it, coherent snr collapses 3394 -> 5.4 and
    /// the residual reproduces the live floor with NO NOISE. A gate has to vary the axis the
    /// defect lives on, and for this whole family of bugs that axis is (small number) x
    /// (huge absolute argument): hop0 supplies the huge argument, this supplies the small one.
    /// Set --seed-dop-rate 0 to recover the old blind behaviour deliberately.
    double seed_dop_rate = -0.36893;
    int quantize = 0;    ///< 1 = 4+4b like the F-engine, 0 = float (noiseless)
    int skip_search = 0; ///< 1 = seed straight from truth (isolates the tracker leg)
    /// Write the detection in /get_detections wire format. e2e_broker.py serves this to the
    /// REAL broker, so its seed arithmetic can be put in the loop too (--seed-file the result).
    const char* emit_det = nullptr;
    bool cuda_refine = false;    ///< also run the GPU refine and compare (A5)
    int max_chips = 0;           ///< truncate the PFB chip gather (0 = full span)
    bool chips_centered = false; ///< center the truncation window on the prototype peak
    /// Additive complex Gaussian noise, std as a MULTIPLE of the synthetic signal's rms.
    /// This is the regime knob that --quantize is not: quantizing a noiseless full-scale
    /// signal changes nothing (measured, byte-identical), because on sky the satellite sits
    /// far UNDER the noise floor and every margin the refine relies on -- notably "the true
    /// grating lobe beats its neighbour by 15%" -- is an infinite-SNR statement without it.
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
        "  --seed-file P      use a REAL captured broker seed (seeds.jsonl) for this PRN\n"
        "  --dll-spacing D    Early/Late tap offset from prompt, COMPONENT chips (default 0.5,\n"
        "                     the node default). With --trim sweeps this measures q on the\n"
        "                     peak and the discriminator slope for a spacing choice.\n"
        "  --emit-detection P write the detection in /get_detections wire format (for e2e_broker)\n"
        "  --s-stride N       search comb stride (default 4 = 2 nodes; 1 = the 8-node agg)\n"
        "  --t-stride N       TRACKER comb stride (default 16 = one GPU's comb; 1 = contiguous)\n"
        "  --t-nchan N        tracker channel count (default 7)\n"
        "  --t-chan0 N        tracker comb first channel (default 5972)\n"
        "  --f-offset HZ      CARRIER, and the band under test. Recentres both combs on it\n"
        "                     unless --t-chan0/--s-chan0 pin them. Without this the harness\n"
        "                     is band-blind: it builds sky AND replica at 1176.45e6, so it is\n"
        "                     self-consistent at any carrier (docs 11.24).\n"
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

// 4+4b offset-encoded round trip, as the F-engine -> GnssChordDequantize does it live.
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
// WHITESPACE-TOLERANT. The first version matched "prn":N literally, and json.dumps writes
// "prn": N -- so EVERY seed file written by Python (e2e_broker's capture of the real
// broker's POST included) failed every lookup and e2e silently fell back to its own seed.
// The harness then reported the broker's contribution as "identical to e2e's own", which
// is exactly what a working broker looks like: a gate that could not fail, for as long as
// it has existed. Found 2026-08-12 when a deliberately WRONG seed (52 chips off) also
// scored 0.000. Now: skip spaces after the colon, and REQUIRE the caller to check the
// return value for fields it depends on (main() logs what it actually took).
static bool seed_field(const std::string& blob, int prn, const char* key, double& out) {
    const std::string pk = "\"prn\":";
    size_t at = std::string::npos;
    for (size_t i = blob.find(pk); i != std::string::npos; i = blob.find(pk, i + 1)) {
        size_t v = i + pk.size();
        while (v < blob.size() && std::isspace((unsigned char)blob[v]))
            ++v;
        if (atoi(blob.c_str() + v) == prn)
            at = i;
    }
    if (at == std::string::npos)
        return false;
    const size_t end = blob.find('}', at);
    const size_t k = blob.find(std::string("\"") + key + "\":", at);
    if (k == std::string::npos || (end != std::string::npos && k > end))
        return false;
    size_t v = k + std::strlen(key) + 3;
    while (v < blob.size() && std::isspace((unsigned char)blob[v]))
        ++v;
    out = atof(blob.c_str() + v);
    return true;
}

int main(int argc, char** argv) {
    Opt o;
    // Whether the caller PINNED these, as opposed to inheriting the L5 defaults. --f-offset
    // recentres the combs on the new carrier, but only over defaults -- an explicit --t-chan0
    // must always win, or a deliberate "carrier here, comb there" test becomes unexpressible.
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
        else if (arg_eq(a, "--seed-file"))
            o.seed_file = argv[++i];
        else if (arg_eq(a, "--emit-detection"))
            o.emit_det = argv[++i];
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

    // -- BAND: recentre the combs on the carrier ------------------------------------------
    //
    // ⚠️ THIS HARNESS WAS BAND-BLIND, AND THAT IS THE WHOLE POINT OF THIS BLOCK. f_offset was
    // a struct field with no flag, hardcoded at 1176.45e6, and it built the synthetic sky AND
    // the replica at that one value -- so it was self-consistent at ANY carrier. Asked to
    // validate GAL_E5B_Q_CS (1207.14 MHz) it ran that signal's CODE at L5's CARRIER and
    // returned a triumphant 0.000 chips. It could not detect a band-dependent fault, which is
    // the entire fault class a second band introduces (docs/gnss_gpu_search.md 11.24).
    //
    // Moving the carrier alone is NOT enough and would be a worse trap than the original: the
    // channel combs default to L5's (t_chan0 5972), so a bare --f-offset would put the replica
    // at 1207.14 MHz and the channels at 1176.45 MHz -- zero overlap, zero correlation, and a
    // "failure" that says nothing about the signal. So the combs FOLLOW the carrier unless the
    // caller pins them explicitly, which keeps --f-offset a statement about the BAND rather
    // than about one of the four numbers a band is made of.
    //
    // The recentring reproduces the node grid rather than inventing one: comb members satisfy
    // freq_id % stride == (default chan0) % stride, so the comb stays on the channels a node
    // actually holds, and the centre is the NEAREST such bin to the carrier. Checked against
    // config/chord_band_plan.py's covering_channels for both bands: L5 1176.45 MHz -> bin
    // 6023.4240 -> centre 6020 -> chan0 5972 (the historical default, reproduced exactly), and
    // E5b 1207.14 MHz -> bin 6180.5568 -> centre 6180 -> chan0 6132, inside the fleet's real
    // covering set 6128..6233.
    //
    // NOTE the two carriers sit +0.4240 and -0.4432 channels off their nearest centre -- same
    // size, OPPOSITE SIGN. That is deliberate to preserve here: it is exactly the asymmetry a
    // floor()-vs-round() bug anywhere in a carrier->bin conversion would expose, being exact
    // for L5 and a whole channel (195.3 kHz) wrong for E5b.
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

    // TWO banks, exactly as production runs them, and the difference between them is the whole
    // reason this harness exists: the SEARCH works in the primary code (e.g. GPS_L5_Q,
    // L = 10230) with the overlay carried separately as an alignment index, while the TRACKER
    // works in the overlaid code (e.g. GPS_L5_Q_NH, L = 204600) with the overlay baked into
    // the table. Everything that has gone wrong in seeding has gone wrong in that translation.
    // The search signal is the tracker name minus its _NH/_CS suffix.
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
    // Two slots holding the SAME PRN: slot 0 takes the tracker's commanded phase, slot 1 the
    // truth -- both ride one kernel launch against one uploaded window (the device job arena is
    // sized by PRN slot count).
    gnss::ChannelizedReplicaBank tbank(*sig_t, o.sample_rate, o.f_offset, o.spectrum_length,
                                       o.num_taps, dsp::Window::Hamming, {o.prn, o.prn});
    sbank.code_doppler_sign = o.code_doppler_sign;
    // TRUNCATE THE ANALYSIS BANK ONLY. sbank builds the SEARCH's replica; the synthetic sky comes
    // from tbank (channels_hoprate at [1]). Capping both would make the measurement
    // self-consistent -- the sky and the replica would be truncated identically, the error would
    // come out ~0, and it would mean nothing. This way the truth keeps the full 210-chip PFB span
    // and the search analyses with a shorter one, which is exactly the on-sky situation.
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
    // Overlay period count from the CODE-LENGTH RATIO, not the search bank's overlay slot:
    // the two agree for shared secondaries (L5: 204600/10230 = 20 = NH20), but per-PRN
    // secondaries (E5a/B2a CS100) are absent from the bank's single-sequence slot, where the
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

    // The truth, in the currency each leg has to get right. The argument is constant over all
    // time (that is what an argument IS), so the true physical phase at any window is one call.
    // In WIRE units: component chips, what cp_at_ref reports and the broker ships in
    // code_phase_at_ref_chips. phase_from_arg returns COMBINED chips (comb_mult x), and feeding
    // that straight into the seed made this harness agree with propagate_seed while the sky
    // disagreed (L2C, comb_mult 2: "ph@ref 11011" on a 10230-chip code was the tell).
    auto truth_phase_at = [&](long long wstart) {
        return tbank.phase_from_arg(o.cp204, wstart, o.dop) / (double)tbank.comb_mult();
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
            const double lo = std::ceil((-o.dop - o.dop_half) / o.dop_step) * o.dop_step;
            for (double f = lo; f <= -o.dop + o.dop_half + 1e-9; f += o.dop_step)
                grid.push_back(f);
        }

        // STAGE TIMERS: measure the stages before optimising any of them.
        double T_acq = 0.0, T_ref = 0.0;
        const auto tnow = [] {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        };
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
            const double t_acq0 = tnow();
            best_nh = -1;
            std::vector<double> surf;
            std::vector<std::vector<cf>> w((size_t)o.s_nchan, std::vector<cf>((size_t)Mp));
            for (int nh = 0; nh < n_nh; ++nh) {
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

            // ---- THE SHIPPED REFINE (de-alias, then localize) ----
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
            // A5 VALIDATION. The GPU refine must return the SAME code phase as the CPU one, on the
            // same data, with the same scan geometry -- it is a reuse of GnssCudaDespread, not a
            // different algorithm, so anything but agreement to float precision is a bug. Run both
            // and print the delta rather than trusting the speedup.
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
                // One engine per <=64-channel group: chan_mask is a uint64_t. The harness comb is
                // 27 channels so this is a single group, but exercise the same code path the live
                // 79-channel aggregator takes.
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
            // ---- THE SHIPPED REPORTING ARITHMETIC ----
            // The coarse lag runs over Mp = 3125 hops = 16 code periods and therefore carries a
            // whole-period count that detection_phase folds back in (the lag-period lift).
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
            // A "wrong lobe" is a within-period error beyond half a grating spacing. The comb's
            // fine-lag response repeats every s_stored samples, so that is the natural unit:
            // anything past half of it means the refine chose the neighbour.
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
    // LEG 2: the seed. Either built directly from the detection (what the broker does when it
    // has nothing to add), or read from a REAL captured broker payload.
    // -----------------------------------------------------------------------------------------
    // The detection in /get_detections wire format, so the REAL broker can be put in the loop:
    // e2e_broker.py serves this file, runs gps_distributed_broker.py against it, and captures
    // the /set_seeds payload for --seed-file. Written even with --skip-search (the fields are
    // then the truth, which is the useful control case for the broker's own arithmetic).
    if (o.emit_det) {
        FILE* fp = fopen(o.emit_det, "wb");
        if (!fp) {
            printf("cannot write %s\n", o.emit_det);
            return 3;
        }
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
    sd.phase_ref_chips = dp.cp_at_ref + o.seed_cp_err;
    sd.doppler_hz = det_dop + o.seed_dop_err;
    sd.cp_rate = o.seed_cp_rate;
    sd.dop_rate = o.seed_dop_rate;
    if (o.seed_file) {
        FILE* fp = fopen(o.seed_file, "rb");
        if (!fp) {
            printf("cannot open %s\n", o.seed_file);
            return 3;
        }
        std::string blob;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, fp)) > 0)
            blob.append(buf, n);
        fclose(fp);
        // POISON FIRST, PARSE SECOND (KV, 2026-08-12). The fields a --seed-file run is
        // ABOUT must not survive a failed parse wearing e2e's own plausible values --
        // that is what made a 52-chip-wrong seed score 0.000 and a whitespace bug look
        // like a healthy broker. NaN cannot be mistaken for a measurement: it propagates
        // to the verdict and is checked below. (cp_rate / dop_rate stay 0: absent means
        // "the producer had none", which is a real and common answer.)
        sd.phase_ref_chips = std::numeric_limits<double>::quiet_NaN();
        sd.cp_chips = std::numeric_limits<double>::quiet_NaN();
        sd.doppler_hz = std::numeric_limits<double>::quiet_NaN();
        sd.cp_rate = 0.0;
        sd.dop_rate = 0.0;
        double v;
        std::string took, missed;
        auto take = [&](const char* key, auto&& apply) {
            double x;
            if (seed_field(blob, o.prn, key, x)) {
                apply(x);
                took += std::string(" ") + key;
            } else
                missed += std::string(" ") + key;
        };
        take("code_phase_at_ref_chips", [&](double x) { sd.phase_ref_chips = x; });
        take("doppler_hz", [&](double x) { sd.doppler_hz = x; });
        take("code_phase_chips", [&](double x) { sd.cp_chips = x; });
        take("code_phase_rate", [&](double x) { sd.cp_rate = x; });
        take("doppler_rate_hz_s", [&](double x) { sd.dop_rate = x; });
        take("ref_hop", [&](double x) { sd.ref_hop = (long long)x; });
        printf("[2] SEED (from %s)\n", o.seed_file);
        // SAY WHAT WAS ACTUALLY TAKEN. A file whose fields all fail to parse leaves e2e
        // running its OWN seed and reporting a perfect score -- which is how the
        // "\"prn\":N" whitespace bug made this harness unfalsifiable. If nothing parsed,
        // that is a broken test, not a passing one: refuse rather than flatter.
        printf("    fields taken:%s\n", took.empty() ? " NONE" : took.c_str());
        if (!missed.empty())
            printf("    fields absent:%s (poisoned to NaN, or 0 for the rate terms --\n"
                   "                   NOT inherited from e2e's own seed)\n",
                   missed.c_str());
        if (took.empty()) {
            printf("\n*** --seed-file parsed NOTHING. The run below would measure e2e's own\n"
                   "*** seed and call it the producer's. Refusing.\n");
            return 4;
        }
        // A seed needs a Doppler and SOME code reference. Either form of the latter is
        // fine -- the phase is preferred, the argument is the fallback -- but a file that
        // supplied neither would run on NaN and print a nonsense verdict rather than
        // failing, which is the same trap one level down.
        if (!std::isfinite(sd.doppler_hz)
            || (!std::isfinite(sd.phase_ref_chips) && !std::isfinite(sd.cp_chips))) {
            printf("\n*** --seed-file supplied no doppler_hz, or neither code_phase_chips\n"
                   "*** nor code_phase_at_ref_chips. Nothing to propagate. Refusing.\n");
            return 4;
        }
        if (!std::isfinite(sd.phase_ref_chips))
            sd.phase_ref_chips = -1.0; // the "no phase, use the argument" sentinel
    } else {
        printf("[2] SEED (direct from the detection)\n");
    }
    printf("    ref_hop %lld  ph@ref %.3f  dop %+.4f  cp_rate %.6e  dop_rate %+.4f\n\n", sd.ref_hop,
           sd.phase_ref_chips, sd.doppler_hz, sd.cp_rate, sd.dop_rate);

    // -----------------------------------------------------------------------------------------
    // LEG 3: the tracker. Propagate through the shipped gnss::propagate_seed(), synthesize the
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

        // ---- THE SHIPPED PROPAGATION ARITHMETIC ----
        // trim defaults to 0 (the OPEN loop, which is what every other bench here wants).
        // --trim commands the actuator, so the sign of the response can be measured rather
        // than reasoned about (#51 F3).
        const gnss::SeedPropagation pr =
            gnss::propagate_seed(tbank, sd, h, o.sample_rate, o.f_offset, o.trim_chips);

        // The error is a difference of PHYSICAL PHASES at this window -- never of arguments.
        //
        // The first version of this harness printed wrap(pr.cp - cp204), i.e. it compared the
        // commanded ARGUMENT against the injected one, and reported 8650 chips of error on a
        // chain that was in fact ~0.4 chips off. Two arguments derived at Dopplers 1.7 Hz apart
        // differ by 1.7 * 5095 = 8650 chips BY CONSTRUCTION and describe the same signal; the
        // difference cancels when each is fed to the generator alongside its own Doppler. That
        // is the exact mistake this whole tool exists to catch, made inside the tool, caught in
        // one run by the despread power disagreeing with the arithmetic. Leave the warning here.
        // pr.phase_now is the bank's currency (COMBINED chips); the truth is in wire units.
        const double err = wrap(pr.phase_now / (double)tbank.comb_mult() - truth_phase_at(W), LL);

        auto rec = tbank.channels_hoprate(0, W, o.cp204, o.dop, o.hops_per_record, t_chans, {}, -1);
        // --noise REACHES THE TRACKER LEG TOO (2026-08-04). It did not before: noise was added
        // only to the search snapshot, so every record despread here was noiseless no matter
        // what --noise said, and a sweep over it returned bit-identical numbers at every level.
        // That makes the whole tracker/deep leg untestable against the one regime that matters.
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

    // ---- THE DEEP FOLD, on the SAME statistic the combiner publishes ----
    // Live, deep_snr sits at 10-14 for EVERY satellite across a 9x range of amp_snr, which
    // means the fold is limited by a per-record phase error rather than by noise: coherent_sum
    // estimates its own noise from the records' scatter about the mean phase, so a systematic
    // phase error makes the signal cancel and the statistic saturate at sqrt(N)/sigma_phi.
    // Run it here with NO noise and a perfect seed: if sigma_phi survives that, the floor is in
    // our arithmetic and not on the sky.
    if (prompts.size() >= 4) {
        const int N = (int)prompts.size();
        // DEROTATE FIRST, as the combiner does. Folding raw prompts measures the RAMP, not the
        // residual: at a 5 Hz seed error the raw fold collapses to snr 0.58 while the residual
        // about the ramp is 0.0115 rad, i.e. clean. Comparing that raw number against the live
        // deep_snr (which IS derotated by the rate search) compares two different quantities.
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
        // ... and again after removing a best-fit LINEAR phase ramp, which is what the
        // combiner's rate search does before folding. Any residual after that is not a ramp.
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
        // ⚠️ IS THE UNWRAP EVEN VALID? sig_res unwraps the per-record phase, which assumes the
        // step between records is under pi. When it is not -- and with a real Doppler rate at
        // days of uptime it routinely is not -- the unwrap aliases and sig_res stops being a
        // measurement: it saturates near the uniform-phase value and stays there no matter how
        // much better or worse the arithmetic gets. That is how this line read 1.1-1.2 rad
        // across THREE DECADES of uptime and a 16x improvement showed as nothing (task #52).
        // Say "unresolved" rather than print a number that looks like data.
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

        // ---- [4a] THE PER-RECORD STEP SERIES, the statistic that discriminates on sky ----
        // A smooth model mismatch (a quadratic phase) gives a per-record step that is nearly
        // CONSTANT from record to record: small step rms. A per-record reference error gives
        // steps that vary randomly: an OFFSET error telescopes (lag-1 autocorrelation -0.5), a
        // WALK does not (lag-1 ~0). The sky reads step rms ~0.06 cycles at lag-1 ~-0.1, so
        // this line is what a reproduction must match, not the about-a-ramp residual (which
        // cannot tell a quadratic from noise).
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
