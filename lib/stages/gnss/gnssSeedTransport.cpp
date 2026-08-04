#ifdef _OPENMP
#include <omp.h>
#endif
#include "gnssSeedTransport.hpp"

#include "gnssChannelizedDespread.hpp" // for channelized_despread

#include <algorithm>
#include <cmath>
#include <vector>

namespace gnss {

double refine_peak(const ChannelizedReplicaBank& bank, int prn_index,
                   const std::vector<std::vector<std::complex<float>>>& data,
                   const std::vector<int>& cov_global, double coarse_cp,
                   const AcquisitionSurface& dims, int nh_phase, double dop, long long anchor,
                   int n_hops, double sample_rate, int refine_span, int refine_step,
                   int n_threads) {
    const double cps = bank.chip_rate_hz() / sample_rate;
    // Build the cumulative channel filter ONCE: the Doppler is fixed across the scan, so the
    // slow half of the hop-rate generator (O(num_taps * fft_len) per channel) is Doppler-only
    // and hoists out. channels_hoprate() would rebuild it every step.
    const ChannelizedReplicaBank::HopRateFilter rf = bank.hoprate_filter(cov_global, dop);
    const int step = std::max(1, refine_step);

    // The scan must simply REACH the true peak; the objective then picks it out unaided. Its
    // shape was dumped over a full hop (e2e --dump-refine) rather than argued about:
    //
    //   off  +5376 samp  +17.186 chips   pw 1.5124e-02   <- truth
    //   off  +1280 samp   +4.092 chips   pw 1.2818e-02   <- grating lobe, 4096 samples away
    //
    // The comb DOES put grating lobes every s_stored samples (the covering channels share a
    // factor with sph, so the fine-lag response is periodic -- see AcquisitionSurface), but
    // they are NOT equal: the true lobe wins by ~15%, so any scan that covers it is correct.
    // The aggregator ran `refine_span: 4096`, could not reach +17.19 chips, and settled on the
    // +4.09 lobe -- an error of exactly 13.09 chips, which is what 4 of 15 noiseless injections
    // showed. It was never an ambiguity to be resolved by cleverness, just a scan that stopped
    // too soon. (Two failed fixes are buried here: evaluating candidates at multiples of
    // s_stored gets 13.198 chips wrong, because the true lobe sits at 4096+1280, not at a
    // multiple of 4096. Reach the peak; do not model it.)
    //
    // So: scan the FULL +-refine_span (default fft_len = one whole hop, which is the entire
    // ambiguity), and pay for it with threads instead of with cleverness. The lobes are only
    // ~sph/(comb span in bins) = 157 samples wide, so the step cannot be opened up to make a
    // coarse-to-fine pass work -- it would step over the lobe it is looking for.
    const int span = std::max(step, refine_span);
    const int n_eval = 2 * span / step + 1;
    (void)dims; // (shape is informational now; the scan no longer depends on s_stored)

    std::vector<double> pw((size_t)n_eval, -1.0);
    // PER-THREAD SCRATCH, allocated once. hoprate_stream returns ~2.6 MB by value, so the
    // by-value form had every one of these threads calling the allocator inside the loop, on
    // every detection. The same churn took the ms-split refine from 89 s to 874 s when it was
    // found in that path (docs/CHORD_GNSS_MS_SPLIT_SEARCH.md section 8); the ordinary refine
    // still had it. Sized by thread id, so no two threads share a buffer.
    const int nthr = n_threads > 0 ? n_threads : 1;
    std::vector<std::vector<std::vector<std::complex<float>>>> scratch((size_t)nthr);
    // n_hops may be SHORTER than the data (a caller trading integration for speed -- the refine
    // only has to localise a peak that already cleared detection). channelized_despread
    // requires matching lengths, so take the leading n_hops of each channel once, here, rather
    // than per trial phase.
    std::vector<std::vector<std::complex<float>>> dview;
    const std::vector<std::vector<std::complex<float>>>* dp = &data;
    if (!data.empty() && (int)data[0].size() > n_hops) {
        dview.resize(data.size());
        for (size_t c = 0; c < data.size(); ++c)
            dview[c].assign(data[c].begin(), data[c].begin() + n_hops);
        dp = &dview;
    }
#pragma omp parallel for schedule(static) num_threads(nthr)
    for (int e = 0; e < n_eval; ++e) {
        const double cp = coarse_cp + (double)(-span + e * step) * cps;
        // Same secondary-code alignment the peak was found at, or the refine despreads an
        // overlay-blind replica against an overlay-aligned peak and walks off it.
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& r = scratch[(size_t)std::min(tid, nthr - 1)];
        bank.hoprate_stream_into(rf, prn_index, anchor, cp, dop, n_hops, {}, nh_phase, r);
        pw[(size_t)e] = std::norm(channelized_despread(*dp, r).amplitude);
    }
    // Reduce serially so the result does not depend on thread scheduling -- a tie broken by
    // whichever thread finished first is exactly the kind of irreproducibility that makes a
    // bug look intermittent.
    int best_e = 0;
    for (int e = 1; e < n_eval; ++e)
        if (pw[(size_t)e] > pw[(size_t)best_e])
            best_e = e;
    return coarse_cp + (double)(-span + best_e * step) * cps;
}

DetectionPhase detection_phase(const ChannelizedReplicaBank& bank, double best_cp, int best_nh,
                               int n_nh, double dop, long long snap_start_hop, long long anchor,
                               double sample_rate, long peak_tau_samples) {
    DetectionPhase out;
    const int fft_len = bank.fft_len();
    const int Mp = bank.repl_period_hops();
    const double cps = bank.chip_rate_hz() / sample_rate;
    const double L = (double)bank.code_length();

    // best_cp is the satellite code phase at the snapshot's absolute start
    // (S_snap = snap_start_hop * fft_len). Reference it to absolute sample 0 so a consumer on
    // the same hop*2N grid can seed directly: cp0 = best_cp - S_snap*chip_per_sample (mod L).
    // Reduce the hop offset modulo the replica period first to keep float precision.
    const double off = std::fmod((double)(snap_start_hop % Mp) * (double)fft_len * cps, L);
    // Code-Doppler drift of the reference over the FULL absolute snapshot start (NOT
    // Mp-periodic, so use the full snap_start_hop): makes cp0 the true sample-0 phase, matching
    // the feed-forward in ChannelizedReplicaBank, so a stale seed no longer drifts the cp.
    const double drift = std::fmod((double)snap_start_hop * (double)fft_len * cps
                                       * (bank.code_doppler_sign * dop / bank.carrier_hz()),
                                   L);
    out.cp0 = std::fmod(best_cp - off - drift, L);
    if (out.cp0 < 0.0)
        out.cp0 += L;

    if (n_nh <= 1 || best_nh < 0)
        return out;

    // LONG-CODE PHASE, mod n_nh*L -- the phase a tracker despreading the overlaid code actually
    // needs. Everything above is reduced mod L, which is right for the primary and DESTROYS the
    // overlay period: a phase known only mod L leaves the tracker a 1-in-n_nh guess, and no
    // constant correction can supply the missing period because it is not constant. Two
    // separate reductions have to change, not one:
    //   * best_cp lifts into the long space by the alignment the acquire MEASURED;
    //   * the `% Mp` shortcut is valid ONLY mod L. One replica period is Mp*fft_len*cps = 16 L
    //     exactly at CHORD -- 0 mod L, but NOT 0 mod 20 L -- so the long form needs the FULL
    //     absolute advance. That product reaches ~7e12 chips and must stay good to a chip,
    //     hence long double (double's ulp there is ~1e-3 chips, fine, but the margin is free).
    // THE CORRELATION LAG'S WHOLE-PERIOD COUNT (2026-08-02). Measured by injection, 22/22.
    //
    // The coarse lag runs over ONE REPLICA PERIOD, Mp*fft_len*cps = 16 primary code periods at
    // CHORD -- but `best_cp` is that lag already reduced mod ONE period, so the count of whole
    // periods inside the lag is discarded before anything downstream can see it. The overlay is
    // 20 periods and 16 is not 0 mod 20, so no `best_nh` alone can carry it: the reported period
    // came out (best_nh - 4) mod 20 in every injection, i.e. a pure function of the alignment,
    // and was wrong by -2..+3 in 8 of the first 15 runs. That is the "period randomised every
    // pass (+1/+3/-4/+3/+3/-1/-2)" seen on sky for a week; it was never random, it is this count,
    // which changes every pass because the peak lands in a different hop.
    //
    // The lag is only determined modulo the 16-period replica span, so its period count must be
    // read as a SIGNED residual: fold [0,16) into (-8,+8]. That is not a free choice -- it is
    // measured. Across 22 injections spanning the full 204600-chip range, |folded| never
    // exceeded 3, because the acquire searches overlay alignment and coarse lag JOINTLY and the
    // winning pair always leaves a small residual lag. The raw (unfolded) count is wrong for
    // every negative-lag case and would put the seed up to 15 periods out.
    //
    // `peak_tau_samples` has been sitting in AcquisitionResult all along; the information was
    // never lost, only never carried across the stage boundary.
    const double tau_chips = (double)peak_tau_samples * bank.chip_rate_hz() / sample_rate;
    const int lag_span_periods =
        (int)std::llround((double)Mp * (double)fft_len * cps / L); // 16 at CHORD
    int lag_periods = 0;
    if (lag_span_periods > 0) {
        const int raw = (int)std::floor(tau_chips / L) % lag_span_periods;
        lag_periods = ((raw % lag_span_periods) + lag_span_periods) % lag_span_periods;
        if (lag_periods > lag_span_periods / 2)
            lag_periods -= lag_span_periods; // fold to the signed residual
    }
    // The lift into the long code: the alignment the acquire measured, PLUS the lag's period
    // count. Both are in units of one primary period.
    const double lift = ((double)best_nh + (double)lag_periods) * L;

    const long double LL = (long double)L * (long double)n_nh;
    const long double adv =
        (long double)snap_start_hop * (long double)fft_len * (long double)cps;
    const long double off_l = adv - std::floor(adv / LL) * LL;
    const double drift_l = std::fmod((double)snap_start_hop * (double)fft_len * cps
                                         * (bank.code_doppler_sign * dop / bank.carrier_hz()),
                                     (double)LL);
    double cpl = std::fmod(best_cp + lift - (double)off_l - drift_l, (double)LL);
    if (cpl < 0.0)
        cpl += (double)LL;
    out.cp_long = cpl;

    // ...and the same phase referenced to the SNAPSHOT, which is what should actually be
    // transported. best_cp is the argument of a replica at `anchor` that matches the data at
    // the snapshot, so the data's phase there is just best_cp lifted plus the ANCHOR's own
    // advance -- a fixed ~163732 chips whose Doppler sensitivity is 1e-4 chips/Hz, against
    // 5903 chips/Hz for the sample-0 route above. Same information, seven orders better
    // conditioned. (Done by hand rather than through bank.phase_from_arg() because THIS bank
    // reduces mod L and would throw the overlay period away.)
    const long double n_anc = (long double)anchor + (long double)fft_len - 1.0L;
    const long double cps_d =
        (long double)cps * (1.0L + (long double)(bank.code_doppler_sign * dop / bank.carrier_hz()));
    const long double a_adv = n_anc * cps_d;
    double phr = std::fmod(best_cp + lift
                               + (double)(a_adv - std::floor(a_adv / LL) * LL),
                           (double)LL);
    if (phr < 0.0)
        phr += (double)LL;
    out.cp_at_ref = phr;
    return out;
}

SeedPropagation propagate_seed(const ChannelizedReplicaBank& bank, const SeedState& sd,
                               long long hop0, double sample_rate, double f_offset_hz,
                               double trim_chips) {
    SeedPropagation out;
    const int fft_len = bank.fft_len();
    const double dh = (double)(hop0 - sd.ref_hop);
    const double dt = dh * (double)fft_len / sample_rate;
    out.doppler_hz = sd.doppler_hz + sd.dop_rate * dt;

    // The phase advances at the TRUE code rate: nominal scaled by the code Doppler. Dropping
    // the scaling loses chip_rate*dop/(carrier*hops_per_sec) chips per hop -- 1.05e-4 at dop
    // 2350, so 41 chips on a seed only 2 s old and thousands on a minute-old one. It is
    // invisible if the seed's `code_phase_rate` happens to carry that same term (which is what
    // hand seeding did, and why hand seeds locked), but the broker's convention -- airspy's,
    // and the right one -- is that code_phase_rate is a RESIDUAL and the geometry is fed
    // forward here, exactly as the replica generator feeds it forward through cps(dop).
    const double chips_per_hop =
        bank.chip_rate_hz() * (double)fft_len / sample_rate
        * (1.0 + bank.code_doppler_sign * sd.doppler_hz / bank.carrier_hz());
    const double quad = 0.5 * (bank.chip_rate_hz() / f_offset_hz) * sd.dop_rate * dt * dt;

    out.phase_ref = (sd.phase_ref_chips >= 0.0)
                        ? sd.phase_ref_chips // transported as a phase: no back-reference at all
                        : bank.phase_from_arg(sd.cp_chips, sd.ref_hop * (long long)fft_len,
                                              sd.doppler_hz);
    out.phase_now = out.phase_ref + (chips_per_hop + sd.cp_rate) * dh + quad + trim_chips;
    out.cp = bank.arg_from_phase(out.phase_now, hop0 * (long long)fft_len, out.doppler_hz);
    return out;
}

} // namespace gnss
