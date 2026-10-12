#ifdef _OPENMP
#include <omp.h>
#endif
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSeedTransport.hpp"
#ifdef GNSS_CUDA
#include "GnssCudaDespread.hpp" // for refine_peak_cuda
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace gnss {

double refine_peak(const ChannelizedReplicaBank& bank, int prn_index,
                   const std::vector<std::vector<std::complex<float>>>& data,
                   const std::vector<int>& cov_global, double coarse_cp,
                   const AcquisitionSurface& dims, int nh_phase, double dop, long long anchor,
                   int n_hops, double sample_rate, int refine_span, int refine_step,
                   int n_threads) {
    const double cps = bank.chip_rate_hz() / sample_rate;
    // The Doppler is fixed across the scan, so the cumulative channel filter (the
    // O(num_taps * fft_len) half of the hop-rate generator) is built once, not per trial.
    const ChannelizedReplicaBank::HopRateFilter rf = bank.hoprate_filter(cov_global, dop);
    const int step = std::max(1, refine_step);

    // Scan the full +-refine_span: the true lobe is the highest of the grating lobes, so the
    // scan only has to reach it (see the header).
    const int span = std::max(step, refine_span);
    const int n_eval = 2 * span / step + 1;
    (void)dims;

    std::vector<double> pw((size_t)n_eval, -1.0);
    // Per-thread replica scratch, indexed by thread id, so the loop does not allocate a
    // multi-MB replica per trial.
    const int nthr = n_threads > 0 ? n_threads : 1;
    std::vector<std::vector<std::vector<std::complex<float>>>> scratch((size_t)nthr);
    // n_hops may be shorter than the data (the refine only localizes a peak that already passed
    // detection). channelized_despread needs matching lengths, so take the leading n_hops of
    // each channel once rather than per trial.
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
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& r = scratch[(size_t)std::min(tid, nthr - 1)];
        bank.hoprate_stream_into(rf, prn_index, anchor, cp, dop, n_hops, {}, nh_phase, r);
        pw[(size_t)e] = std::norm(channelized_despread(*dp, r).amplitude);
    }
    // Serial reduction, first strict maximum, so a tie does not depend on thread scheduling.
    int best_e = 0;
    for (int e = 1; e < n_eval; ++e)
        if (pw[(size_t)e] > pw[(size_t)best_e])
            best_e = e;
    return coarse_cp + (double)(-span + best_e * step) * cps;
}

#ifdef GNSS_CUDA
double refine_peak_cuda(std::vector<CudaRefineGroup>& groups, const ChannelizedReplicaBank& bank,
                        int prn_index, const std::complex<float>* window, int window_stride,
                        long long anchor, double coarse_cp, double dop, double sample_rate,
                        int refine_span, int refine_step) {
    // Same scan as refine_peak (cps, step/span clamping, n_eval, ascending first-strict-max
    // tie-break), so the two paths differ only in the despread.
    const double cps = bank.chip_rate_hz() / sample_rate;
    const int step = std::max(1, refine_step);
    const int span = std::max(step, refine_span);
    const int n_eval = 2 * span / step + 1;
    if (groups.empty())
        return coarse_cp;

    // One engine per group of <= 64 channels (DespreadJob::chan_mask is a uint64_t).
    // correlation and replica_energy (sum_c sum_m |R|^2) both add over channels, so the group
    // totals recombine exactly. Each group's channels are de-interleaved into its own window.
    for (CudaRefineGroup& g : groups) {
        const int ng = (int)g.local.size();
        g.stage.resize((size_t)g.n_hops * (size_t)ng);
        for (int m = 0; m < g.n_hops; ++m)
            for (int i = 0; i < ng; ++i)
                g.stage[(size_t)m * ng + i] =
                    window[(size_t)m * window_stride + g.local[(size_t)i]];
        g.gpu->upload_window(g.stage.data(), anchor);
    }

    // Three trials per Spec: with the prompt on the middle trial and the spacing one scan step,
    // Spec gg's {early, prompt, late} are trials {3gg, 3gg+1, 3gg+2}. Chunked to the engine's
    // job arena (GnssCudaDespread::max_batch_specs).
    int cap = 1 << 30;
    for (const CudaRefineGroup& g : groups)
        cap = std::min(cap, std::max(1, g.gpu->max_batch_specs()));
    const int n_group = (n_eval + 2) / 3;

    std::vector<std::complex<double>> corr((size_t)n_eval, {0.0, 0.0});
    std::vector<double> energy((size_t)n_eval, 0.0);

    for (int g0 = 0; g0 < n_group; g0 += cap) {
        const int ng2 = std::min(cap, n_group - g0);
        for (CudaRefineGroup& grp : groups) {
            std::vector<::GnssCudaDespread::Spec> specs;
            specs.reserve((size_t)ng2);
            std::vector<int> all(grp.local.size());
            for (size_t i = 0; i < grp.local.size(); ++i)
                all[i] = (int)i; // every channel of this group is covering, in group-local order
            for (int gg = g0; gg < g0 + ng2; ++gg) {
                ::GnssCudaDespread::Spec sp;
                sp.p = prn_index;
                sp.cp_seed = coarse_cp + (double)(-span + (3 * gg + 1) * step) * cps;
                sp.spacing_chips = (double)step * cps;
                sp.doppler_hz = dop;
                sp.covering = all;
                specs.push_back(sp);
            }
            const std::vector<std::array<DespreadResult, 3>> res = grp.gpu->despread_batch(specs);
            for (size_t k = 0; k < res.size(); ++k)
                for (int j = 0; j < 3; ++j) {
                    const int e = (g0 + (int)k) * 3 + j;
                    if (e >= n_eval) // the last group's padding: computed, never scored
                        break;
                    corr[(size_t)e] += res[k][(size_t)j].correlation;
                    energy[(size_t)e] += res[k][(size_t)j].replica_energy;
                }
        }
    }

    int best_e = 0;
    double best_pw = -1.0;
    for (int e = 0; e < n_eval; ++e) {
        const double en = energy[(size_t)e];
        const double pw = (en > 0.0) ? std::norm(corr[(size_t)e] / en) : 0.0;
        if (pw > best_pw) { // strict >, ascending e: same first-max rule as refine_peak
            best_pw = pw;
            best_e = e;
        }
    }
    return coarse_cp + (double)(-span + best_e * step) * cps;
}
#endif // GNSS_CUDA

DetectionPhase detection_phase(const ChannelizedReplicaBank& bank, double best_cp, int best_nh,
                               int n_nh, double dop, long long snap_start_hop, long long anchor,
                               double sample_rate, long peak_tau_samples) {
    DetectionPhase out;
    const int fft_len = bank.fft_len();
    const int Mp = bank.repl_period_hops();
    const double cps = bank.chip_rate_hz() / sample_rate;
    const double L = (double)bank.code_length();

    // best_cp is the code phase at the snapshot's start (S_snap = snap_start_hop * fft_len).
    // Refer it to absolute sample 0: cp0 = best_cp - S_snap*chip_per_sample (mod L). The hop
    // offset is reduced modulo the replica period first (one period advances 0 mod L) to keep
    // precision.
    const double off = std::fmod((double)(snap_start_hop % Mp) * (double)fft_len * cps, L);
    // Code-Doppler advance over the full snapshot start (not Mp-periodic, so not reduced),
    // matching the code-Doppler feed-forward in ChannelizedReplicaBank.
    const double drift = std::fmod((double)snap_start_hop * (double)fft_len * cps
                                       * (bank.code_doppler_sign * dop / bank.carrier_hz()),
                                   L);
    out.cp0 = std::fmod(best_cp - off - drift, L);
    if (out.cp0 < 0.0)
        out.cp0 += L;

    if (n_nh <= 1 || best_nh < 0)
        return out;

    // Long-code phase, mod n_nh*L: the phase a tracker despreading the overlaid code needs. A
    // phase known only mod L leaves the overlay period unknown. Two reductions differ from the
    // primary-code ones above:
    //   * best_cp is lifted into the long code by the alignment the acquire measured, plus the
    //     whole primary periods inside the correlation lag;
    //   * the `% Mp` shortcut is valid only mod L. One replica period is Mp*fft_len*cps = 16 L
    //     at CHORD, 0 mod L but not 0 mod 20 L, so the long form needs the full absolute
    //     advance. That product reaches ~7e12 chips and must stay good to a chip; long double
    //     keeps a margin over double's ~1e-3-chip ulp there.
    //
    // The lag spans one replica period (16 primary periods at CHORD), but best_cp is reduced
    // mod one period, so its whole-period count is taken from peak_tau_samples. The overlay is
    // 20 periods and 16 is not 0 mod 20, so best_nh alone cannot carry it. The lag is known
    // only modulo the replica span, so the count is read as a signed residual, [0,16) folded to
    // (-8,+8]: the acquire searches alignment and lag jointly, and the winning pair leaves a
    // small residual lag. The unfolded count would be wrong for every negative lag.
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
    // Lift into the long code: measured alignment plus the lag's period count, in primary
    // periods.
    const double lift = ((double)best_nh + (double)lag_periods) * L;

    const long double LL = (long double)L * (long double)n_nh;
    const long double adv = (long double)snap_start_hop * (long double)fft_len * (long double)cps;
    const long double off_l = adv - std::floor(adv / LL) * LL;
    const double drift_l = std::fmod((double)snap_start_hop * (double)fft_len * cps
                                         * (bank.code_doppler_sign * dop / bank.carrier_hz()),
                                     (double)LL);
    double cpl = std::fmod(best_cp + lift - (double)off_l - drift_l, (double)LL);
    if (cpl < 0.0)
        cpl += (double)LL;
    out.cp_long = cpl;

    // The same phase referred to the snapshot, which is what should be transported. best_cp is
    // the argument of a replica at `anchor` that matches the data at the snapshot, so the
    // data's phase there is best_cp lifted plus the anchor's own advance: a fixed ~1.6e5 chips
    // (L5 at CHORD) with a Doppler sensitivity of ~1e-4 chips/Hz, against thousands of
    // chips/Hz for cp0. Done here rather than with bank.phase_from_arg(), which reduces mod L.
    //
    // The epoch is the hop's last sample (anchor + fft_len - 1), the reference every phase on
    // the C++ side shares (ChannelizedReplicaBank::window_advance_chips, phase_from_arg,
    // arg_from_phase, propagate_seed and the e2e harness), so cp_at_ref is comparable with the
    // tracker's own propagation. cp0 above is referred back from the hop's first sample, so a
    // phase rebuilt from cp0 at ref_hop sits (fft_len-1)*cps below cp_at_ref by construction;
    // consumers that mix the two fields convert (gnss_broker/fits.py seed_phase_at_ref).
    const long double n_anc = (long double)anchor + (long double)fft_len - 1.0L;
    const long double cps_d =
        (long double)cps * (1.0L + (long double)(bank.code_doppler_sign * dop / bank.carrier_hz()));
    const long double a_adv = n_anc * cps_d;
    double phr =
        std::fmod(best_cp + lift + (double)(a_adv - std::floor(a_adv / LL) * LL), (double)LL);
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

    // The phase advances at the true code rate: nominal scaled by the code Doppler. The seed's
    // cp_rate is a residual; the geometric code Doppler is fed forward here, as the replica
    // generator feeds it forward through cps(dop).
    const double chips_per_hop =
        // eff_chip_rate, not chip_rate_hz: phase_now goes to arg_from_phase, which works in
        // combined chips (it subtracts window_advance_chips, built from the combined rate, and
        // divides by comb_mult). The two rates differ only when comb_mult > 1 (TDM and BOC).
        bank.eff_chip_rate() * (double)fft_len / sample_rate
        * (1.0 + bank.code_doppler_sign * sd.doppler_hz / bank.carrier_hz());
    const double quad = 0.5 * (bank.chip_rate_hz() / f_offset_hz) * sd.dop_rate * dt * dt;

    // The seed's phase_ref_chips is in component chips (CM chips for GPS_L2C_CM), the unit of
    // code_phase_chips, cp_at_ref and the broker (gnss_broker/fits.py seed_phase_at_ref).
    // phase_from_arg returns combined chips (comb_mult * arg + advance), as are chips_per_hop
    // and arg_from_phase's input, so the phase branch scales by comb_mult on entry. cp_rate,
    // quad and trim_chips are added unscaled.
    out.phase_ref =
        (sd.phase_ref_chips >= 0.0)
            ? (double)bank.comb_mult() * sd.phase_ref_chips
            : bank.phase_from_arg(sd.cp_chips, sd.ref_hop * (long long)fft_len, sd.doppler_hz);
    out.phase_now = out.phase_ref + (chips_per_hop + sd.cp_rate) * dh + quad + trim_chips;
    out.cp = bank.arg_from_phase(out.phase_now, hop0 * (long long)fft_len, out.doppler_hz);
    return out;
}

} // namespace gnss
