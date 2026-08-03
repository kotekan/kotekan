#include "gnssChannelizedAcquire.hpp"

#include "fftwPlannerLock.hpp" // for fftw_planner_mutex

#include <algorithm> // for max
#include <cmath>   // for cos, sin, fmod, M_PI, round
#include <chrono>  // for the opt-in ms-split stage timers
#include <cstdio>  // for fprintf (ditto)
#include <cstdlib> // for abs, getenv
#include <mutex>   // for lock_guard
#include <numeric> // for gcd
#include <thread>  // for thread (aggregate d-parallelism)

namespace gnss {

using cd = std::complex<double>;
using cf = std::complex<float>;

void AcquireWorkspace::ensure(int n) {
    if (n == Mp)
        return;
    std::lock_guard<std::mutex> lk(fftw_planner_mutex());
    if (fwd)
        fftwf_destroy_plan(fwd);
    if (inv)
        fftwf_destroy_plan(inv);
    if (in)
        fftwf_free(in);
    if (out)
        fftwf_free(out);
    Mp = n;
    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Mp);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Mp);
    fwd = fftwf_plan_dft_1d(Mp, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    inv = fftwf_plan_dft_1d(Mp, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
}

AcquireWorkspace::~AcquireWorkspace() {
    std::lock_guard<std::mutex> lk(fftw_planner_mutex());
    if (fwd)
        fftwf_destroy_plan(fwd);
    if (inv)
        fftwf_destroy_plan(inv);
    if (in)
        fftwf_free(in);
    if (out)
        fftwf_free(out);
}

std::vector<std::vector<cf>>
channel_correlate(const std::vector<cf>& data, const std::vector<cf>& repl0,
                  const std::vector<double>& doppler_grid, double sample_rate, int samples_per_hop,
                  AcquireWorkspace& ws) {
    const int M = (int)data.size();   // data window length, hops
    const int Mp = (int)repl0.size(); // replica hop-period (>= M)
    const int nd = (int)doppler_grid.size();
    const int sph = (samples_per_hop > 0) ? samples_per_hop : Mp;

    ws.ensure(Mp);
    cf* IN = reinterpret_cast<cf*>(ws.in);
    cf* OUT = reinterpret_cast<cf*>(ws.out);

    // conj(FFT(repl0)) -- Doppler-independent, computed once.
    std::vector<cf> conjB(Mp);
    for (int k = 0; k < Mp; ++k)
        IN[k] = repl0[k];
    fftwf_execute(ws.fwd);
    for (int k = 0; k < Mp; ++k)
        conjB[k] = std::conj(OUT[k]);

    const float inv_mp = 1.0f / (float)Mp; // FFTW backward is unnormalized
    std::vector<std::vector<cf>> P(nd, std::vector<cf>(Mp));

    // BIN-ALIGNED DOPPLER GRID: wiping the data by f_d multiplies it by e^{-i 2pi f_d m sph/Fs},
    // which is a pure cyclic SHIFT of its spectrum -- exactly, with no approximation -- whenever
    // f_d is a whole number of this transform's own bins, Fs/(Mp*sph) = 62.5 Hz at CHORD. Then
    // ONE forward transform serves every Doppler trial instead of one per trial, halving the
    // per-channel correlation, which is the dominant cost once the surface is decimated.
    //
    // Detected rather than configured: a caller whose doppler_step lands on the bin spacing gets
    // it for free, and one whose does not (the historical 31.25 Hz = HALF a bin) keeps the
    // original path bit-for-bit. Halving the grid costs Doppler resolution the parabolic refine
    // in channelized_peak already recovers to ~step/20.
    const double bin_hz = (Mp > 0 && sph > 0) ? sample_rate / ((double)Mp * (double)sph) : 0.0;
    std::vector<int> bshift((size_t)nd);
    bool on_bins = (nd > 0 && bin_hz > 0.0);
    for (int d = 0; d < nd && on_bins; ++d) {
        const double b = doppler_grid[d] / bin_hz;
        const double rb = std::round(b);
        if (std::fabs(b - rb) > 1e-6 * std::max(1.0, std::fabs(rb)))
            on_bins = false;
        else
            bshift[(size_t)d] = (int)((((long long)rb % Mp) + Mp) % Mp);
    }
    if (on_bins) {
        std::vector<cf> A(Mp);
        for (int m = 0; m < M; ++m)
            IN[m] = data[m];
        for (int m = M; m < Mp; ++m)
            IN[m] = cf(0.0f, 0.0f);
        fftwf_execute(ws.fwd);
        for (int k = 0; k < Mp; ++k)
            A[k] = OUT[k];
        for (int d = 0; d < nd; ++d) {
            const int b = bshift[(size_t)d];
            for (int k = 0; k < Mp - b; ++k)
                IN[k] = A[k + b] * conjB[k];
            for (int k = Mp - b; k < Mp; ++k)
                IN[k] = A[k + b - Mp] * conjB[k];
            fftwf_execute(ws.inv);
            for (int q = 0; q < Mp; ++q)
                P[d][q] = OUT[q] * inv_mp;
        }
        return P;
    }

    std::vector<cf> wipe(M);
    for (int d = 0; d < nd; ++d) {
        // Per-hop Doppler wipe e^{-i 2pi fd (m*sph)/Fs} via a phasor recurrence.
        const double dphi = -2.0 * M_PI * doppler_grid[d] * (double)sph / sample_rate;
        cd rot(std::cos(dphi), std::sin(dphi));
        cd ph(1.0, 0.0);
        for (int m = 0; m < M; ++m) {
            wipe[m] = cf((float)ph.real(), (float)ph.imag());
            ph *= rot;
        }
        for (int m = 0; m < M; ++m)
            IN[m] = data[m] * wipe[m];
        for (int m = M; m < Mp; ++m)
            IN[m] = cf(0.0f, 0.0f);
        fftwf_execute(ws.fwd); // OUT = FFT(wiped0) = A
        for (int k = 0; k < Mp; ++k)
            IN[k] = OUT[k] * conjB[k]; // A * conj(B)
        fftwf_execute(ws.inv);         // OUT = Mp * P_c(q)
        for (int q = 0; q < Mp; ++q)
            P[d][q] = OUT[q] * inv_mp;
    }
    return P;
}

AcquisitionSurface
aggregate_accumulate(const std::vector<std::vector<std::vector<cf>>>& P,
                     const std::vector<int>& chan_freq, int samples_per_hop,
                     std::vector<double>& surf, int n_threads, int fine_step) {
    const int nc = (int)P.size();
    const int nd = nc ? (int)P[0].size() : 0;
    const int Mp = (nc && nd) ? (int)P[0][0].size() : 0;
    const int sph = samples_per_hop;

    // The fine-lag axis is exactly periodic, so only store one period of it.
    //
    // D[s] = sum_ci P_ci e^{+i 2pi f_ci s / sph}. Factor out the first channel's ramp: it has
    // unit modulus, so |D[s]|^2 -- all this function keeps -- depends only on the channel
    // DIFFERENCES f_ci - f_c0. If those share a common factor g with sph, then |D[s]|^2 repeats
    // with period sph/g and the remaining g-1 copies are bit-for-bit identical.
    //
    // This is not a corner case, it is the normal situation for a combed band, and g is large:
    // CHORD's node comb is stride 8, split across two GPUs, so each search instance sees
    // stride 16 and g = 16 -- 15/16 of both the compute and the 2 GB surface was recomputing
    // numbers it already had. (Airspy's covering set is stride 2, so g = 2.)
    //
    // It is also the SAME redundancy as the known code-phase ambiguity: period sph/g samples is
    // 1/(g * channel_width) in time, = 320 ns for g=16 here, the comb ambiguity resolved by the
    // BRDC model with ~16x margin. So nothing is lost that was ever independent information.
    //
    // Behaviour is preserved EXACTLY, not approximately:
    //   - channelized_peak takes the first STRICT maximum, so among g identical copies it
    //     already returned the s in [0, sph/g); the peak cell is unchanged.
    //   - its noise floor is a mean over the surface, and dropping g uniform copies of every
    //     value leaves the mean identical -- so the reported SNR is unchanged too.
    int g = 0;
    for (int ci = 1; ci < nc; ++ci)
        g = std::gcd(g, std::abs(chan_freq[ci] - chan_freq[0]));
    // g == 0 means a single covering channel: |D|^2 is then flat in s (one unit-modulus ramp),
    // so one column suffices. std::gcd(0, sph) == sph gives exactly that.
    const int s_stored = (nc > 0 && sph > 0) ? sph / std::gcd(g, sph) : sph;
    // Decimate the fine axis: it is sampled every sample but its lobe is sph/(comb span)
    // samples wide (157 here), so the stored width is ~157x finer than the peak it resolves.
    const int s_step = std::max(1, fine_step);
    const AcquisitionSurface dims{nd, Mp, sph, s_stored, s_step};
    const int s_cols = dims.fine();
    if ((long)surf.size() != dims.size())
        surf.assign(dims.size(), 0.0);
    if (nc == 0)
        return dims;

    // Cross-channel fine-lag phase ramp e^{+i 2pi f_c s / sph}; SoA float so the
    // inner s-loop vectorizes (std::complex<float> does not auto-vectorize).
    // NOTE the ramp argument still divides by the FULL sph -- s_stored bounds the range of s,
    // it does not rescale the frequency.
    std::vector<std::vector<float>> rampRe(nc, std::vector<float>(s_cols));
    std::vector<std::vector<float>> rampIm(nc, std::vector<float>(s_cols));
    for (int ci = 0; ci < nc; ++ci) {
        const int fc = chan_freq[ci];
        for (int i = 0; i < s_cols; ++i) {
            const double a = 2.0 * M_PI * fc * (double)(i * s_step) / sph;
            rampRe[ci][i] = (float)std::cos(a);
            rampIm[ci][i] = (float)std::sin(a);
        }
    }

    // Doppler bins write DISJOINT surface slices, so the d-loop is trivially parallel. This
    // matters at aggregator scale, where the aggregate is the whole cost of a pass: 27 union
    // channels x 4096 stored lags x 11 Doppler bins is ~3e10 flops per window, ~10 s on one
    // core -- which multiplied by windows x NH alignments x PRNs is HOURS per pass. n_threads=1
    // (the default everywhere it is not explicitly configured) runs the identical serial loop.
    // Split the (Doppler x coarse-lag) PRODUCT, not the Doppler axis alone. Each (d,q) owns one
    // surface row, so the write is disjoint either way -- but the d axis is short exactly when
    // the search is working well (the broker's hints narrow it to a handful of bins), which
    // capped the old split at min(n_threads, n_dop) and left most cores idle in the case that
    // matters. n_rows = n_dop*Mp is tens of thousands, so every core gets work regardless.
    const long n_rows = (long)nd * Mp;
    const auto run_rows = [&](long r0, long r1) {
        std::vector<float> Dre(s_cols), Dim(s_cols);
        for (long r = r0; r < r1; ++r) {
            const int d = (int)(r / Mp), q = (int)(r % Mp);
            for (int s = 0; s < s_cols; ++s) {
                Dre[s] = 0.0f;
                Dim[s] = 0.0f;
            }
            for (int ci = 0; ci < nc; ++ci) {
                const float pre = P[ci][d][q].real(), pim = P[ci][d][q].imag();
                const float* rr = rampRe[ci].data();
                const float* ri = rampIm[ci].data();
                for (int s = 0; s < s_cols; ++s) {
                    Dre[s] += pre * rr[s] - pim * ri[s];
                    Dim[s] += pre * ri[s] + pim * rr[s];
                }
            }
            double* row = surf.data() + r * s_cols;
            for (int s = 0; s < s_cols; ++s)
                row[s] += (double)(Dre[s] * Dre[s] + Dim[s] * Dim[s]);
        }
    };
    const int nt = (int)std::max(1L, std::min((long)std::max(1, n_threads), n_rows));
    if (nt == 1) {
        run_rows(0, n_rows);
    } else {
        std::vector<std::thread> pool;
        for (int t = 0; t < nt; ++t) {
            const long r0 = n_rows * t / nt, r1 = n_rows * (t + 1) / nt;
            if (r1 > r0)
                pool.emplace_back(run_rows, r0, r1);
        }
        for (auto& th : pool)
            th.join();
    }
    return dims;
}

AcquisitionSurface
channelized_accumulate(const std::vector<std::vector<std::complex<float>>>& data_ch,
                       const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                       const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                       double sample_rate, int num_chan, std::vector<double>& surf,
                       AcquireWorkspace& ws, const std::vector<int>& chan_freq, int samples_per_hop,
                       int n_threads, int fine_step) {
    // The decimation (samples per hop): N for a critically-sampled complex bank,
    // 2N for an r2c real-FFT bank. This is now just per-channel correlate (the
    // distributable part) + cross-channel aggregate (see channel_correlate /
    // aggregate_accumulate); the distributed search runs the two halves as
    // separate stages.
    const int sph = (samples_per_hop > 0) ? samples_per_hop : num_chan;
    const int nc = (int)covering.size();
    std::vector<std::vector<std::vector<cf>>> P(nc);
    // Per-channel correlation, in parallel over CHANNELS. Once the fine axis is decimated this
    // is the dominant cost (measured on the archived P32 frame at fine_step 32: 6.9 s here vs
    // 1.9 s of replica generation), and it was entirely serial -- not for any ordering reason,
    // but because every channel shared the one caller-owned FFTW workspace. Each worker gets
    // its OWN workspace; the caller's is reused by worker 0 so the single-threaded path keeps
    // its persistent plans exactly as before. The extra workspaces are function-local, so they
    // re-plan per call: FFTW_ESTIMATE at this length is cheap next to the transforms it saves,
    // and the alternative (a caller-owned pool) would change the signature for every caller.
    const int nt_c = std::max(1, std::min(n_threads, nc));
    if (nt_c == 1) {
        for (int ci = 0; ci < nc; ++ci)
            P[ci] = channel_correlate(data_ch[covering[ci]], repl0_ch[covering[ci]], doppler_grid,
                                      sample_rate, sph, ws);
    } else {
        std::vector<AcquireWorkspace> extra((size_t)nt_c - 1);
        const auto run_ch = [&](int t) {
            AcquireWorkspace& mine = (t == 0) ? ws : extra[(size_t)t - 1];
            for (int ci = t; ci < nc; ci += nt_c)
                P[ci] = channel_correlate(data_ch[covering[ci]], repl0_ch[covering[ci]],
                                          doppler_grid, sample_rate, sph, mine);
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < nt_c; ++t)
            pool.emplace_back(run_ch, t);
        run_ch(0);
        for (auto& th : pool)
            th.join();
    }
    std::vector<int> fc(nc);
    for (int ci = 0; ci < nc; ++ci)
        fc[ci] = chan_freq.empty() ? covering[ci] : chan_freq[ci];
    return aggregate_accumulate(P, fc, sph, surf, n_threads, fine_step);
}

// ---------------------------------------------------------------------------------------------
// Sub-window ("ms-split") accumulation. Header carries the argument; this is the mechanics.
// ---------------------------------------------------------------------------------------------
AcquisitionSurface
ms_split_accumulate(gnss::ChannelizedReplicaBank& bank, int prn_index,
                    const std::vector<std::vector<std::complex<float>>>& data_ch,
                    const std::vector<int>& chan_ids, long long window_start_sample, int sub_hops,
                    int n_sub, const std::vector<double>& doppler_grid, double sample_rate,
                    std::vector<double>& surf, AcquireWorkspace& ws, int fine_step,
                    int n_threads) {
    const int nc = (int)chan_ids.size();
    const int N = sub_hops;
    const int fft_len = bank.fft_len();
    const int sph = fft_len;
    AcquisitionSurface dims{};
    if (nc == 0 || N <= 0 || n_sub <= 0)
        return dims;

    // Stage timers, opt-in via GNSS_MSSPLIT_PROFILE=1. This function was written and validated
    // for CORRECTNESS and never timed, and the design plan's "258x cheaper" is an OP COUNT --
    // measured end to end it came out 5.7x SLOWER than the shipped path. Op counts are not
    // seconds, so the breakdown ships with the function.
    const bool prof = std::getenv("GNSS_MSSPLIT_PROFILE") != nullptr;
    double t_repl = 0.0, t_corr = 0.0, t_agg = 0.0;
    const auto now = [] {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    for (int k = 0; k < n_sub; ++k) {
        const long long W = window_start_sample + (long long)k * N * fft_len;
        // TWO code periods of replica, starting ONE period EARLY, against one period of data.
        // Early start is what makes lag 0 reachable: channel_correlate forms
        // P[q] = sum_m data[m] conj(repl[m - q]), so a replica beginning at W would need
        // negative indices for any positive delay. Beginning at W - N*fft_len puts zero delay
        // at q = N and the whole [0, one period) lag range inside [N, 2N) -- with every one of
        // the N data hops overlapped, which is the point.
        const long long W_repl = W - (long long)N * fft_len;
        double t0 = prof ? now() : 0.0;
        std::vector<std::vector<std::complex<float>>> repl =
            bank.channels_hoprate(prn_index, W_repl, 0.0, 0.0, 2 * N, chan_ids, {}, -1);
        if (prof)
            t_repl += now() - t0;

        // PARALLEL OVER CHANNELS, exactly as channelized_accumulate does it. This loop was
        // serial on the caller's single workspace, which is why the ms-split measured SLOWER
        // than the path it was meant to replace: ~10x fewer ops, run on 1/12 of the machine.
        // Each worker gets its own AcquireWorkspace (worker 0 reuses the caller's, so the
        // single-threaded path keeps its persistent FFTW plans bit-for-bit).
        t0 = prof ? now() : 0.0;
        std::vector<std::vector<std::vector<std::complex<float>>>> P((size_t)nc);
        const int nt_c = std::max(1, std::min(n_threads, nc));
        const auto run_ch = [&](int t) {
            AcquireWorkspace local;
            AcquireWorkspace& mine = (t == 0) ? ws : local;
            std::vector<std::complex<float>> d((size_t)N);
            for (int c = t; c < nc; c += nt_c) {
                for (int m = 0; m < N; ++m)
                    d[(size_t)m] = data_ch[(size_t)c][(size_t)(k * N + m)];
                // data (N) shorter than replica (2N) -> channel_correlate zero-pads it to the
                // transform length, which is exactly the linear correlation we want.
                P[(size_t)c] =
                    channel_correlate(d, repl[(size_t)c], doppler_grid, sample_rate, sph, mine);
            }
        };
        if (nt_c == 1) {
            run_ch(0);
        } else {
            std::vector<std::thread> pool;
            pool.reserve((size_t)nt_c - 1);
            for (int t = 1; t < nt_c; ++t)
                pool.emplace_back(run_ch, t);
            run_ch(0);
            for (auto& th : pool)
                th.join();
        }
        if (prof)
            t_corr += now() - t0;

        t0 = prof ? now() : 0.0;
        dims = aggregate_accumulate(P, chan_ids, sph, surf, n_threads, fine_step);
        if (prof)
            t_agg += now() - t0;
    }
    if (prof)
        fprintf(stderr,
                "[ms-split profile] K=%d N=%d nc=%d threads=%d | replica %.2fs  correlate %.2fs  "
                "aggregate %.2fs  (total %.2fs)\n",
                n_sub, N, nc, n_threads, t_repl, t_corr, t_agg, t_repl + t_corr + t_agg);
    return dims;
}

AcquisitionResult ms_split_peak(const std::vector<double>& surf, const AcquisitionSurface& dims,
                                const std::vector<double>& doppler_grid, double sample_rate,
                                double chip_rate, long code_length, double phi_r0, int sub_hops) {
    AcquisitionResult best{0.0, 0.0, 0, -1.0, 0.0};
    const int sfine = dims.fine();
    const int sph = dims.sph;
    const double cps = chip_rate / sample_rate;
    const double L = (double)code_length;

    // THE PHYSICAL LAG WINDOW: q in [N, 2N), and nothing outside it is even looked at.
    //
    // The replica spans 2N hops starting N hops before the data's N, so lag q reads replica
    // indices (m-q) mod 2N for m in [0,N). Those are contiguous -- one unbroken stretch of code
    // -- exactly when q >= N. For q in [1, N) the sum straddles the replica's own wrap, joining
    // its tail to its head across a 2.007-period jump: a partial correlation over only the
    // (N-q) hops that still line up.
    //
    // Those partial lags are not a small effect to be thresholded away. At the bench point the
    // strongest of them stood at 26% of the true peak's power -- second place on the whole
    // surface, above every real sidelobe -- and it TRACKS the satellite (q moves 19.09 hops per
    // 1000 chips of code phase, exactly 1/cph), so it looks like a detection in every way a
    // statistic can see. It is also nearly a code period away from the truth, so accepting one
    // seeds the tracker a full period off.
    //
    // [N, 2N) still covers the entire ambiguity: one code period is 195.3125 hops <= N = 196,
    // so every physical delay has exactly one representative in the window.
    const int q_lo = std::max(0, std::min(sub_hops, dims.Mp));
    const int q_hi = dims.Mp;

    double surf_sum = 0.0;
    long surf_n = 0;
    int best_d = 0, best_q = 0, best_i = 0;
    std::vector<double> dop_peak(dims.n_dop, 0.0);
    for (int d = 0; d < dims.n_dop; ++d) {
        const double* base = surf.data() + (long)d * dims.Mp * sfine;
        for (int q = q_lo; q < q_hi; ++q) {
            const double* row = base + (long)q * sfine;
            for (int s = 0; s < sfine; ++s) {
                const double pw = row[s];
                // Mean over the SEARCHED cells only. Including the partial-overlap half would
                // put correlation energy into the noise estimate and quietly deflate the SNR
                // this surface reports -- the acceptance threshold is compared against it.
                surf_sum += pw;
                surf_n++;
                if (pw > dop_peak[d])
                    dop_peak[d] = pw;
                if (pw > best.peak) {
                    best.peak = pw;
                    best_d = d;
                    best_q = q;
                    best_i = s;
                }
            }
        }
    }

    best.doppler_hz = doppler_grid.empty() ? 0.0 : doppler_grid[best_d];
    const int nd = (int)doppler_grid.size();
    if (nd >= 3 && best_d > 0 && best_d < nd - 1) {
        const double sm = dop_peak[best_d - 1], s0 = dop_peak[best_d], sp = dop_peak[best_d + 1];
        const double denom = sm - 2.0 * s0 + sp;
        if (denom < 0.0) {
            double delta = 0.5 * (sm - sp) / denom;
            delta = std::max(-0.5, std::min(0.5, delta));
            best.doppler_hz += delta * (doppler_grid[best_d + 1] - doppler_grid[best_d]);
        }
    }
    const double mean = (surf_n > 0) ? surf_sum / (double)surf_n : 0.0;
    best.snr = (mean > 0.0) ? best.peak / mean : 0.0;

    // THE FINE HALF OF THE LAG CARRIES THE OPPOSITE SIGN TO THE COARSE HALF, so AcquisitionSurface
    // ::tau()'s q*sph + i*s_step is not the delay here and cannot be used. Measured directly:
    // injecting +5 chips of code phase moves the peak's fine index +49 columns at s_step 32,
    // = 9.8 columns/chip = 312.8 samples/chip = exactly 1/cps, while q moves the other way.
    // The shipped path never had to face this because refine_peak re-scans a full hop either
    // way; here it is the difference between 0.05 chips and 52.4.
    //
    // Fold to the signed half-open hop first: the cross-channel DFT reports the fine lag modulo
    // sph, so a small NEGATIVE offset comes back as i near the top of the axis. Read literally
    // that is a whole extra hop of error -- every +52.35 chip outlier in the bench sweeps was
    // this and only this.
    long fine = (long)best_i * (long)(dims.s_step > 0 ? dims.s_step : 1);
    if (fine > sph / 2)
        fine -= sph;
    const long tau_eff = (long)best_q * (long)sph - fine;

    // The peak condition, all three terms: phi_r0 (the replica's own start phase) + Ns*cps (the
    // cyclic wrap, since the peak reads indices a whole Mp hops on) - tau_eff*cps.
    const long Ns = (long)dims.Mp * (long)sph;
    best.peak_tau_samples = ((tau_eff % Ns) + Ns) % Ns;
    double cp = std::fmod(phi_r0 + (double)(Ns - tau_eff) * cps, L);
    if (cp < 0.0)
        cp += L;
    best.code_phase_chips = cp;
    return best;
}

AcquisitionResult channelized_peak(const std::vector<double>& surf, const AcquisitionSurface& dims,
                                   const std::vector<double>& doppler_grid, double sample_rate,
                                   double chip_rate, long code_length) {
    AcquisitionResult best{0.0, 0.0, 0, -1.0, 0.0};
    double surf_sum = 0.0;
    long surf_n = 0;
    int best_d = 0;
    long best_tau = 0;
    std::vector<double> dop_peak(dims.n_dop, 0.0); // per-Doppler max-over-tau (for the sub-grid fit)

    // Index by the STORED fine-lag COLUMNS, and form the absolute delay through dims.tau(),
    // which reapplies both the coarse stride sph and the fine decimation s_step. The surface
    // may hold one period of a periodic fine-lag axis, decimated (see channelized_accumulate);
    // dropping identical copies changes neither the peak cell (first strict max, which already
    // fell in the first period) nor the mean (uniform copies), and the decimation costs only
    // the sub-lobe interpolation the following refine redoes exactly anyway.
    const int sfine = dims.fine();
    for (int d = 0; d < dims.n_dop; ++d) {
        const double* base = surf.data() + (long)d * dims.Mp * sfine;
        for (int q = 0; q < dims.Mp; ++q) {
            const double* row = base + (long)q * sfine;
            for (int s = 0; s < sfine; ++s) {
                const double pw = row[s];
                surf_sum += pw;
                surf_n++;
                if (pw > dop_peak[d])
                    dop_peak[d] = pw;
                if (pw > best.peak) {
                    best.peak = pw;
                    best_d = d;
                    best_tau = dims.tau(q, s);
                }
            }
        }
    }

    best.doppler_hz = doppler_grid.empty() ? 0.0 : doppler_grid[best_d];
    // Sub-grid Doppler refine: the correlation-vs-Doppler peak is smooth but sampled on a
    // coarse grid (doppler_step), so the cell max is only +-step/2 accurate. Fit a parabola to
    // the peak and its two Doppler neighbours -- using each cell's OWN max-over-tau, not the
    // power at the centre cell's tau (the code-Doppler shifts the peak tau slightly cell to
    // cell, so a fixed-tau slice is asymmetric and biases the vertex) -- and take the vertex.
    // Lifts the seed to ~+-step/20: a tighter Doppler for the tracker FLL to pull in from and a
    // sharper input to the broker's clock-bias solve. Skipped at the grid edges / non-concave.
    const int nd = (int)doppler_grid.size();
    if (nd >= 3 && best_d > 0 && best_d < nd - 1) {
        const double sm = dop_peak[best_d - 1], s0 = dop_peak[best_d], sp = dop_peak[best_d + 1];
        const double denom = sm - 2.0 * s0 + sp;
        if (denom < 0.0) {
            double delta = 0.5 * (sm - sp) / denom; // parabola vertex, in grid cells (|delta|<=0.5)
            delta = std::max(-0.5, std::min(0.5, delta));
            best.doppler_hz += delta * (doppler_grid[best_d + 1] - doppler_grid[best_d]);
        }
    }
    const double mean = (surf_n > 0) ? surf_sum / (double)surf_n : 0.0;
    best.snr = (mean > 0.0) ? best.peak / mean : 0.0;

    // The correlation peaks at the negative of the replica's code-phase advance
    // (data = repl0 advanced by the delay). Report the delay so it can seed the
    // measurement replica generator directly (same "code[floor(cp + n*r)]"
    // convention).
    const long Ns = (long)dims.Mp * dims.sph;
    best.peak_tau_samples = ((Ns - best_tau) % Ns + Ns) % Ns;
    const double chips = (double)best.peak_tau_samples * chip_rate / sample_rate;
    best.code_phase_chips = std::fmod(chips, (double)code_length);
    if (best.code_phase_chips < 0.0)
        best.code_phase_chips += code_length;
    return best;
}

AcquisitionResult
channelized_acquire(const std::vector<std::vector<std::complex<float>>>& data_ch,
                    const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                    const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                    double sample_rate, double chip_rate, int num_chan, long code_length,
                    const std::vector<int>& chan_freq, int samples_per_hop) {
    // Single-window acquire = one accumulation + reduce (local FFT workspace).
    std::vector<double> surf;
    AcquireWorkspace ws;
    const auto dims = channelized_accumulate(data_ch, repl0_ch, covering, doppler_grid, sample_rate,
                                             num_chan, surf, ws, chan_freq, samples_per_hop);
    return channelized_peak(surf, dims, doppler_grid, sample_rate, chip_rate, code_length);
}

} // namespace gnss
