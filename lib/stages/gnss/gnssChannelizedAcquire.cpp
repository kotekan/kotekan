#include "gnssChannelizedAcquire.hpp"

#include "fftwPlannerLock.hpp" // for fftw_planner_mutex

#include <algorithm> // for max
#include <cmath>     // for cos, sin, fmod, M_PI, round
#include <cstdlib>   // for abs
#include <mutex>     // for lock_guard
#include <numeric>   // for gcd
#include <thread>    // for thread

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

std::vector<std::vector<cf>> channel_correlate(const std::vector<cf>& data,
                                               const std::vector<cf>& repl0,
                                               const std::vector<double>& doppler_grid,
                                               double sample_rate, int samples_per_hop,
                                               AcquireWorkspace& ws) {
    std::vector<std::vector<cf>> P;
    channel_correlate_into(data, repl0, doppler_grid, sample_rate, samples_per_hop, ws, P);
    return P;
}

void channel_correlate_into(const std::vector<cf>& data, const std::vector<cf>& repl0,
                            const std::vector<double>& doppler_grid, double sample_rate,
                            int samples_per_hop, AcquireWorkspace& ws,
                            std::vector<std::vector<cf>>& P) {
    const int M = (int)data.size();   // data window length, hops
    const int Mp = (int)repl0.size(); // replica hop-period (>= M)
    const int nd = (int)doppler_grid.size();
    const int sph = (samples_per_hop > 0) ? samples_per_hop : Mp;

    ws.ensure(Mp);
    cf* IN = reinterpret_cast<cf*>(ws.in);
    cf* OUT = reinterpret_cast<cf*>(ws.out);

    // conj(FFT(repl0)): Doppler-independent, computed once per call.
    std::vector<cf> conjB(Mp);
    for (int k = 0; k < Mp; ++k)
        IN[k] = repl0[k];
    fftwf_execute(ws.fwd);
    for (int k = 0; k < Mp; ++k)
        conjB[k] = std::conj(OUT[k]);

    const float inv_mp = 1.0f / (float)Mp; // FFTW backward is unnormalized
    // Resized only on a shape change, so a caller reusing P across windows does not reallocate.
    if ((int)P.size() != nd)
        P.assign((size_t)nd, std::vector<cf>((size_t)Mp));
    for (auto& row : P)
        if ((int)row.size() != Mp)
            row.assign((size_t)Mp, cf(0.0f, 0.0f));

    // Bin-aligned Doppler grid: wiping the data by f_d multiplies it by e^{-i 2pi f_d m sph/Fs},
    // which is exactly a cyclic shift of its spectrum when f_d is a whole number of this
    // transform's bins, Fs/(Mp*sph) (62.5 Hz at CHORD). One forward transform then serves every
    // Doppler trial. Detected rather than configured: a grid not on the bins takes the per-trial
    // path below.
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
        return;
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
}

AcquisitionSurface surface_dims(const std::vector<int>& chan_freq, int samples_per_hop, int n_dop,
                                int Mp, int fine_step) {
    const int nc = (int)chan_freq.size();
    const int sph = samples_per_hop;
    int g = 0;
    for (int ci = 1; ci < nc; ++ci)
        g = std::gcd(g, std::abs(chan_freq[ci] - chan_freq[0]));
    // g == 0 means a single covering channel: |D|^2 is then flat in s (one unit-modulus ramp),
    // so one column suffices. std::gcd(0, sph) == sph gives exactly that.
    const int s_stored = (nc > 0 && sph > 0) ? sph / std::gcd(g, sph) : sph;
    // Decimate the fine axis (see AcquisitionSurface::s_step).
    const int s_step = std::max(1, fine_step);
    return AcquisitionSurface{n_dop, Mp, sph, s_stored, s_step};
}

AcquisitionSurface aggregate_accumulate(const std::vector<std::vector<std::vector<cf>>>& P,
                                        const std::vector<int>& chan_freq, int samples_per_hop,
                                        std::vector<double>& surf, int n_threads, int fine_step) {
    const int nc = (int)P.size();
    const int nd = nc ? (int)P[0].size() : 0;
    const int Mp = (nc && nd) ? (int)P[0][0].size() : 0;
    const int sph = samples_per_hop;

    // The fine-lag axis is exactly periodic, so store one period of it.
    //
    // D[s] = sum_ci P_ci e^{+i 2pi f_ci s / sph}. Factoring out the first channel's unit-modulus
    // ramp shows that |D[s]|^2, all this function keeps, depends only on the differences
    // f_ci - f_c0. If those share a factor g with sph, |D[s]|^2 repeats with period sph/g and
    // the other g-1 copies are identical. For a combed band g is large (16 for a stride-16
    // comb). The period sph/g samples is 1/(g * channel spacing) in time, the comb's own delay
    // ambiguity, so no independent information is dropped.
    //
    // Storing one period gives the same peak cell and SNR as the full axis: channelized_peak
    // takes the first strict maximum, which lies in [0, sph/g), and dropping g uniform copies
    // leaves the surface mean the same.
    const AcquisitionSurface dims = surface_dims(chan_freq, sph, nd, Mp, fine_step);
    const int s_step = dims.s_step;
    const int s_cols = dims.fine();
    if ((long)surf.size() != dims.size())
        surf.assign(dims.size(), 0.0);
    if (nc == 0)
        return dims;

    // Cross-channel fine-lag ramp e^{+i 2pi f_c s / sph}, as split float arrays so the inner
    // s loop vectorizes (std::complex<float> does not). The ramp divides by the full sph:
    // s_stored bounds the range of s, it does not rescale the frequency.
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

    // Each (d, q) surface row is written by exactly one thread. Splitting the Doppler x lag
    // product rather than the Doppler axis alone keeps every core busy when the Doppler grid is
    // only a few bins wide. n_threads = 1 runs the same loop serially.
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
    // Samples per hop: N for a critically sampled complex bank, 2N for an r2c bank.
    const int sph = (samples_per_hop > 0) ? samples_per_hop : num_chan;
    const int nc = (int)covering.size();
    std::vector<std::vector<std::vector<cf>>> P(nc);
    // Correlate channels in parallel, each worker with its own FFTW workspace; worker 0 uses
    // the caller's, so the single-threaded path keeps its persistent plans. The extra
    // workspaces are local and re-plan on each call, which FFTW_ESTIMATE keeps cheap.
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

AcquisitionResult channelized_peak(const std::vector<double>& surf, const AcquisitionSurface& dims,
                                   const std::vector<double>& doppler_grid, double sample_rate,
                                   double chip_rate, long code_length, bool pairsum_select,
                                   int data_hops) {
    AcquisitionResult best{0.0, 0.0, 0, -1.0, 0.0, -1.0};
    double surf_sum = 0.0;
    long surf_n = 0;
    int best_d = 0, best_q = 0, best_i = 0;
    // per-Doppler max over tau: the pair-sum selection, the sub-grid refine and peak_interp
    std::vector<double> dop_peak(dims.n_dop, 0.0);

    // Index by the stored fine-lag columns, fine(): the surface may hold one decimated period of
    // a periodic fine-lag axis (see aggregate_accumulate). peak_from_reduction turns the cell
    // into a delay.
    const int sfine = dims.fine();
    std::vector<int> pd_q(dims.n_dop, 0), pd_i(dims.n_dop, 0); // per-plane argmax cell
    for (int d = 0; d < dims.n_dop; ++d) {
        const double* base = surf.data() + (long)d * dims.Mp * sfine;
        for (int q = 0; q < dims.Mp; ++q) {
            const double* row = base + (long)q * sfine;
            for (int s = 0; s < sfine; ++s) {
                const double pw = row[s];
                surf_sum += pw;
                surf_n++;
                if (pw > dop_peak[d]) {
                    dop_peak[d] = pw;
                    pd_q[d] = q;
                    pd_i[d] = s;
                }
                if (pw > best.peak) {
                    best.peak = pw;
                    best_d = d;
                    best_q = q;
                    best_i = s;
                }
            }
        }
    }
    // pairsum_select: choose the Doppler plane by the mainlobe pair-sum
    // dop_peak[d] + max(neighbours), the statistic peak_interp uses, rather than by the raw
    // argmax. Otherwise an overlay-mismatch sideband that lands on a grid bin can beat a true
    // peak that scallops between bins (a single bin keeps as little as 41% of the continuum
    // power at mid-bin; the sideband family reaches ~50%), shifting tau by whole code periods.
    // Only interior local maxima compete; without that the plane beside a strong peak ties it
    // (small + big against big + small) and the chosen Doppler dithers by one bin. The returned
    // peak and snr stay the chosen cell's raw values, which downstream thresholds expect.
    if (pairsum_select && dims.n_dop >= 3) {
        double bestps = -1.0;
        int sel = -1;
        for (int d = 1; d < dims.n_dop - 1; ++d) {
            if (dop_peak[d] < dop_peak[d - 1] || dop_peak[d] < dop_peak[d + 1])
                continue; // not a local max: a neighbouring peak's shoulder
            const double ps = dop_peak[d] + std::max(dop_peak[d - 1], dop_peak[d + 1]);
            if (ps > bestps) {
                bestps = ps;
                sel = d;
            }
        }
        if (sel >= 0) { // no interior local max -> keep the raw argmax
            best_d = sel;
            best.peak = dop_peak[best_d];
            best_q = pd_q[best_d];
            best_i = pd_i[best_d];
        }
    }

    // The neighbour Doppler planes at the winning tau (max over a 3x3 cell window; the peak
    // moves less than a stored column across +-1 bin of code Doppler), for the ratio refine in
    // peak_from_reduction. The plane-wide dop_peak remains the selection statistic.
    double dop_loc_m = -1.0, dop_loc_p = -1.0;
    if (dims.n_dop >= 3 && best_d > 0 && best_d < dims.n_dop - 1) {
        auto local_max = [&](int d) {
            const double* base = surf.data() + (long)d * dims.Mp * sfine;
            double m = 0.0;
            for (int dq = -1; dq <= 1; ++dq) {
                const int q = ((best_q + dq) % dims.Mp + dims.Mp) % dims.Mp;
                for (int ds = -1; ds <= 1; ++ds) {
                    const int si = best_i + ds;
                    if (si < 0 || si >= sfine)
                        continue;
                    m = std::max(m, base[(long)q * sfine + si]);
                }
            }
            return m;
        };
        dop_loc_m = local_max(best_d - 1);
        dop_loc_p = local_max(best_d + 1);
    }

    // The rest is arithmetic on the reduction, shared with the GPU path, which reduces on the
    // device.
    const double mean0 = (surf_n > 0) ? surf_sum / (double)surf_n : 0.0;
    const double dop_u =
        (data_hops > 0 && data_hops < dims.Mp) ? (double)data_hops / (double)dims.Mp : 1.0;
    return peak_from_reduction(dims, doppler_grid, sample_rate, chip_rate, code_length, best.peak,
                               mean0, best_d, best_q, best_i, dop_peak, dop_loc_m, dop_loc_p,
                               dop_u);
}

AcquisitionResult peak_from_reduction(const AcquisitionSurface& dims,
                                      const std::vector<double>& doppler_grid, double sample_rate,
                                      double chip_rate, long code_length, double peak, double mean,
                                      int best_d, int best_q, int best_i,
                                      const std::vector<double>& dop_peak, double dop_loc_m,
                                      double dop_loc_p, double dop_u) {
    AcquisitionResult best{0.0, 0.0, 0, peak, 0.0, peak};
    best.doppler_hz = doppler_grid.empty() ? 0.0 : doppler_grid[best_d];
    // Sub-grid Doppler refine: the grid cell max is only +-step/2 accurate, so place the peak
    // between bins from the peak plane and its two neighbours. Skipped at the grid edges.
    const int nd = (int)doppler_grid.size();
    if (nd >= 3 && best_d > 0 && best_d < nd - 1 && (int)dop_peak.size() == nd) {
        const double sm = dop_peak[best_d - 1], s0 = dop_peak[best_d], sp = dop_peak[best_d + 1];
        // With dop_loc_* (the neighbour planes at the winning tau) and both below the peak,
        // invert the amplitude ratio r of the larger neighbour to the peak. The data window is
        // data_hops of signal in an Mp-hop transform, so the response to an offset of x bins is
        // g(x) = sin(pi x u)/x with u = data_hops/Mp, and r = |g(1-delta)|/|g(delta)| is
        // monotone on [0, 0.5]; solve it by bisection. On-grid r is g(1)/g(0) = sin(pi u)/(pi u),
        // not 0 for u < 1, so the single-tone form delta = r/(1+r) would misplace on-grid peaks.
        //
        // Otherwise fit a parabola to dop_peak, each plane's own max over tau (the code Doppler
        // shifts the peak tau between planes, so a fixed-tau slice would be asymmetric). It is
        // the weaker estimator: on this response a parabola under-recovers the offset, and for a
        // strong peak the far neighbour's plane-wide max is a noise order statistic rather than
        // the peak's shoulder. Either way only the Doppler is placed here; peak_interp below
        // stays the pair-sum.
        if (dop_loc_m >= 0.0 && dop_loc_p >= 0.0 && s0 > 0.0
            && std::max(dop_loc_m, dop_loc_p) < s0) {
            const bool up = dop_loc_p >= dop_loc_m;
            const double r_meas = std::sqrt(std::max(dop_loc_m, dop_loc_p) / s0);
            auto g_abs = [&](double x) {
                if (std::fabs(x) < 1e-9)
                    return M_PI * dop_u; // lim sin(pi x u)/x
                return std::fabs(std::sin(M_PI * x * dop_u) / x);
            };
            auto r_model = [&](double d) { return g_abs(1.0 - d) / std::max(g_abs(d), 1e-30); };
            double delta = 0.0;
            if (r_meas >= 1.0) {
                delta = 0.5; // at/beyond mid-bin: the neighbour ties the peak
            } else if (r_meas > r_model(0.0)) {
                double lo = 0.0, hi = 0.5;
                for (int it = 0; it < 40; ++it) {
                    const double mid = 0.5 * (lo + hi);
                    (r_model(mid) < r_meas ? lo : hi) = mid;
                }
                delta = 0.5 * (lo + hi);
            } // r_meas at or below the on-grid ratio: indistinguishable from 0, keep the bin
            best.doppler_hz +=
                (up ? delta : -delta) * (doppler_grid[best_d + 1] - doppler_grid[best_d]);
        } else if (const double denom = sm - 2.0 * s0 + sp; denom < 0.0) {
            double delta = 0.5 * (sm - sp) / denom; // parabola vertex, in grid cells (|delta|<=0.5)
            delta = std::max(-0.5, std::min(0.5, delta));
            best.doppler_hz += delta * (doppler_grid[best_d + 1] - doppler_grid[best_d]);
        }
        // peak_interp: the mainlobe pair-sum, the statistic for comparing hypotheses. It has
        // no division and is monotone in the underlying spectrum, so it cannot over-recover.
        // For one tone at any grid phase the pair holds |sinc(d)|^2 + |sinc(1-d)|^2, 0.81-1 of
        // the continuum power (a single bin: 0.405-1), while an overlay-mismatch spectrum is
        // bounded by its best adjacent pair, <= ~0.5 by Parseval on a +-1 sequence. A parabola's
        // vertex value under-recovers the peak 2.2x at mid-bin, and a two-bin sinc ratio
        // over-recovers when two adjacent mismatch tones mimic one mid-bin tone.
        best.peak_interp = s0 + std::max(sm, sp);
    }
    best.snr = (mean > 0.0) ? best.peak / mean : 0.0;

    // The correlation peaks at the negative of the replica's code-phase advance (data = repl0
    // advanced by the delay). Report the delay, so it seeds the replica generators directly
    // (same code[floor(cp + n*r)] convention).
    //
    // The fine half of the lag runs opposite to the coarse half (FINE_LAG_SIGN_PFB, a
    // newest-first bank), so q*sph + i*s_step is not the delay. With the wrong sign the coarse
    // phase is off by up to sph (52.4 chips of a 10.23 Mchip/s code at CHORD), which only a
    // refine spanning a full hop absorbs. Fold the fine index to a signed half period first: the
    // cross-channel DFT reports it modulo the stored period, so a small negative offset comes
    // back near the top of the axis and would read as a whole period too far. Fold by s_stored,
    // not sph: the surface holds only sph/g samples of fine lag when the comb shares a factor g
    // with sph, and folding by sph would step outside them.
    const int fine_period = (dims.s_stored > 0) ? dims.s_stored : dims.sph;
    long fine = (long)best_i * (long)(dims.s_step > 0 ? dims.s_step : 1);
    if (fine > fine_period / 2)
        fine -= fine_period;
    const long best_tau = (long)best_q * (long)dims.sph + (long)FINE_LAG_SIGN_PFB * fine;
    const long Ns = (long)dims.Mp * dims.sph;
    best.peak_tau_samples = ((Ns - best_tau) % Ns + Ns) % Ns;
    const double chips = (double)best.peak_tau_samples * chip_rate / sample_rate;
    best.code_phase_chips = std::fmod(chips, (double)code_length);
    if (best.code_phase_chips < 0.0)
        best.code_phase_chips += code_length;
    return best;
}

} // namespace gnss
