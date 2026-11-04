/**
 * @file
 * @brief The host side of cudaDtvSubtract's rolling fit: the least-squares impulse responses g
 *        from decaying sums of the lag correlations, refreshed every second or so, with the
 *        guards that decide whether a refreshed g goes into the stage.
 *
 * With R_xs,i(lag_j) = sum_t x_i(t) conj(s(t - lag_j)) and R_ss(d) = sum_t s(t) conj(s(t - d))
 * summed over the same frames (cudaDtvSubtractKernel.hpp), subtracting
 * sum_m g_i(lag_m) s(t - lag_m) from input i changes its power over those frames by
 *     cost_i(g) = sum_{m,j} conj(g_m) A[m][j] g_j - 2 Re sum_m conj(g_m) R_xs,i(lag_m),
 * A[m][j] = R_ss(lag_m - lag_j) (Hermitian Toeplitz; T^T in scripts/gnss/dtv_state.py). The
 * least-squares g solves A g_i = R_xs,i: dtv_state.solve_g, with the same 1e-9 x trace
 * regularisation, solved here by a Cholesky factorisation in double precision. For that g,
 * cost = -Re g^H R_xs: minus the power it explains (dtv_state.py's "fraction explained" divides
 * this by the input power).
 *
 * The sums decay: each frame S <- lam_k S + R with lam_k = exp(-frame / tau_k) per source, so a
 * fit uses the last ~tau_k of data (docs/CHORD_DTV_SUBTRACTION_20261102.md: g goes stale on every
 * timescale from 1 s; a fit refreshed each second from tau ~1-2 s of data leaves ~1e-4 of the
 * removed power where the 6.5-min refit loop leaves 1e-2). dtv_roll_step() is one refresh of
 * every (channel, source); it is a pure function of a snapshot of the sums so the stage, the GPU
 * test and the offline replay (dtv_roll_replay) run the same code.
 */

#ifndef DTV_ROLLING_FIT_HPP
#define DTV_ROLLING_FIT_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace gnss_cuda {

/// One (channel, source)'s normal equations: A from R_ss, then g from R_xs for every input.
class DtvLsq {
public:
    using cd = std::complex<double>;

    /// Factor A + 1e-9 tr(A) I from R_ss(d), d = 0..n-1 (interleaved re, im). False when A is
    /// not positive definite (no signal in the sums).
    bool factor(const float* rss, int n) {
        _n = n;
        _r.assign((size_t)n, cd(0.0, 0.0));
        for (int d = 0; d < n; d++)
            _r[(size_t)d] = cd(rss[2 * d], rss[2 * d + 1]);
        const double diag = _r[0].real();
        if (!(diag > 0.0) || !std::isfinite(diag))
            return false;
        const double eps = 1e-9 * n * diag;
        _L.assign((size_t)n * n, cd(0.0, 0.0));
        for (int j = 0; j < n; j++) {
            // A[i][j] for i >= j is R_ss(i - j)
            double djj = diag + eps;
            for (int k = 0; k < j; k++)
                djj -= std::norm(_L[(size_t)j * n + k]);
            if (!(djj > 0.0))
                return false;
            const double ljj = std::sqrt(djj);
            _L[(size_t)j * n + j] = cd(ljj, 0.0);
            for (int i = j + 1; i < n; i++) {
                cd v = _r[(size_t)(i - j)];
                for (int k = 0; k < j; k++)
                    v -= _L[(size_t)i * n + k] * std::conj(_L[(size_t)j * n + k]);
                _L[(size_t)i * n + j] = v / ljj;
            }
        }
        return true;
    }

    /// g[j][i] (interleaved, i < ni) from rxs[j][i]: the layout of one (channel, source) block.
    /// False if the solution is not finite.
    bool solve(const float* rxs, int ni, float* g) {
        const int n = _n;
        _y.assign((size_t)n * ni, cd(0.0, 0.0));
        // L y = b
        for (int m = 0; m < n; m++) {
            cd* ym = &_y[(size_t)m * ni];
            for (int i = 0; i < ni; i++)
                ym[i] = cd(rxs[2 * ((size_t)m * ni + i)], rxs[2 * ((size_t)m * ni + i) + 1]);
            for (int k = 0; k < m; k++) {
                const cd l = _L[(size_t)m * n + k];
                const cd* yk = &_y[(size_t)k * ni];
                for (int i = 0; i < ni; i++)
                    ym[i] -= l * yk[i];
            }
            const double inv = 1.0 / _L[(size_t)m * n + m].real();
            for (int i = 0; i < ni; i++)
                ym[i] *= inv;
        }
        // L^H g = y, in place
        for (int m = n - 1; m >= 0; m--) {
            cd* ym = &_y[(size_t)m * ni];
            for (int k = m + 1; k < n; k++) {
                const cd l = std::conj(_L[(size_t)k * n + m]);
                const cd* yk = &_y[(size_t)k * ni];
                for (int i = 0; i < ni; i++)
                    ym[i] -= l * yk[i];
            }
            const double inv = 1.0 / _L[(size_t)m * n + m].real();
            for (int i = 0; i < ni; i++)
                ym[i] *= inv;
        }
        bool ok = true;
        for (size_t q = 0; q < (size_t)n * ni; q++) {
            g[2 * q] = (float)_y[q].real();
            g[2 * q + 1] = (float)_y[q].imag();
            ok = ok && std::isfinite(g[2 * q]) && std::isfinite(g[2 * q + 1]);
        }
        return ok;
    }

    /// sum over the inputs of cost_i(g) (see the file comment) with the unregularised A of the
    /// last factor(): negative when g removes power from these frames.
    double cost(const float* rxs, const float* g, int ni) const {
        const int n = _n;
        double c = 0.0;
        std::vector<cd> ag((size_t)ni);
        for (int m = 0; m < n; m++) {
            for (int i = 0; i < ni; i++)
                ag[(size_t)i] = cd(0.0, 0.0);
            for (int j = 0; j < n; j++) {
                const cd a = m >= j ? _r[(size_t)(m - j)] : std::conj(_r[(size_t)(j - m)]);
                const float* gj = g + 2 * (size_t)j * ni;
                for (int i = 0; i < ni; i++)
                    ag[(size_t)i] += a * cd(gj[2 * i], gj[2 * i + 1]);
            }
            const float* gm = g + 2 * (size_t)m * ni;
            const float* rm = rxs + 2 * (size_t)m * ni;
            for (int i = 0; i < ni; i++) {
                const cd gv(gm[2 * i], gm[2 * i + 1]);
                const cd rv(rm[2 * i], rm[2 * i + 1]);
                c += (std::conj(gv) * ag[(size_t)i]).real() - 2.0 * (std::conj(gv) * rv).real();
            }
        }
        return c;
    }

private:
    int _n = 0;
    std::vector<cd> _r; ///< R_ss(d), d = 0..n-1, unregularised
    std::vector<cd> _L; ///< Cholesky factor of A + eps I, row-major, lower
    std::vector<cd> _y; ///< solve workspace [n][ni]
};

/// A snapshot of the stage's decaying sums, as the GPU left them (interleaved float re, im).
struct DtvRollSnapshot {
    int n_chan = 0, n_src = 0, n_lag = 0, ni = 0;
    const float* rxs = nullptr; ///< [n_chan][n_src][n_lag][ni] complex: lam_k-decayed R_xs
    const float* rss = nullptr; ///< [n_chan][n_src][n_lag] complex: lam_k-decayed R_ss
    /// [3][n_chan]: sum over inputs of |x|^2 before subtraction, lam_0-decayed; the same
    /// undecayed since the last snapshot; after every subtracting source, since the last snapshot
    const float* pw = nullptr;
    const int* act = nullptr;    ///< [n_chan][n_src] at the snapshot: 2 = subtracting
    const long* since = nullptr; ///< [n_chan][n_src] frames in the sums since they were zeroed
};

/// The refresh's settings.
struct DtvRollParams {
    std::vector<long> warm;      ///< [n_src] frames a source's sums need before its first fit
    std::vector<int> freeze;     ///< [n_src] nonzero: hold this source's g (the sums keep going)
    double min_explained = 0.05; ///< source 0 must explain this much of the input power...
    double max_power = 2.0; ///< ...and residual / input power over the interval stay below this
};

/// What one refresh decided.
struct DtvRollOutcome {
    std::vector<char> trip;   ///< [n_chan] the stage added power (> max_power): stop subtracting
    std::vector<char> weak;   ///< [n_chan] source 0 explains < min_explained: hold the channel
    std::vector<char> upd;    ///< [n_chan][n_src] a new g is in g_out
    std::vector<float> ratio; ///< [n_chan] residual / input power over the interval (-1: none)
    std::vector<float> expl0; ///< [n_chan] fraction of the input power the new source-0 g explains
    std::vector<float> gain;  ///< [n_chan][n_src] (cost(g_cur) - cost(g_new)) / -cost(g_new)
    std::vector<float> step;  ///< [n_chan][n_src] |g_new - g_cur|^2 / |g_cur|^2
    int n_upd = 0, n_bad = 0;
};

/// One refresh: for each channel with a subtracting source, check the residual power of the
/// last interval (trip), fit every subtracting, unfrozen, warmed-up source from the sums, hold
/// the channel if source 0's fit explains too little (no signal: sky or analog chain off), and
/// write the accepted fits into g_out (same layout as the stage's g: [n_chan][n_src][n_lag][ni]
/// complex; blocks not updated are left as they are).
inline DtvRollOutcome dtv_roll_step(const DtvRollSnapshot& s, const DtvRollParams& p,
                                    const float* g_cur, float* g_out) {
    const int n = s.n_chan, ns = s.n_src, nl = s.n_lag, ni = s.ni;
    const size_t blk = (size_t)nl * ni * 2; // floats per (channel, source) g block
    DtvRollOutcome o;
    o.trip.assign((size_t)n, 0);
    o.weak.assign((size_t)n, 0);
    o.upd.assign((size_t)n * ns, 0);
    o.ratio.assign((size_t)n, -1.0f);
    o.expl0.assign((size_t)n, NAN);
    o.gain.assign((size_t)n * ns, NAN);
    o.step.assign((size_t)n * ns, NAN);
    DtvLsq lsq;
    std::vector<float> gn(blk);
    for (int c = 0; c < n; c++) {
        bool sub = false;
        for (int k = 0; k < ns; k++)
            sub = sub || s.act[(size_t)c * ns + k] >= 2;
        if (!sub)
            continue;
        const float raw_int = s.pw[(size_t)n + c], res_int = s.pw[(size_t)2 * n + c];
        if (raw_int > 0.0f) {
            o.ratio[(size_t)c] = res_int / raw_int;
            if (!(o.ratio[(size_t)c] <= p.max_power)) {
                o.trip[(size_t)c] = 1;
                continue;
            }
        }
        // fit first, accept after: source 0's fit decides for the channel
        std::vector<std::vector<float>> fits((size_t)ns);
        for (int k = 0; k < ns; k++) {
            const size_t ck = (size_t)c * ns + k;
            if (s.act[ck] < 2 || p.freeze[(size_t)k] || s.since[ck] < p.warm[(size_t)k])
                continue;
            const float* rxs = s.rxs + ck * blk;
            const float* gc = g_cur + ck * blk;
            if (!lsq.factor(s.rss + ck * nl * 2, nl) || !lsq.solve(rxs, ni, gn.data())) {
                o.n_bad++;
                continue;
            }
            const double cn = lsq.cost(rxs, gn.data(), ni), cc = lsq.cost(rxs, gc, ni);
            o.gain[ck] = (float)((cc - cn) / std::max(-cn, 1e-30));
            double dg = 0.0, g2 = 0.0;
            for (size_t q = 0; q < blk; q++) {
                dg += (double)(gn[q] - gc[q]) * (gn[q] - gc[q]);
                g2 += (double)gc[q] * gc[q];
            }
            o.step[ck] = (float)(g2 > 0.0 ? dg / g2 : INFINITY);
            if (k == 0) {
                const float pw0 = s.pw[(size_t)c];
                o.expl0[(size_t)c] = pw0 > 0.0f ? (float)(-cn / pw0) : 0.0f;
            }
            fits[(size_t)k] = gn;
        }
        if (!fits[0].empty() && !(o.expl0[(size_t)c] >= p.min_explained)) {
            o.weak[(size_t)c] = 1;
            continue;
        }
        for (int k = 0; k < ns; k++) {
            if (fits[(size_t)k].empty())
                continue;
            const size_t ck = (size_t)c * ns + k;
            std::copy(fits[(size_t)k].begin(), fits[(size_t)k].end(), g_out + ck * blk);
            o.upd[ck] = 1;
            o.n_upd++;
        }
    }
    return o;
}

} // namespace gnss_cuda

#endif
