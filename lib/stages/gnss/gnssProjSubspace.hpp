#ifndef GNSS_PROJ_SUBSPACE_HPP
#define GNSS_PROJ_SUBSPACE_HPP
/**
 * @file gnssProjSubspace.hpp
 * @brief Bright-satellite spatial projection for the GNSS despread combine: the interferer
 *        subspace tracker, the per-channel projector, and the process-wide board that hands a
 *        chain's subspace to its siblings.
 *  - gnss::ProjBasis
 *  - gnss::ProjSubspace
 *  - gnss::ProjBoard
 *
 * A satellite near boresight leaks into every other satellite's per-element despread row
 * through the code cross-correlation. Within a couple of degrees the leak dominates the weak
 * satellites' element vectors, and a per-satellite learner then calibrates the array onto the
 * interferer.
 *
 * Per channel the leak is spatially low rank: rank one (the interferer's own per-element
 * response a) for one bright satellite, rank two with a second one nearby. With Q an
 * orthonormal basis of that subspace, v' = v - Q (Q^H v) removes it from every other row
 * exactly, at 2kn complex multiply-adds per (row, channel), and costs each victim only the
 * fraction of its own response inside the span (~1/n_live for an unrelated direction).
 * Projection acts on the element axis and the despread on time, so projecting the record rows
 * is projecting the voltages, with no requantisation.
 *
 * Sources of a, in the order the basis takes them:
 *   1. the bright satellite's own prompt row, when this chain tracks it (zero latency);
 *   2. a sibling chain on the same GPU (same channels and element axis), via the board;
 *   3. the top eigenvectors of the probe rows' covariance. The probes (below-horizon PRNs) carry
 *      only noise and leak, so their stacked covariance is the interferer's a a^H for any
 *      emitter, including satellites the sky model drops and objects in no BRDC file.
 * A geometric model of a fails near boresight (the dish's E/H-plane sidelobe pattern), so every
 * vector here is measured from the current records.
 *
 * The covariance is stored without its autos. The per-record rows carry a common random phase
 * (the NCO/replica anchor), so a is estimated from EMA<v v^H>, which is phase-invariant; with
 * the autos left in, the top eigenvector of a noisy EMA is the noisiest element (often the
 * reference), and projecting that out destroys every cal. Zeroing the diagonal instead biases
 * the direction (see solve()), so the solver completes the diagonal from its own rank-k model:
 * the unbiased fit to the off-diagonals.
 *
 * Header-only; used by GnssGpuRecordAssemble (record rows) and GnssN2Project (push_cov()).
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace gnss {

/// An orthonormal basis Q [k][n] (row-major, k <= kmax) in the unsteered element frame of one
/// channel, and the projector v -> v - Q (Q^H v).
struct ProjBasis {
    using cd = std::complex<double>;
    int n = 0;
    int kmax = 0;
    int k = 0;
    std::vector<cd> q; ///< [kmax][n]

    void init(int n_elem, int k_max) {
        n = n_elem;
        kmax = std::max(0, k_max);
        k = 0;
        q.assign((size_t)n * kmax, cd(0.0, 0.0));
    }
    void clear() {
        k = 0;
    }

    /// Gram-Schmidt one candidate direction into the basis. A candidate whose component outside
    /// the current span carries less than @p min_new_frac of its energy is the same emitter seen
    /// twice (own row, sibling row, probe stack) and is dropped. Returns true if a basis vector
    /// was added.
    bool add(const cd* v, double min_new_frac = 0.5) {
        if (k >= kmax || n <= 0 || (size_t)(k + 1) * (size_t)n > q.size())
            return false; // last clause: a kmax raised above the allocation cannot overflow q
        cd* r = &q[(size_t)k * n];
        double e_in = 0.0;
        for (int i = 0; i < n; ++i) {
            r[i] = v[i];
            e_in += std::norm(v[i]);
        }
        if (!(e_in > 0.0))
            return false;
        for (int j = 0; j < k; ++j) {
            const cd* qj = &q[(size_t)j * n];
            cd s(0.0, 0.0);
            for (int i = 0; i < n; ++i)
                s += std::conj(qj[i]) * r[i];
            for (int i = 0; i < n; ++i)
                r[i] -= qj[i] * s;
        }
        double e_out = 0.0;
        for (int i = 0; i < n; ++i)
            e_out += std::norm(r[i]);
        if (e_out < min_new_frac * e_in)
            return false;
        const double inv = 1.0 / std::sqrt(e_out);
        for (int i = 0; i < n; ++i)
            r[i] *= inv;
        ++k;
        return true;
    }

    /// v <- v - Q (Q^H v). 2 k n complex multiply-adds.
    void project(cd* v) const {
        for (int j = 0; j < k; ++j) {
            const cd* qj = &q[(size_t)j * n];
            cd s(0.0, 0.0);
            for (int i = 0; i < n; ++i)
                s += std::conj(qj[i]) * v[i];
            for (int i = 0; i < n; ++i)
                v[i] -= qj[i] * s;
        }
    }

    /// Fraction of |w|^2 inside the span: sum_j |q_j^H w|^2 / |w|^2, the capture and cost
    /// measure (~1/n_live for a random direction). -1 if w is zero.
    double cos2(const cd* w) const {
        double e = 0.0;
        for (int i = 0; i < n; ++i)
            e += std::norm(w[i]);
        if (!(e > 0.0))
            return -1.0;
        double c = 0.0;
        for (int j = 0; j < k; ++j) {
            const cd* qj = &q[(size_t)j * n];
            cd s(0.0, 0.0);
            for (int i = 0; i < n; ++i)
                s += std::conj(qj[i]) * w[i];
            c += std::norm(s);
        }
        return c / e;
    }
};

/// Per-channel interferer-subspace tracker: an EMA covariance with a zero diagonal, solved by
/// warm-started power iteration with deflation. One instance covers every channel; the caller
/// pushes vectors (a bright satellite's own row, or each probe row) with push(), or whole
/// covariance matrices with push_cov().
class ProjSubspace {
public:
    using cd = std::complex<double>;

    ProjSubspace() = default;
    ProjSubspace(int n_elem, int n_chan, double tau_s, int rank_max) :
        _n(n_elem), _nc(n_chan), _kmax(std::max(1, rank_max)), _tau(tau_s > 0.0 ? tau_s : 0.5) {
        reset();
    }

    void reset() {
        _C.assign((size_t)_nc * _n * _n, cd(0.0, 0.0));
        _q.assign((size_t)_nc * _kmax * _n, cd(0.0, 0.0));
        _lam.assign((size_t)_nc * _kmax, 0.0);
        _frac.assign((size_t)_nc * _kmax, 0.0);
        _k.assign((size_t)_nc, 0);
        _warm.assign((size_t)_nc, 0.0);
        _nz.assign((size_t)_n, 0);
        _x.assign((size_t)_n, cd(0.0, 0.0));
        _y.assign((size_t)_n, cd(0.0, 0.0));
        _d.assign((size_t)_n, 0.0);
    }

    /// Accumulate one vector for channel @p ch: C <- (1 - alpha) C + alpha v v^H off the
    /// diagonal, alpha = 1 - exp(-dt_s/tau) clamped at 0.25. The horizon is tau in time only
    /// when @p dt_s is the time each push represents: m pushes per record, each passing the
    /// record's dt, shorten it to ~tau/m. Entries of elements that are exactly zero (dark feeds)
    /// are skipped, so the cost is n_live^2.
    void push(int ch, const cd* v, double dt_s) {
        if (ch < 0 || ch >= _nc || !(dt_s > 0.0))
            return;
        const double alpha = std::min(0.25, 1.0 - std::exp(-dt_s / _tau));
        int m = 0;
        for (int i = 0; i < _n; ++i)
            if (std::norm(v[i]) > 0.0)
                _nz[(size_t)m++] = i;
        cd* C = &_C[(size_t)ch * _n * _n];
        for (int a = 0; a < m; ++a) {
            const int i = _nz[(size_t)a];
            cd* Ci = C + (size_t)i * _n;
            const cd vi = v[i];
            for (int b = 0; b < m; ++b) {
                const int j = _nz[(size_t)b];
                if (j == i)
                    continue; // diagonal stays zero (file note)
                Ci[j] += alpha * (vi * std::conj(v[j]) - Ci[j]);
            }
        }
        _warm[(size_t)ch] += alpha * (1.0 - _warm[(size_t)ch]);
    }

    /// Accumulate one Hermitian covariance @p V (n x n, row-major, any diagonal) for channel
    /// @p ch: the EMA of its off-diagonal entries, the diagonal staying zero as for push().
    /// GnssN2Project feeds the correlator's matrix here once per integration, so the per-input
    /// autos, whose noise levels differ by feed, never enter the eigenproblem.
    void push_cov(int ch, const cd* V, double dt_s) {
        if (ch < 0 || ch >= _nc || !(dt_s > 0.0))
            return;
        const double alpha = std::min(0.25, 1.0 - std::exp(-dt_s / _tau));
        cd* C = &_C[(size_t)ch * _n * _n];
        for (int i = 0; i < _n; ++i) {
            cd* Ci = C + (size_t)i * _n;
            const cd* Vi = V + (size_t)i * _n;
            for (int j = 0; j < _n; ++j)
                if (j != i)
                    Ci[j] += alpha * (Vi[j] - Ci[j]);
        }
        _warm[(size_t)ch] += alpha * (1.0 - _warm[(size_t)ch]);
    }

    /// Re-solve channel @p ch: @p iters power iterations per component, warm-started from the
    /// previous solution and deflated by the components before it. Sets k(ch) by the rank gates
    /// below. Cost ~ iters * kmax * n^2. Returns k.
    ///
    /// The diagonal is completed from the model, not left at zero. The stored matrix has no
    /// autos (file note). The top eigenvector of a rank-one matrix with its diagonal removed is
    /// biased toward uniform element gains, the more so the more the gains differ (a 14 dB
    /// spread over 16 elements limits the null to ~-26 dB), and the matrix has a spurious
    /// negative second component. Each iteration therefore adds back the diagonal the current
    /// rank-k model predicts, diag(sum_i lam_i |q_i|^2), with the eigenvalue taken as the
    /// least-squares fit of lam x x^H to the off-diagonal entries (for the first component,
    /// lam = x^H C x / (1 - sum |x_r|^4)). The fixed point is the unbiased rank-one fit of the
    /// off-diagonals, and the energy fractions below are likewise taken over the off-diagonal
    /// part only.
    int solve(int ch, int iters = 2) {
        if (ch < 0 || ch >= _nc)
            return 0;
        const cd* C = &_C[(size_t)ch * _n * _n];
        double fro2 = 0.0;
        for (size_t i = 0; i < (size_t)_n * _n; ++i)
            fro2 += std::norm(C[i]);
        std::vector<cd>& x = _x;
        std::vector<cd>& y = _y;
        std::vector<double>& d = _d; // model diagonal of the components before j
        std::fill(d.begin(), d.end(), 0.0);
        double remaining = fro2;
        int k = 0;
        for (int j = 0; j < _kmax; ++j) {
            cd* qj = &_q[((size_t)ch * _kmax + j) * _n];
            double e = 0.0;
            for (int i = 0; i < _n; ++i)
                e += std::norm(qj[i]);
            if (e > 0.0) {
                for (int i = 0; i < _n; ++i)
                    x[(size_t)i] = qj[i] / std::sqrt(e);
            } else {
                // Cold start: the column of C with the most energy (a deterministic, signal-
                // aligned seed; a random one would need many more iterations).
                int best = 0;
                double eb = -1.0;
                for (int c = 0; c < _n; ++c) {
                    double ec = 0.0;
                    for (int i = 0; i < _n; ++i)
                        ec += std::norm(C[(size_t)i * _n + c]);
                    if (ec > eb) {
                        eb = ec;
                        best = c;
                    }
                }
                if (!(eb > 0.0))
                    break;
                for (int i = 0; i < _n; ++i)
                    x[(size_t)i] = C[(size_t)i * _n + best];
                const double inv = 1.0 / std::sqrt(eb);
                for (int i = 0; i < _n; ++i)
                    x[(size_t)i] *= inv;
            }
            double lam = 0.0;
            for (int it = 0; it < iters; ++it) {
                // The least-squares eigenvalue of lam x x^H against the OFF-diagonal entries
                // (x unit): lam = (x^H C_off x + sum_r d_r |x_r|^2) / (1 - sum_r |x_r|^4).
                cd rq(0.0, 0.0);
                double s4 = 0.0, sd = 0.0;
                for (int r = 0; r < _n; ++r) {
                    const cd* Cr = C + (size_t)r * _n;
                    cd s(0.0, 0.0);
                    for (int c = 0; c < _n; ++c)
                        s += Cr[c] * x[(size_t)c];
                    y[(size_t)r] = s; // C_off x, reused below
                    rq += std::conj(x[(size_t)r]) * s;
                    const double x2 = std::norm(x[(size_t)r]);
                    s4 += x2 * x2;
                    sd += d[(size_t)r] * x2;
                }
                const double den = 1.0 - s4;
                lam = (den > 1e-6) ? (rq.real() + sd) / den : rq.real();
                // y = (C_off + diag(d + lam |x|^2)) x - sum_{i<j} lam_i q_i q_i^H x, then
                // orthogonalised against q_i (i<j).
                for (int r = 0; r < _n; ++r)
                    y[(size_t)r] += (d[(size_t)r] + lam * std::norm(x[(size_t)r])) * x[(size_t)r];
                for (int i = 0; i < j; ++i) {
                    const cd* qi = &_q[((size_t)ch * _kmax + i) * _n];
                    cd s(0.0, 0.0);
                    for (int r = 0; r < _n; ++r)
                        s += std::conj(qi[r]) * x[(size_t)r];
                    const cd t = s * _lam[(size_t)ch * _kmax + i];
                    for (int r = 0; r < _n; ++r)
                        y[(size_t)r] -= qi[r] * t;
                }
                for (int i = 0; i < j; ++i) {
                    const cd* qi = &_q[((size_t)ch * _kmax + i) * _n];
                    cd s(0.0, 0.0);
                    for (int r = 0; r < _n; ++r)
                        s += std::conj(qi[r]) * y[(size_t)r];
                    for (int r = 0; r < _n; ++r)
                        y[(size_t)r] -= qi[r] * s;
                }
                double ny = 0.0;
                for (int r = 0; r < _n; ++r)
                    ny += std::norm(y[(size_t)r]);
                if (!(ny > 0.0)) {
                    lam = 0.0;
                    break;
                }
                const double inv = 1.0 / std::sqrt(ny);
                for (int r = 0; r < _n; ++r)
                    x[(size_t)r] = y[(size_t)r] * inv;
            }
            // Pin the global phase so consecutive solves (and siblings) agree on a convention:
            // the largest-magnitude element real positive.
            int imax = 0;
            double amax = -1.0;
            for (int r = 0; r < _n; ++r)
                if (std::abs(x[(size_t)r]) > amax) {
                    amax = std::abs(x[(size_t)r]);
                    imax = r;
                }
            if (amax > 0.0) {
                const cd pin = std::conj(x[(size_t)imax]) / amax;
                for (int r = 0; r < _n; ++r)
                    x[(size_t)r] *= pin;
            }
            double s4 = 0.0;
            for (int r = 0; r < _n; ++r) {
                qj[r] = x[(size_t)r];
                const double x2 = std::norm(x[(size_t)r]);
                s4 += x2 * x2;
                d[(size_t)r] += lam * x2;
            }
            _lam[(size_t)ch * _kmax + j] = lam;
            // Off-diagonal energy of this component: |lam|^2 (1 - sum |q_r|^4).
            const double l2 = lam * lam * std::max(0.0, 1.0 - s4);
            _frac[(size_t)ch * _kmax + j] = (remaining > 0.0) ? std::min(1.0, l2 / remaining) : 0.0;
            remaining = std::max(0.0, remaining - l2);
            // Rank gates. The first component is accepted whenever it exists (any trigger on it
            // is the caller's). Each further one must carry rel_min of the first eigenvalue and
            // frac_next_min of the energy left after the ones before it; a noise-only remainder
            // gives ~4/n_live (about 0.15 at 26).
            const double lam0 = std::fabs(_lam[(size_t)ch * _kmax]);
            if (j > 0
                && (std::fabs(lam) < rel_min * lam0
                    || _frac[(size_t)ch * _kmax + j] < frac_next_min))
                break;
            ++k;
        }
        _k[(size_t)ch] = k;
        return k;
    }

    int k(int ch) const {
        return (ch >= 0 && ch < _nc) ? _k[(size_t)ch] : 0;
    }
    const cd* q(int ch, int j) const {
        return &_q[((size_t)ch * _kmax + j) * _n];
    }
    double lambda(int ch, int j) const {
        return _lam[(size_t)ch * _kmax + j];
    }
    /// Fraction of the (deflated) matrix energy carried by component j: ~1 for a clean rank-1
    /// leak, ~0.15 for noise alone at 26 live elements; the callers trigger on it.
    double frac(int ch, int j) const {
        return _frac[(size_t)ch * _kmax + j];
    }
    /// Approaches 1 with tau: 0.63 after one time constant of pushes.
    double warmth(int ch) const {
        return (ch >= 0 && ch < _nc) ? _warm[(size_t)ch] : 0.0;
    }
    bool warm(int ch) const {
        return warmth(ch) > 0.6;
    }

    double rel_min = 0.2;       ///< component j >= 1 needs |lam_j| >= rel_min * |lam_0|
    double frac_next_min = 0.3; ///< ... and this fraction of the remaining energy

private:
    int _n = 0, _nc = 0, _kmax = 1;
    double _tau = 0.5;
    std::vector<cd> _C;        ///< [nc][n][n], diagonal zero
    std::vector<cd> _q;        ///< [nc][kmax][n]
    std::vector<double> _lam;  ///< [nc][kmax]
    std::vector<double> _frac; ///< [nc][kmax]
    std::vector<int> _k;       ///< [nc]
    std::vector<double> _warm; ///< [nc]
    std::vector<int> _nz;      ///< scratch: indices of the non-zero elements of the last push
    std::vector<cd> _x, _y;    ///< solve() scratch (no per-call allocation on the record path)
    std::vector<double> _d;
};

/// One chain's published subspace for one channel.
struct ProjEntry {
    std::string owner;  ///< the publishing stage's unique_name (+ "/probe" for a probe stack,
                        ///< + "/lane/<prn>" for a #190 lane direction)
    int src = 0;        ///< 0 = the satellite's own row, 1 = probe-stack eigenvector, 2 = lane
                        ///< direction (gnssLaneDirection.hpp: a tracked PRN's averaged prompt row)
    double t_pub = 0.0; ///< steady seconds of publication
    double power = 0.0; ///< src 2: the row's EMA power (despread units; orders the lanes)
    double coh = 0.0;   ///< src 2: LaneDirection::coherence, the direction's quality in [0, 1]
    int n = 0, k = 0;
    std::vector<std::complex<float>> q; ///< [k][n]
    // src 2: the track the row was steered with (ElemSteer::track), so the direction can be
    // re-steered elsewhere (gnssFleetLane.hpp); xfer 1 = a lane_transfer_freq_ids copy.
    double t_utc = 0.0;                   ///< the instant los describes (unix s; 0 = none)
    double los[3] = {0.0, 0.0, 0.0};      ///< line of sight, ENU unit vector
    double los_rate[3] = {0.0, 0.0, 0.0}; ///< its rate per second
    int xfer = 0;
    double held_s = 0.0; ///< src 2: seconds since the row was last coherent (0 = live)
};

/// Process-wide board. The sibling assemblers of one GPU are threads of one kotekan process and
/// despread the same channels from the same ring with the same element axis, so a subspace one
/// chain measures applies to the others directly, matched by freq_id, never by position. No
/// same-record rendezvous is needed: a direction a second old still nulls to ~-30 dB, so
/// readers use every entry younger than max_age_s. One mutex; writers copy k x n
/// complex<float> per channel per record.
class ProjBoard {
public:
    static ProjBoard& instance() {
        static ProjBoard b;
        return b;
    }
    void publish(const std::string& group, int freq_id, const ProjEntry& e) {
        std::lock_guard<std::mutex> lk(_m);
        _e[std::make_tuple(group, freq_id, e.owner)] = e;
    }
    /// Drop every entry of @p owner in @p group (the source went away).
    void retire(const std::string& group, const std::string& owner) {
        std::lock_guard<std::mutex> lk(_m);
        for (auto it = _e.begin(); it != _e.end();) {
            if (std::get<0>(it->first) == group && std::get<2>(it->first) == owner)
                it = _e.erase(it);
            else
                ++it;
        }
    }
    /// Visit every fresh entry for (group, freq_id) not published by @p not_owner (nor any
    /// "<not_owner>/..." owner, i.e. its probe stack) under the lock, without copying.
    template<typename F>
    void for_each(const std::string& group, int freq_id, const std::string& not_owner, double now_s,
                  double max_age_s, F&& f) const {
        std::lock_guard<std::mutex> lk(_m);
        for (const auto& [key, e] : _e) {
            if (std::get<0>(key) != group || std::get<1>(key) != freq_id)
                continue;
            if (e.k <= 0 || now_s - e.t_pub > max_age_s)
                continue;
            if (e.owner == not_owner
                || (e.owner.size() > not_owner.size()
                    && e.owner.compare(0, not_owner.size(), not_owner) == 0
                    && e.owner[not_owner.size()] == '/'))
                continue;
            f(e);
        }
    }

private:
    ProjBoard() = default;
    mutable std::mutex _m;
    std::map<std::tuple<std::string, int, std::string>, ProjEntry> _e;
};

} // namespace gnss

#endif // GNSS_PROJ_SUBSPACE_HPP
