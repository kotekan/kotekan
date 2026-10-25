/**
 * @file
 * @brief Per (slot, channel) direction of a tracked satellite from its own prompt row, for the
 *        science-N^2 projection of every known satellite (#190), and the per-element
 *        instrumental delay that carries it across frequency. Header-only, no CUDA.
 */
#ifndef GNSS_LANE_DIRECTION_HPP
#define GNSS_LANE_DIRECTION_HPP
#include "gnssElemSteer.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace gnss {

/**
 * @class LaneDirection
 * @brief EMA of the STEERED prompt row per (slot, channel): the satellite's element vector.
 *
 * The tracker's prompt row for one channel is the satellite's element response times the
 * despread gain, plus noise. Multiplying by the steering phasor D (what the assembler does
 * before combining) removes the geometric phase, so in the steered frame the row is a slowly
 * varying vector and a plain exponential average of it converges to the response at the
 * record cadence with no eigensolve: O(n) per record per (slot, channel), where the phase-1
 * own-row trackers (ProjSubspace::push, v v^H) cost O(n^2) and run only for boresight
 * candidates. The direction in the raw correlator frame is conj(D) o mean, unit-normalised.
 *
 * Quality: coherence = |mean|^2 / mean(|row|^2), both summed over the elements. A satellite
 * at per-record SNR rho per element reads ~rho / (1 + rho), and its averaged direction
 * carries SNR rho x 2 tau / dt per element. Pure noise does not read zero: the phase
 * alignment below projects every record onto the mean, so noise accumulates ~1/n of its
 * energy along it and the floor is ~dt / (2 tau) + 1 / n (0.016 at n 96, tau 1 s; 0.027
 * measured at n 48 in the boost test). Readers gate at a few times that; nothing here
 * decides.
 *
 * Phase: steering removes the geometric phase only. Each record's row still carries the
 * satellite's residual carrier phase and its symbol sign, common to all elements and
 * arbitrary from record to record, so a plain average would cancel. push() aligns every
 * record to the running mean first, by the phase of mean^H row (the O(n) form of the rank-1
 * power iteration ProjSubspace does on v v^H): the direction is then phase-invariant and
 * only its own phase origin is arbitrary. A PRN change on the slot is a reset.
 */
class LaneDirection {
public:
    using cd = std::complex<double>;
    LaneDirection() = default;
    LaneDirection(int n_elem, int n_chan, int n_slot, double tau_s) :
        _n(n_elem), _nc(n_chan), _ns(n_slot), _tau(tau_s > 0.0 ? tau_s : 0.5),
        _m((size_t)n_slot * n_chan * n_elem, cd(0.0, 0.0)), _p((size_t)n_slot * n_chan, 0.0),
        _age((size_t)n_slot * n_chan, 0.0) {}

    int n_elem() const {
        return _n;
    }
    int n_chan() const {
        return _nc;
    }
    int n_slot() const {
        return _ns;
    }
    double tau_s() const {
        return _tau;
    }

    /// Feed one record's steered row v (v[i] = row_i * D_i) for (slot, ch), dt_s after the
    /// previous one. The first record after a reset sets the state outright.
    void push(int slot, int ch, const cd* v, double dt_s) {
        if (!ok(slot, ch))
            return;
        cd* m = &_m[idx(slot, ch) * (size_t)_n];
        double& p = _p[idx(slot, ch)];
        double& age = _age[idx(slot, ch)];
        double e = 0.0;
        for (int i = 0; i < _n; ++i)
            e += std::norm(v[i]);
        if (age <= 0.0 || !(dt_s > 0.0)) {
            for (int i = 0; i < _n; ++i)
                m[i] = v[i];
            p = e;
            age = (dt_s > 0.0) ? dt_s : 1e-3;
            return;
        }
        // Align the record's common phase to the mean: rotate v by conj(phase(mean^H v)).
        // A record orthogonal to the mean (pure noise, or the first after a sign flip of
        // a cold mean) keeps its phase; its weight in the mean is small anyway.
        cd ip(0.0, 0.0);
        for (int i = 0; i < _n; ++i)
            ip += std::conj(m[i]) * v[i];
        const double ipa = std::abs(ip);
        const cd rot = ipa > 0.0 ? std::conj(ip) / ipa : cd(1.0, 0.0);
        const double a = 1.0 - std::exp(-dt_s / _tau);
        for (int i = 0; i < _n; ++i)
            m[i] += a * (v[i] * rot - m[i]);
        p += a * (e - p);
        age += dt_s;
    }

    /// Forget a slot (its PRN changed or stopped): every channel's state goes cold.
    void reset(int slot) {
        if (slot < 0 || slot >= _ns)
            return;
        for (int ch = 0; ch < _nc; ++ch) {
            _age[idx(slot, ch)] = 0.0;
            _p[idx(slot, ch)] = 0.0;
        }
    }

    /// At least 2 tau of records have been averaged.
    bool warm(int slot, int ch) const {
        return ok(slot, ch) && _age[idx(slot, ch)] >= 2.0 * _tau;
    }
    /// Seconds averaged since the last reset.
    double age_s(int slot, int ch) const {
        return ok(slot, ch) ? _age[idx(slot, ch)] : 0.0;
    }
    /// EMA of sum_i |row_i|^2 (the row's power, the despread's units).
    double power(int slot, int ch) const {
        return ok(slot, ch) ? _p[idx(slot, ch)] : 0.0;
    }
    /// |mean|^2 / power, in [0, 1]; 0 when cold or powerless.
    double coherence(int slot, int ch) const {
        if (!ok(slot, ch) || !(_p[idx(slot, ch)] > 0.0))
            return 0.0;
        const cd* m = &_m[idx(slot, ch) * (size_t)_n];
        double e = 0.0;
        for (int i = 0; i < _n; ++i)
            e += std::norm(m[i]);
        return e / _p[idx(slot, ch)];
    }
    /// The averaged steered row (not normalised).
    const cd* mean(int slot, int ch) const {
        return ok(slot, ch) ? &_m[idx(slot, ch) * (size_t)_n] : nullptr;
    }
    /// Unit direction in the raw correlator element frame: out[i] = conj(D_i) mean_i / |mean|.
    /// D is the steering table row for (slot, ch) (ElemSteer::cf, n_elem long). Returns false
    /// (out untouched) when cold or the mean is zero.
    bool direction(int slot, int ch, const ElemSteer::cf* D, cd* out) const {
        if (!ok(slot, ch) || _age[idx(slot, ch)] <= 0.0)
            return false;
        const cd* m = &_m[idx(slot, ch) * (size_t)_n];
        double e = 0.0;
        for (int i = 0; i < _n; ++i)
            e += std::norm(m[i]);
        if (!(e > 0.0))
            return false;
        const double inv = 1.0 / std::sqrt(e);
        for (int i = 0; i < _n; ++i)
            out[i] = std::conj(cd(D[i].real(), D[i].imag())) * m[i] * inv;
        return true;
    }

private:
    bool ok(int slot, int ch) const {
        return slot >= 0 && slot < _ns && ch >= 0 && ch < _nc && _n > 0;
    }
    size_t idx(int slot, int ch) const {
        return (size_t)slot * _nc + ch;
    }
    int _n = 0, _nc = 0, _ns = 0;
    double _tau = 0.5;
    std::vector<cd> _m;
    std::vector<double> _p, _age;
};

/**
 * @class InstDelay
 * @brief Per-element instrumental delay from the lane directions of adjacent channels, for
 *        lane_transfer_freq_ids.
 *
 * A lane direction q(f) is the raw-frame element vector: geometric phase times each element's
 * instrumental response. ElemSteer::shift moves only the geometric phase across frequency; the
 * instrumental phase has its own per-element slope, a delay tau_e of a few ns, so a transfer
 * q(f2) = q(f1) conj(shift(f2 - f1)) turns each element by 2 pi (f2 - f1) tau_e too.
 *
 * For one slot's channels a and b = a + df0, z_e = q_b[e] conj(q_a[e]) shift_e(df0) is
 * exp(i 2 pi df0 tau_e) |q_a[e] q_b[e]| times a phase common to the elements (each channel's
 * direction has its own phase origin), removed by referencing z to its sum over the elements.
 * add() accumulates the referenced z per element, pooled over pairs and slots and forgotten
 * with time constant tau_s; tau_e = arg(sum) / (2 pi df0). The delays are relative to the
 * array: a delay common to every element is a common phase, which QQ^H does not see.
 * apply() turns a transferred direction by exp(i 2 pi df tau_e).
 */
class InstDelay {
public:
    using cd = std::complex<double>;
    InstDelay() = default;
    /// @param df0_hz the channel spacing of the pairs fed to add(), Hz.
    /// @param tau_s forgetting time constant, seconds.
    InstDelay(int n_elem, double df0_hz, double tau_s) :
        _n(n_elem), _df0(df0_hz), _tau(tau_s > 0.0 ? tau_s : 300.0),
        _c((size_t)n_elem, cd(0.0, 0.0)), _z((size_t)n_elem) {}

    /// Move the clock to @p t_s (seconds), forgetting by exp(-dt / tau_s). Call before add().
    void advance(double t_s) {
        if (_t_first < 0.0) {
            _t_first = _t_last = t_s;
            return;
        }
        const double dt = t_s - _t_last;
        if (!(dt > 0.0))
            return;
        const double f = std::exp(-dt / _tau);
        for (cd& c : _c)
            c *= f;
        _w *= f;
        _t_last = t_s;
    }

    /// One channel pair of one slot: unit directions @p qa at f and @p qb at f + df0, with
    /// @p shift = ElemSteer::shift(los, df0) for the slot's line of sight, weighted by @p w.
    void add(const cd* qa, const cd* qb, const ElemSteer::cf* shift, double w) {
        if (_n <= 0 || !(w > 0.0))
            return;
        cd sum(0.0, 0.0);
        for (int i = 0; i < _n; ++i) {
            _z[(size_t)i] = qb[i] * std::conj(qa[i]) * cd(shift[i].real(), shift[i].imag());
            sum += _z[(size_t)i];
        }
        const double a = std::abs(sum);
        if (!(a > 0.0))
            return;
        const cd ref = w * std::conj(sum) / a;
        for (int i = 0; i < _n; ++i)
            _c[(size_t)i] += _z[(size_t)i] * ref;
        _w += w;
    }

    /// The estimate has averaged for at least tau_s.
    bool warm() const {
        return _w > 0.0 && _t_last - _t_first >= _tau;
    }
    /// Pair weight accumulated (forgotten like the estimate).
    double weight() const {
        return _w;
    }
    /// Element @p i's instrumental delay, seconds, up to a delay common to all elements.
    double delay_s(int i) const {
        return std::arg(_c[(size_t)i]) / (2.0 * M_PI * _df0);
    }
    /// Spread of the delays over the elements, seconds: rms about the mean, each element
    /// weighted by its accumulated magnitude (dead elements read near zero weight).
    double rms_s() const {
        double sw = 0.0, s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < _n; ++i) {
            const double w = std::abs(_c[(size_t)i]), d = delay_s(i);
            sw += w;
            s1 += w * d;
            s2 += w * d * d;
        }
        return sw > 0.0 ? std::sqrt(std::max(0.0, s2 / sw - (s1 / sw) * (s1 / sw))) : 0.0;
    }
    /// Turn a direction transferred by @p df_hz: q[i] *= exp(i 2 pi df tau_i).
    void apply(double df_hz, cd* q) const {
        for (int i = 0; i < _n; ++i)
            q[i] *= std::polar(1.0, 2.0 * M_PI * df_hz * delay_s(i));
    }

private:
    int _n = 0;
    double _df0 = 1.0, _tau = 300.0;
    double _w = 0.0, _t_first = -1.0, _t_last = -1.0;
    std::vector<cd> _c; ///< [n_elem] sum of the referenced pair products
    std::vector<cd> _z; ///< [n_elem] scratch: one pair's products
};

} // namespace gnss
#endif
