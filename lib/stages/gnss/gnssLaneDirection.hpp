/**
 * @file
 * @brief Per (slot, channel) direction of a tracked satellite from its own prompt row, for the
 *        science-N^2 projection of every known satellite (#190). Header-only, no CUDA.
 */
#ifndef GNSS_LANE_DIRECTION_HPP
#define GNSS_LANE_DIRECTION_HPP
#include "gnssElemSteer.hpp"

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
 * Quality: coherence = |mean|^2 / mean(|row|^2), both summed over the elements. A pure-noise
 * row averages to dt / (2 tau) (0.5% at 10.49 ms / 1 s); a satellite at per-record SNR rho per
 * element reads ~rho / (1 + rho), and its averaged direction carries SNR rho x 2 tau / dt per
 * element. Readers gate on it; nothing here decides.
 *
 * Phase: the steered frame pins the phase to the reference element's, so consecutive records
 * add coherently; a reanchor or PRN change on the slot is a reset.
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
        const double a = 1.0 - std::exp(-dt_s / _tau);
        for (int i = 0; i < _n; ++i)
            m[i] += a * (v[i] - m[i]);
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

} // namespace gnss
#endif
