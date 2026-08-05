#ifndef GNSS_ELEM_CAL_HPP
#define GNSS_ELEM_CAL_HPP
/**
 * @file gnssElemCal.hpp
 * @brief Self-calibrated weighted coherent element sum for the record header (CHORD).
 *
 * The record header's correlation slots historically carried ONE reference element of
 * n_elements (gnssRecord.hpp: "a coherent sum needs the per-element phases that are the very
 * thing being measured"). This closes that loop: the per-element complex gains are estimated
 * FROM THE SATELLITE ITSELF (bootstrap maximum-ratio combining), and the header becomes the
 * calibrated weighted mean -- per-record SNR up ~sqrt(N_healthy), which is what makes the
 * per-record carrier phase estimable from one instance (the phase-floor fix, CHORD_GNSS_STATE
 * 8.21.5) and hands the broker's DLL/carrier loops the same gain for free.
 *
 * DESIGN
 *  - Weights: u_e = EMA[ G_e * conj(H) ] where H is the header prompt actually emitted
 *    (the reference element until warm, the sum afterwards -- bootstrap MRC). The satellite's
 *    common phase wander cancels in the product (it is common to G_e and H), so the cal is
 *    IMMUNE to the very effect it exists to fix.
 *  - Phase convention: weights are re-anchored so the REFERENCE element's weight is real-
 *    positive, i.e. the sum keeps the reference element's phase convention exactly. Downstream
 *    phase consumers (carrier loop, ADR arcs) see a quieter version of the same observable,
 *    not a new one. If the reference dies (|u_ref| ~ 0 vs the fleet), the strongest element
 *    takes over as anchor -- a one-time phase step, flagged via anchor_moved().
 *  - Scale: the sum is normalized by the total |weight|, so H stays in "one element" units --
 *    all-equal elements give the plain mean, and every downstream ratio (q = 2P/(E+L), A = G/E
 *    scaling) is unchanged.
 *  - Absent/unpowered elements: their u_e is an EMA of pure noise ~ 0, so MRC naturally
 *    weights them out; min_w_frac additionally hard-gates anything below that fraction of the
 *    strongest weight so a mostly-dead array (the 512-element future) does not sum hundreds of
 *    noise weights.
 *  - Warm-up: until ~3 time constants of updates the caller should emit the reference element
 *    unchanged (warm() == false) -- byte-identical to the historical behaviour.
 *
 * Header-only and stage-independent so the offline tools (scripts/gnss) can drive it with
 * synthetic per-element data; GnssGpuRecordAssemble is the production consumer.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace gnss {

class ElemCal {
public:
    ElemCal() = default;
    ElemCal(int n_elem, int ref, double tau_s, double min_w_frac) :
        _n(n_elem), _ref(ref), _tau(tau_s > 0.0 ? tau_s : 0.5),
        _min_w(std::max(0.0, min_w_frac)), _u((size_t)n_elem, {0.0, 0.0}) {}

    /// Drop all state (fresh acquisition: the fringes may have moved arbitrarily).
    void reset() {
        std::fill(_u.begin(), _u.end(), std::complex<double>(0.0, 0.0));
        _warmth = 0.0;
        _anchor = _ref;
    }

    /// True once the weights have integrated ~3 time constants and the sum should be used.
    bool warm() const {
        return _warmth > 0.95;
    }

    /// True if the phase anchor is currently NOT the configured reference element
    /// (reference too weak): the sum's phase convention moved. Log-worthy, once.
    bool anchor_moved() const {
        return _anchor != _ref;
    }

    /// Combine one correlator row (E/P/L/PH/RES -- any row, same weights): the calibrated
    /// weighted mean in "one element" units, reference-element phase convention. Call only
    /// when warm(); g is the per-element row, stride 1, length n_elem.
    std::complex<double> combine(const std::complex<double>* g) const {
        std::complex<double> s(0.0, 0.0);
        for (int e = 0; e < _n; ++e)
            if (_w[(size_t)e] != 0.0 || _wim[(size_t)e] != 0.0)
                s += std::complex<double>(_w[(size_t)e], -_wim[(size_t)e]) * g[(size_t)e];
        return s;
    }

    /// One record's cal update from the PROMPT row. @p g_prompt: per-element prompt
    /// correlations; @p h the header prompt actually emitted this record (reference element
    /// until warm, the sum afterwards); @p dt_s seconds since this PRN's previous record.
    /// Also refreshes the gated, anchored, normalized weights combine() uses.
    void update(const std::complex<double>* g_prompt, std::complex<double> h, double dt_s) {
        if (_n <= 0 || !(dt_s > 0.0) || std::norm(h) <= 0.0)
            return;
        // Clamped: after a coast/drop gap dt can span seconds, and an unclamped EMA would let
        // ONE noisy record replace the whole cal. 0.25 caps any single record's influence while
        // still converging in a handful of records after a genuine change.
        const double alpha = std::min(0.25, 1.0 - std::exp(-dt_s / _tau));
        const std::complex<double> hc = std::conj(h);
        for (int e = 0; e < _n; ++e)
            _u[(size_t)e] += alpha * (g_prompt[(size_t)e] * hc - _u[(size_t)e]);
        _warmth += alpha * (1.0 - _warmth);
        rebuild_weights();
    }

private:
    void rebuild_weights() {
        _w.assign((size_t)_n, 0.0);
        _wim.assign((size_t)_n, 0.0);
        // Anchor selection: the reference element unless it has collapsed relative to the
        // strongest -- the anchor sets the sum's phase convention, so it must itself be a
        // measurable phase. Hysteresis-free by intent: the min_w gate below uses the same
        // scale, so a reference healthy enough to be summed is healthy enough to anchor.
        double wmax = 0.0;
        int strongest = _ref;
        for (int e = 0; e < _n; ++e) {
            const double m = std::abs(_u[(size_t)e]);
            if (m > wmax) {
                wmax = m;
                strongest = e;
            }
        }
        if (wmax <= 0.0)
            return; // no signal yet: combine() contributes nothing, caller stays on warm()
        _anchor = (std::abs(_u[(size_t)_ref]) >= _min_w * wmax) ? _ref : strongest;
        // Re-anchor: v_e = u_e * conj(u_anchor)/|u_anchor| makes v_anchor real-positive and
        // gives sum(v_e^* G_e) exactly the anchor element's phase (the u's common bootstrap
        // phase cancels in the product).
        const std::complex<double> pin =
            std::conj(_u[(size_t)_anchor]) / std::abs(_u[(size_t)_anchor]);
        double norm = 0.0;
        for (int e = 0; e < _n; ++e) {
            const std::complex<double> v = _u[(size_t)e] * pin;
            const double m = std::abs(v);
            if (m < _min_w * wmax)
                continue; // dead/absent element: EMA of pure noise, gate it out
            _w[(size_t)e] = v.real();
            _wim[(size_t)e] = v.imag();
            norm += m;
        }
        if (norm > 0.0) {
            const double inv = 1.0 / norm;
            for (int e = 0; e < _n; ++e) {
                _w[(size_t)e] *= inv;
                _wim[(size_t)e] *= inv;
            }
        }
    }

    int _n = 0;
    int _ref = 0;
    int _anchor = 0;
    double _tau = 0.5;
    double _min_w = 0.02;
    double _warmth = 0.0;               ///< approaches 1 with ~tau; warm at > 0.95 (~3 tau)
    std::vector<std::complex<double>> _u; ///< EMA of G_e * conj(H) -- the raw cal products
    std::vector<double> _w, _wim;         ///< gated, anchored, normalized weights (re, im)
};

} // namespace gnss
#endif // GNSS_ELEM_CAL_HPP
