#ifndef GNSS_ELEM_CAL_HPP
#define GNSS_ELEM_CAL_HPP
/**
 * @file gnssElemCal.hpp
 * @brief Self-calibrated element combining for the record header (CHORD): noise-weighted MRC,
 *        plus the SPLIT-APERTURE per-record sky-phase estimator.
 *
 * TWO JOBS, one set of weights.
 *
 * (1) COMBINE. The header's correlation slots historically carried ONE reference element of
 *     n_elements (gnssRecord.hpp). Here they become the calibrated weighted mean, with the
 *     per-element complex gains estimated FROM THE SATELLITE ITSELF (bootstrap MRC).
 *
 * (2) THE SKY PHASE. Measured 2026-08-05: the per-record prompt phase carries a common error of
 *     ~0.75 rad that is 0.98-coherent across instances (and across elements) but only ~0.57
 *     autocorrelated record-to-record -- i.e. it is COMMON IN SPACE and nearly WHITE IN TIME.
 *     That is why the temporal tracker (gnss::phase_track_loo) buys nothing on sky data: a
 *     record's neighbours do not predict it. But the OTHER ELEMENTS do, instantly, because they
 *     see the same sky with independent thermal noise. @ref combine_split does that, by SPLIT
 *     APERTURE -- two disjoint halves, each derotated by the other's phase. See its note for why
 *     it must be disjoint halves and NOT leave-one-element-out (LOO is unbiased in the mean but
 *     silently destroys the noise estimate, and was caught doing so on synthetic data).
 *
 * NOISE-WEIGHTED MRC. Weighting by measured signal alone (w_e = u_e) is optimal only if every
 * element has the SAME noise. CHORD's do not -- some feeds have oscillating amplifiers, and an
 * oscillating element carries real signal (so it earns a weight) while injecting excess noise.
 * So each element's noise power is estimated as the part of its power NOT correlated with the
 * (leave-one-out) reference, and the weight is w_e = u_e / sigma_e^2. A noisy element then
 * down-weights ITSELF, with no threshold to tune and no list of bad feeds to maintain.
 *
 * Every reference used to build the WEIGHTS is leave-one-out, so no element's noise enters its
 * own gain or its own noise estimate. The per-record PHASE correction is a stricter construction
 * still (disjoint halves) -- see @ref combine_split.
 *
 * Header-only and stage-independent so the offline tools can drive it with synthetic
 * per-element data; GnssGpuRecordAssemble is the production consumer.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace gnss {

class ElemCal {
public:
    using cd = std::complex<double>;

    ElemCal() = default;
    ElemCal(int n_elem, int ref, double tau_s, double min_w_frac) :
        _n(n_elem), _ref(ref), _tau(tau_s > 0.0 ? tau_s : 0.5),
        _min_w(std::max(0.0, min_w_frac)) {
        reset();
    }

    /// Drop all state (fresh acquisition: the fringes may have moved arbitrarily).
    void reset() {
        _u.assign((size_t)_n, cd(0.0, 0.0));
        _p.assign((size_t)_n, 0.0);
        _q.assign((size_t)_n, 0.0);
        _w.assign((size_t)_n, cd(0.0, 0.0));
        _half.assign((size_t)_n, 0);
        _split_ok = 0;
        _wsum = 0.0;
        _warmth = 0.0;
        _anchor = _ref;
    }

    /// True once the weights have integrated ~3 time constants and the sum should be used.
    bool warm() const {
        return _warmth > 0.95 && _wsum > 0.0;
    }

    /// True if the phase anchor is currently NOT the configured reference element
    /// (reference too weak): the sum's phase convention moved. Log-worthy, once.
    bool anchor_moved() const {
        return _anchor != _ref;
    }

    /// Calibrated weighted mean of one correlator row (E/P/L/PH/RES -- same weights, same
    /// antennas), in "one element" units and the reference element's phase convention.
    /// This is what the record HEADER carries: it must stay a plain linear combination, because
    /// its phase IS the carrier-phase/ADR observable (gnssRecord.hpp).
    cd combine(const cd* g) const {
        if (_wsum <= 0.0)
            return cd(0.0, 0.0);
        cd s(0.0, 0.0);
        for (int e = 0; e < _n; ++e)
            s += std::conj(_w[(size_t)e]) * g[(size_t)e];
        return s / _wsum;
    }

    /// SKY-PHASE-CORRECTED combine, by SPLIT APERTURE: the array is partitioned into two
    /// disjoint halves A and B (balanced in weight), and each half is derotated by the OTHER
    /// half's phase:
    ///
    ///     corrected = A * e^{-i arg(B)}  +  B * e^{-i arg(A)}
    ///
    /// ⚠️ IT MUST BE DISJOINT HALVES, NOT LEAVE-ONE-ELEMENT-OUT. Per-element LOO looks stronger
    /// (each estimate uses N-1 elements instead of N/2) and IS unbiased in the mean -- but it
    /// destroys the NOISE ESTIMATE, which is what every downstream significance number is built
    /// on. With N elements, every element's leave-one-out reference is nearly the same vector
    /// (rest_e ~ tot), so every term gets rotated by ~arg(tot) and the record collapses to
    /// ~|tot|: real, positive, every record. coherent_sum then takes its noise from the
    /// component orthogonal to the sum, finds it artificially tiny, and reports a wildly
    /// inflated SNR. Measured on synthetic: 47.3 against a GENIE (true phase removed) of 17.7 --
    /// i.e. it "beat" perfect knowledge, the signature of a self-referential estimator.
    /// Two disjoint halves have no such collapse: arg(A) - arg(B) stays random under noise, so
    /// the orthogonal component survives and the statistic keeps its meaning.
    ///
    /// This is the deep fold's input -- NOT the header's, which must keep the raw phase for ADR.
    cd combine_split(const cd* g) const {
        if (_wsum <= 0.0 || _split_ok == 0)
            return cd(0.0, 0.0);
        cd A(0.0, 0.0), B(0.0, 0.0);
        for (int e = 0; e < _n; ++e) {
            const cd c = std::conj(_w[(size_t)e]) * g[(size_t)e];
            (_half[(size_t)e] ? B : A) += c;
        }
        if (std::norm(A) <= 0.0 || std::norm(B) <= 0.0)
            return cd(0.0, 0.0);
        // ONE-WAY: A is integrated, B is ONLY a phase reference. NOT the symmetric
        // A*e^{-i arg B} + B*e^{-i arg A}, which expands to
        // (|A|+|B|) cos d + i (|A|-|B|) sin d and so annihilates the imaginary part whenever the
        // halves are balanced -- the same trap as the fleet combine's, and it was still here:
        // measured 18.0 against a GENIE of 16.5, i.e. beating perfect knowledge by 9%.
        // The price of one-way is that half the aperture measures while half integrates (~sqrt2);
        // that is the honest cost of not knowing the phase.
        return A * std::polar(1.0, -std::arg(B)) / _wsum;
    }

    /// Per-element gate diagnostics (offline tools; see the gate in rebuild_weights). @p rho2 is
    /// the squared correlation of this element with its leave-one-out reference, @p thresh the
    /// significance bar it must clear, @p mag the resulting MRC weight magnitude (0 = gated out).
    void diag(int e, double& rho2, double& thresh, double& mag) const {
        const double p = _p[(size_t)e], q = _q[(size_t)e];
        const double n_eff = (_alpha > 0.0) ? (2.0 - _alpha) / _alpha : 1.0;
        rho2 = (p > 0.0 && q > 0.0) ? std::norm(_u[(size_t)e]) / (p * q) : 0.0;
        thresh = 9.0 / n_eff;
        mag = std::abs(_w[(size_t)e]);
    }

    /// One record's cal update from the PROMPT row. @p dt_s is seconds since this PRN's previous
    /// record. Every per-element statistic is formed against that element's LEAVE-ONE-OUT
    /// reference, so no element's noise contaminates its own gain or its own noise estimate.
    void update(const cd* g, double dt_s) {
        if (_n <= 0 || !(dt_s > 0.0))
            return;
        // Clamped: after a coast/drop gap dt can span seconds, and an unclamped EMA would let
        // ONE noisy record replace the whole cal.
        const double alpha = std::min(0.25, 1.0 - std::exp(-dt_s / _tau));
        _alpha = alpha;
        // Current best combination, for the leave-one-out references. Before any weight exists
        // the reference element bootstraps it (and cannot reference itself -- its own weight
        // appears once the others have one, a few records later).
        // ⚠️ THE CAL REFERENCE IS FIXED (the reference ELEMENT), NOT the evolving weighted sum.
        // The self-consistent version -- reference each element against the current combination
        // of the others -- looks strictly better (more SNR in the reference) and DOES NOT WORK:
        // the weights are rebuilt from noisy statistics every record, so the combination's phase
        // jitters record to record, and the EMA of G_e*conj(reference) averages that jitter to
        // ZERO. Measured on synthetic data: the per-element correlation collapsed to
        // rho^2 = 0.0007 against 1/n_eff = 0.00105 -- i.e. exactly the value uncorrelated data
        // gives, so the gate then threw out every live element and left one. The bootstrap ate
        // itself. A FIXED reference cannot jitter, and it is still leave-one-out for every
        // element except the reference (G_ref does not contain G_e), which is what keeps each
        // element's noise out of its own gain and its own noise estimate.
        //
        // The reference element itself is the one special case: it has no fixed external
        // reference, so it is calibrated against the leave-it-out weighted sum of the others,
        // which exists once they are warm. Until then it keeps its bootstrap weight of 1 (the
        // header is then exactly the reference element -- the historical behaviour).
        cd tot(0.0, 0.0);
        const bool have_w = _wsum > 0.0;
        if (have_w)
            for (int e = 0; e < _n; ++e)
                tot += std::conj(_w[(size_t)e]) * g[(size_t)e];
        for (int e = 0; e < _n; ++e) {
            const cd ge = g[(size_t)e];
            const cd rest = (e != _ref)
                                ? g[(size_t)_ref]
                                : (have_w ? (tot - std::conj(_w[(size_t)e]) * ge) : cd(0.0, 0.0));
            if (std::norm(rest) <= 0.0)
                continue; // no independent reference for this element yet
            _u[(size_t)e] += alpha * (ge * std::conj(rest) - _u[(size_t)e]);
            _p[(size_t)e] += alpha * (std::norm(ge) - _p[(size_t)e]);
            _q[(size_t)e] += alpha * (std::norm(rest) - _q[(size_t)e]);
        }
        _warmth += alpha * (1.0 - _warmth);
        rebuild_weights();
    }

private:
    void rebuild_weights() {
        std::fill(_w.begin(), _w.end(), cd(0.0, 0.0));
        _wsum = 0.0;
        // Per-element gain (u_e) and NOISE power: the part of |G_e|^2 that is NOT correlated
        // with the leave-one-out reference. |u_e|^2/_q is the reference-correlated (signal)
        // power; whatever is left is this element's own noise, oscillating amplifiers included.
        std::vector<double> mag((size_t)_n, 0.0);
        double mmax = 0.0;
        int strongest = _ref;
        int self_ref = -1;   // element whose only reference is the sum it belongs to (see below)
        // STATISTICAL GATE, not a fraction-of-the-strongest. u_e is an EMA of G_e*conj(ref) over
        // ~n_eff records; a DEAD element's u_e is not zero, it is a random walk of magnitude
        // ~sqrt(p_e*q_e/n_eff). Gating on a fraction of the max let dead elements in at ~2/3 the
        // weight of live ones (measured), which is exactly the "mixing crap in" failure: their
        // noise enters the sum and their random phases cost coherence. So each element must beat
        // its OWN noise-only expectation by GATE sigma -- self-calibrating, no per-feed list.
        const double n_eff = (_alpha > 0.0) ? (2.0 - _alpha) / _alpha : 1.0;
        constexpr double GATE = 3.0;   // 3 sigma: measured live rho^2 >= 0.0114, dead <= 0.0024,
                                       // so the bar at 0.0095 keeps every live feed with 4x
                                       // margin over the worst dead one. 4 sigma lost 2 of 8.
        for (int e = 0; e < _n; ++e) {
            if (_q[(size_t)e] <= 0.0 || _p[(size_t)e] <= 0.0)
                continue;
            const double u2 = std::norm(_u[(size_t)e]);
            const double rho2 = u2 / (_p[(size_t)e] * _q[(size_t)e]);
            if (rho2 < GATE * GATE / n_eff)
                continue; // indistinguishable from an element with no signal
            // rho2 -> 1 means the element is correlated with ITSELF: the reference element,
            // whose only available reference is the sum it is itself part of. Its variance then
            // collapses to the floor and its weight blows up (measured: 4133 against ~10 for a
            // healthy element, which then dragged every other element under a
            // fraction-of-the-strongest gate). Handled after the loop by convention instead.
            if (rho2 > 0.99) {
                self_ref = e;
                continue;
            }
            const double sig = u2 / _q[(size_t)e];
            // Floor at a small fraction of the total power: an element whose fitted signal
            // exceeds its power (noise fluctuation early in the EMA) must not get an unbounded
            // weight from a near-zero denominator.
            const double var = std::max(_p[(size_t)e] - sig, 0.05 * _p[(size_t)e]);
            mag[(size_t)e] = std::abs(_u[(size_t)e]) / var; // |MRC weight| = |u_e|/sigma_e^2
            if (mag[(size_t)e] > mmax) {
                mmax = mag[(size_t)e];
                strongest = e;
            }
        }
        // THE REFERENCE ELEMENT'S OWN WEIGHT, by convention. Every u_e is measured against it,
        // so its gain is the scale the whole solution is expressed in and cannot be recovered
        // from statistics that all contain it. For a homogeneous array the defensible convention
        // is the MEDIAN of the elements that did solve: it keeps the reference in the sum (it is
        // usually one of the strongest feeds) without letting a degenerate variance hand it an
        // unbounded weight. A wrong-by-a-factor reference weight costs a little combining
        // efficiency; an unbounded one costs every other element.
        if (self_ref >= 0) {
            std::vector<double> live;
            for (int e = 0; e < _n; ++e)
                if (mag[(size_t)e] > 0.0)
                    live.push_back(mag[(size_t)e]);
            if (!live.empty()) {
                std::sort(live.begin(), live.end());
                mag[(size_t)self_ref] = live[live.size() / 2];
                if (mag[(size_t)self_ref] > mmax) {
                    mmax = mag[(size_t)self_ref];
                    strongest = self_ref;
                }
            }
        }
        if (mmax <= 0.0)
            return;
        // The anchor sets the sum's phase convention: the reference element unless it has
        // collapsed relative to the strongest (then the strongest takes over -- a one-time
        // phase step downstream, reported by anchor_moved()).
        _anchor = (mag[(size_t)_ref] >= _min_w * mmax) ? _ref : strongest;
        const cd pin = std::conj(_u[(size_t)_anchor]) / std::abs(_u[(size_t)_anchor]);
        for (int e = 0; e < _n; ++e) {
            if (mag[(size_t)e] <= 0.0)
                continue; // failed the significance gate above: absent / unpowered / too noisy
            // Re-anchored MRC weight: direction from u_e (rotated so the anchor is real
            // positive), magnitude |u_e|/sigma_e^2.
            const cd dir = _u[(size_t)e] * pin;
            const double d = std::abs(dir);
            if (d <= 0.0)
                continue;
            _w[(size_t)e] = (dir / d) * mag[(size_t)e];
            _wsum += mag[(size_t)e];
        }
        rebuild_split();
    }

    /// Partition the surviving elements into two disjoint halves of near-equal WEIGHT (greedy
    /// largest-first onto the lighter side), so combine_split()'s two references have comparable
    /// SNR -- an unbalanced split is limited by its weaker half.
    void rebuild_split() {
        _half.assign((size_t)_n, 0);
        _split_ok = 0;
        std::vector<std::pair<double, int>> live;
        for (int e = 0; e < _n; ++e)
            if (std::abs(_w[(size_t)e]) > 0.0)
                live.emplace_back(std::abs(_w[(size_t)e]), e);
        if (live.size() < 2)
            return; // one live element: no independent reference exists, no correction possible
        std::sort(live.begin(), live.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        double wa = 0.0, wb = 0.0;
        for (const auto& [w, e] : live) {
            if (wa <= wb) {
                wa += w;
                _half[(size_t)e] = 0;
            } else {
                wb += w;
                _half[(size_t)e] = 1;
            }
        }
        _split_ok = (wa > 0.0 && wb > 0.0) ? 1 : 0;
    }

    int _n = 0;
    int _ref = 0;
    int _anchor = 0;
    double _tau = 0.5;
    double _min_w = 0.02;
    double _warmth = 0.0;  ///< approaches 1 with ~tau; warm at > 0.95 (~3 tau)
    double _wsum = 0.0;    ///< sum |w_e| -- the "one element" normalization
    std::vector<cd> _u;    ///< EMA of G_e * conj(LOO reference) -- the per-element complex gain
    std::vector<double> _p; ///< EMA of |G_e|^2 -- total per-element power
    std::vector<double> _q; ///< EMA of |LOO reference|^2 -- normalizes the signal-power estimate
    std::vector<cd> _w;     ///< anchored MRC weights u_e/sigma_e^2 (magnitude), gated
    std::vector<uint8_t> _half; ///< split-aperture side per element (0=A, 1=B)
    int _split_ok = 0;      ///< both halves non-empty -> combine_split() is available
    double _alpha = 0.0;    ///< last EMA coefficient (sets n_eff for the significance gate)
};

} // namespace gnss
#endif // GNSS_ELEM_CAL_HPP
