#ifndef GNSS_ELEM_CAL_HPP
#define GNSS_ELEM_CAL_HPP
/**
 * @file gnssElemCal.hpp
 * @brief Self-calibrated element combining for the record header: noise-weighted MRC, plus a
 *        split-aperture estimate of the per-record sky phase.
 *  - gnss::ElemCal
 *
 * One set of per-element weights serves two outputs. @ref ElemCal::combine gives the header's
 * correlation rows as the calibrated weighted mean of the elements, with the per-element complex
 * gains estimated from the satellite itself (bootstrap MRC) or installed by seed() or hold().
 * @ref ElemCal::combine_split gives the prompt with the per-record sky phase removed. That phase
 * is common to all elements but close to white in time, so a record's neighbours do not predict
 * it, while the other elements, with independent thermal noise, do: one disjoint half of the
 * array is derotated by the other half's phase.
 *
 * Noise-weighted MRC: w_e = u_e / sigma_e^2, with u_e the element's gain against its reference
 * and sigma_e^2 the part of its power not correlated with that reference. Weighting by signal
 * alone is optimal only when every element has the same noise; an element with an oscillating
 * amplifier carries real signal but excess noise, and this weighting discounts it with no
 * per-feed threshold or list of bad feeds. Each element's reference excludes it, so no
 * element's noise enters its own gain or noise estimate.
 *
 * Header-only and stage-independent so offline tools can drive it with synthetic per-element
 * data; GnssGpuRecordAssemble is the production consumer.
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
        _n(n_elem), _ref(ref), _tau(tau_s > 0.0 ? tau_s : 0.5), _min_w(std::max(0.0, min_w_frac)) {
        reset();
    }

    /// Drop all state (fresh acquisition: the fringes may have moved arbitrarily). A prior set
    /// by seed() is re-applied as a warm start: the instrumental phase does not change between a
    /// drop and its re-acquisition, so the combine is coherent from the first record instead of
    /// bootstrapping for ~3 tau.
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
        if (!_prior.empty())
            apply_prior();
    }

    /// Inject a per-element complex gain prior (e.g. from the N^2 eigendecomposition, sky
    /// removed). Its phase is what matters: the instrumental cable and component phase, which
    /// the raw-voltage pipeline does not correct. Its relative magnitude sets the initial MRC
    /// weighting, and update() refines both. Every later reset() re-applies it. A null
    /// @p g_prior, an @p n_elem other than the cal's, or an all-zero prior clears it (cold
    /// self-bootstrap).
    void seed(const cd* g_prior, int n_elem) {
        if (g_prior == nullptr || n_elem != _n) {
            _prior.clear();
            return;
        }
        double s = 0.0;
        for (int e = 0; e < _n; ++e)
            s += std::norm(g_prior[(size_t)e]);
        if (!(s > 0.0)) { // all-zero prior = "no prior"
            _prior.clear();
            return;
        }
        _prior.assign(g_prior, g_prior + _n);
        apply_prior();
    }

    /// Install @p w (n entries, anchored and scaled by the caller) as this cal's weights, warm at
    /// once; combine() and combine_split() then use them as they would learned weights. The
    /// caller skips update() while holding, since it would rebuild the weights from the learned
    /// statistics. This is how a shared instrument model (one per-element gain for every
    /// satellite) reaches the per-PRN combine.
    void hold(const cd* w, int n_elem) {
        if (w == nullptr || n_elem != _n)
            return;
        _w.assign(w, w + _n);
        _wsum = 0.0;
        for (int e = 0; e < _n; ++e)
            _wsum += std::abs(_w[(size_t)e]);
        _warmth = 1.0;
        _anchor = _ref;
        rebuild_split();
    }

    /// True once the weights have integrated ~3 time constants (at once after seed() or hold()),
    /// so the combine should be used.
    bool warm() const {
        return _warmth > 0.95 && _wsum > 0.0;
    }

    /// True if the phase anchor is not the configured reference element, so the combine's phase
    /// convention has moved: the reference is below min_w_frac of the strongest, which includes
    /// a cold bootstrap before the reference has a weight of its own. Worth logging once.
    bool anchor_moved() const {
        return _anchor != _ref;
    }

    /// The anchored MRC weights as they stand (n entries): for comparing cals, building a shared
    /// model, or another cal's hold().
    const std::vector<cd>& weights() const {
        return _w;
    }

    /// Calibrated weighted mean of one correlator row over the elements (every row uses the same
    /// weights), in one-element units and the anchor's phase convention. The record header
    /// carries this, so it must stay a plain linear combination: its phase is the carrier-phase
    /// (ADR) observable (gnssRecord.hpp).
    cd combine(const cd* g) const {
        if (_wsum <= 0.0)
            return cd(0.0, 0.0);
        cd s(0.0, 0.0);
        for (int e = 0; e < _n; ++e)
            s += std::conj(_w[(size_t)e]) * g[(size_t)e];
        return s / _wsum;
    }

    /// Sky-phase-corrected combine by split aperture. The live elements are split into two
    /// disjoint halves A and B, balanced in weight, and A is derotated by B's phase:
    ///
    ///     corrected = A * e^{-i arg(B)} / sum|w|
    ///
    /// The halves must be disjoint. Leave-one-element-out is unbiased in the mean but destroys
    /// the noise estimate every downstream significance is built on: each element's reference is
    /// then nearly the whole sum, so every record comes out near |sum|, real and positive, and
    /// coherent_sum, which takes its noise from the component orthogonal to the sum, reports a
    /// grossly inflated SNR. With disjoint halves arg(A) - arg(B) stays random under noise, so
    /// the orthogonal component survives.
    ///
    /// This feeds the deep folds (record slots 24/25), not the header, which keeps the raw phase
    /// for ADR.
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
        // One-way: A integrates and B is only a phase reference. The symmetric
        // A*e^{-i arg B} + B*e^{-i arg A} expands to (|A|+|B|) cos d + i (|A|-|B|) sin d, which
        // cancels the imaginary part when the halves are balanced and so understates the noise.
        // The cost is that only half the aperture integrates (~sqrt(2) in SNR).
        return A * std::polar(1.0, -std::arg(B)) / _wsum;
    }

    /// One record's cal update from the prompt row. @p dt_s is seconds since this PRN's previous
    /// record. Each element's statistics are formed against a reference that excludes it, so no
    /// element's noise enters its own gain or noise estimate.
    void update(const cd* g, double dt_s) {
        if (_n <= 0 || !(dt_s > 0.0))
            return;
        // Clamped: after a coast or drop gap dt can span seconds, and an unclamped EMA would let
        // one noisy record replace the whole cal.
        const double alpha = std::min(0.25, 1.0 - std::exp(-dt_s / _tau));
        _alpha = alpha;
        // Every element but the reference is calibrated against the reference element, which is
        // fixed. Referencing each element against the current weighted sum of the others would
        // give more SNR but does not work: the weights are rebuilt from noisy statistics every
        // record, so the sum's phase jitters from record to record and the EMA of
        // G_e*conj(reference) averages towards zero, below the gate. G_ref does not contain G_e,
        // so the fixed reference is still leave-one-out.
        //
        // The reference element is calibrated against the leave-it-out weighted mean of the
        // others, which exists once they have weights; until then it has no weight of its own.
        // A mean, not a sum, so its reference is in the same one-element units as g_ref is for
        // every other element: rho^2 and sig are scale-free, but the MRC magnitude |u|/sigma^2
        // is not, and against the sum the reference's weight would swamp the others'.
        cd tot(0.0, 0.0);
        double w_rest = 0.0; // sum of |w_e| over e != ref: the leave-it-out mean's normalization
        if (_wsum > 0.0)
            for (int e = 0; e < _n; ++e) {
                tot += std::conj(_w[(size_t)e]) * g[(size_t)e];
                if (e != _ref)
                    w_rest += std::abs(_w[(size_t)e]);
            }
        for (int e = 0; e < _n; ++e) {
            const cd ge = g[(size_t)e];
            const cd rest = (e != _ref)      ? g[(size_t)_ref]
                            : (w_rest > 0.0) ? (tot - std::conj(_w[(size_t)e]) * ge) / w_rest
                                             : cd(0.0, 0.0);
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
    /// Load the stored prior so warm() is true and combine() is coherent at once. u_e carries
    /// the prior's phase and magnitude; a nominal seed SNR sets the noise floor so the MRC
    /// weights are finite. update() then refines both from the data.
    void apply_prior() {
        if ((int)_prior.size() != _n)
            return;
        constexpr double SEED_SNR = 4.0; // signal/noise the prior is trusted at, per element
        for (int e = 0; e < _n; ++e) {
            const cd g = _prior[(size_t)e];
            _u[(size_t)e] = g;                                     // per-element gain direction+mag
            _q[(size_t)e] = 1.0;                                   // reference power normalization
            _p[(size_t)e] = std::norm(g) * (1.0 + 1.0 / SEED_SNR); // total = signal + floor
        }
        // rebuild_weights gates each element at rho^2 >= GATE^2 / n_eff, with n_eff set by
        // _alpha. A trusted prior stands in for many integrated records, so give it a small
        // alpha: with the default n_eff = 1 the bar is 9, which rho^2 (<= 1) can never clear.
        // update() overwrites _alpha on the first record.
        _alpha = 0.1;   // n_eff = 19: bar ~0.47, below the seed's rho^2 = 0.8
        _warmth = 0.96; // just past warm(), so the seeded combine is used at once
        rebuild_weights();
    }

    void rebuild_weights() {
        std::fill(_w.begin(), _w.end(), cd(0.0, 0.0));
        _wsum = 0.0;
        // Per-element gain u_e and noise power: |u_e|^2/_q is the power correlated with the
        // reference (signal), and the rest of |G_e|^2 is the element's own noise, oscillating
        // amplifiers included.
        std::vector<double> mag((size_t)_n, 0.0);
        double mmax = 0.0;
        int strongest = _ref;
        int self_ref = -1; // element with rho2 > 0.99 (see below)
        // Statistical gate, not a fraction of the strongest. u_e is an EMA of G_e*conj(ref) over
        // ~n_eff records, so an element with no signal has |u_e| ~ sqrt(p_e*q_e/n_eff), not 0; a
        // fraction-of-max gate admits such elements, and their noise and random phases cost
        // coherence. Each element must beat its own noise-only expectation by GATE sigma.
        const double n_eff = (_alpha > 0.0) ? (2.0 - _alpha) / _alpha : 1.0;
        constexpr double GATE = 3.0; // sigma; a higher bar starts rejecting weak live feeds
        for (int e = 0; e < _n; ++e) {
            if (_q[(size_t)e] <= 0.0 || _p[(size_t)e] <= 0.0)
                continue;
            const double u2 = std::norm(_u[(size_t)e]);
            const double rho2 = u2 / (_p[(size_t)e] * _q[(size_t)e]);
            if (rho2 < GATE * GATE / n_eff)
                continue; // indistinguishable from an element with no signal
            // rho2 near 1 leaves almost no measured noise (statistics from a single record, or a
            // very bright source), so the variance would sit at the floor below and the weight
            // at the largest that floor allows. Such an element is held back and given the
            // median weight after the loop. Only the last one found is kept (any others get no
            // weight), and if every gated element is held back the cal has no weights.
            if (rho2 > 0.99) {
                self_ref = e;
                continue;
            }
            const double sig = u2 / _q[(size_t)e];
            // Floor the noise at 5% of the total power: early in the EMA the fitted signal can
            // exceed the power, and a near-zero denominator would give an unbounded weight.
            const double var = std::max(_p[(size_t)e] - sig, 0.05 * _p[(size_t)e]);
            mag[(size_t)e] = std::abs(_u[(size_t)e]) / var; // |MRC weight| = |u_e|/sigma_e^2
            if (mag[(size_t)e] > mmax) {
                mmax = mag[(size_t)e];
                strongest = e;
            }
        }
        // The held-back element's weight, by convention: the median of the elements that did
        // solve, which keeps it in the sum without letting its degenerate variance set the
        // weight. A weight wrong by a factor costs only a little combining efficiency.
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
        // The anchor sets the combine's phase convention: the reference element, unless it is
        // below _min_w of the strongest, when the strongest takes over (a one-time phase step
        // downstream, reported by anchor_moved()).
        _anchor = (mag[(size_t)_ref] >= _min_w * mmax) ? _ref : strongest;
        // The reference element's direction is the convention, not a measurement. Every other
        // u_e is G_e*conj(G_ref), a phase relative to the reference, so the reference's own is 0
        // by definition. Its u_ref is measured against the leave-it-out mean, whose phase
        // depends on the current weights; taking the reference's weight phase from it closes a
        // loop with two stable fixed points, aligned and anti-aligned with the other elements.
        // A shared model built from these cals and pinned to the reference would then carry an
        // arbitrary sign per instance, and the instances would cancel in the fleet's lobe sum.
        auto dir_of = [&](int e) -> cd {
            if (e == _ref)
                return cd(1.0, 0.0);
            const cd u = _u[(size_t)e];
            const double d = std::abs(u);
            return (d > 0.0) ? u / d : cd(0.0, 0.0);
        };
        const cd pin = std::conj(dir_of(_anchor));
        for (int e = 0; e < _n; ++e) {
            if (mag[(size_t)e] <= 0.0)
                continue; // failed the significance gate above: absent / unpowered / too noisy
            // Re-anchored MRC weight: direction from u_e (rotated so the anchor is real
            // positive), magnitude |u_e|/sigma_e^2.
            const cd dir = dir_of(e) * pin;
            const double d = std::abs(dir);
            if (d <= 0.0)
                continue;
            _w[(size_t)e] = (dir / d) * mag[(size_t)e];
            _wsum += mag[(size_t)e];
        }
        rebuild_split();
    }

    /// Partition the live elements into two disjoint halves of near-equal weight (greedy,
    /// largest first, onto the lighter side), so combine_split()'s halves have comparable SNR;
    /// an unbalanced split is limited by its weaker half.
    void rebuild_split() {
        _half.assign((size_t)_n, 0);
        _split_ok = 0;
        std::vector<std::pair<double, int>> live;
        for (int e = 0; e < _n; ++e)
            if (std::abs(_w[(size_t)e]) > 0.0)
                live.emplace_back(std::abs(_w[(size_t)e]), e);
        if (live.size() < 2)
            return; // one live element: no independent reference exists, no correction possible
        std::sort(live.begin(), live.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
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
    double _warmth = 0.0;       ///< approaches 1 with ~tau; warm at > 0.95 (~3 tau)
    double _wsum = 0.0;         ///< sum |w_e|: the one-element normalization
    std::vector<cd> _u;         ///< EMA of G_e * conj(reference): the per-element complex gain
    std::vector<double> _p;     ///< EMA of |G_e|^2: total per-element power
    std::vector<double> _q;     ///< EMA of |reference|^2: normalizes the signal-power estimate
    std::vector<cd> _w;         ///< anchored MRC weights, magnitude |u_e|/sigma_e^2; 0 = gated out
    std::vector<uint8_t> _half; ///< split-aperture side per element (0 = A, 1 = B)
    int _split_ok = 0;          ///< both halves non-empty: combine_split() is available
    double _alpha = 0.0;        ///< last EMA coefficient (sets n_eff for the significance gate)
    std::vector<cd> _prior;     ///< seed() prior; empty = cold bootstrap
};

} // namespace gnss
#endif // GNSS_ELEM_CAL_HPP
