#ifndef GNSS_SHARED_PIN_HPP
#define GNSS_SHARED_PIN_HPP

/// @file gnssSharedPin.hpp
/// @brief The fleet-common phase pin of the shared element model, the formation gate near
///        boresight, and the re-seed from the fleet reference.
///
/// Every assembler instance of a band learns the same instrument, but the model's global phase
/// per polarisation is a convention: the rotation that makes <ref, G> real positive. With ref
/// the instance's own first model, formed from one shadow learner on whatever sky the node
/// started into, a poor first shadow leaves that instance at an arbitrary phase against the
/// others, and its records cancel in the fleet sum until it restarts. With ref = F, one vector
/// per band shared by every instance (config elem_sum_shared_ref), the convention is the
/// fleet's and does not depend on the first shadow.
///
/// The pin never steps a warm model. Its records are being combined, and a step in the model's
/// phase is a step in every record's carrier phase downstream, like a cycle slip; a slow ramp
/// (elem_sum_shared_ref_slew_deg_s, default 1 deg/s) is not. A first model is pinned in full:
/// installing a model is already a one-time step, with no earlier phase to keep continuous.

#include <cmath>   // for sqrt
#include <complex> // for complex, conj, norm, arg, polar

namespace gnss {

/// Where a model's half [e0, e1) sits against the reference F.
struct RefOffset {
    bool ok = false;      ///< false: <F, G> is zero over the half (F or G zero, or orthogonal)
    double err_rad = 0.0; ///< arg<F, G>: rotating G by -err_rad makes <F, G> real positive
    double sim = 0.0;     ///< |<F, G>| / (|F| |G|): how well F describes this model's shape
};

inline RefOffset ref_offset(const std::complex<double>* F, const std::complex<double>* G, int e0,
                            int e1) {
    RefOffset o;
    std::complex<double> y(0.0, 0.0);
    double nf = 0.0, ng = 0.0;
    for (int e = e0; e < e1; ++e) {
        y += std::conj(F[e]) * G[e];
        nf += std::norm(F[e]);
        ng += std::norm(G[e]);
    }
    if (!(std::abs(y) > 0.0) || !(nf > 0.0) || !(ng > 0.0))
        return o;
    o.ok = true;
    o.err_rad = std::arg(y);
    o.sim = std::abs(y) / std::sqrt(nf * ng);
    return o;
}

/// The rotation to apply to G's half: the full -err_rad when @p limit is false (a first model),
/// at most @p max_step_rad either way when it is true (a warm model; no rotation if
/// max_step_rad <= 0).
inline std::complex<double> pin_rotation(const RefOffset& o, bool limit, double max_step_rad) {
    double a = -o.err_rad;
    if (limit) {
        if (!(max_step_rad > 0.0))
            return std::complex<double>(1.0, 0.0);
        if (a > max_step_rad)
            a = max_step_rad;
        else if (a < -max_step_rad)
            a = -max_step_rad;
    }
    return std::polar(1.0, a);
}

/// True while no first model may be formed: the last post puts the nearest satellite inside
/// @p form_deg of boresight. The transit freeze stops a model in force from learning a bright
/// satellite's leakage; a model formed just outside the freeze cone learns that leakage as its
/// shape and keeps it until the consensus EMA or a re-seed replaces it. A warm model keeps
/// adapting; only forming waits. A post older than 120 s is no evidence, as for the freeze.
inline bool formation_frozen(double now_s, double post_t, double sep_deg, double form_deg) {
    constexpr double max_age_s = 120.0;
    return now_s - post_t <= max_age_s && sep_deg < form_deg;
}

/// Re-seed state of one half (pol) of the model. A warm half that stays below the reference's
/// shape-similarity floor for reseed_s takes the reference's shape: the pin refuses such a half
/// (F does not describe it), so a mis-shaped model has no other way back.
struct ReseedState {
    double low_since = -1.0; ///< steady time the similarity first fell below the floor, or -1
    double last = -1.0e18;   ///< steady time of the last re-seed (rate limit)
    int n = 0;               ///< re-seeds so far (diagnostics)
};

/// True once per episode: the half (the caller passes only warm ones) has been below @p min_sim
/// for @p reseed_s, its sibling pol (same assembler, other half) still describes F (@p sib_ok),
/// and no re-seed happened within the last @p reseed_s. A similarity at or above the floor, one
/// that cannot be measured (@p ok false), a sibling off F too, or reseed_s <= 0 ends the
/// episode. When both pols stop matching at once it is F that changed (new F-engine gains), and
/// re-seeding would reset every healthy model to the stale F before elem_shared_ref.py's
/// re-take replaces it; only one pol off F with the other on it re-seeds.
inline bool reseed_due(ReseedState& s, bool ok, double sim, double min_sim, bool sib_ok,
                       double now_s, double reseed_s) {
    if (!(reseed_s > 0.0) || !ok || sim >= min_sim || !sib_ok) {
        s.low_since = -1.0;
        return false;
    }
    if (s.low_since < 0.0)
        s.low_since = now_s;
    if (now_s - s.low_since < reseed_s || now_s - s.last < reseed_s)
        return false;
    s.last = now_s;
    s.low_since = -1.0;
    ++s.n;
    return true;
}

} // namespace gnss

#endif // GNSS_SHARED_PIN_HPP
