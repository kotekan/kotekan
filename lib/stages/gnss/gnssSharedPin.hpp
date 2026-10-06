#ifndef GNSS_SHARED_PIN_HPP
#define GNSS_SHARED_PIN_HPP

/// THE FLEET-COMMON PHASE PIN of the shared element model (#154).
///
/// Every assembler instance learns the same instrument (one model per band per GPU), but the
/// model's GLOBAL phase per polarisation is a convention: the rotation that makes <ref, G>
/// real positive. With ref = the instance's own first model -- formed from one shadow, warmed
/// on whatever sky the node started into -- a poor first shadow leaves that instance at an
/// arbitrary phase against every other one, and its records cancel in the fleet sum until a
/// restart (an F-engine re-base does this to every node at once). With ref = F, one vector per
/// band shared by every instance, the convention is the fleet's: it does not depend on the
/// first shadow, so once the shape re-learns the pin follows on its own.
///
/// A WARM model is never stepped. Its records are being combined, and a step in the model's
/// phase is a step in every record's carrier phase downstream (cycle-slip class); a ramp of
/// ~1 deg/s is not. A FIRST model is pinned in full: installing a model is already a
/// one-time step, and there is no earlier phase of it to keep continuous.

#include <cmath>   // for sqrt
#include <complex> // for complex, conj, norm, arg, polar

namespace gnss {

/// Where a model's half [e0, e1) sits against the reference F.
struct RefOffset {
    bool ok = false;      ///< false: F has no projection on G (all-zero F or G over the half)
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

/// The rotation to apply to G's half: the full -err_rad when `limit` is false (a first model),
/// at most max_step_rad either way when it is true (a warm model).
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

} // namespace gnss

#endif // GNSS_SHARED_PIN_HPP
