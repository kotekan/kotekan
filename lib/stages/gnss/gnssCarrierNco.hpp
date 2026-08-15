#ifndef GNSS_CARRIER_NCO_HPP
#define GNSS_CARRIER_NCO_HPP

#include <cmath>

namespace gnss {

/**
 * @brief The replica carrier phase as an ACCUMULATOR -- i.e. an actual NCO (task #71).
 *
 * ⚠️ ONE EXPRESSION, ONE PLACE. GnssCudaDespread::Impl calls this, and
 * scripts/gnss/carrier_nco_gate.cpp drives THIS function rather than a second copy of the
 * arithmetic, so the gate cannot pass against a formula the production path does not use.
 * Same discipline as gnss::dll_integrate.
 *
 * THE DEFECT IT REPLACES. The despread used to evaluate the phase as `f * n0` with n0 an
 * ABSOLUTE sample index, so the replica's entire phase history hung off the CURRENT frequency
 * estimate. n0/fs is the uptime -- ~5.85e5 s after a week -- which makes that a lever of
 * catastrophic length: a Doppler change of 2.7e-7 Hz rotates the phase a full radian. The
 * Doppler is re-propagated every record (dop_rate * 10.5 ms = 1e-4..6e-3 Hz), so every record
 * got a large and essentially arbitrary phase offset. Invisible to |A| -- tracking, q, the DLL
 * and the incoherent C/N0 are all POWER -- and fatal to everything cross-record.
 *
 * ⚠️ THE L1 PATH FIXED THIS ON 2026-07-10 and the CHORD path never got the same treatment;
 * GnssChannelizedTracker.cpp states it outright ("retuning its frequency by df mid-stream
 * rotates the whole phase history by 2*pi*df*t_abs ... the root cause of the L1 deep decay").
 *
 * TRAPEZOIDAL IN f. The Doppler moves linearly between records -- dop_rate is precisely that
 * model -- so averaging the interval's endpoint frequencies integrates it EXACTLY rather than
 * to first order, and a dropped record still integrates correctly across the hole.
 *
 * ⚠️ THE ABSOLUTE VALUE IS MEANINGLESS, AND THAT IS THE POINT. Only phase DIFFERENCES between
 * records are observable in a correlation, so the constant of integration cancels. What this
 * buys is that the difference is the physical one instead of 2*pi*df*uptime of bookkeeping.
 */
struct CarrierNco {
    bool valid = false;
    long long n0 = 0;   ///< absolute sample of the last reference
    long double ph = 0; ///< accrued phase there, radians, reduced to [0, 2pi)
    double f = 0.0;     ///< frequency in force at that reference, Hz
};

/// Advance @p a to sample @p n0 at frequency @p f_now and return the phase there, radians.
///
/// @param seed_phase  the value to adopt when re-anchoring (first sight, backwards jump, or a
///                    gap longer than @p gap_max). Any value is admissible -- see the note on
///                    the constant of integration -- so callers pass whatever keeps them
///                    comparable to the arm they came from.
/// @param reanchored  incremented (never reset) when a re-anchor happens. Each one is a REAL
///                    phase discontinuity, so it is counted rather than hidden: a cross-record
///                    estimator must be able to ask whether one fell inside its span.
inline double carrier_nco_advance(CarrierNco& a, double f_now, long long n0, double fs,
                                  double seed_phase, long long gap_max,
                                  unsigned long long* reanchored = nullptr) {
    constexpr long double TWO_PI_L = 6.283185307179586476925286766559005768L;
    // ⚠️ IDEMPOTENT PER n0. This is stateful and GnssCudaDespread::build_jobs is shared by
    // three enqueue paths, so a second call for the SAME record must return the same phase
    // rather than advance again -- otherwise the replica's phase would depend on how many code
    // paths happened to ask for it.
    if (a.valid && n0 == a.n0)
        return (double)a.ph;
    if (!a.valid || n0 < a.n0 || (n0 - a.n0) > gap_max) {
        a.valid = true;
        a.n0 = n0;
        a.f = f_now;
        a.ph = (long double)seed_phase;
        if (reanchored)
            (*reanchored)++;
        return (double)a.ph;
    }
    const long double dn = (long double)(n0 - a.n0);
    const long double fbar = 0.5L * ((long double)a.f + (long double)f_now);
    // Reduce the CYCLE count before scaling to radians: fbar*dn/fs is ~1.2e7 cycles for one
    // record, where a long double's ulp is ~1e-12 cycles. Radians-first would throw that away.
    long double cyc = fmodl(fbar * dn / (long double)fs, 1.0L);
    a.ph = fmodl(a.ph + TWO_PI_L * cyc, TWO_PI_L);
    if (a.ph < 0.0L)
        a.ph += TWO_PI_L;
    a.n0 = n0;
    a.f = f_now;
    return (double)a.ph;
}

/// The arm this replaces: phase at an ABSOLUTE sample. Kept here beside its successor so the
/// gate can drive both from one place and show the lever rather than assert it.
inline double carrier_phase_absolute(double f, long long n0, double fs) {
    constexpr long double TWO_PI_L = 6.283185307179586476925286766559005768L;
    long double fr = fmodl((long double)f / (long double)fs * (long double)n0, 1.0L);
    if (fr < 0.0L)
        fr += 1.0L;
    return (double)(TWO_PI_L * fr);
}

} // namespace gnss

#endif
