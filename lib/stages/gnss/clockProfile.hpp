#ifndef CLOCK_PROFILE_HPP
#define CLOCK_PROFILE_HPP
/**
 * @file clockProfile.hpp
 * @brief Receiver clock-quality profile: the single knob that sizes the GNSS pipeline for a
 *        given reference oscillator (airspy TCXO ... OCXO ... GPSDO ... rubidium ... maser).
 *
 * "Clock uncertainty" is really TWO independent physical axes, driving different stages:
 *   accuracy_ppm : fractional frequency OFFSET bound (ppm). Sizes the carrier-Doppler SEARCH
 *                  extent -- the clock offset appears as a common carrier shift accuracy_ppm*f_c
 *                  on top of the sky Doppler, so a worse clock needs a wider search.
 *   coherence_s  : coherent-integration ceiling (s), Allan/phase-noise limited. Sizes how long
 *                  the deep integration can cohere -> integration_length and the Doppler step.
 * (The LO-vs-ADC code/carrier split l-a is a third quantity but the broker's cp_rate slope fit
 *  absorbs it adaptively, so it is NOT a user knob here.)
 *
 * This header is the CANONICAL preset table. The C++ stages read it directly; the Python broker
 * and run_live.sh parse this same file (as they already do for gnssSignal.hpp) so there is one
 * source of truth in one place. Named presets expand to numbers; an explicit number overrides its
 * preset; "auto" (the default) is a conservative cold bound that the broker's clock-bias estimator
 * narrows live -- so any explicit setting is just an override/seed on the automatic behaviour.
 */
#include <cmath>
#include <string>

namespace gnss {

struct ClockProfile {
    double accuracy_ppm; ///< fractional frequency-offset bound (ppm) -> Doppler search extent
    double coherence_s;  ///< coherent-integration ceiling (s) -> integration_length / step
};

/// Canonical preset table. Keep the columns in sync with the doc block above; run_live.sh /
/// gps_distributed_broker.py parse THESE literals. "auto"/unknown -> a conservative cold bound.
inline ClockProfile clock_profile_preset(const std::string& name) {
    // name       accuracy_ppm  coherence_s
    if (name == "tcxo")     return {2.0,   0.08};  // airspy stock TCXO, free-running
    if (name == "ocxo")     return {0.3,   1.0};   // plain oven-controlled xtal
    if (name == "gpsdo")    return {0.06,  1.5};   // GPS-disciplined OCXO (measured l-a ~0.06 ppm)
    if (name == "rubidium") return {1e-3,  30.0};  // Rb standard
    if (name == "maser")    return {1e-6,  300.0}; // H-maser
    return {2.0, 0.2};                             // auto / unknown: start wide, narrowed live
}

/// Resolve a profile name plus optional explicit overrides (NaN = unset) to concrete numbers.
inline ClockProfile resolve_clock_profile(const std::string& name, double accuracy_ppm_override,
                                          double coherence_s_override) {
    ClockProfile p = clock_profile_preset(name);
    if (!std::isnan(accuracy_ppm_override))
        p.accuracy_ppm = accuracy_ppm_override;
    if (!std::isnan(coherence_s_override))
        p.coherence_s = coherence_s_override;
    return p;
}

/// Carrier-Doppler search HALF-range (Hz) for a band: max sky Doppler (GPS radial velocity) +
/// the clock frequency-offset term + a small margin. Scales with the carrier, so it is correct
/// per band without per-band constants.
inline double clock_doppler_half_range_hz(double carrier_hz, double accuracy_ppm,
                                          double margin_hz = 400.0) {
    constexpr double V_MAX_MPS = 929.0;       // GPS max line-of-sight velocity (near horizon)
    constexpr double C_MPS = 299792458.0;
    return V_MAX_MPS / C_MPS * carrier_hz + accuracy_ppm * 1e-6 * carrier_hz + margin_hz;
}

} // namespace gnss
#endif // CLOCK_PROFILE_HPP
