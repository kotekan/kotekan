#ifndef CLOCK_PROFILE_HPP
#define CLOCK_PROFILE_HPP
/**
 * @file clockProfile.hpp
 * @brief Receiver clock-quality profile: presets for the reference oscillator (airspy TCXO,
 *        OCXO, GPSDO, rubidium, maser) and the Doppler search range they imply.
 *
 * accuracy_ppm is the fractional frequency-offset bound. The offset shifts every carrier by
 * accuracy_ppm*f_c on top of the sky Doppler, so it sets the Doppler search extent
 * (GnssChannelizedSearch, from the /clock_profile config block). The LO-vs-ADC code/carrier
 * split (l-a) is another quantity, but the broker estimates it from the code-phase rate, so it
 * is not a setting here.
 */
#include <string>

namespace gnss {

struct ClockProfile {
    double accuracy_ppm; ///< fractional frequency-offset bound (ppm)
};

/// Preset table; an unknown name gives a conservative cold bound.
inline ClockProfile clock_profile_preset(const std::string& name) {
    if (name == "tcxo")
        return {2.0}; // airspy stock TCXO, free-running
    if (name == "ocxo")
        return {0.3}; // plain oven-controlled xtal
    if (name == "gpsdo")
        return {0.06}; // GPS-disciplined OCXO
    if (name == "rubidium")
        return {1e-3}; // Rb standard
    if (name == "maser")
        return {1e-6}; // H-maser
    return {2.0};      // unknown
}

/// Carrier-Doppler search half-range (Hz) for a band: max sky Doppler (GPS radial velocity) plus
/// the clock frequency-offset term plus a margin. Every term but the margin scales with the
/// carrier, so no per-band constants are needed.
inline double clock_doppler_half_range_hz(double carrier_hz, double accuracy_ppm,
                                          double margin_hz = 400.0) {
    constexpr double V_MAX_MPS = 929.0; // GPS max line-of-sight velocity (near horizon)
    constexpr double C_MPS = 299792458.0;
    return V_MAX_MPS / C_MPS * carrier_hz + accuracy_ppm * 1e-6 * carrier_hz + margin_hz;
}

} // namespace gnss
#endif // CLOCK_PROFILE_HPP
