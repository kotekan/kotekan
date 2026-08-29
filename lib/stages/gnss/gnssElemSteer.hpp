#ifndef GNSS_ELEM_STEER_HPP
#define GNSS_ELEM_STEER_HPP
/**
 * @file gnssElemSteer.hpp
 * @brief Per-element geometric steering for the GNSS despread combine (#102).
 *
 * THE PROBLEM. The despread's channel->element combine sums per-channel correlations with a
 * single fleet-common code phase. An element displaced by r along the satellite line of sight
 * sees the code EARLIER by tau = (r . e_sat)/c: 200 m of array is ~0.6 us ~ 6 chips of
 * differential delay, and the sharp L5 peak drops elements past +-0.5 chip (+-15 m). Today's
 * clustered dishes sit at that margin; the build-out makes it fatal.
 *
 * THE FIX LIVES IN THE CHANNELIZATION. A true time delay tau is EXACTLY a per-channel phase
 * e^{-i 2 pi f_ch tau} (channel bandwidth << 1/tau), so steering is one complex multiply per
 * (channel, element) inside the combine the assembler already runs -- code and carrier both,
 * no kernel changes. The sparse comb's delay aliasing does not matter here: we APPLY a known
 * tau, we never estimate one.
 *
 * ⚠️ THE SIGN IS A MEASURED QUANTITY, NOT A DERIVATION. This codebase has paid for derived
 * signs three times (fine-lag, rrate-phase, the 08-29 position step). `sign` is a config
 * parameter, and the arming procedure calibrates it from the elem archive (per-element
 * cross-channel phase slope vs geometry) or by an A/B on p/noise -- never from this comment.
 *
 * Geometry arrives per satellite over REST (az/el, degrees; slow-moving, ~0.5 deg/min) and
 * HOLDS for hold_s: stale geometry degrades gracefully to no steering rather than steering
 * with yesterday's sky.
 *
 * Header-only and stage-independent so the offline gate drives exactly this arithmetic.
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace gnss {

class ElemSteer {
public:
    using cf = std::complex<float>;

    ElemSteer() = default;

    /// @param positions_enu flat [n_elem][3] element positions, metres East/North/Up of the
    ///        array reference point (the same point the broker's station coordinates should
    ///        eventually revert to -- see buglist #102).
    /// @param freq_mhz per-channel RF centre frequencies, MHz (freq_id x 0.1953125 at CHORD).
    /// @param sign +1 or -1: the measured phase-convention sign (see file note).
    /// @param hold_s geometry validity horizon, seconds.
    ElemSteer(std::vector<double> positions_enu, std::vector<double> freq_mhz, int n_prn,
              double sign, double hold_s) :
        _pos(std::move(positions_enu)), _f_mhz(std::move(freq_mhz)),
        _n_elem((int)(_pos.size() / 3)), _n_chan((int)_f_mhz.size()), _sign(sign),
        _hold_s(hold_s), _tab((size_t)n_prn * _n_chan * _n_elem, cf(1.0f, 0.0f)),
        _fresh_t((size_t)n_prn, -1.0e18) {}

    bool enabled() const {
        return _n_elem > 0 && _n_chan > 0;
    }
    int n_elem() const {
        return _n_elem;
    }

    /// New geometry for one satellite slot: rebuild its [n_chan][n_elem] phasor table.
    /// az/el in degrees (az from North, clockwise/east-positive -- the broker's predict_all
    /// convention); t_now in the caller's steady seconds (only differenced against itself).
    void update(int slot, double az_deg, double el_deg, double t_now) {
        if (!enabled() || slot < 0 || (size_t)slot >= _fresh_t.size())
            return;
        const double az = az_deg * M_PI / 180.0, el = el_deg * M_PI / 180.0;
        // Unit vector receiver -> satellite, ENU.
        const double e[3] = {std::cos(el) * std::sin(az), std::cos(el) * std::cos(az),
                             std::sin(el)};
        cf* tab = &_tab[(size_t)slot * _n_chan * _n_elem];
        for (int el_i = 0; el_i < _n_elem; ++el_i) {
            // Delay of this element relative to the reference point: an element displaced
            // TOWARD the satellite receives the code EARLIER by (r.e)/c.
            const double* r = &_pos[(size_t)el_i * 3];
            const double tau_s = (r[0] * e[0] + r[1] * e[1] + r[2] * e[2]) / 299792458.0;
            for (int ch = 0; ch < _n_chan; ++ch) {
                const double ph = _sign * 2.0 * M_PI * (_f_mhz[(size_t)ch] * 1e6) * tau_s;
                tab[(size_t)ch * _n_elem + el_i] = cf((float)std::cos(ph), (float)std::sin(ph));
            }
        }
        _fresh_t[(size_t)slot] = t_now;
    }

    /// Is this slot's geometry fresh enough to steer with?
    bool warm(int slot, double t_now) const {
        return enabled() && slot >= 0 && (size_t)slot < _fresh_t.size()
               && t_now - _fresh_t[(size_t)slot] <= _hold_s;
    }

    /// The [n_elem] phasor row for (slot, channel). Valid only when warm().
    const cf* row(int slot, int ch) const {
        return &_tab[((size_t)slot * _n_chan + ch) * _n_elem];
    }

private:
    std::vector<double> _pos;   // [n_elem][3] ENU metres
    std::vector<double> _f_mhz; // [n_chan] RF MHz
    int _n_elem = 0;
    int _n_chan = 0;
    double _sign = 1.0;
    double _hold_s = 120.0;
    std::vector<cf> _tab;         // [n_prn][n_chan][n_elem]
    std::vector<double> _fresh_t; // [n_prn] last update, steady seconds
};

} // namespace gnss

#endif
