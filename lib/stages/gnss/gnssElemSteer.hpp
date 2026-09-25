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
 * Geometry arrives per satellite over REST (az/el, degrees) every ~30 s and HOLDS for hold_s:
 * stale geometry degrades gracefully to no steering rather than steering with yesterday's sky.
 *
 * ⚠️ A 30-s SNAPSHOT IS NOT STEADY ENOUGH TO HOLD. A GPS satellite turns ~1.5e-4 rad/s, so
 * the line-of-sight path difference across a 44 m baseline moves ~7 mm/s: ~0.2 m, most of an
 * L-band wavelength, by the end of a 30-s hold. Under a slowly-adapting shared element model
 * that is a 30-s sawtooth of 0.3-1.6 dB on every satellite. So each post also carries the
 * az/el RATES and the epoch they describe, and the table is re-evaluated at the record's own
 * time, every rebuild_s, from the line-of-sight UNIT VECTOR extrapolated linearly (az/el
 * themselves are not linear near zenith, where the az rate diverges; the vector is).
 *
 * Header-only and stage-independent so the offline gate drives exactly this arithmetic.
 */

#include <algorithm>
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
    /// @param rebuild_s how often a slot with rates is re-evaluated along its track, seconds
    ///        (the steering error between rebuilds is at most rate x rebuild_s: ~3 mm at 0.5 s).
    ElemSteer(std::vector<double> positions_enu, std::vector<double> freq_mhz, int n_prn,
              double sign, double hold_s, double rebuild_s = 0.5) :
        _pos(std::move(positions_enu)), _f_mhz(std::move(freq_mhz)),
        _n_elem((int)(_pos.size() / 3)), _n_chan((int)_f_mhz.size()), _sign(sign),
        _hold_s(hold_s), _rebuild_s(rebuild_s > 0.0 ? rebuild_s : 0.5),
        _tab((size_t)n_prn * _n_chan * _n_elem, cf(1.0f, 0.0f)),
        _fresh_t((size_t)n_prn, -1.0e18), _trk((size_t)n_prn) {}

    bool enabled() const {
        return _n_elem > 0 && _n_chan > 0;
    }
    int n_elem() const {
        return _n_elem;
    }
    int n_chan() const {
        return _n_chan;
    }

    /// Forget one slot's geometry. A slot is a PRN SLOT, not a satellite: when the tracker
    /// swaps the PRN in a slot, the table still holds the departed satellite's phasors and
    /// warm() would keep steering the newcomer with them for up to hold_s -- a wrong steer,
    /// not a weaker one (it derotates every element by another direction's delay).
    void invalidate(int slot) {
        if (slot >= 0 && (size_t)slot < _fresh_t.size()) {
            _fresh_t[(size_t)slot] = -1.0e18;
            _trk[(size_t)slot] = Track{};
        }
    }

    /// Copy one slot's whole [n_chan][n_elem] table into `out` (n_chan*n_elem entries).
    /// The caller holds whatever lock guards update(); the copy is what lets it drop that
    /// lock before the combine instead of reading rows a concurrent update() may rewrite.
    void copy_slot(int slot, cf* out) const {
        const cf* src = &_tab[(size_t)slot * _n_chan * _n_elem];
        std::copy(src, src + (size_t)_n_chan * _n_elem, out);
    }

    /// New geometry for one satellite slot: rebuild its [n_chan][n_elem] phasor table.
    /// az/el in degrees (az from North, clockwise/east-positive -- the broker's predict_all
    /// convention); t_now in the caller's steady seconds (only differenced against itself).
    /// @p az_rate_dps / @p el_rate_dps (deg/s) and @p t_utc (unix s, the instant az/el
    /// describe) enable extrapolation along the track by refresh(); t_utc <= 0 means a bare
    /// snapshot (held, as before).
    void update(int slot, double az_deg, double el_deg, double t_now, double az_rate_dps = 0.0,
                double el_rate_dps = 0.0, double t_utc = 0.0) {
        if (!enabled() || slot < 0 || (size_t)slot >= _fresh_t.size())
            return;
        const double az = az_deg * M_PI / 180.0, el = el_deg * M_PI / 180.0;
        const double daz = az_rate_dps * M_PI / 180.0, del = el_rate_dps * M_PI / 180.0;
        Track& k = _trk[(size_t)slot];
        // Unit vector receiver -> satellite, ENU, and its time derivative (chain rule).
        k.e[0] = std::cos(el) * std::sin(az);
        k.e[1] = std::cos(el) * std::cos(az);
        k.e[2] = std::sin(el);
        k.edot[0] = -std::sin(el) * std::sin(az) * del + std::cos(el) * std::cos(az) * daz;
        k.edot[1] = -std::sin(el) * std::cos(az) * del - std::cos(el) * std::sin(az) * daz;
        k.edot[2] = std::cos(el) * del;
        k.t0_utc = t_utc;
        k.moving = t_utc > 0.0 && (az_rate_dps != 0.0 || el_rate_dps != 0.0);
        build(slot, k.e);
        k.built_utc = t_utc;
        _fresh_t[(size_t)slot] = t_now;
    }

    /// Re-evaluate one slot's table at record time @p t_utc if it carries rates and the table
    /// is more than rebuild_s old. Extrapolation is capped at hold_s past the post: beyond
    /// that warm() refuses the slot anyway. Cheap when nothing is due (a compare).
    void refresh(int slot, double t_utc) {
        if (!enabled() || slot < 0 || (size_t)slot >= _trk.size() || !(t_utc > 0.0))
            return;
        Track& k = _trk[(size_t)slot];
        if (!k.moving || std::fabs(t_utc - k.built_utc) < _rebuild_s)
            return;
        const double dt = std::max(-_hold_s, std::min(_hold_s, t_utc - k.t0_utc));
        double e[3] = {k.e[0] + k.edot[0] * dt, k.e[1] + k.edot[1] * dt,
                       k.e[2] + k.edot[2] * dt};
        const double n = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        if (!(n > 0.0))
            return;
        for (double& x : e)
            x /= n;
        build(slot, e);
        k.built_utc = t_utc;
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
    /// One slot's posted direction and its rate: e(t) = e + edot * (t - t0_utc), renormalised.
    struct Track {
        double e[3] = {0.0, 0.0, 1.0};
        double edot[3] = {0.0, 0.0, 0.0};
        double t0_utc = 0.0;     ///< the instant e describes (unix s); 0 = no epoch
        double built_utc = 0.0;  ///< the instant the table currently describes
        bool moving = false;     ///< rates and epoch present: refresh() extrapolates
    };

    /// Fill one slot's [n_chan][n_elem] table for line-of-sight unit vector @p e (ENU).
    void build(int slot, const double* e) {
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
    }

    std::vector<double> _pos;   // [n_elem][3] ENU metres
    std::vector<double> _f_mhz; // [n_chan] RF MHz
    int _n_elem = 0;
    int _n_chan = 0;
    double _sign = 1.0;
    double _hold_s = 120.0;
    double _rebuild_s = 0.5;
    std::vector<cf> _tab;         // [n_prn][n_chan][n_elem]
    std::vector<double> _fresh_t; // [n_prn] last update, steady seconds
    std::vector<Track> _trk;      // [n_prn] posted direction + rate
};

} // namespace gnss

#endif
