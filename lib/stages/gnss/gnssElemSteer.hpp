#ifndef GNSS_ELEM_STEER_HPP
#define GNSS_ELEM_STEER_HPP
/**
 * @file gnssElemSteer.hpp
 * @brief Per-element geometric steering for the GNSS despread combine.
 *  - gnss::ElemSteer
 *
 * The despread's channel-to-element combine uses one code phase for every element. An element
 * displaced by r along the satellite line of sight e_sat receives the signal earlier by
 * tau = (r . e_sat)/c: 200 m of array is ~0.67 us, ~7 chips at L5, and the L5 correlation peak
 * loses an element offset by more than +-0.5 chip (+-15 m).
 *
 * Across a channel narrow compared with 1/tau, a delay tau is a per-channel phase
 * 2*pi*f_ch*tau, so steering is one complex multiply per (channel, element) inside the combine,
 * correcting code and carrier together. The sparse comb's delay aliasing does not matter: tau
 * is applied, never estimated.
 *
 * The sign of that phase depends on conventions earlier in the signal chain (the F-engine's FFT
 * and output conjugation), so it is not derived here: it is a config parameter
 * (elem_steer_sign) calibrated from data, e.g. each element's cross-channel phase slope against
 * geometry.
 *
 * Geometry arrives per satellite over REST (/set_sat_geometry, az/el in degrees, every ~30 s)
 * and is held for hold_s; a slot with no post within hold_s is not steered. A snapshot drifts
 * within a post interval: a GPS satellite turns ~1.5e-4 rad/s, so the path difference across a
 * 44 m baseline moves ~7 mm/s, ~0.2 m (most of an L-band wavelength) in 30 s. So a post may
 * carry the az/el rates and the epoch they describe, and refresh() re-evaluates the table at
 * the record's own time, every rebuild_s, from the line-of-sight unit vector extrapolated
 * linearly (az/el themselves are not linear near zenith, where the az rate diverges; the
 * vector is).
 *
 * Header-only and independent of the stage; GnssGpuRecordAssemble is the consumer.
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

    /// @param positions_enu flat [n_elem][3] element positions, metres East/North/Up from the
    ///        point the delays are referred to (an element there gets a unity phasor).
    /// @param freq_mhz per-channel RF centre frequencies, MHz (freq_id x 0.1953125 at CHORD).
    /// @param n_prn number of PRN slots.
    /// @param sign +1 or -1, the calibrated phase sign (see file note).
    /// @param hold_s how long a post's geometry stays valid, seconds.
    /// @param rebuild_s how often a slot with rates is re-evaluated along its track, seconds;
    ///        between rebuilds the path error grows at the path rate (~3.5 mm in 0.5 s at 7 mm/s).
    ElemSteer(std::vector<double> positions_enu, std::vector<double> freq_mhz, int n_prn,
              double sign, double hold_s, double rebuild_s = 0.5) :
        _pos(std::move(positions_enu)), _f_mhz(std::move(freq_mhz)),
        _n_elem((int)(_pos.size() / 3)), _n_chan((int)_f_mhz.size()), _sign(sign), _hold_s(hold_s),
        _rebuild_s(rebuild_s > 0.0 ? rebuild_s : 0.5),
        _tab((size_t)n_prn * _n_chan * _n_elem, cf(1.0f, 0.0f)), _fresh_t((size_t)n_prn, -1.0e18),
        _trk((size_t)n_prn) {}

    bool enabled() const {
        return _n_elem > 0 && _n_chan > 0;
    }
    int n_elem() const {
        return _n_elem;
    }
    int n_chan() const {
        return _n_chan;
    }

    /// Forget one slot's geometry. The table is keyed by PRN slot, not satellite: after a PRN
    /// swap it still holds the departed satellite's phasors, and warm() would steer the
    /// newcomer with them for up to hold_s, derotating every element by another direction's
    /// delays.
    void invalidate(int slot) {
        if (slot >= 0 && (size_t)slot < _fresh_t.size()) {
            _fresh_t[(size_t)slot] = -1.0e18;
            _trk[(size_t)slot] = Track{};
        }
    }

    /// Copy one slot's [n_chan][n_elem] table into @p out (n_chan*n_elem entries). The caller
    /// holds the lock that guards update() and combines from the copy after releasing it, so a
    /// concurrent update() cannot rewrite rows mid-combine.
    void copy_slot(int slot, cf* out) const {
        const cf* src = &_tab[(size_t)slot * _n_chan * _n_elem];
        std::copy(src, src + (size_t)_n_chan * _n_elem, out);
    }

    /// New geometry for one slot: rebuild its [n_chan][n_elem] phasor table. az/el in degrees
    /// (az from North, east-positive, as the broker's predict_all); t_now in the caller's steady
    /// seconds (only differenced against itself). @p az_rate_dps and @p el_rate_dps (deg/s) with
    /// @p t_utc (unix s, the instant az/el describe) let refresh() extrapolate along the track;
    /// t_utc <= 0, or both rates zero, gives a snapshot, held as posted.
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
    /// is at least rebuild_s old. Extrapolation is capped at hold_s from the post's epoch:
    /// beyond that warm() refuses the slot anyway. Cheap when nothing is due (a compare).
    void refresh(int slot, double t_utc) {
        if (!enabled() || slot < 0 || (size_t)slot >= _trk.size() || !(t_utc > 0.0))
            return;
        Track& k = _trk[(size_t)slot];
        if (!k.moving || std::fabs(t_utc - k.built_utc) < _rebuild_s)
            return;
        const double dt = std::max(-_hold_s, std::min(_hold_s, t_utc - k.t0_utc));
        double e[3] = {k.e[0] + k.edot[0] * dt, k.e[1] + k.edot[1] * dt, k.e[2] + k.edot[2] * dt};
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

    /// The slot's line-of-sight unit vector (ENU) at @p t_utc, extrapolated along its track as
    /// refresh() does but without rebuilding the table. False if the slot has no geometry
    /// (never posted, or invalidated); freshness is the caller's check (warm()).
    bool direction(int slot, double t_utc, double* e_out) const {
        if (!enabled() || slot < 0 || (size_t)slot >= _trk.size()
            || _fresh_t[(size_t)slot] <= -1.0e17)
            return false;
        const Track& k = _trk[(size_t)slot];
        double dt = 0.0;
        if (k.moving && t_utc > 0.0)
            dt = std::max(-_hold_s, std::min(_hold_s, t_utc - k.t0_utc));
        double e[3] = {k.e[0] + k.edot[0] * dt, k.e[1] + k.edot[1] * dt, k.e[2] + k.edot[2] * dt};
        const double n = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        if (!(n > 0.0))
            return false;
        for (int i = 0; i < 3; ++i)
            e_out[i] = e[i] / n;
        return true;
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
        double t0_utc = 0.0;    ///< the instant e describes (unix s); 0 = no epoch
        double built_utc = 0.0; ///< the instant the table currently describes
        bool moving = false;    ///< rates and epoch present: refresh() extrapolates
    };

    /// Fill one slot's [n_chan][n_elem] table for line-of-sight unit vector @p e (ENU).
    void build(int slot, const double* e) {
        cf* tab = &_tab[(size_t)slot * _n_chan * _n_elem];
        for (int el_i = 0; el_i < _n_elem; ++el_i) {
            // An element displaced toward the satellite receives the signal earlier, by
            // tau = (r.e)/c; its phasor is exp(i * sign * 2*pi*f*tau).
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
