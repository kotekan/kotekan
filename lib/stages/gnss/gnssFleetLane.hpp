/**
 * @file
 * @brief A lane direction received from another node and its evaluation on a channel of this
 *        GPU (docs/CHORD_FLEET_LANES.md). Header-only, no CUDA.
 */
#ifndef GNSS_FLEET_LANE_HPP
#define GNSS_FLEET_LANE_HPP

#include "gnssElemSteer.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace gnss {

/**
 * @struct FleetLane
 * @brief One satellite's published lane direction and the track it was steered with.
 *
 * q is the raw-frame direction LaneDirection::direction returned on the source channel:
 * conj(D(f_src, e)) m, with D the geometric phasor ElemSteer::build uses for line of sight e
 * at f_src and m the steered prompt-row mean, the per-element instrument response, which
 * varies slowly. On any channel f at time t the direction is conj(D(f, e(t))) D(f_src, e) q,
 * times the instrumental delays from f_src to f (gnssLaneDirection.hpp InstDelay): the
 * lane_transfer_freq_ids transfer, with the line of sight moved along its track.
 */
struct FleetLane {
    using cd = std::complex<double>;
    std::string id;                        ///< "<node>/<owner>": logs, duplicates
    double f_src_hz = 0.0;                 ///< the channel q was measured on
    double t_utc = 0.0;                    ///< the instant los describes (unix s; 0 = held)
    double los[3] = {0.0, 0.0, 1.0};       ///< line of sight at t_utc, ENU unit vector
    double los_rate[3] = {0.0, 0.0, 0.0};  ///< its rate per second
    double coh = 0.0;                      ///< LaneDirection::coherence at the source
    double power = 0.0;                    ///< the source row's EMA power
    std::vector<std::pair<int, int>> fids; ///< target freq_id ranges, inclusive
    std::vector<cd> q;                     ///< [n_elem]

    bool covers(int fid) const {
        for (const auto& r : fids)
            if (fid >= r.first && fid <= r.second)
                return true;
        return false;
    }

    /// The line of sight at @p t along the track, renormalised. False if degenerate.
    bool los_at(double t, double* e_out) const {
        const double dt = t_utc > 0.0 ? t - t_utc : 0.0;
        double e[3], n2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            e[i] = los[i] + los_rate[i] * dt;
            n2 += e[i] * e[i];
        }
        if (!(n2 > 0.0))
            return false;
        const double inv = 1.0 / std::sqrt(n2);
        for (int i = 0; i < 3; ++i)
            e_out[i] = e[i] * inv;
        return true;
    }
};

/// Per element, the factor D(f, los) conj(D(f, los(t))) that turns a raw-frame direction on a
/// channel at @p f_hz from line of sight @p los (the instant @p t_los) along its track
/// (@p los_rate per second) to time @p t: fleet_lane_direction's turn with f_src = f. False if
/// the track is degenerate (out untouched).
inline bool lane_turn(const ElemSteer& geo, const double* los, const double* los_rate, double t_los,
                      double t, double f_hz, ElemSteer::cf* out) {
    const double dt = t - t_los;
    double s[3], n2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        s[i] = los[i] + los_rate[i] * dt;
        n2 += s[i] * s[i];
    }
    if (!(n2 > 0.0))
        return false;
    const double inv = 1.0 / std::sqrt(n2);
    double d[3];
    for (int i = 0; i < 3; ++i)
        d[i] = los[i] - s[i] * inv;
    geo.shift(d, f_hz, out); // tau is linear in the line of sight
    return true;
}

/// The lane's raw-frame direction on a channel at @p f_hz at time @p t (see FleetLane), turned
/// by the instrumental delays from f_src to f (@p inst_delay_s per element, or null for none).
/// @p geo holds the element positions and the steering sign; @p sh is scratch. False if the
/// track is degenerate (out untouched).
inline bool fleet_lane_direction(const FleetLane& L, const ElemSteer& geo, double f_hz, double t,
                                 const double* inst_delay_s, std::vector<ElemSteer::cf>& sh,
                                 std::complex<double>* out) {
    double e[3];
    if (!L.los_at(t, e))
        return false;
    const int n = (int)L.q.size();
    sh.resize((size_t)n * 2);
    geo.shift(L.los, L.f_src_hz, sh.data()); // D(f_src, e): undoes the source's steering
    geo.shift(e, f_hz, sh.data() + n);       // D(f, e(t))
    const double df = f_hz - L.f_src_hz;
    for (int i = 0; i < n; ++i) {
        const ElemSteer::cf Ds = sh[(size_t)i], Dt = sh[(size_t)n + i];
        out[i] = L.q[(size_t)i] * std::complex<double>(Ds.real(), Ds.imag())
                 * std::complex<double>(Dt.real(), -Dt.imag());
        if (inst_delay_s)
            out[i] *= std::polar(1.0, 2.0 * M_PI * df * inst_delay_s[i]);
    }
    return true;
}

/**
 * @struct InstPhaseTable
 * @brief Each element's instrumental phase at knots in freq_id, from a Tau A calibration
 *        (python/scripts/gnss/inst_phase_table.py; docs/CHORD_FLEET_LANES.md "The instrumental
 *        phase table").
 *
 * The phases are in the science N^2's frame, a lane's after lane_conj: a lane carried there from
 * channel f_src to f is turned per element by exp(i(phi(f) - phi(f_src))).
 */
struct InstPhaseTable {
    int64_t made_frame0 = 0;   ///< the F-engine start the phases hold for
    int n = 0;                 ///< elements
    std::vector<int> fids;     ///< knots, ascending
    std::vector<double> phase; ///< [knot][n], radians; NaN where the knot does not hold the element

    bool empty() const {
        return fids.empty();
    }

    /// Element @p el's phase at @p fid: linear in freq_id between the two nearest knots that hold
    /// it, their difference wrapped into (-pi, pi] (the knots are ~1 MHz apart), and the nearest
    /// one's value outside them. NaN if no knot holds it.
    double phase_at(double fid, int el) const {
        const int nk = (int)fids.size();
        const int j = (int)(std::lower_bound(fids.begin(), fids.end(), fid) - fids.begin());
        int lo = j - 1, hi = j;
        while (lo >= 0 && std::isnan(phase[(size_t)lo * n + el]))
            --lo;
        while (hi < nk && std::isnan(phase[(size_t)hi * n + el]))
            ++hi;
        if (lo < 0 && hi >= nk)
            return std::numeric_limits<double>::quiet_NaN();
        if (lo < 0)
            return phase[(size_t)hi * n + el];
        if (hi >= nk)
            return phase[(size_t)lo * n + el];
        const double p0 = phase[(size_t)lo * n + el];
        double d = phase[(size_t)hi * n + el] - p0;
        d -= 2.0 * M_PI * std::ceil((d - M_PI) / (2.0 * M_PI));
        return p0 + d * (fid - fids[(size_t)lo]) / (fids[(size_t)hi] - fids[(size_t)lo]);
    }
};

/// Per element, the phasor u(fid) whose ratio u(f) conj(u(f_src)) carries a lane in the science
/// N^2's frame from channel f_src to f: exp(i phi(fid)) where @p tab holds the element (the
/// table's carry, exp(i(phi(f) - phi(f_src)))); else the pooled delay @p inst_delay_s (null for
/// none), exp(i 2 pi f tau) in the raw frame that InstDelay::apply and fleet_lane_direction turn,
/// conjugated with @p lane_conj; else 1. Whether the table holds an element does not depend on
/// the channel, so u(f) and u(f_src) always take the same branch.
inline void inst_carry_phasors(const InstPhaseTable& tab, const double* inst_delay_s,
                               bool lane_conj, int n, double fid, std::complex<double>* out) {
    const double w = (lane_conj ? -2.0 : 2.0) * M_PI * fid * 195312.5;
    for (int el = 0; el < n; ++el) {
        const double ph =
            tab.n == n ? tab.phase_at(fid, el) : std::numeric_limits<double>::quiet_NaN();
        if (!std::isnan(ph))
            out[el] = std::polar(1.0, ph);
        else if (inst_delay_s)
            out[el] = std::polar(1.0, w * inst_delay_s[el]);
        else
            out[el] = 1.0;
    }
}

} // namespace gnss

#endif // GNSS_FLEET_LANE_HPP
