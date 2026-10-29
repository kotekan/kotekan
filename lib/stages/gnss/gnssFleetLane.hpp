/**
 * @file
 * @brief A lane direction received from another node and its evaluation on a channel of this
 *        GPU (docs/CHORD_FLEET_LANES.md). Header-only, no CUDA.
 */
#ifndef GNSS_FLEET_LANE_HPP
#define GNSS_FLEET_LANE_HPP

#include "gnssElemSteer.hpp"

#include <cmath>
#include <complex>
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

} // namespace gnss

#endif // GNSS_FLEET_LANE_HPP
