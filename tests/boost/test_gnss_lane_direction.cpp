#define BOOST_TEST_MODULE "test_gnss_lane_direction"
#include "gnssFleetLane.hpp"     // for FleetLane, fleet_lane_direction
#include "gnssLaneDirection.hpp" // for LaneDirection, InstDelay

#include <algorithm> // for max
#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for exp, sqrt
#include <complex> // for complex, conj, norm
#include <random>  // for mt19937, normal_distribution
#include <vector>
using cd = std::complex<double>;
using gnss::ElemSteer;
using gnss::LaneDirection;

namespace {
// A satellite with element response s and steering table D: the raw row is s + noise, the
// steered row (s + noise) * D. The direction the helper must return is the raw response
// s / |s| (conj(D) undoes the steering), up to a common phase.
struct Sim {
    int n;
    std::vector<cd> s;
    std::vector<ElemSteer::cf> D;
    std::mt19937 rng{3};
    std::normal_distribution<double> g{0.0, 1.0};
    explicit Sim(int n_elem, double amp) : n(n_elem), s(n_elem), D(n_elem) {
        for (int i = 0; i < n; ++i) {
            const double ph = 0.37 * i * i, ps = -1.1 * i;
            s[i] = amp * std::polar(1.0, ph);
            D[i] = ElemSteer::cf((float)std::cos(ps), (float)std::sin(ps));
        }
    }
    // one steered row: (s_i + sigma * noise) * D_i
    void row(double sigma, cd* v) {
        for (int i = 0; i < n; ++i)
            v[i] =
                (s[i] + sigma * cd(g(rng), g(rng)) / std::sqrt(2.0)) * cd(D[i].real(), D[i].imag());
    }
    // |<dir, truth>|^2 with truth = s / |s|: the cos^2 of the returned direction
    double cos2(const cd* dir) const {
        cd acc(0.0, 0.0);
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            acc += std::conj(s[i]) * dir[i];
            e += std::norm(s[i]);
        }
        return std::norm(acc) / e;
    }
};
} // namespace

// Noise-free: the first push sets the state; the direction is the un-steered response at
// cos^2 = 1, unit norm, and the coherence is exactly 1.
BOOST_AUTO_TEST_CASE(noise_free_direction_is_the_unsteered_response) {
    Sim sim(8, 2.0);
    LaneDirection L(8, 3, 2, 0.5);
    std::vector<cd> v(8), out(8);
    BOOST_CHECK(!L.warm(1, 2));
    BOOST_CHECK(!L.direction(1, 2, sim.D.data(), out.data()));
    for (int r = 0; r < 200; ++r) {
        sim.row(0.0, v.data());
        L.push(1, 2, v.data(), 0.01049);
    }
    BOOST_CHECK(L.warm(1, 2));
    BOOST_CHECK(L.direction(1, 2, sim.D.data(), out.data()));
    double e = 0.0;
    for (const cd& x : out)
        e += std::norm(x);
    BOOST_CHECK_CLOSE(e, 1.0, 1e-5); // D is float: |D| = 1 to 1e-7
    BOOST_CHECK_CLOSE(sim.cos2(out.data()), 1.0, 1e-5);
    BOOST_CHECK_CLOSE(L.coherence(1, 2), 1.0, 1e-9);
    BOOST_CHECK_CLOSE(L.power(1, 2), 8 * 4.0, 1e-4); // D is float: unit modulus to 1e-7
    // other (slot, channel) cells are untouched
    BOOST_CHECK(!L.warm(0, 2));
    BOOST_CHECK(!L.warm(1, 0));
}

// Noisy: per-record SNR rho per element -> coherence ~ rho / (1 + rho) and a direction whose
// cos^2 improves with tau / dt (the averaged SNR), from rho = 0.1 (where a single record's
// direction has cos^2 ~ 0.1) to above 0.9.
BOOST_AUTO_TEST_CASE(noisy_rows_average_to_the_direction) {
    const int n = 48;
    const double dt = 0.01049, tau = 1.0;
    for (double rho : {0.1, 1.0}) {
        Sim sim(n, 1.0);
        const double sigma = 1.0 / std::sqrt(rho);
        LaneDirection L(n, 1, 1, tau);
        std::vector<cd> v(n), out(n);
        double c1 = -1.0;
        for (int r = 0; r < 1000; ++r) {
            sim.row(sigma, v.data());
            L.push(0, 0, v.data(), dt);
            if (r == 0) {
                L.direction(0, 0, sim.D.data(), out.data());
                c1 = sim.cos2(out.data());
            }
        }
        BOOST_CHECK(L.warm(0, 0));
        BOOST_CHECK(L.direction(0, 0, sim.D.data(), out.data()));
        const double c = sim.cos2(out.data());
        const double coh = L.coherence(0, 0);
        BOOST_TEST_MESSAGE("rho " << rho << ": single-record cos2 " << c1 << " averaged " << c
                                  << " coherence " << coh);
        BOOST_CHECK(c > 0.9);
        BOOST_CHECK(c > c1);
        // coherence within a factor 2 of rho / (1 + rho) + the noise floor dt / (2 tau)
        const double expect = rho / (1.0 + rho) + dt / (2.0 * tau);
        BOOST_CHECK(coh > 0.5 * expect && coh < 2.0 * expect);
    }
}

// The record phase is arbitrary: a random common phase and a random sign per record (the
// residual carrier and the symbol) must not cancel the average. The direction still converges
// and the coherence still reads the SNR.
BOOST_AUTO_TEST_CASE(random_record_phase_and_sign_do_not_cancel) {
    const int n = 48;
    const double dt = 0.01049, tau = 1.0;
    for (double rho : {0.3, 3.0}) {
        Sim sim(n, 1.0);
        const double sigma = 1.0 / std::sqrt(rho);
        LaneDirection L(n, 1, 1, tau);
        std::mt19937 prng(11);
        std::uniform_real_distribution<double> u(0.0, 6.283185307);
        std::vector<cd> v(n), out(n);
        for (int r = 0; r < 1500; ++r) {
            sim.row(sigma, v.data());
            const cd ph = std::polar(1.0, u(prng)) * ((r % 3 == 0) ? -1.0 : 1.0);
            for (cd& x : v)
                x *= ph;
            L.push(0, 0, v.data(), dt);
        }
        BOOST_CHECK(L.direction(0, 0, sim.D.data(), out.data()));
        const double c = sim.cos2(out.data());
        const double coh = L.coherence(0, 0);
        BOOST_TEST_MESSAGE("random phase, rho " << rho << ": cos2 " << c << " coherence " << coh);
        BOOST_CHECK(c > 0.9);
        const double expect = rho / (1.0 + rho) + dt / (2.0 * tau);
        BOOST_CHECK(coh > 0.5 * expect && coh < 2.0 * expect);
    }
}

// Pure noise: the coherence sits at the floor dt / (2 tau) + 1 / n (the phase alignment keeps
// ~1/n of each record's energy along the mean), far below any satellite, and the power is the
// noise power.
BOOST_AUTO_TEST_CASE(pure_noise_coherence_is_the_floor) {
    const int n = 48;
    const double dt = 0.01049, tau = 1.0;
    Sim sim(n, 0.0);
    LaneDirection L(n, 1, 1, tau);
    std::vector<cd> v(n);
    for (int r = 0; r < 3000; ++r) {
        sim.row(1.0, v.data());
        L.push(0, 0, v.data(), dt);
    }
    const double coh = L.coherence(0, 0);
    const double floor = dt / (2.0 * tau) + 1.0 / n;
    BOOST_TEST_MESSAGE("noise coherence " << coh << " floor " << floor);
    BOOST_CHECK(coh < 2.0 * floor);
    BOOST_CHECK(coh > 0.25 * floor);
    BOOST_CHECK_CLOSE(L.power(0, 0), (double)n, 15.0);
}

// reset() makes the cell cold and the next push starts over (no memory of the old PRN);
// out-of-range cells are ignored.
BOOST_AUTO_TEST_CASE(reset_and_bounds) {
    Sim a(4, 1.0), b(4, 3.0);
    LaneDirection L(4, 2, 2, 0.5);
    std::vector<cd> v(4), out(4);
    for (int r = 0; r < 100; ++r) {
        a.row(0.0, v.data());
        L.push(1, 1, v.data(), 0.02);
    }
    BOOST_CHECK(L.warm(1, 1));
    L.reset(1);
    BOOST_CHECK(!L.warm(1, 1));
    BOOST_CHECK_EQUAL(L.age_s(1, 1), 0.0);
    b.row(0.0, v.data());
    L.push(1, 1, v.data(), 0.02);
    BOOST_CHECK_CLOSE(L.power(1, 1), 4 * 9.0, 1e-4); // the new row outright, no EMA with the old
    L.push(7, 1, v.data(), 0.02);                    // ignored
    L.push(1, -1, v.data(), 0.02);                   // ignored
    BOOST_CHECK(!L.direction(7, 1, b.D.data(), out.data()));
    BOOST_CHECK_EQUAL(L.coherence(5, 0), 0.0);
}

// A lane transferred across frequency: the steering table at f1 times shift(e, f2 - f1) is the
// table at f2, for the slot's own line of sight.
BOOST_AUTO_TEST_CASE(shift_takes_the_table_across_frequency) {
    const int n = 16;
    std::vector<double> pos((size_t)n * 3);
    for (int i = 0; i < n; ++i) {
        pos[(size_t)i * 3] = 6.3 * (i % 4) - 9.0;
        pos[(size_t)i * 3 + 1] = 8.5 * (i / 4) - 12.0;
        pos[(size_t)i * 3 + 2] = 0.05 * i;
    }
    const double f1 = 1278.75, f2 = 1268.52;
    ElemSteer S(pos, {f1, f2}, 1, 1.0, 120.0);
    S.update(0, 141.0, 37.0, 0.0);
    std::vector<ElemSteer::cf> tab((size_t)2 * n), sh((size_t)n);
    S.copy_slot(0, tab.data());
    double e[3];
    BOOST_REQUIRE(S.direction(0, 0.0, e));
    S.shift(e, (f2 - f1) * 1e6, sh.data());
    double err = 0.0;
    for (int i = 0; i < n; ++i)
        err = std::max(err, (double)std::abs(tab[(size_t)i] * sh[(size_t)i] - tab[(size_t)n + i]));
    BOOST_CHECK_LT(err, 1e-5);
    // and the shift is not trivial over a 10 MHz move at these baselines
    double moved = 0.0;
    for (int i = 0; i < n; ++i)
        moved = std::max(moved, (double)std::abs(sh[(size_t)i] - ElemSteer::cf(1.0f, 0.0f)));
    BOOST_CHECK_GT(moved, 0.1);
}

// Instrumental delays: each element's response carries exp(i 2 pi f tau_e) on top of the
// geometry. InstDelay measures tau_e (up to a common delay) from adjacent comb channels of two
// satellites, and a transfer 10 MHz off the comb matches the true direction there only with
// the measured delays applied.
BOOST_AUTO_TEST_CASE(inst_delay_carries_the_transfer) {
    const int n = 32, n_comb = 7;
    std::vector<double> pos((size_t)n * 3);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> tau_inst(n), amp(n), ph0(n);
    for (int i = 0; i < n; ++i) {
        pos[(size_t)i * 3] = 6.3 * (i % 8) - 22.0;
        pos[(size_t)i * 3 + 1] = 7.9 * (i / 8) - 12.0;
        pos[(size_t)i * 3 + 2] = 0.05 * i;
        tau_inst[i] = 5e-9 * u(rng);
        amp[i] = 1.0 + 0.5 * u(rng);
        ph0[i] = M_PI * u(rng);
    }
    const double df0 = 16 * 0.1953125e6;
    std::vector<double> f_mhz;
    for (int k = 0; k < n_comb; ++k)
        f_mhz.push_back((5975 + 16 * k) * 0.1953125);
    const double f_x = f_mhz.back() + 9.96; // the transfer target, off the comb
    f_mhz.push_back(f_x);
    const int n_ch = (int)f_mhz.size();
    ElemSteer S(pos, f_mhz, 2, 1.0, 120.0);
    S.update(0, 141.0, 37.0, 0.0);
    S.update(1, 230.0, 62.0, 0.0);
    std::vector<ElemSteer::cf> tab((size_t)n_ch * n), sh((size_t)n);

    // Unit raw-frame direction for (slot, channel): conj(D) x instrument x a random common phase.
    auto truth = [&](int ch, double common, std::vector<cd>& q) {
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            const ElemSteer::cf D = tab[(size_t)ch * n + i];
            q[i] =
                std::conj(cd(D.real(), D.imag()))
                * std::polar(amp[i], ph0[i] + 2.0 * M_PI * f_mhz[ch] * 1e6 * tau_inst[i] + common);
            e += std::norm(q[i]);
        }
        for (cd& x : q)
            x /= std::sqrt(e);
    };

    gnss::InstDelay est(n, df0, 10.0);
    std::vector<std::vector<cd>> q(n_ch, std::vector<cd>(n));
    for (int t = 0; t <= 20; ++t) {
        est.advance(t);
        for (int slot = 0; slot < 2; ++slot) {
            S.copy_slot(slot, tab.data());
            double los[3] = {};
            BOOST_REQUIRE(S.direction(slot, 0.0, los));
            S.shift(los, df0, sh.data());
            for (int ch = 0; ch < n_comb; ++ch)
                truth(ch, M_PI * u(rng), q[ch]);
            for (int ch = 1; ch < n_comb; ++ch)
                est.add(q[ch - 1].data(), q[ch].data(), sh.data(), 0.5);
        }
        BOOST_CHECK_EQUAL(est.warm(), t >= 10);
    }

    // Slopes: estimate minus truth is one delay common to every element, to the float
    // phasors' precision (1e-7 rad over 2 pi x 3.125 MHz is ~5e-15 s).
    const double d0 = est.delay_s(0) - tau_inst[0];
    double slope_err = 0.0, rms_true = 0.0;
    for (int i = 0; i < n; ++i) {
        slope_err = std::max(slope_err, std::abs(est.delay_s(i) - tau_inst[i] - d0));
        rms_true += tau_inst[i] * tau_inst[i];
    }
    BOOST_TEST_MESSAGE("instrumental delay: max error " << slope_err << " s, truth rms "
                                                        << std::sqrt(rms_true / n) << " s, rms_s "
                                                        << est.rms_s() << " s");
    BOOST_CHECK_LT(slope_err, 1e-13);

    // Transfer slot 0 from the last comb channel to f_x: geometric shift, then the delays.
    S.copy_slot(0, tab.data());
    double los[3] = {};
    BOOST_REQUIRE(S.direction(0, 0.0, los));
    const double df = (f_x - f_mhz[n_comb - 1]) * 1e6;
    S.shift(los, df, sh.data());
    std::vector<cd> src(n), want(n), geo(n), full(n);
    truth(n_comb - 1, 0.3, src);
    truth(n_comb, -1.2, want);
    for (int i = 0; i < n; ++i)
        geo[i] = src[i] * std::conj(cd(sh[i].real(), sh[i].imag()));
    full = geo;
    est.apply(df, full.data());
    // max element error after aligning the common phase to the truth
    auto err = [&](const std::vector<cd>& x) {
        cd ip(0.0, 0.0);
        for (int i = 0; i < n; ++i)
            ip += std::conj(x[i]) * want[i];
        const cd rot = ip / std::abs(ip);
        double m = 0.0;
        for (int i = 0; i < n; ++i)
            m = std::max(m, std::abs(x[i] * rot - want[i]));
        return m;
    };
    BOOST_TEST_MESSAGE("transfer by " << df * 1e-6 << " MHz: max element error " << err(full)
                                      << " with the delays, " << err(geo) << " without");
    BOOST_CHECK_LT(err(full), 1e-6);
    BOOST_CHECK_GT(err(geo), 1e-2);
}

// A lane shipped between nodes (gnssFleetLane.hpp): the source publishes q with the track it
// steered with; the receiver's fleet_lane_direction at f is the local transfer
// (q conj(shift(f - f_src)), then the instrumental delays) at the source's instant, equals the
// truth at f, and re-steered later along the track still equals the truth there.
BOOST_AUTO_TEST_CASE(fleet_lane_is_the_local_transfer_moved_along_the_track) {
    const int n = 24;
    std::vector<double> pos((size_t)n * 3);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> tau_inst(n), amp(n), ph0(n);
    for (int i = 0; i < n; ++i) {
        pos[(size_t)i * 3] = 6.3 * (i % 6) - 15.0;
        pos[(size_t)i * 3 + 1] = 8.5 * (i / 6) - 12.0;
        pos[(size_t)i * 3 + 2] = 0.05 * i;
        tau_inst[i] = 5e-9 * u(rng);
        amp[i] = 1.0 + 0.5 * u(rng);
        ph0[i] = M_PI * u(rng);
    }
    const double f1 = 1227.6, f2 = 1268.52; // MHz: L2C's lane carried to B3I's channel
    ElemSteer S(pos, {f1, f2}, 1, 1.0, 120.0);
    const double t0 = 1.7e9;
    S.update(0, 141.0, 37.0, 0.0, 0.01, -0.004, t0); // a moving track with an epoch
    std::vector<ElemSteer::cf> tab((size_t)2 * n), sh((size_t)n);
    S.copy_slot(0, tab.data());
    auto truth = [&](int ch, double common, std::vector<cd>& q) {
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            const ElemSteer::cf D = tab[(size_t)ch * n + i];
            const double f = (ch == 0 ? f1 : f2) * 1e6;
            q[i] = std::conj(cd(D.real(), D.imag()))
                   * std::polar(amp[i], ph0[i] + 2.0 * M_PI * f * tau_inst[i] + common);
            e += std::norm(q[i]);
        }
        for (cd& x : q)
            x /= std::sqrt(e);
    };
    auto err = [&](const std::vector<cd>& x, const std::vector<cd>& want) {
        cd ip(0.0, 0.0);
        for (int i = 0; i < n; ++i)
            ip += std::conj(x[i]) * want[i];
        const cd rot = ip / std::abs(ip);
        double m = 0.0;
        for (int i = 0; i < n; ++i)
            m = std::max(m, std::abs(x[i] * rot - want[i]));
        return m;
    };

    gnss::FleetLane L;
    L.f_src_hz = f1 * 1e6;
    BOOST_REQUIRE(S.track(0, L.los, L.los_rate, &L.t_utc));
    BOOST_CHECK_CLOSE(L.t_utc, t0, 1e-9);
    BOOST_CHECK_GT(std::abs(L.los_rate[0]) + std::abs(L.los_rate[1]), 0.0);
    L.q.resize(n);
    truth(0, 0.3, L.q);
    L.fids = {{6233, 6338}, {6442, 6573}};
    BOOST_CHECK(L.covers(6500));
    BOOST_CHECK(!L.covers(6400));

    ElemSteer geo(pos, {}, 0, 1.0, 0.0); // positions and sign only: shift()
    std::vector<ElemSteer::cf> scratch;
    std::vector<cd> out(n), want(n), local(n);
    // (a) at the source's instant, geometry only: the local transfer
    BOOST_REQUIRE(gnss::fleet_lane_direction(L, geo, f2 * 1e6, t0, nullptr, scratch, out.data()));
    double e0[3];
    BOOST_REQUIRE(S.direction(0, t0, e0));
    S.shift(e0, (f2 - f1) * 1e6, sh.data());
    for (int i = 0; i < n; ++i)
        local[i] = L.q[i] * std::conj(cd(sh[(size_t)i].real(), sh[(size_t)i].imag()));
    double d = 0.0;
    for (int i = 0; i < n; ++i)
        d = std::max(d, std::abs(out[i] - local[i]));
    BOOST_CHECK_LT(d, 1e-5);
    // (b) with the delays: the truth at f2
    truth(1, -1.2, want);
    BOOST_REQUIRE(
        gnss::fleet_lane_direction(L, geo, f2 * 1e6, t0, tau_inst.data(), scratch, out.data()));
    BOOST_CHECK_LT(err(out, want), 1e-5);
    BOOST_CHECK_GT(err(local, want), 1e-2);
    // (c) 20 s on: the track has moved; evaluated there it is the truth there, held it is not
    const double t1 = t0 + 20.0;
    S.refresh(0, t1);
    S.copy_slot(0, tab.data());
    std::vector<cd> want1(n), held(n);
    truth(1, 0.8, want1);
    held = out;
    BOOST_REQUIRE(
        gnss::fleet_lane_direction(L, geo, f2 * 1e6, t1, tau_inst.data(), scratch, out.data()));
    BOOST_TEST_MESSAGE("20 s along the track: error re-steered " << err(out, want1) << ", held "
                                                                 << err(held, want1));
    BOOST_CHECK_LT(err(out, want1), 1e-5);
    BOOST_CHECK_GT(err(held, want1), 1e-2);
    // a degenerate track is refused
    gnss::FleetLane bad = L;
    std::fill(bad.los, bad.los + 3, 0.0);
    std::fill(bad.los_rate, bad.los_rate + 3, 0.0);
    BOOST_CHECK(!gnss::fleet_lane_direction(bad, geo, f2 * 1e6, t0, nullptr, scratch, out.data()));
}
