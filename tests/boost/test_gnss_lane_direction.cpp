#define BOOST_TEST_MODULE "test_gnss_lane_direction"
#include "gnssFleetLane.hpp"     // for FleetLane, fleet_lane_direction
#include "gnssLaneDirection.hpp" // for LaneDirection, InstDelay

#include <algorithm> // for max
#include <atomic>    // for atomic
#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for exp, sqrt
#include <complex> // for complex, conj, norm
#include <memory>  // for shared_ptr, make_shared
#include <random>  // for mt19937, normal_distribution
#include <thread>  // for thread
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

// A held direction outlives the satellite's coherence: after the rows turn to noise (the tracker
// lost the satellite) the live direction is lost, while held_direction() with the CURRENT table
// is the satellite's raw response at the new geometry, g o conj(D2) with g = s o D the steered
// mean. A reset drops the hold.
BOOST_AUTO_TEST_CASE(peak_hold_keeps_the_best_mean) {
    // The snapshot follows the slot's best recent coherence: a lane that sinks into a
    // bright pass keeps its pre-pass mean; once the best has decayed, a lesser mean is
    // accepted again; a reset forgets the best.
    const int n = 48;
    const double dt = 0.01049, tau = 1.0;
    Sim sim(n, 1.0);
    LaneDirection L(n, 1, 1, tau);
    L.set_best_tau_s(1000.0);
    std::vector<cd> v(n);
    BOOST_CHECK_EQUAL(L.best_coherence(0, 0, 100.0), 0.0);
    BOOST_CHECK(!L.hold_if_best(0, 0, 100.0, 0.9, 0.08)); // cold: nothing to hold
    for (int r = 0; r < 1000; ++r) {
        sim.row(1.0 / std::sqrt(3.0), v.data());
        L.push(0, 0, v.data(), dt);
    }
    const double c_good = L.coherence(0, 0);
    BOOST_CHECK_GT(c_good, 0.5);
    BOOST_CHECK(L.hold_if_best(0, 0, 100.0, 0.9, 0.08));
    BOOST_CHECK_EQUAL(L.held_coherence(0, 0), c_good);
    BOOST_CHECK_CLOSE(L.best_coherence(0, 0, 100.0), c_good, 1e-9);
    // The lane sinks (a bright pass leaks in): still above the floor, well under 0.9 of
    // its best, so no new snapshot and the held mean stays the good one.
    for (int r = 0; r < 1000; ++r) {
        sim.row(2.0, v.data());
        L.push(0, 0, v.data(), dt);
    }
    const double c_low = L.coherence(0, 0);
    BOOST_CHECK_GT(c_low, 0.08);
    BOOST_CHECK_LT(c_low, 0.9 * c_good);
    BOOST_CHECK(!L.hold_if_best(0, 0, 110.0, 0.9, 0.08));
    BOOST_CHECK_EQUAL(L.held_coherence(0, 0), c_good);
    BOOST_CHECK_EQUAL(L.held_age_s(0, 0, 110.0), 10.0);
    // Below the floor: never.
    BOOST_CHECK(!L.hold_if_best(0, 0, 110.0, 0.9, c_low + 0.01));
    // Long after, the best has decayed (e-folding 1000 s) and the lesser mean is accepted.
    const double t_late = 100.0 + 1000.0 * std::log(c_good / (0.5 * c_low));
    BOOST_CHECK_LT(L.best_coherence(0, 0, t_late), c_low);
    BOOST_CHECK(L.hold_if_best(0, 0, t_late, 0.9, 0.08));
    BOOST_CHECK_EQUAL(L.held_coherence(0, 0), c_low);
    BOOST_CHECK_CLOSE(L.best_coherence(0, 0, t_late), c_low, 1e-9);
    // Recovery: the coherence climbs back and the best follows it up.
    for (int r = 0; r < 1000; ++r) {
        sim.row(1.0 / std::sqrt(3.0), v.data());
        L.push(0, 0, v.data(), dt);
    }
    BOOST_CHECK(L.hold_if_best(0, 0, t_late + 10.0, 0.9, 0.08));
    BOOST_CHECK_GT(L.held_coherence(0, 0), 0.9 * c_good);
    BOOST_CHECK_GT(L.best_coherence(0, 0, t_late + 10.0), 0.9 * c_good);
    L.reset(0);
    BOOST_CHECK_EQUAL(L.best_coherence(0, 0, t_late + 10.0), 0.0);
    BOOST_CHECK_LT(L.held_age_s(0, 0, t_late + 10.0), 0.0);
}

BOOST_AUTO_TEST_CASE(capture_needs_a_coherence_jump_and_a_new_direction) {
    // A weak lane's live and held means overlap little (both noisy) but its coherence does
    // not jump: not captured. A bright pass leaking in raises the coherence AND moves the
    // mean onto another direction: captured; when it leaves, no longer.
    const int n = 48;
    const double dt = 0.01049, tau = 1.0;
    Sim weak(n, 1.0), bright(n, 1.0);
    for (int i = 0; i < n; ++i) // the bright satellite's response: another direction
        bright.s[i] = std::polar(1.0, 0.37 * i + 1.0);
    LaneDirection L(n, 1, 1, tau);
    L.set_best_tau_s(6000.0);
    std::vector<cd> v(n);
    BOOST_CHECK(!L.captured(0, 0, 100.0, 1.5, 0.5)); // nothing held
    for (int r = 0; r < 1000; ++r) {
        weak.row(2.0, v.data()); // noisy: coherence well below 1
        L.push(0, 0, v.data(), dt);
    }
    const double c_weak = L.coherence(0, 0);
    BOOST_CHECK(L.hold_if_best(0, 0, 100.0, 0.9, 0.08));
    // More of the same: the mean wanders (overlap with the held one below 1) but the
    // coherence stays near its best -> never captured.
    int false_captures = 0;
    double ov_min = 1.0;
    for (int blk = 0; blk < 20; ++blk) {
        for (int r = 0; r < 200; ++r) {
            weak.row(2.0, v.data());
            L.push(0, 0, v.data(), dt);
        }
        ov_min = std::min(ov_min, L.overlap(0, 0));
        if (L.captured(0, 0, 100.0 + blk, 1.5, 0.5))
            ++false_captures;
    }
    BOOST_TEST_MESSAGE("weak lane: coherence " << c_weak << ", min overlap with the held mean "
                                               << ov_min);
    BOOST_CHECK_EQUAL(false_captures, 0);
    // The bright pass: strong rows in the other direction take the mean over.
    for (int r = 0; r < 1000; ++r) {
        bright.row(0.2, v.data());
        L.push(0, 0, v.data(), dt);
    }
    BOOST_CHECK_GT(L.coherence(0, 0), 1.5 * c_weak);
    BOOST_CHECK_LT(L.overlap(0, 0), 0.5);
    BOOST_CHECK(L.captured(0, 0, 200.0, 1.5, 0.5));
    BOOST_CHECK_EQUAL(L.held_coherence(0, 0), c_weak); // the snapshot was not refreshed
    // It leaves: back to the weak rows, the coherence falls back -> not captured.
    for (int r = 0; r < 2000; ++r) {
        weak.row(2.0, v.data());
        L.push(0, 0, v.data(), dt);
    }
    BOOST_CHECK(!L.captured(0, 0, 300.0, 1.5, 0.5));
}

BOOST_AUTO_TEST_CASE(held_direction_survives_lost_coherence) {
    const int n = 48;
    const double dt = 0.01049, tau = 1.0;
    Sim sim(n, 1.0), noise(n, 0.0);
    LaneDirection L(n, 1, 1, tau);
    std::vector<cd> v(n), out(n);
    BOOST_CHECK_LT(L.held_age_s(0, 0, 100.0), 0.0);
    BOOST_CHECK(!L.held_direction(0, 0, sim.D.data(), out.data()));
    for (int r = 0; r < 1000; ++r) {
        sim.row(1.0 / std::sqrt(3.0), v.data());
        L.push(0, 0, v.data(), dt);
    }
    const double coh0 = L.coherence(0, 0), p0 = L.power(0, 0);
    L.hold(0, 0, 100.0);
    BOOST_CHECK_EQUAL(L.held_age_s(0, 0, 130.0), 30.0);
    BOOST_CHECK_EQUAL(L.held_coherence(0, 0), coh0);
    BOOST_CHECK_EQUAL(L.held_power(0, 0), p0);
    for (int r = 0; r < 1000; ++r) {
        noise.row(1.0, v.data());
        L.push(0, 0, v.data(), dt);
    }
    BOOST_CHECK(L.direction(0, 0, sim.D.data(), out.data()));
    const double c_live = sim.cos2(out.data());
    // The satellite moved: a new table D2, and its raw response is g o conj(D2).
    std::vector<ElemSteer::cf> D2(n);
    std::vector<cd> s2(n);
    for (int i = 0; i < n; ++i) {
        D2[i] = ElemSteer::cf((float)std::cos(0.21 * i), (float)std::sin(0.21 * i));
        s2[i] = sim.s[i] * cd(sim.D[i].real(), sim.D[i].imag())
                * std::conj(cd(D2[i].real(), D2[i].imag()));
    }
    BOOST_CHECK(L.held_direction(0, 0, D2.data(), out.data()));
    cd acc(0.0, 0.0);
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += std::conj(s2[i]) * out[i];
        e += std::norm(s2[i]);
    }
    const double c_held = std::norm(acc) / e;
    BOOST_TEST_MESSAGE("lost coherence " << L.coherence(0, 0) << " (held " << coh0
                                         << "): live cos2 " << c_live << ", held at the new "
                                         << "geometry " << c_held << ", the old one "
                                         << sim.cos2(out.data()));
    BOOST_CHECK_LT(L.coherence(0, 0), 0.05);
    BOOST_CHECK_LT(c_live, 0.5);
    BOOST_CHECK_GT(c_held, 0.95);
    BOOST_CHECK_LT(sim.cos2(out.data()), 0.5);
    L.reset(0);
    BOOST_CHECK_LT(L.held_age_s(0, 0, 130.0), 0.0);
    BOOST_CHECK(!L.held_direction(0, 0, D2.data(), out.data()));
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

namespace {
// An 8 x 8 grid at CHORD's spacing, ~44 m by ~60 m about the reference point.
std::vector<double> grid_positions() {
    std::vector<double> pos;
    for (int i = 0; i < 64; ++i) {
        pos.push_back(6.3 * (i % 8) - 22.0);
        pos.push_back(8.5 * (i / 8) - 30.0);
        pos.push_back(0.05 * i);
    }
    return pos;
}

// The largest phase difference between two tables, rad.
double max_phase_err(const std::vector<ElemSteer::cf>& a, const std::vector<ElemSteer::cf>& b) {
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(std::arg(cd(a[i].real(), a[i].imag())
                                              * std::conj(cd(b[i].real(), b[i].imag())))));
    return err;
}
} // namespace

// A GuoWang satellite turns ~5e-3 rad/s, ~35 times a GPS satellite. Refreshed at the record
// cadence, both tables stay within 0.1 rad of the track's own line of sight; the fast track is
// rebuilt every second record or so, the slow one every REBUILD_S. A table 0.5 s old would be more
// than 1 rad off for the fast track.
BOOST_AUTO_TEST_CASE(refresh_keeps_a_fast_track_current) {
    const std::vector<double> pos = grid_positions();
    const int n = (int)pos.size() / 3;
    const double f_mhz = 1203.125, t0 = 1.7939e9, dt_rec = 0.0105;
    ElemSteer S(pos, {f_mhz}, 2, 1.0, 120.0);
    S.update(0, 180.0, 62.7, 0.0, 0.0, 0.31, t0);  // LEO
    S.update(1, 180.0, 62.7, 0.0, 0.0, 0.008, t0); // MEO
    ElemSteer exact(pos, {f_mhz}, 1, 1.0, 120.0);
    std::vector<ElemSteer::cf> tab((size_t)n), want((size_t)n);
    for (int slot = 0; slot < 2; ++slot) {
        double err = 0.0, built_prev = t0, e[3], edot[3], built;
        int rebuilds = 0;
        for (int k = 1; k <= 190; ++k) {
            const double t = t0 + k * dt_rec;
            S.refresh(slot, t);
            S.copy_slot(slot, tab.data());
            BOOST_REQUIRE(S.direction(slot, t, e));
            exact.shift(e, f_mhz * 1e6, want.data());
            err = std::max(err, max_phase_err(tab, want));
            BOOST_REQUIRE(S.track(slot, e, edot, &built));
            rebuilds += built != built_prev;
            built_prev = built;
        }
        BOOST_TEST_MESSAGE("slot " << slot << ": max phase error " << err << " rad, " << rebuilds
                                   << " rebuilds in 2 s");
        BOOST_CHECK_LT(err, 0.1);
        if (slot == 0)
            BOOST_CHECK_GT(rebuilds, 80);
        else
            BOOST_CHECK_LE(rebuilds, 6);
    }
    // The fast track 0.5 s apart: what a REBUILD_S-only table would be off by.
    double e0[3], e1[3];
    BOOST_REQUIRE(S.direction(0, t0, e0));
    BOOST_REQUIRE(S.direction(0, t0 + 0.5, e1));
    exact.shift(e0, f_mhz * 1e6, tab.data());
    exact.shift(e1, f_mhz * 1e6, want.data());
    BOOST_CHECK_GT(max_phase_err(tab, want), 1.0);
}

// A lane stamped with its track is turned to a later time: the turned direction is the true
// one there, the stale one is not, and the turn is fleet_lane_direction's on the same channel.
BOOST_AUTO_TEST_CASE(lane_turn_follows_the_track) {
    const std::vector<double> pos = grid_positions();
    const int n = (int)pos.size() / 3;
    const double f = 1203.125e6, t0 = 1.7939e9, t1 = t0 + 0.3;
    ElemSteer geo(pos, {}, 0, 1.0, 0.0); // as GnssN2Project builds it
    const double el = 62.7 * M_PI / 180.0, del = 0.31 * M_PI / 180.0;
    const double los[3] = {0.0, -std::cos(el), std::sin(el)};
    const double rate[3] = {0.0, std::sin(el) * del, std::cos(el) * del};
    double s1[3], n2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        s1[i] = los[i] + rate[i] * (t1 - t0);
        n2 += s1[i] * s1[i];
    }
    for (double& x : s1)
        x /= std::sqrt(n2);
    // raw-frame direction conj(D(f, e)) m, m a fixed instrument response
    std::vector<ElemSteer::cf> D0((size_t)n), D1((size_t)n), fac((size_t)n);
    geo.shift(los, f, D0.data());
    geo.shift(s1, f, D1.data());
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<cd> q0((size_t)n), q1((size_t)n), turned((size_t)n);
    for (int i = 0; i < n; ++i) {
        const cd m = std::polar(1.0 + 0.3 * u(rng), M_PI * u(rng));
        q0[i] = std::conj(cd(D0[i].real(), D0[i].imag())) * m;
        q1[i] = std::conj(cd(D1[i].real(), D1[i].imag())) * m;
    }
    BOOST_REQUIRE(gnss::lane_turn(geo, los, rate, t0, t1, f, fac.data()));
    for (int i = 0; i < n; ++i)
        turned[i] = q0[i] * cd(fac[i].real(), fac[i].imag());
    auto ovl = [&](const std::vector<cd>& a) {
        cd acc(0.0, 0.0);
        double ea = 0.0, eb = 0.0;
        for (int i = 0; i < n; ++i) {
            acc += std::conj(q1[i]) * a[i];
            ea += std::norm(a[i]);
            eb += std::norm(q1[i]);
        }
        return std::norm(acc) / (ea * eb);
    };
    BOOST_CHECK_GT(ovl(turned), 0.9999);
    // 0.3 s stale: an overlap under 0.9 caps the null at 10 dB (0.58 on this grid)
    BOOST_CHECK_LT(ovl(q0), 0.9);

    gnss::FleetLane L;
    L.f_src_hz = f;
    L.t_utc = t0;
    std::copy(los, los + 3, L.los);
    std::copy(rate, rate + 3, L.los_rate);
    L.q = q0;
    std::vector<ElemSteer::cf> sh;
    std::vector<cd> fl((size_t)n);
    BOOST_REQUIRE(gnss::fleet_lane_direction(L, geo, f, t1, nullptr, sh, fl.data()));
    double diff = 0.0;
    for (int i = 0; i < n; ++i)
        diff = std::max(diff, std::abs(fl[i] - turned[i]));
    BOOST_CHECK_LT(diff, 1e-5);
}

// The instrumental phase table (#246): per element, linear in freq_id between the nearest knots
// that hold the element, wrapped, the nearest knot's value outside them, none where no knot holds
// it; the carry phasors fall back to the pooled delay in the N^2 frame, then to 1.
BOOST_AUTO_TEST_CASE(inst_phase_table_interpolates_per_element) {
    const double nan = std::nan("");
    gnss::InstPhaseTable tab;
    tab.n = 5;
    tab.fids = {100, 104, 108};
    // elements: all knots; the middle missing; none; across +-pi; only the last knot
    tab.phase = {0.0, 0.5, nan, 3.0,  nan, //
                 0.4, nan, nan, -3.0, nan, //
                 1.2, 1.5, nan, -2.0, 2.0};
    BOOST_CHECK_CLOSE(tab.phase_at(102, 0), 0.2, 1e-9);
    BOOST_CHECK_CLOSE(tab.phase_at(104, 0), 0.4, 1e-9);
    BOOST_CHECK_CLOSE(tab.phase_at(106.5, 0), 0.4 + 0.8 * 2.5 / 4, 1e-9);
    BOOST_CHECK_EQUAL(tab.phase_at(90, 0), 0.0);
    BOOST_CHECK_EQUAL(tab.phase_at(130, 0), 1.2);
    BOOST_CHECK_CLOSE(tab.phase_at(101, 1), 0.5 + 1.0 / 8, 1e-9);
    BOOST_CHECK_CLOSE(tab.phase_at(104, 1), 1.0, 1e-9);
    BOOST_CHECK(std::isnan(tab.phase_at(104, 2)));
    // 3.0 -> -3.0 is +0.283 rad, not -6
    const double d = 2.0 * M_PI - 6.0;
    BOOST_CHECK_CLOSE(tab.phase_at(102, 3), 3.0 + d / 2, 1e-9);
    BOOST_CHECK_LT(std::abs(std::arg(std::polar(1.0, tab.phase_at(104, 3) + 3.0))), 1e-12);
    BOOST_CHECK_EQUAL(tab.phase_at(90, 4), 2.0);
    BOOST_CHECK_EQUAL(tab.phase_at(106, 4), 2.0);
    gnss::InstPhaseTable none;
    BOOST_CHECK(none.empty() && !tab.empty());

    // u(f) conj(u(f_src)): the table's carry where it holds the element, else the pooled delay
    // as the N^2 frame sees it (the conjugate of InstDelay::apply's raw-frame turn), else 1.
    const std::vector<double> tau = {1e-9, 2e-9, -3e-9, 4e-9, 5e-9};
    std::vector<cd> uf(5), us(5), uf0(5), us0(5), ufr(5), usr(5);
    gnss::inst_carry_phasors(tab, tau.data(), true, 5, 106, uf.data());
    gnss::inst_carry_phasors(tab, tau.data(), true, 5, 101, us.data());
    gnss::inst_carry_phasors(tab, nullptr, true, 5, 106, uf0.data());
    gnss::inst_carry_phasors(tab, nullptr, true, 5, 101, us0.data());
    gnss::inst_carry_phasors(tab, tau.data(), false, 5, 106, ufr.data());
    gnss::inst_carry_phasors(tab, tau.data(), false, 5, 101, usr.data());
    const double df = 5 * 195312.5;
    for (int el = 0; el < 5; ++el) {
        const cd c = uf[el] * std::conj(us[el]);
        if (el == 2) {
            BOOST_CHECK_LT(std::abs(c - std::polar(1.0, -2.0 * M_PI * df * tau[2])), 1e-9);
            BOOST_CHECK_LT(
                std::abs(ufr[el] * std::conj(usr[el]) - std::polar(1.0, 2.0 * M_PI * df * tau[2])),
                1e-9);
            BOOST_CHECK_LT(std::abs(uf0[el] - 1.0) + std::abs(us0[el] - 1.0), 1e-15);
        } else {
            const double want = tab.phase_at(106, el) - tab.phase_at(101, el);
            BOOST_CHECK_LT(std::abs(c - std::polar(1.0, want)), 1e-12);
            BOOST_CHECK_LT(std::abs(uf0[el] * std::conj(us0[el]) - c), 1e-12);
            BOOST_CHECK_LT(std::abs(ufr[el] * std::conj(usr[el]) - c), 1e-12);
        }
    }
    // a table of another element count is not used
    gnss::inst_carry_phasors(tab, nullptr, true, 4, 106, uf.data());
    for (int el = 0; el < 4; ++el)
        BOOST_CHECK_EQUAL(uf[el], cd(1.0, 0.0));
}

namespace {
// A synthetic instrument in the N^2 frame, as python/scripts/gnss/test_inst_phase_table.py: a
// delay per element plus a slow ripple, and an offset.
struct SynthInst {
    int n;
    std::vector<double> tau, rip, ph0;
    SynthInst(int n_elem, unsigned seed) : n(n_elem), tau(n_elem), rip(n_elem), ph0(n_elem) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (int i = 0; i < n; ++i) {
            tau[i] = 5e-9 * u(rng);
            rip[i] = M_PI * u(rng);
            ph0[i] = M_PI * u(rng);
        }
    }
    double phase(double fid, int el) const {
        const double f = fid * 195312.5;
        return ph0[el] + 2.0 * M_PI * f * tau[el]
               + 8.0 * M_PI / 180.0 * std::sin(2.0 * M_PI * f / 40e6 + rip[el]);
    }
    // the table: knots every fid_step channels from fid_lo to fid_hi, wrapped as the calibration
    // reads them; every element in every drop-th knot (offset by the element) left out
    gnss::InstPhaseTable table(int fid_lo, int fid_hi, int fid_step, int drop) const {
        gnss::InstPhaseTable t;
        t.n = n;
        for (int fid = fid_lo, k = 0; fid <= fid_hi; fid += fid_step, ++k) {
            t.fids.push_back(fid);
            for (int el = 0; el < n; ++el)
                t.phase.push_back(drop > 0 && (k + el) % drop == 0
                                      ? std::nan("")
                                      : std::arg(std::polar(1.0, phase(fid, el))));
        }
        return t;
    }
};

// Largest element phase error of x against want after aligning their common phase, degrees.
double max_err_deg(const std::vector<cd>& x, const std::vector<cd>& want) {
    cd ip(0.0, 0.0);
    for (size_t i = 0; i < x.size(); ++i)
        ip += std::conj(x[i]) * want[i];
    const cd rot = ip / std::abs(ip);
    double m = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::abs(std::arg(x[i] * rot * std::conj(want[i]))));
    return m * 180.0 / M_PI;
}
} // namespace

// Knots ~1 MHz apart (some missing per element) across 1160-1300 MHz carry an element vector
// measured on one channel to the instrument's own on another, near and far.
BOOST_AUTO_TEST_CASE(inst_phase_table_carries_the_instrument) {
    const int n = 32;
    const SynthInst inst(n, 246);
    const gnss::InstPhaseTable tab = inst.table(5940, 6655, 5, 7);
    std::vector<cd> us(n), uf(n), v(n), want(n), geo(n);
    const int pairs[4][2] = {{6233, 6300}, {6400, 5980}, {5946, 6650}, {6270, 6271}};
    for (const auto& p : pairs) {
        gnss::inst_carry_phasors(tab, nullptr, true, n, p[0], us.data());
        gnss::inst_carry_phasors(tab, nullptr, true, n, p[1], uf.data());
        for (int el = 0; el < n; ++el) {
            geo[el] = std::polar(1.0, inst.phase(p[0], el));
            v[el] = geo[el] * uf[el] * std::conj(us[el]);
            want[el] = std::polar(1.0, inst.phase(p[1], el));
        }
        BOOST_TEST_MESSAGE("carried " << p[0] << " -> " << p[1] << ": max error "
                                      << max_err_deg(v, want) << " deg, uncarried "
                                      << max_err_deg(geo, want) << " deg");
        BOOST_CHECK_LT(max_err_deg(v, want), 1.0);
        if (std::abs(p[1] - p[0]) > 10)
            BOOST_CHECK_GT(max_err_deg(geo, want), 10.0);
    }
}

// The build paths with lane_inst_phase (GnssN2Project::build_fleet and build_lanes): the lane in
// the N^2 frame (after lane_conj) times u(f) conj(u(f_src)). With the table it is the truth at
// the target; with no table it is today's path, the pooled delay in fleet_lane_direction; with a
// table that holds some elements, each element takes its own.
BOOST_AUTO_TEST_CASE(fleet_lane_carried_by_the_table_or_the_pooled_delay) {
    const int n = 24;
    std::vector<double> pos((size_t)n * 3);
    for (int i = 0; i < n; ++i) {
        pos[(size_t)i * 3] = 6.3 * (i % 6) - 15.0;
        pos[(size_t)i * 3 + 1] = 8.5 * (i / 6) - 12.0;
        pos[(size_t)i * 3 + 2] = 0.05 * i;
    }
    const SynthInst inst(n, 11);
    const int fid1 = 6285, fid2 = 6346; // L2C's centre carried 11.9 MHz up
    const double f1 = fid1 * 0.1953125, f2 = fid2 * 0.1953125;
    ElemSteer S(pos, {f1, f2}, 1, 1.0, 120.0);
    const double t0 = 1.7e9;
    S.update(0, 141.0, 37.0, 0.0, 0.01, -0.004, t0);
    std::vector<ElemSteer::cf> tab((size_t)2 * n), sh((size_t)n);
    S.copy_slot(0, tab.data());
    // The raw-frame lane at each channel is conj(D) times the instrument's raw response, the
    // conjugate of its N^2-frame phase.
    auto raw = [&](int ch, std::vector<cd>& q) {
        for (int i = 0; i < n; ++i) {
            const ElemSteer::cf D = tab[(size_t)ch * n + i];
            q[i] = std::conj(cd(D.real(), D.imag()))
                   * std::polar(1.0 / std::sqrt((double)n), -inst.phase(ch ? fid2 : fid1, i));
        }
    };
    gnss::FleetLane L;
    L.f_src_hz = f1 * 1e6;
    BOOST_REQUIRE(S.track(0, L.los, L.los_rate, &L.t_utc));
    L.q.resize(n);
    raw(0, L.q);
    std::vector<cd> want(n);
    raw(1, want);
    for (cd& x : want)
        x = std::conj(x);

    ElemSteer geo(pos, {}, 0, 1.0, 0.0);
    std::vector<ElemSteer::cf> scratch;
    std::vector<cd> out(n), us(n), uf(n);
    auto carried = [&](const gnss::InstPhaseTable& t, const double* tau) {
        // build_fleet: no delay in fleet_lane_direction, conj, then the carry
        BOOST_REQUIRE(
            gnss::fleet_lane_direction(L, geo, f2 * 1e6, t0, nullptr, scratch, out.data()));
        gnss::inst_carry_phasors(t, tau, true, n, fid1, us.data());
        gnss::inst_carry_phasors(t, tau, true, n, fid2, uf.data());
        std::vector<cd> v(n);
        for (int i = 0; i < n; ++i)
            v[i] = std::conj(out[i]) * uf[i] * std::conj(us[i]);
        return v;
    };
    // today's path: the pooled delay in the raw frame, then conj
    auto pooled = [&](const double* tau) {
        BOOST_REQUIRE(gnss::fleet_lane_direction(L, geo, f2 * 1e6, t0, tau, scratch, out.data()));
        std::vector<cd> v(n);
        for (int i = 0; i < n; ++i)
            v[i] = std::conj(out[i]);
        return v;
    };
    // The pooled delay is the raw-frame slope: the N^2 phase falls with f.
    std::vector<double> tau(n);
    for (int i = 0; i < n; ++i)
        tau[i] = -inst.tau[i];

    const gnss::InstPhaseTable full = inst.table(6200, 6400, 5, 0);
    const std::vector<cd> v_tab = carried(full, tau.data());
    const std::vector<cd> v_pool = pooled(tau.data());
    const std::vector<cd> v_geo = pooled(nullptr);
    BOOST_TEST_MESSAGE("carried 11.9 MHz: table "
                       << max_err_deg(v_tab, want) << " deg, pooled delay "
                       << max_err_deg(v_pool, want) << " deg, geometry " << max_err_deg(v_geo, want)
                       << " deg");
    BOOST_CHECK_LT(max_err_deg(v_tab, want), 0.5);
    BOOST_CHECK_GT(max_err_deg(v_pool, want), 2.0);
    BOOST_CHECK_GT(max_err_deg(v_geo, want), 10.0);

    // No table: exactly today's path; neither: geometry only.
    const gnss::InstPhaseTable empty;
    double d = 0.0;
    const std::vector<cd> v_none = carried(empty, tau.data()), v_zero = carried(empty, nullptr);
    for (int i = 0; i < n; ++i)
        d = std::max(d, std::abs(v_none[i] - v_pool[i]) + std::abs(v_zero[i] - v_geo[i]));
    BOOST_CHECK_LT(d, 1e-9);

    // A table holding the even elements only: those take the table, the odd ones the delay.
    gnss::InstPhaseTable half = full;
    for (size_t k = 0; k < half.fids.size(); ++k)
        for (int i = 1; i < n; i += 2)
            half.phase[k * n + i] = std::nan("");
    const std::vector<cd> v_half = carried(half, tau.data());
    d = 0.0;
    for (int i = 0; i < n; ++i)
        d = std::max(d, std::abs(v_half[i] - (i % 2 ? v_pool[i] : v_tab[i])));
    BOOST_CHECK_LT(d, 1e-9);

    // build_lanes: the board's transfer, the assembler's geometric shift with no delays of its
    // own, carried the same way, is the relay's copy at the source's instant.
    double e0[3];
    BOOST_REQUIRE(S.direction(0, t0, e0));
    S.shift(e0, (f2 - f1) * 1e6, sh.data());
    gnss::inst_carry_phasors(full, tau.data(), true, n, fid1, us.data());
    gnss::inst_carry_phasors(full, tau.data(), true, n, fid2, uf.data());
    d = 0.0;
    for (int i = 0; i < n; ++i) {
        const cd board = std::conj(L.q[i] * std::conj(cd(sh[i].real(), sh[i].imag()))) * uf[i]
                         * std::conj(us[i]);
        d = std::max(d, std::abs(board - v_tab[i]));
    }
    BOOST_CHECK_LT(d, 1e-5);
}

namespace {
// A table over 6200-6400 with a few elements no knot holds (they take the delay), pooled delays,
// and the given lane_conj.
gnss::InstCarry synth_carry(int n, bool lane_conj) {
    const SynthInst inst(n, 7);
    gnss::InstCarry c;
    c.tab = inst.table(6200, 6400, 5, 7);
    for (int el = 3; el < n; el += 9)
        for (size_t k = 0; k < c.tab.fids.size(); ++k)
            c.tab.phase[k * n + el] = std::nan("");
    for (int el = 0; el < n; ++el)
        c.delay_s.push_back(-inst.tau[el]);
    c.lane_conj = lane_conj;
    return c;
}

// The N^2-frame lane GnssN2Project reads from a raw-frame entry: conj with lane_conj.
cd to_n2(cd x, bool lane_conj) {
    return lane_conj ? std::conj(x) : x;
}
} // namespace

// The assemblers carry a transfer in the raw frame; read into the N^2 frame it must be what
// GnssN2Project's own carry, u(f) conj(u(f_src)) after lane_conj, gives, for either lane_conj.
BOOST_AUTO_TEST_CASE(inst_carry_raw_is_the_n2_carry) {
    const int n = 32;
    std::mt19937 rng(246);
    std::uniform_real_distribution<double> u(-M_PI, M_PI);
    std::vector<cd> x(n), raw(n), uf(n), us(n);
    for (const bool conj : {true, false}) {
        const gnss::InstCarry c = synth_carry(n, conj);
        const int pairs[4][2] = {{6233, 6300}, {6400, 6210}, {6150, 6450}, {6270, 6271}};
        double d = 0.0, moved = 0.0;
        for (const auto& p : pairs) {
            gnss::inst_carry_raw(c, n, p[0], p[1], raw.data());
            gnss::inst_carry_phasors(c.tab, c.delay_s.data(), conj, n, p[1], uf.data());
            gnss::inst_carry_phasors(c.tab, c.delay_s.data(), conj, n, p[0], us.data());
            for (int el = 0; el < n; ++el) {
                x[el] = std::polar(1.0 / std::sqrt((double)n), u(rng));
                const cd n2 = to_n2(x[el], conj) * uf[el] * std::conj(us[el]);
                d = std::max(d, std::abs(to_n2(x[el] * raw[el], conj) - n2));
                moved = std::max(moved, std::abs(raw[el] - 1.0));
            }
        }
        BOOST_TEST_MESSAGE("lane_conj " << conj << ": raw-frame carry against the N^2 one " << d);
        BOOST_CHECK_LT(d, 1e-9);
        BOOST_CHECK_GT(moved, 0.1);
    }
    // Delays of another element count are not used: those elements are not turned.
    gnss::InstCarry c = synth_carry(n, true);
    c.delay_s.pop_back();
    gnss::inst_carry_raw(c, n, 6233, 6300, raw.data());
    for (int el = 3; el < n; el += 9)
        BOOST_CHECK_EQUAL(raw[el], cd(1.0, 0.0));
}

// The process-wide holder: set, clear, and readers that always see one whole content while it
// is replaced.
BOOST_AUTO_TEST_CASE(inst_carry_board_holds_one_content) {
    gnss::InstCarryBoard& b = gnss::InstCarryBoard::instance();
    BOOST_CHECK(!b.get());
    auto a = std::make_shared<gnss::InstCarry>(synth_carry(16, true));
    b.set(a);
    BOOST_CHECK(b.get() == a);
    b.set(nullptr);
    BOOST_CHECK(!b.get());

    // Content k holds k + 1 knots and delay k on every element.
    std::atomic<bool> stop{false};
    std::atomic<int> bad{0}, seen{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r)
        readers.emplace_back([&] {
            while (!stop.load()) {
                const std::shared_ptr<const gnss::InstCarry> c = b.get();
                if (!c)
                    continue;
                const double k = c->delay_s[0];
                if ((double)c->tab.fids.size() != k + 1 || c->delay_s.back() != k)
                    ++bad;
                ++seen;
            }
        });
    for (int k = 0; k < 2000; ++k) {
        auto c = std::make_shared<gnss::InstCarry>();
        c->tab.n = 4;
        for (int j = 0; j <= k % 50; ++j)
            c->tab.fids.push_back(j);
        c->tab.phase.assign(c->tab.fids.size() * 4, 0.0);
        c->delay_s.assign(4, (double)(k % 50));
        b.set(std::move(c));
    }
    stop = true;
    for (std::thread& t : readers)
        t.join();
    BOOST_TEST_MESSAGE("snapshots read: " << seen.load());
    BOOST_CHECK_EQUAL(bad.load(), 0);
    b.set(nullptr);
}

// A transfer published while the holder has content is carried and marked; GnssN2Project skips
// the carried one (build_lanes), so the lane it reads is the geometric transfer carried once,
// whoever carried it. With no content nothing is touched.
BOOST_AUTO_TEST_CASE(carried_transfer_is_not_carried_twice) {
    const int n = 32, fid_src = 6233, fid = 6300;
    gnss::ProjEntry geo;
    geo.src = 2;
    geo.n = n;
    geo.k = 1;
    geo.xfer = 1;
    geo.fid_src = fid_src;
    for (int el = 0; el < n; ++el)
        geo.q.push_back(std::polar(1.0f, 0.37f * el * el));

    gnss::InstCarryCache cache;
    gnss::ProjEntry e = geo;
    BOOST_CHECK(!cache.refresh());
    cache.carry(e, fid);
    BOOST_CHECK(!e.carried);
    BOOST_CHECK(e.q == geo.q);

    for (const bool conj : {true, false}) {
        auto c = std::make_shared<gnss::InstCarry>(synth_carry(n, conj));
        gnss::InstCarryBoard::instance().set(c);
        BOOST_REQUIRE(cache.refresh());
        e = geo;
        cache.carry(e, fid);
        BOOST_CHECK(e.carried);
        // build_lanes' read: the N^2 frame, carried by the stage unless the entry is.
        std::vector<cd> uf(n), us(n);
        gnss::inst_carry_phasors(c->tab, c->delay_s.data(), conj, n, fid, uf.data());
        gnss::inst_carry_phasors(c->tab, c->delay_s.data(), conj, n, fid_src, us.data());
        double d_once = 0.0, d_twice = 0.0;
        for (int el = 0; el < n; ++el) {
            const cd want = to_n2(cd(geo.q[el]), conj) * uf[el] * std::conj(us[el]);
            const cd got = to_n2(cd(e.q[el]), conj);
            d_once = std::max(d_once, std::abs(got - want));
            d_twice = std::max(d_twice, std::abs(got * uf[el] * std::conj(us[el]) - want));
        }
        BOOST_CHECK_LT(d_once, 1e-6);
        BOOST_CHECK_GT(d_twice, 0.05);
        // The cache is rebuilt when the content changes: the same pair, another table.
        auto c2 = std::make_shared<gnss::InstCarry>(*c);
        for (double& ph : c2->tab.phase)
            ph = -ph;
        gnss::InstCarryBoard::instance().set(c2);
        BOOST_REQUIRE(cache.refresh());
        gnss::ProjEntry e2 = geo;
        cache.carry(e2, fid);
        BOOST_CHECK(e2.q != e.q);
    }
    gnss::InstCarryBoard::instance().set(nullptr);
    BOOST_CHECK(!cache.refresh());
    e = geo;
    cache.carry(e, fid);
    BOOST_CHECK(!e.carried && e.q == geo.q);
}
