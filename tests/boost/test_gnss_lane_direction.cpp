#define BOOST_TEST_MODULE "test_gnss_lane_direction"
#include "gnssLaneDirection.hpp" // for LaneDirection
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
            v[i] = (s[i] + sigma * cd(g(rng), g(rng)) / std::sqrt(2.0))
                   * cd(D[i].real(), D[i].imag());
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
    L.push(7, 1, v.data(), 0.02);                     // ignored
    L.push(1, -1, v.data(), 0.02);                    // ignored
    BOOST_CHECK(!L.direction(7, 1, b.D.data(), out.data()));
    BOOST_CHECK_EQUAL(L.coherence(5, 0), 0.0);
}
