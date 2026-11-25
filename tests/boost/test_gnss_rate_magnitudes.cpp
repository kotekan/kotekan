#define BOOST_TEST_MODULE "test_gnss_rate_magnitudes"

#include "gnssCombinerChain.hpp" // for rate_magnitudes

#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for abs, polar
#include <complex> // for complex
#include <random>  // for mt19937, normal_distribution
#include <vector>  // for vector

using cd = std::complex<double>;

// The definition, summed term by term.
static std::vector<double> reference(const std::vector<cd>& x, const std::vector<long long>& m,
                                     size_t nb) {
    std::vector<double> mag(nb);
    for (size_t i = 0; i < nb; ++i) {
        cd acc(0.0, 0.0);
        for (size_t k = 0; k < x.size(); ++k)
            acc += x[k] * std::polar(1.0, -2.0 * M_PI * (double)((i * m[k]) % nb) / (double)nb);
        mag[i] = std::abs(acc);
    }
    return mag;
}

// A deep window as the combiner sees it: a tone in noise on a record grid with dropped records.
static void window(size_t L, int drop_one_in, std::vector<cd>& x, std::vector<long long>& m) {
    std::mt19937 rng(11 + L);
    std::normal_distribution<double> g;
    x.clear();
    m.clear();
    for (long long t = 0; x.size() < L; ++t) {
        if (drop_one_in > 0 && rng() % drop_one_in == 0)
            continue;
        x.push_back(cd(g(rng), g(rng)) + 0.5 * std::polar(1.0, 0.37 * (double)t));
        m.push_back(t);
    }
}

BOOST_AUTO_TEST_CASE(matches_the_direct_sum) {
    std::vector<cd> x;
    std::vector<long long> m;
    std::vector<double> mag;
    // The rate search's sizes (4 (span + 1), capped at 4096), a size that is not a multiple of
    // 4, and a cap below the span, where grid indices wrap.
    struct Case {
        size_t L;
        int drop;
        size_t nb;
    };
    for (const Case& c : {Case{200, 50, 0}, Case{1000, 50, 0}, Case{64, 0, 0}, Case{100, 9, 401},
                          Case{1200, 20, 4096}}) {
        window(c.L, c.drop, x, m);
        const size_t nb = c.nb ? c.nb : (size_t)std::min<long long>(4 * (m.back() + 1), 4096);
        const std::vector<double> want = reference(x, m, nb);
        // twice, so the second call runs on the cached plan
        for (int pass = 0; pass < 2; ++pass) {
            gnss::rate_magnitudes(x, m, nb, mag);
            BOOST_REQUIRE_EQUAL(mag.size(), nb);
            double peak = 0.0, worst = 0.0;
            for (size_t i = 0; i < nb; ++i) {
                peak = std::max(peak, want[i]);
                worst = std::max(worst, std::abs(mag[i] - want[i]));
            }
            BOOST_TEST_INFO("L " << c.L << " nb " << nb << " pass " << pass);
            BOOST_CHECK_LT(worst, 1e-9 * peak);
        }
    }
}
