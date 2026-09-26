#define BOOST_TEST_MODULE "test_frb1IntensityBound"

#include "DataType.hpp"           // for float16_t
#include "frb1IntensityBound.hpp" // for frb1_intensity_bound, frb1_intensity_limit

#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for isnan, cos, sin, NAN
#include <complex> // for complex
#include <random>  // for mt19937, uniform_real_distribution
#include <vector>  // for vector

using kotekan::frb1_intensity_bound;
using kotekan::frb1_intensity_limit;

namespace {

constexpr int P = 2;

// Weights for one frequency, laid out as [P][dishN][dishM][re/im]
std::vector<float16_t> make_weights(const int M, const int N,
                                    const std::vector<std::complex<float>>& W) {
    BOOST_REQUIRE(W.size() == std::size_t(P) * M * N);
    std::vector<float16_t> W16(2 * W.size());
    for (std::size_t i = 0; i < W.size(); ++i) {
        W16.at(2 * i + 0) = float16_t(W.at(i).real());
        W16.at(2 * i + 1) = float16_t(W.at(i).imag());
    }
    return W16;
}

std::vector<float16_t> make_constant_weights(const int M, const int N,
                                             const std::complex<float> W) {
    return make_weights(M, N, std::vector<std::complex<float>>(std::size_t(P) * M * N, W));
}

} // namespace

// Unit weights on the full CHIME grid reach 49·M·N, just below the limit
BOOST_AUTO_TEST_CASE(unit_weights_chime) {
    const int M = 256, N = 4;
    const auto W = make_constant_weights(M, N, 1);
    const double bound = frb1_intensity_bound(W.data(), P, M, N);
    BOOST_CHECK_CLOSE(bound, 49.0 * M * N, 1.0e-10);
    BOOST_CHECK_EQUAL(bound, 50176);
    BOOST_CHECK(bound <= frb1_intensity_limit);
}

// Weights of magnitude 1.2 (the GPU stress test that overflowed) exceed the limit
BOOST_AUTO_TEST_CASE(large_weights_chime) {
    const int M = 256, N = 4;
    const auto W = make_constant_weights(M, N, 1.2f);
    const double bound = frb1_intensity_bound(W.data(), P, M, N);
    BOOST_CHECK(bound > frb1_intensity_limit);
}

// Only the magnitudes matter, not the phases
BOOST_AUTO_TEST_CASE(phases_do_not_matter) {
    const int M = 24, N = 24;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> phase(0, 2 * M_PI);
    std::vector<std::complex<float>> Wc(std::size_t(P) * M * N);
    for (auto& w : Wc)
        w = std::polar(1.0f, phase(rng));
    const auto W = make_weights(M, N, Wc);
    // Float16 rounds each component, so the magnitudes are 1 only to Float16 precision
    BOOST_CHECK_CLOSE(frb1_intensity_bound(W.data(), P, M, N), 49.0 * M * N, 0.1);
}

// The bound depends on the mean magnitude per polarization, so a few large gains are fine
BOOST_AUTO_TEST_CASE(mean_magnitude_matters) {
    const int M = 256, N = 4;
    std::vector<std::complex<float>> Wc(std::size_t(P) * M * N);
    for (std::size_t i = 0; i < Wc.size(); ++i)
        Wc.at(i) = i % 2 == 0 ? 2 : 0;
    const auto W = make_weights(M, N, Wc);
    BOOST_CHECK_EQUAL(frb1_intensity_bound(W.data(), P, M, N), 50176);
}

// Masked (zero) weights reduce the bound
BOOST_AUTO_TEST_CASE(zero_weights) {
    const int M = 8, N = 8;
    const auto W = make_constant_weights(M, N, 0);
    BOOST_CHECK_EQUAL(frb1_intensity_bound(W.data(), P, M, N), 0);
}

// NaN weights (unused frequencies in setFRB1Phase) give a NaN bound, which passes the check
BOOST_AUTO_TEST_CASE(nan_weights) {
    const int M = 8, N = 8;
    const auto W = make_constant_weights(M, N, NAN);
    const double bound = frb1_intensity_bound(W.data(), P, M, N);
    BOOST_CHECK(std::isnan(bound));
    BOOST_CHECK(!(bound > frb1_intensity_limit));
}
