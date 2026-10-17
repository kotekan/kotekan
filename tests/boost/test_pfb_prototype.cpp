#define BOOST_TEST_MODULE "test_pfb_prototype"

#include "pfbPrototype.hpp" // for pfb_prototype, pfb_fold, pfb_push, pfb_window_from_string

#include <algorithm>                         // for max, min
#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK_SMALL, BOOST_AUTO_TEST_CASE
#include <cmath>                             // for M_PI, abs, fabs, log10
#include <complex>                           // for complex, polar
#include <stdexcept>                         // for invalid_argument
#include <vector>                            // for vector

using cf = std::complex<float>;

namespace {

constexpr int N = 16; // channels
constexpr int P = 4;  // taps per channel

// Reference N-point forward DFT.
std::vector<cf> dft(const std::vector<cf>& v) {
    std::vector<cf> X(N);
    for (int k = 0; k < N; ++k) {
        std::complex<double> x(0.0, 0.0);
        for (int p = 0; p < N; ++p)
            x += std::complex<double>(v[p]) * std::polar(1.0, -2.0 * M_PI * k * p / N);
        X[k] = cf(x);
    }
    return X;
}

// Channels for a unit complex tone at `f` bins, through the given prototype
// (length N * num_taps): pushes num_taps blocks of the tone, folds, and takes a
// reference DFT, scaled by 1/N so a channel-centre tone reads 1.
std::vector<cf> channels(const std::vector<float>& proto, int num_taps, double f) {
    std::vector<cf> hist(N * num_taps), block(N);
    for (int hop = 0; hop < num_taps; ++hop) {
        for (int i = 0; i < N; ++i)
            block[i] = std::polar(1.0f, static_cast<float>(2.0 * M_PI * f * (hop * N + i) / N));
        pfb_push(hist.data(), block.data(), N, num_taps);
    }
    std::vector<cf> v(N);
    pfb_fold(hist.data(), proto.data(), N, num_taps, v.data());
    std::vector<cf> X = dft(v);
    for (cf& x : X)
        x /= static_cast<float>(N);
    return X;
}

// Largest channel magnitude at least `guard` channels from a tone at `f`.
double far_leakage(const std::vector<float>& proto, int num_taps, double f, double guard) {
    std::vector<cf> X = channels(proto, num_taps, f);
    double m = 0.0;
    for (int k = 0; k < N; ++k) {
        double d = std::fabs(k - f);
        d = std::min(d, N - d); // circular distance
        if (d >= guard)
            m = std::max(m, static_cast<double>(std::abs(X[k])));
    }
    return m;
}

} // namespace

BOOST_AUTO_TEST_CASE(pfb_push_keeps_delay_line_in_time_order) {
    const int n = 4, p = 3; // L = 12
    std::vector<cf> hist(n * p, cf(0.0f, 0.0f));
    // Four hops of blocks 0..3, 4..7, 8..11, 12..15; the line keeps the last three.
    for (int hop = 0; hop < 4; ++hop) {
        std::vector<cf> block(n);
        for (int i = 0; i < n; ++i)
            block[i] = cf(static_cast<float>(hop * n + i), 0.0f);
        pfb_push(hist.data(), block.data(), n, p);
    }
    for (int j = 0; j < n * p; ++j)
        BOOST_CHECK_EQUAL(hist[j].real(), static_cast<float>(n + j));
}

BOOST_AUTO_TEST_CASE(one_flat_tap_is_plain_dft) {
    // With one tap and a flat prototype the fold must hand the DFT the newest
    // block unchanged, so the channels match a plain FFT, including the sign of
    // the frequency axis.
    std::vector<cf> block(N), hist(N, cf(0.0f, 0.0f)), v(N);
    for (int i = 0; i < N; ++i)
        block[i] = cf(std::sin(1.0f + 3.0f * i), std::cos(2.0f * i * i));
    pfb_push(hist.data(), block.data(), N, 1);
    pfb_fold(hist.data(), std::vector<float>(N, 1.0f).data(), N, 1, v.data());
    std::vector<cf> got = dft(v), want = dft(block);
    for (int k = 0; k < N; ++k)
        BOOST_CHECK_SMALL(std::abs(got[k] - want[k]), 1e-4f);
}

BOOST_AUTO_TEST_CASE(off_centre_tone_matches_direct_response) {
    // For a tone x[n] = exp(2 pi i f n / N) over the window n = 0..L-1, channel k
    // is sum_r h[r] exp(2 pi i (f - k) r / N) / N.
    const double f = 4.25;
    auto h = pfb_prototype(N, P, PfbWindow::Hamming);
    std::vector<cf> got = channels(h, P, f);
    for (int k = 0; k < N; ++k) {
        std::complex<double> want(0.0, 0.0);
        for (int r = 0; r < N * P; ++r)
            want += static_cast<double>(h[r]) * std::polar(1.0, 2.0 * M_PI * (f - k) * r / N);
        BOOST_CHECK_SMALL(std::abs(std::complex<double>(got[k]) - want / double(N)), 1e-4);
    }
}

BOOST_AUTO_TEST_CASE(prototype_taps_sum_to_num_chan) {
    // So that a tone at a channel centre has the gain of a plain DFT.
    auto h = pfb_prototype(N, P, PfbWindow::Hamming);
    double sum = 0.0;
    for (float v : h)
        sum += v;
    BOOST_CHECK_CLOSE(sum, (double)N, 1e-3);
}

BOOST_AUTO_TEST_CASE(prototype_is_symmetric_linear_phase) {
    auto h = pfb_prototype(N, P, PfbWindow::Hamming);
    const int L = N * P;
    for (int r = 0; r < L / 2; ++r)
        BOOST_CHECK_CLOSE(h[r], h[L - 1 - r], 1e-3);
}

BOOST_AUTO_TEST_CASE(channel_edge_and_stopband) {
    // With hamming a tone at a channel edge is 6 dB down. For each window, the worst leakage
    // into channels a channel spacing or more from the tone is within 1 dB of its level here.
    BOOST_CHECK_CLOSE(std::abs(channels(pfb_prototype(N, P, PfbWindow::Hamming), P, 4.5)[4]),
                      0.4968, 1.0);
    const struct {
        PfbWindow window;
        double level_db;
    } levels[] = {{PfbWindow::Rectangular, -29.1},
                  {PfbWindow::Hann, -44.4},
                  {PfbWindow::Hamming, -50.2},
                  {PfbWindow::Blackman, -38.1}};
    for (const auto& l : levels) {
        auto h = pfb_prototype(N, P, l.window);
        double worst = 0.0;
        for (int i = 0; i < 64; ++i)
            worst = std::max(worst, far_leakage(h, P, 4.0 + i / 64.0, 1.0));
        BOOST_CHECK_SMALL(20.0 * std::log10(worst) - l.level_db, 1.0);
    }
}

BOOST_AUTO_TEST_CASE(blackman_leaks_less_far_from_channel) {
    const double f = 4.5;
    BOOST_CHECK_LT(far_leakage(pfb_prototype(N, P, PfbWindow::Blackman), P, f, 3.0),
                   far_leakage(pfb_prototype(N, P, PfbWindow::Hamming), P, f, 3.0));
}

BOOST_AUTO_TEST_CASE(window_names) {
    BOOST_CHECK(pfb_window_from_string("rect") == PfbWindow::Rectangular);
    BOOST_CHECK(pfb_window_from_string("hann") == PfbWindow::Hann);
    BOOST_CHECK(pfb_window_from_string("hamming") == PfbWindow::Hamming);
    BOOST_CHECK(pfb_window_from_string("blackman") == PfbWindow::Blackman);
    BOOST_CHECK_THROW(pfb_window_from_string("kaiser"), std::invalid_argument);
}
