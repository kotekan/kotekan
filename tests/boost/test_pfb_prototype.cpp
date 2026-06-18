#define BOOST_TEST_MODULE "test_pfb_prototype"

#include "pfbPrototype.hpp" // for pfb_prototype, pfb_fold, Window

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <complex>
#include <vector>

using dsp::pfb_fold;
using dsp::pfb_prototype;
using dsp::Window;
using cf = std::complex<float>;

namespace {

constexpr int N = 16; // channels
constexpr int P = 4;  // taps per channel

// Channel magnitudes for a unit complex tone at `f` bins, through the given
// prototype (length N*num_taps). Builds the delay line, folds, and takes a
// reference N-point DFT -- no FFTW.
std::vector<double> channel_response(const std::vector<float>& proto, int num_taps, double f) {
    const int L = N * num_taps;
    std::vector<cf> hist(L);
    for (int j = 0; j < L; ++j) {
        const double ph = 2.0 * M_PI * f * j / N; // +freq tone -> lands in channel f
        hist[j] = cf(std::cos(ph), std::sin(ph));
    }
    std::vector<cf> v(N);
    pfb_fold(hist.data(), proto.data(), N, num_taps, v.data());

    std::vector<double> mag(N);
    for (int k = 0; k < N; ++k) {
        cf x(0.0f, 0.0f);
        for (int p = 0; p < N; ++p) {
            const double ph = -2.0 * M_PI * k * p / N;
            x += v[p] * cf(std::cos(ph), std::sin(ph));
        }
        mag[k] = std::abs(x);
    }
    return mag;
}

// The straight-FFT channelizer = a length-N rectangular prototype (P=1).
std::vector<float> straight_fft_prototype() {
    return std::vector<float>(N, 1.0f / N);
}

// Largest channel magnitude at least `guard` bins from the tone (the leakage).
double far_leakage(const std::vector<double>& mag, double f, int guard) {
    double m = 0.0;
    for (int k = 0; k < N; ++k) {
        double d = std::fabs(k - f);
        d = std::min(d, N - d); // circular distance
        if (d >= guard)
            m = std::max(m, mag[k]);
    }
    return m;
}

} // namespace

BOOST_AUTO_TEST_CASE(pfb_push_orders_delay_line_newest_first) {
    const int n = 4, p = 3; // L = 12
    std::vector<cf> hist(n * p, cf(0.0f, 0.0f));
    // Three hops of chronological blocks: 0..3, 4..7, 8..11.
    for (int hop = 0; hop < 3; ++hop) {
        std::vector<cf> block(n);
        for (int i = 0; i < n; ++i)
            block[i] = cf(static_cast<float>(hop * n + i), 0.0f);
        dsp::pfb_push(hist.data(), block.data(), n, p);
    }
    // Newest sample (11) is at hist[0], counting down to oldest kept (0).
    for (int j = 0; j < n * p; ++j)
        BOOST_CHECK_CLOSE(hist[j].real(), static_cast<float>(11 - j), 1e-4);
}

BOOST_AUTO_TEST_CASE(prototype_has_unit_dc_gain) {
    auto h = pfb_prototype(N, P, Window::Hamming);
    double sum = 0.0;
    for (float v : h)
        sum += v;
    BOOST_CHECK_CLOSE(sum, 1.0, 1e-3);
}

BOOST_AUTO_TEST_CASE(prototype_is_symmetric_linear_phase) {
    auto h = pfb_prototype(N, P, Window::Hamming);
    const int L = N * P;
    for (int r = 0; r < L / 2; ++r)
        BOOST_CHECK_CLOSE(h[r], h[L - 1 - r], 1e-3);
}

BOOST_AUTO_TEST_CASE(unit_passband_gain_at_channel_center) {
    auto h = pfb_prototype(N, P, Window::Hamming);
    auto mag = channel_response(h, P, 4.0); // tone at channel-4 center
    BOOST_CHECK_CLOSE(mag[4], 1.0, 1.0); // within 1%
}

BOOST_AUTO_TEST_CASE(dc_tone_isolated_to_channel_zero) {
    auto h = pfb_prototype(N, P, Window::Hamming);
    auto mag = channel_response(h, P, 0.0);
    BOOST_CHECK_CLOSE(mag[0], 1.0, 1.0);
    for (int k = 1; k < N; ++k)
        BOOST_CHECK_SMALL(mag[k], 0.03); // all other channels in the stopband
}

BOOST_AUTO_TEST_CASE(pfb_far_leakage_beats_straight_fft) {
    // Worst case: a tone on a channel boundary (f = 4.5) spills into both
    // neighbours legitimately; the win is how little reaches *far* channels.
    const double f = 4.5;
    auto pfb = channel_response(pfb_prototype(N, P, Window::Hamming), P, f);
    auto fft = channel_response(straight_fft_prototype(), 1, f);

    const double pfb_leak = far_leakage(pfb, f, 2);
    const double fft_leak = far_leakage(fft, f, 2);

    BOOST_CHECK_SMALL(pfb_leak, 0.05);          // deep stopband
    BOOST_CHECK_GT(fft_leak, 0.1);              // sinc sidelobes leak badly
    BOOST_CHECK_LT(pfb_leak * 3.0, fft_leak);   // PFB >> straight FFT
}

BOOST_AUTO_TEST_CASE(stronger_window_gives_deeper_stopband) {
    const double f = 4.5;
    auto ham = channel_response(pfb_prototype(N, P, Window::Hamming), P, f);
    auto blk = channel_response(pfb_prototype(N, P, Window::Blackman), P, f);
    BOOST_CHECK_LT(far_leakage(blk, f, 3), far_leakage(ham, f, 3));
}
