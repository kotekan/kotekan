#define BOOST_TEST_MODULE "test_gnss_channelized_acquire"

#include "gnssChannelizedAcquire.hpp" // for channelized_acquire
#include "gpsCACode.hpp"              // for generate_ca_code
#include "pfbPrototype.hpp"           // for pfb_prototype

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <complex>
#include <vector>

using cf = std::complex<float>;

namespace {

constexpr double CHIP_RATE = 1.023e6;
constexpr int SP = 4;                          // samples per chip
constexpr double FS = CHIP_RATE * SP;          // 4.092 MHz
constexpr int N = 31;                          // channels (31 | 4092, one period)
constexpr int P = 4;                           // PFB taps
constexpr long CODE_LEN = 1023;
constexpr int NS = CODE_LEN * SP;              // 4092 = one code period
constexpr int M = NS / N;                      // 132 hops

std::vector<int8_t> ca(int prn) {
    auto a = gps::generate_ca_code(prn);
    return std::vector<int8_t>(a.begin(), a.end());
}

// Wideband baseband signal at code phase `cp` chips, Doppler `fd` Hz, amp `amp`.
std::vector<cf> gen(int prn, double cp, double fd, cf amp) {
    auto code = ca(prn);
    std::vector<cf> s(NS);
    for (int n = 0; n < NS; ++n) {
        long idx = (long)std::floor(cp + n * CHIP_RATE / FS) % CODE_LEN;
        if (idx < 0)
            idx += CODE_LEN;
        const double ph = 2.0 * M_PI * fd * n / FS;
        s[n] = amp * (float)code[idx] * cf(std::cos(ph), std::sin(ph));
    }
    return s;
}

// Clean windowed-DFT filterbank: X_c[m] = sum_l proto[l] x[(mN+l) mod NS]
// e^{-i 2pi c l / N}, chronological, channel c <-> +frequency c. Matches the
// convention the acquisition reconstruction assumes.
std::vector<std::vector<cf>> stft(const std::vector<cf>& x, const std::vector<float>& proto) {
    std::vector<std::vector<cf>> ch(N, std::vector<cf>(M));
    std::vector<cf> v(N);
    for (int m = 0; m < M; ++m) {
        for (int p = 0; p < N; ++p) {
            cf acc(0.0f, 0.0f);
            for (int r = 0; r < P; ++r) {
                const int l = r * N + p;
                acc += proto[l] * x[((long)m * N + l) % NS];
            }
            v[p] = acc;
        }
        for (int c = 0; c < N; ++c) {
            cf X(0.0f, 0.0f);
            for (int p = 0; p < N; ++p) {
                const double a = -2.0 * M_PI * c * p / N;
                X += v[p] * cf(std::cos(a), std::sin(a));
            }
            ch[c][m] = X;
        }
    }
    return ch;
}

std::vector<int> energy_covering(const std::vector<std::vector<cf>>& repl) {
    std::vector<double> e(N, 0.0);
    double emax = 0.0;
    for (int c = 0; c < N; ++c) {
        for (const cf& v : repl[c])
            e[c] += std::norm(v);
        emax = std::max(emax, e[c]);
    }
    std::vector<int> cov;
    for (int c = 0; c < N; ++c)
        if (e[c] > 0.05 * emax)
            cov.push_back(c);
    return cov;
}

} // namespace

BOOST_AUTO_TEST_CASE(recovers_code_phase_and_doppler) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 50;   // not hop-aligned: q=1, s=19 -> exercises fine lag
    const double true_dop = 200.0;

    const double cp = true_tau * CHIP_RATE / FS;
    auto data = stft(gen(5, cp, true_dop, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), proto);
    const auto cov = energy_covering(repl0);

    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    auto r = gnss::channelized_acquire(data, repl0, cov, grid, FS, CHIP_RATE, N, CODE_LEN);

    BOOST_CHECK_EQUAL(r.doppler_hz, true_dop);
    BOOST_CHECK_LE(std::abs(r.peak_tau_samples - true_tau), (long)SP); // within one chip
    BOOST_CHECK_GT(r.snr, 20.0);                                        // sharp peak
}

BOOST_AUTO_TEST_CASE(hop_aligned_phase_recovered) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 2 * N; // exactly two hops: s=0
    auto data = stft(gen(12, true_tau * CHIP_RATE / FS, 0.0, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(12, 0.0, 0.0, cf(1.0f, 0.0f)), proto);

    const std::vector<double> grid = {-200, 0, 200};
    auto r = gnss::channelized_acquire(data, repl0, energy_covering(repl0), grid, FS, CHIP_RATE, N,
                                       CODE_LEN);
    BOOST_CHECK_EQUAL(r.doppler_hz, 0.0);
    BOOST_CHECK_LE(std::abs(r.peak_tau_samples - true_tau), (long)SP);
}

BOOST_AUTO_TEST_CASE(wrong_prn_has_low_snr) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    auto data = stft(gen(5, 40.0 * CHIP_RATE / FS, 0.0, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(7, 0.0, 0.0, cf(1.0f, 0.0f)), proto); // different PRN

    const std::vector<double> grid = {-200, 0, 200};
    auto r = gnss::channelized_acquire(data, repl0, energy_covering(repl0), grid, FS, CHIP_RATE, N,
                                       CODE_LEN);
    BOOST_CHECK_LT(r.snr, 12.0); // no coherent peak for the wrong code
}
