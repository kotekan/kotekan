#define BOOST_TEST_MODULE "test_gnss_channelized_acquire"

#include "gnssChannelizedAcquire.hpp"  // for channelized_accumulate, channelized_peak
#include "gnssChannelizedDespread.hpp" // for channelized_despread (cp refine, as the stage does)
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank (real production replica)
#include "gnssSignal.hpp"              // for signal_by_name, SignalDescriptor
#include "gpsCACode.hpp"               // for generate_ca_code
#include "pfbPrototype.hpp"            // for pfb_prototype

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

using cf = std::complex<float>;

namespace {

constexpr double CHIP_RATE = 1.023e6;
constexpr int SP = 4;                 // samples per chip
constexpr double FS = CHIP_RATE * SP; // 4.092 MHz
constexpr int N = 31;                 // channels (31 | 4092, one period)
constexpr int P = 4;                  // PFB taps
constexpr long CODE_LEN = 1023;
constexpr int NS = CODE_LEN * SP; // 4092 = one code period
constexpr int M = NS / N;         // 132 hops

std::vector<int8_t> ca(int prn) {
    auto a = gps::generate_ca_code(prn);
    return std::vector<int8_t>(a.begin(), a.end());
}

// Reproducible standard-normal sample by Box-Muller on the engine's raw output;
// std::normal_distribution gives different sequences under libc++ and libstdc++.
double gauss(std::mt19937& rng) {
    const double u1 = (rng() + 1.0) / 4294967297.0; // (0,1), excludes 0 for log
    const double u2 = (rng() + 1.0) / 4294967297.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
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

// Windowed-DFT filterbank folded newest first (dsp::pfb_push / pfb_fold), as
// ChannelizedReplicaBank and the CHORD F-engine fold, so its fine-lag axis runs against the coarse
// one as gnss::FINE_LAG_SIGN_PFB says. Channels come out in natural DFT order, or with DC moved
// to the centre (fftshift, the fftwEngine convention). The delay line is warmed from the periodic
// tail so the first hop is circular, matching the acquire's circular correlation.
std::vector<std::vector<cf>> analyze(const std::vector<cf>& x, int Nc,
                                     const std::vector<float>& proto, bool fftshift) {
    const int Ns = (int)x.size();
    const int Mh = Ns / Nc;
    std::vector<cf> hist((size_t)Nc * P, cf(0.0f, 0.0f)), block(Nc), v(Nc), spec(Nc);
    std::vector<std::vector<cf>> ch(Nc, std::vector<cf>(Mh));
    auto do_hop = [&](int m, bool store) {
        for (int i = 0; i < Nc; ++i)
            block[i] = x[(((long)m * Nc + i) % Ns + Ns) % Ns];
        dsp::pfb_push(hist.data(), block.data(), Nc, P);
        if (!store)
            return;
        dsp::pfb_fold(hist.data(), proto.data(), Nc, P, v.data());
        for (int k = 0; k < Nc; ++k) {
            cf X(0.0f, 0.0f);
            for (int p = 0; p < Nc; ++p) {
                const double a = -2.0 * M_PI * k * p / Nc;
                X += v[p] * cf(std::cos(a), std::sin(a));
            }
            spec[k] = X;
        }
        for (int j = 0; j < Nc; ++j)
            ch[j][m] = fftshift ? spec[(j + Nc / 2) % Nc] : spec[j]; // fftshift: DC at j = Nc/2
    };
    for (int w = 0; w < P - 1; ++w)
        do_hop(Mh - (P - 1) + w, false); // warm history from the tail
    for (int m = 0; m < Mh; ++m)
        do_hop(m, true);
    return ch;
}

// The N-channel bank over one code period (M hops), in natural order.
std::vector<std::vector<cf>> stft(const std::vector<cf>& x, const std::vector<float>& proto) {
    return analyze(x, N, proto, false);
}

std::vector<int> energy_covering_n(const std::vector<std::vector<cf>>& repl, int Nc) {
    std::vector<double> e(Nc, 0.0);
    double emax = 0.0;
    for (int c = 0; c < Nc; ++c) {
        for (const cf& v : repl[c])
            e[c] += std::norm(v);
        emax = std::max(emax, e[c]);
    }
    std::vector<int> cov;
    for (int c = 0; c < Nc; ++c)
        if (e[c] > 0.05 * emax)
            cov.push_back(c);
    return cov;
}

// Single-window acquire: one channelized_accumulate into a fresh surface, then channelized_peak.
gnss::AcquisitionResult
channelized_acquire(const std::vector<std::vector<cf>>& data_ch,
                    const std::vector<std::vector<cf>>& repl0_ch, const std::vector<int>& covering,
                    const std::vector<double>& doppler_grid, double sample_rate, double chip_rate,
                    int num_chan, long code_length, const std::vector<int>& chan_freq = {},
                    int samples_per_hop = 0) {
    std::vector<double> surf;
    gnss::AcquireWorkspace ws;
    const auto dims =
        gnss::channelized_accumulate(data_ch, repl0_ch, covering, doppler_grid, sample_rate,
                                     num_chan, surf, ws, chan_freq, samples_per_hop);
    return gnss::channelized_peak(surf, dims, doppler_grid, sample_rate, chip_rate, code_length);
}

} // namespace

BOOST_AUTO_TEST_CASE(recovers_with_fftwengine_convention) {
    // Even Nc dividing one code period (4092 = 2^2*3*11*31): Nc=12, DC at bin 6.
    constexpr int Nc = 12;
    const auto proto = dsp::pfb_prototype(Nc, P, dsp::Window::Hamming);
    const long true_tau = 20; // q=1, s=8 -> exercises fine lag
    const double true_dop = 200.0;

    auto data =
        analyze(gen(5, true_tau * CHIP_RATE / FS, true_dop, cf(1.0f, 0.0f)), Nc, proto, true);
    auto repl0 = analyze(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), Nc, proto, true);
    const auto cov = energy_covering_n(repl0, Nc);

    // fftshifted bank: DC at channel Nc/2, so chan_freq = j - Nc/2.
    std::vector<int> chan_freq;
    for (int c : cov)
        chan_freq.push_back(c - Nc / 2);

    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    auto r = channelized_acquire(data, repl0, cov, grid, FS, CHIP_RATE, Nc, CODE_LEN, chan_freq);
    BOOST_CHECK_LT(std::abs(r.doppler_hz - true_dop),
                   100.0); // sub-grid refine: within half a 200 Hz cell
    BOOST_CHECK_LE(std::abs(r.peak_tau_samples - true_tau), (long)SP);
    BOOST_CHECK_GT(r.snr, 15.0);
}

BOOST_AUTO_TEST_CASE(recovers_code_phase_and_doppler) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 50; // not hop-aligned: q=1, s=19 -> exercises fine lag
    const double true_dop = 200.0;

    const double cp = true_tau * CHIP_RATE / FS;
    auto data = stft(gen(5, cp, true_dop, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), proto);
    const auto cov = energy_covering(repl0);

    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    auto r = channelized_acquire(data, repl0, cov, grid, FS, CHIP_RATE, N, CODE_LEN);

    BOOST_CHECK_LT(std::abs(r.doppler_hz - true_dop),
                   100.0); // sub-grid refine: within half a 200 Hz cell
    BOOST_CHECK_LE(std::abs(r.peak_tau_samples - true_tau), (long)SP); // within one chip
    BOOST_CHECK_GT(r.snr, 20.0);                                       // sharp peak
}

BOOST_AUTO_TEST_CASE(hop_aligned_phase_recovered) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 2 * N; // exactly two hops: s=0
    auto data = stft(gen(12, true_tau * CHIP_RATE / FS, 0.0, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(12, 0.0, 0.0, cf(1.0f, 0.0f)), proto);

    const std::vector<double> grid = {-200, 0, 200};
    auto r =
        channelized_acquire(data, repl0, energy_covering(repl0), grid, FS, CHIP_RATE, N, CODE_LEN);
    BOOST_CHECK_LT(std::abs(r.doppler_hz), 100.0); // sub-grid refine: ~0 within half a cell
    BOOST_CHECK_LE(std::abs(r.peak_tau_samples - true_tau), (long)SP);
}

BOOST_AUTO_TEST_CASE(wrong_prn_has_low_snr) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    auto data = stft(gen(5, 40.0 * CHIP_RATE / FS, 0.0, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(7, 0.0, 0.0, cf(1.0f, 0.0f)), proto); // different PRN

    const std::vector<double> grid = {-200, 0, 200};
    auto r =
        channelized_acquire(data, repl0, energy_covering(repl0), grid, FS, CHIP_RATE, N, CODE_LEN);
    BOOST_CHECK_LT(r.snr, 12.0); // no coherent peak for the wrong code
}

// channelized_acquire (above) is channelized_accumulate over one window plus channelized_peak;
// calling the two directly with a caller-owned AcquireWorkspace must reproduce it exactly and
// report the expected surface dimensions.
BOOST_AUTO_TEST_CASE(accumulate_peak_matches_wrapper) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 50;
    const double true_dop = 200.0;
    auto data = stft(gen(5, true_tau * CHIP_RATE / FS, true_dop, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), proto);
    const auto cov = energy_covering(repl0);
    const std::vector<double> grid = {-400, -200, 0, 200, 400};

    auto rw = channelized_acquire(data, repl0, cov, grid, FS, CHIP_RATE, N, CODE_LEN);

    gnss::AcquireWorkspace ws;
    std::vector<double> surf;
    const auto dims = gnss::channelized_accumulate(data, repl0, cov, grid, FS, N, surf, ws);
    auto rs = gnss::channelized_peak(surf, dims, grid, FS, CHIP_RATE, CODE_LEN);

    BOOST_CHECK_EQUAL(dims.n_dop, (int)grid.size());
    BOOST_CHECK_EQUAL(dims.Mp, M);  // repl0 hop-period
    BOOST_CHECK_EQUAL(dims.sph, N); // complex bank: hop == channel count (samples_per_hop=0)
    BOOST_CHECK_EQUAL(rs.peak_tau_samples, rw.peak_tau_samples);
    BOOST_CHECK_EQUAL(rs.doppler_hz, rw.doppler_hz);
    BOOST_CHECK_CLOSE(rs.snr, rw.snr, 1e-2);
}

// channelized_accumulate must read only the covering rows of repl0: GnssChannelizedSearch builds
// those rows alone and leaves the rest empty (7 rows of 8192 on a CHORD bank). Checked with a
// strong wrong-code replica in the non-covering rows and with empty ones; both must give the
// full replica's surface bit for bit.
BOOST_AUTO_TEST_CASE(accumulate_reads_only_covering_rows) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    auto data = stft(gen(5, 50 * CHIP_RATE / FS, 200.0, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), proto);
    const auto cov = energy_covering(repl0);
    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    BOOST_REQUIRE_LT(cov.size(), (size_t)N); // otherwise the test proves nothing

    gnss::AcquireWorkspace ws;
    std::vector<double> ref_surf;
    const auto ref_dims = gnss::channelized_accumulate(data, repl0, cov, grid, FS, N, ref_surf, ws);

    std::vector<bool> is_cov(N, false);
    for (int c : cov)
        is_cov[(size_t)c] = true;

    // (a) poison: a different PRN's replica in every non-covering row.
    auto poisoned = repl0;
    auto wrong = stft(gen(17, 300.0, -1000.0, cf(50.0f, 0.0f)), proto);
    for (int c = 0; c < N; ++c)
        if (!is_cov[(size_t)c])
            poisoned[(size_t)c] = wrong[(size_t)c];
    std::vector<double> p_surf;
    gnss::channelized_accumulate(data, poisoned, cov, grid, FS, N, p_surf, ws);

    // (b) banded: non-covering rows empty, as the search builds them.
    auto banded = std::vector<std::vector<cf>>((size_t)N);
    for (int c : cov)
        banded[(size_t)c] = repl0[(size_t)c];
    std::vector<double> b_surf;
    const auto b_dims = gnss::channelized_accumulate(data, banded, cov, grid, FS, N, b_surf, ws);

    BOOST_CHECK_EQUAL(b_dims.Mp, ref_dims.Mp);
    BOOST_REQUIRE_EQUAL(p_surf.size(), ref_surf.size());
    BOOST_REQUIRE_EQUAL(b_surf.size(), ref_surf.size());
    for (size_t i = 0; i < ref_surf.size(); ++i) {
        BOOST_REQUIRE_EQUAL(p_surf[i], ref_surf[i]); // bit-identical: poison never read
        BOOST_REQUIRE_EQUAL(b_surf[i], ref_surf[i]); // bit-identical: banded == full
    }
}

// With combed covering channels the fine-lag axis is stored for one period only, and that must
// be exact: the stored period must equal the full-width surface computed from the definition,
// and channelized_peak must give the same peak cell, Doppler and SNR on both. On CHORD's
// stride-16 comb this saves 16x.
BOOST_AUTO_TEST_CASE(fine_lag_period_is_exact) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    auto data = stft(gen(5, 50 * CHIP_RATE / FS, 200.0, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), proto);
    const std::vector<double> grid = {-400, -200, 0, 200, 400};

    // A combed covering set: stride 3 with sph = 30, so gcd(3, 30) = 3 and the stored period is
    // 10. The stft bank's own sph, N = 31, is prime and would store the full width.
    const std::vector<int> cov = {0, 3, 6, 9};
    const int sph = 30;
    std::vector<std::vector<std::vector<cf>>> Pall(cov.size());
    for (size_t ci = 0; ci < cov.size(); ++ci) {
        gnss::AcquireWorkspace w;
        Pall[ci] = gnss::channel_correlate(data[(size_t)cov[ci]], repl0[(size_t)cov[ci]], grid, FS,
                                           sph, w);
    }

    std::vector<double> surf;
    const auto dims = gnss::aggregate_accumulate(Pall, cov, sph, surf);
    BOOST_REQUIRE_EQUAL(dims.sph, sph);
    BOOST_REQUIRE_EQUAL(dims.s_stored, 10); // gcd(3, 30) = 3 -> period 10
    BOOST_REQUIRE_EQUAL((long)surf.size(), (long)dims.n_dop * dims.Mp * 10);

    // Compute the full-width surface from the definition and check the stored one is exactly one
    // period of it.
    const int nd = dims.n_dop, Mp = dims.Mp;
    for (int d = 0; d < nd; ++d)
        for (int q = 0; q < Mp; ++q)
            for (int s = 0; s < sph; ++s) {
                std::complex<double> D(0.0, 0.0);
                for (size_t ci = 0; ci < cov.size(); ++ci) {
                    const double a = 2.0 * M_PI * cov[ci] * s / sph;
                    D += std::complex<double>(Pall[ci][d][q].real(), Pall[ci][d][q].imag())
                         * std::complex<double>(std::cos(a), std::sin(a));
                }
                const double want = std::norm(D);
                const double got = surf[((long)d * Mp + q) * 10 + (s % 10)];
                // 1e-5 relative: production builds the ramp and D in float32 so the s-loop
                // vectorizes, ~2e-6 from this double reference. A wrong period is O(1).
                BOOST_REQUIRE_SMALL(got - want, 1e-5 * std::max(1.0, want));
            }

    // channelized_peak must pick the same cell on the stored surface as on the full-width one
    // tiled from it. SNR is a surface mean, so it would shift if the copies were not uniform.
    std::vector<double> wide((size_t)nd * Mp * sph);
    for (int d = 0; d < nd; ++d)
        for (int q = 0; q < Mp; ++q)
            for (int s = 0; s < sph; ++s)
                wide[((size_t)d * Mp + q) * sph + s] = surf[((long)d * Mp + q) * 10 + (s % 10)];

    const gnss::AcquisitionSurface wide_dims{nd, Mp, sph, sph};
    const auto r_red = gnss::channelized_peak(surf, dims, grid, FS, CHIP_RATE, CODE_LEN);
    const auto r_wide = gnss::channelized_peak(wide, wide_dims, grid, FS, CHIP_RATE, CODE_LEN);
    // The same cell gives the same delay up to the fold: each surface folds the fine index to a
    // signed half of its own period (10 stored, 30 wide), so a peak column in the upper half of
    // the stored period reads exactly one stored period away. The comb cannot tell those apart.
    const long dtau = r_red.peak_tau_samples - r_wide.peak_tau_samples;
    BOOST_CHECK_EQUAL(dtau % 10, 0L);
    BOOST_CHECK_LE(std::abs(dtau), 10L);
    BOOST_CHECK_EQUAL(r_red.doppler_hz, r_wide.doppler_hz);
    BOOST_CHECK_CLOSE(r_red.snr, r_wide.snr, 1e-9);
}

// Re-accumulating the same noiseless window K times scales the |D|^2 surface (and
// its peak) by K, leaving the peak/mean ratio and the recovered lag/Doppler
// unchanged -- the basic invariant the incoherent accumulator must satisfy.
BOOST_AUTO_TEST_CASE(accumulation_scales_surface_linearly) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 50;
    const double true_dop = 200.0;
    auto data = stft(gen(5, true_tau * CHIP_RATE / FS, true_dop, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), proto);
    const auto cov = energy_covering(repl0);
    const std::vector<double> grid = {-400, -200, 0, 200, 400};

    gnss::AcquireWorkspace ws;
    std::vector<double> surf1, surfK;
    const auto dims1 = gnss::channelized_accumulate(data, repl0, cov, grid, FS, N, surf1, ws);
    constexpr int K = 5;
    gnss::AcquisitionSurface dimsK{};
    for (int k = 0; k < K; ++k)
        dimsK = gnss::channelized_accumulate(data, repl0, cov, grid, FS, N, surfK, ws);

    auto r1 = gnss::channelized_peak(surf1, dims1, grid, FS, CHIP_RATE, CODE_LEN);
    auto rK = gnss::channelized_peak(surfK, dimsK, grid, FS, CHIP_RATE, CODE_LEN);

    BOOST_CHECK_EQUAL(rK.peak_tau_samples, r1.peak_tau_samples);
    BOOST_CHECK_EQUAL(rK.doppler_hz, r1.doppler_hz);
    BOOST_CHECK_CLOSE(rK.peak, K * r1.peak, 1e-2); // surface scales by K
    BOOST_CHECK_CLOSE(rK.snr, r1.snr, 1e-2);       // ratio invariant
}

// A signal too weak to acquire in one window (a noise spike wins the surface) is
// recovered once enough independent-noise windows are integrated incoherently --
// the sensitivity mechanism the accumulation exists for.
BOOST_AUTO_TEST_CASE(accumulation_recovers_weak_signal_under_noise) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 50;
    const double true_dop = 200.0;
    const double cp = true_tau * CHIP_RATE / FS;
    constexpr float AMP = 0.30f;  // per-sample signal amplitude
    constexpr float SIGMA = 6.0f; // per-I/Q noise sigma (signal well below noise)
    constexpr int K = 32;

    auto repl0 = stft(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), proto);
    const auto cov = energy_covering(repl0);
    const std::vector<double> grid = {-400, -200, 0, 200, 400};

    std::mt19937 rng(2024u);

    gnss::AcquireWorkspace ws;
    std::vector<double> surf;
    gnss::AcquisitionSurface dims{};
    long tau1 = 0;
    for (int k = 0; k < K; ++k) {
        auto wb = gen(5, cp, true_dop, cf(AMP, 0.0f));
        for (auto& v : wb)
            v += cf((float)(SIGMA * gauss(rng)), (float)(SIGMA * gauss(rng)));
        dims = gnss::channelized_accumulate(stft(wb, proto), repl0, cov, grid, FS, N, surf, ws);
        if (k == 0) {
            std::vector<double> s1(surf); // first-window-only surface
            tau1 = gnss::channelized_peak(s1, dims, grid, FS, CHIP_RATE, CODE_LEN).peak_tau_samples;
        }
    }
    auto rK = gnss::channelized_peak(surf, dims, grid, FS, CHIP_RATE, CODE_LEN);

    BOOST_CHECK_GT(std::abs(tau1 - true_tau), (long)SP); // single window: misses
    BOOST_CHECK_LT(std::abs(rK.doppler_hz - true_dop),
                   100.0); // K windows: Doppler right (sub-grid refine)
    BOOST_CHECK_LE(std::abs(rK.peak_tau_samples - true_tau), (long)SP); // ...and code phase
}

// --------------------------------------------------------------------------------------------
// L2C through the real ChannelizedReplicaBank (time-multiplexed CM/CL) and a single-window acquire,
// at a 5 MSPS real front end with N = 12 and f_off = Fs/4.
namespace {
constexpr double FS_L2C = 5.0e6;    // 5 MSPS real
constexpr double FOFF_L2C = 1.25e6; // IF = Fs/4
constexpr int N_L2C = 12;
constexpr int TAPS_L2C = 4;
} // namespace

// A known L2C code phase and Doppler, generated and matched through the replica bank over one
// replica period (an integer number of code periods), is recovered with a sharp peak.
BOOST_AUTO_TEST_CASE(l2c_single_window_acquire_recovers) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L2C_CM");
    BOOST_REQUIRE(sig != nullptr);
    gnss::ChannelizedReplicaBank bank(*sig, FS_L2C, FOFF_L2C, N_L2C, TAPS_L2C, dsp::Window::Hamming,
                                      {1});
    const int fft_len = 2 * N_L2C; // 24 real samples / hop (r2c)
    const int Mp =
        bank.repl_period_hops(); // 12500 hops = 3 code periods (24 does not divide 100000)
    const long long anchor = (long long)Mp * fft_len; // warm-up reads the periodic code

    // repl0 (code 0, Doppler 0) over one replica period; covering = channels carrying energy.
    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp); // [N][Mp], natural-order r2c
    const auto cov = energy_covering_n(repl0, N_L2C);
    BOOST_REQUIRE(!cov.empty());

    const double true_cp = 137.0;  // CM chips (sub-chip / sub-hop lag exercised)
    const double true_dop = 200.0; // Hz
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);

    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    // As the stage does: samples_per_hop = fft_len (r2c), chan_freq = covering (natural order),
    // code phase in component CM chips (sig->chip_rate_hz, code_length).
    auto r = channelized_acquire(data, repl0, cov, grid, FS_L2C, sig->chip_rate_hz, N_L2C,
                                 sig->code_length, cov, fft_len);

    BOOST_TEST_MESSAGE("L2C single-window: snr=" << r.snr << " dop=" << r.doppler_hz
                                                 << " cp=" << r.code_phase_chips << " (true "
                                                 << true_cp << ")");
    BOOST_CHECK_LT(std::abs(std::abs(r.doppler_hz) - true_dop),
                   100.0);       // grid value (sign per bank convention)
    BOOST_CHECK_GT(r.snr, 15.0); // sharp coherent peak
    double cp_err = std::fabs(r.code_phase_chips - true_cp);
    cp_err = std::min(cp_err, (double)sig->code_length - cp_err);
    BOOST_CHECK_LT(cp_err, 2.0); // within ~1 CM chip
}

// The incoherent search sums contiguous coherent windows, so a window must span an integer
// number of code periods. At N = 12, Fs = 5 MHz one L2C CM period is 100000 samples
// = 4166.67 hops: a 4166-hop window advances the code phase by 99984 mod 100000 = -16 samples
// per window, so the |D|^2 peak walks and the sum smears it away. The 12500-hop replica period
// (3 code periods) keeps the peak stationary.
BOOST_AUTO_TEST_CASE(l2c_incoherent_window_must_span_integer_code_periods) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L2C_CM");
    BOOST_REQUIRE(sig != nullptr);
    gnss::ChannelizedReplicaBank bank(*sig, FS_L2C, FOFF_L2C, N_L2C, TAPS_L2C, dsp::Window::Hamming,
                                      {1});
    const int fft_len = 2 * N_L2C;
    const int Mp = bank.repl_period_hops(); // 12500 hops = 3 code periods
    const long long anchor = (long long)Mp * fft_len;

    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp);
    const auto cov = energy_covering_n(repl0, N_L2C);
    BOOST_REQUIRE(!cov.empty());

    const double true_cp = 137.0, true_dop = 200.0;
    const int nwin_int = 4;              // 4 integer-period windows
    const int snap_hops = nwin_int * Mp; // 50000 hops = 12 code periods of continuous signal
    auto snap = bank.channels(0, anchor, true_cp, true_dop, snap_hops);
    const std::vector<double> grid = {-400, -200, 0, 200, 400};

    // Slice contiguous windows of `hpr` hops out of the snapshot and integrate |D|^2, as
    // GnssChannelizedSearch::search_snapshot does. Returns the surface peak/mean SNR.
    auto run = [&](int hpr) {
        gnss::AcquireWorkspace ws;
        std::vector<double> surf;
        gnss::AcquisitionSurface dims{};
        const int nwin = snap_hops / hpr;
        std::vector<std::vector<cf>> dch(N_L2C, std::vector<cf>(hpr));
        for (int w = 0; w < nwin; ++w) {
            for (int c = 0; c < N_L2C; ++c)
                for (int m = 0; m < hpr; ++m)
                    dch[c][m] = snap[c][(size_t)w * hpr + m];
            dims = gnss::channelized_accumulate(dch, repl0, cov, grid, FS_L2C, N_L2C, surf, ws, cov,
                                                fft_len);
        }
        return gnss::channelized_peak(surf, dims, grid, FS_L2C, sig->chip_rate_hz, sig->code_length)
            .snr;
    };

    const double snr_int = run(Mp);    // 12500 hops = 3 code periods (integer) -> stationary peak
    const double snr_live = run(4166); // ~1 period, not an integer number of periods -> smears

    BOOST_TEST_MESSAGE("L2C incoherent snr: integer-period(12500)=" << snr_int
                                                                    << "  live(4166)=" << snr_live);
    BOOST_CHECK_GT(snr_int, 15.0);           // legal window detects
    BOOST_CHECK_LT(snr_live, snr_int * 0.5); // non-integer window smears the peak away
}

// L5 Q5 at a 20 MSPS real front end (N = 10, f_off = Fs/4). It passes the central ~10 MHz of
// L5's ~20 MHz main lobe; the bank band-limits the replica the same way, so the despread stays
// matched. fft_len = 20 divides the 20000-sample code period, so the search window is exactly
// one 1 ms period, within one NH20 chip.
BOOST_AUTO_TEST_CASE(l5_single_window_acquire_recovers) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    BOOST_REQUIRE(sig != nullptr);
    constexpr double FS_L5 = 20.0e6;  // 20 MSPS real
    constexpr double FOFF_L5 = 5.0e6; // = Fs/4
    constexpr int N_L5 = 10;
    gnss::ChannelizedReplicaBank bank(*sig, FS_L5, FOFF_L5, N_L5, 4, dsp::Window::Hamming, {1});
    const int fft_len = 2 * N_L5;           // 20
    const int Mp = bank.repl_period_hops(); // 20000 / gcd(20,20000) = 1000 hops = 1 ms (1 period)
    const long long anchor = (long long)Mp * fft_len;
    BOOST_REQUIRE_EQUAL(Mp, 1000); // one code period is an integer number of hops at N=10

    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp);
    const auto cov = energy_covering_n(repl0, N_L5);
    BOOST_REQUIRE(!cov.empty());

    const double true_cp = 137.0; // L5 chips (~1.96 samples/chip -> sub-hop lag exercised)
    const double true_dop = 200.0;
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);

    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    auto r = channelized_acquire(data, repl0, cov, grid, FS_L5, sig->chip_rate_hz, N_L5,
                                 sig->code_length, cov, fft_len);

    BOOST_TEST_MESSAGE("L5 acquire: snr=" << r.snr << " dop=" << r.doppler_hz
                                          << " cp=" << r.code_phase_chips << " (true " << true_cp
                                          << ")  cov=" << cov.size() << " chans");
    BOOST_CHECK_LT(std::abs(std::abs(r.doppler_hz) - true_dop),
                   100.0);       // detects with the right Doppler
    BOOST_CHECK_GT(r.snr, 15.0); // strong peak
    // The acquire places the coarse lag to ~1 hop; with only ~half of L5's main lobe the
    // correlation peak is ~2x broader.
    const long true_tau = (long)std::llround(true_cp * FS_L5 / sig->chip_rate_hz);
    BOOST_CHECK_LE(std::abs(r.peak_tau_samples - true_tau), (long)(2 * fft_len));

    // The stage refines cp after the acquire with an exact-despread scan of +-1 hop
    // (refine_peak); the same scan here must land within a chip.
    const double cps = sig->chip_rate_hz / FS_L5; // chips per sample
    const double dop = -r.doppler_hz;             // r2c sign flip, as the stage applies
    double best_cp = r.code_phase_chips, best_pw = -1.0;
    for (int off = -fft_len; off <= fft_len; ++off) {
        const double cp = r.code_phase_chips + off * cps;
        const auto repl = bank.channels(0, anchor, cp, dop, Mp);
        std::vector<std::vector<cf>> d, rr;
        for (int c : cov) {
            d.push_back(data[c]);
            rr.push_back(repl[c]);
        }
        const double pw = std::norm(gnss::channelized_despread(d, rr).amplitude);
        if (pw > best_pw) {
            best_pw = pw;
            best_cp = cp;
        }
    }
    double cp_err = std::fabs(best_cp - true_cp);
    cp_err = std::min(cp_err, (double)sig->code_length - cp_err);
    BOOST_TEST_MESSAGE("L5 refined cp=" << best_cp << " (true " << true_cp << "), err " << cp_err
                                        << " chips");
    BOOST_CHECK_LT(cp_err, 1.0); // sub-chip after the refine
}

// Doppler-hinted search: the broker sends the physical predicted Doppler, and
// GnssChannelizedSearch scans a narrow grid around -hint because the r2c bank reports the
// conjugate sign. A narrow grid around -true_dop must still detect and one around +true_dop must
// miss. The Doppler is large so the wrong-sign grid, 2*dop away, misses cleanly; at small Doppler
// the 1 ms coherent resolution would hide a sign error.
BOOST_AUTO_TEST_CASE(narrowed_doppler_grid_recovers_and_sign_is_right) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    BOOST_REQUIRE(sig != nullptr);
    constexpr double FS_L5 = 20.0e6, FOFF_L5 = 5.0e6;
    constexpr int N_L5 = 10;
    gnss::ChannelizedReplicaBank bank(*sig, FS_L5, FOFF_L5, N_L5, 4, dsp::Window::Hamming, {1});
    const int fft_len = 2 * N_L5;
    const int Mp = bank.repl_period_hops();
    const long long anchor = (long long)Mp * fft_len;
    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp);
    const auto cov = energy_covering_n(repl0, N_L5);

    const double true_cp = 137.0, true_dop = 3000.0; // physical Doppler, as the broker predicts
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);

    const double hint = true_dop, margin = 100.0, step = 50.0;
    auto grid_around = [&](double centre) {
        std::vector<double> g;
        for (double f = centre - margin; f <= centre + margin + 1e-6; f += step)
            g.push_back(f);
        return g;
    };
    // Centred at -hint, as the search does: detects.
    const auto narrow = grid_around(-hint);
    auto r = channelized_acquire(data, repl0, cov, narrow, FS_L5, sig->chip_rate_hz, N_L5,
                                 sig->code_length, cov, fft_len);
    BOOST_TEST_MESSAGE("narrow(-hint): snr=" << r.snr << " dop=" << r.doppler_hz);
    BOOST_CHECK_LT(std::abs(r.doppler_hz + true_dop),
                   25.0);        // ~-physical within half a 50 Hz cell (r2c + sub-grid refine)
    BOOST_CHECK_GT(r.snr, 15.0); // narrowing kept the detection

    // Wrong sign: centre at +hint (2*dop from the true peak) -> a clean miss.
    const auto wrong = grid_around(hint);
    auto rw = channelized_acquire(data, repl0, cov, wrong, FS_L5, sig->chip_rate_hz, N_L5,
                                  sig->code_length, cov, fft_len);
    BOOST_TEST_MESSAGE("narrow(+hint, wrong): snr=" << rw.snr);
    BOOST_CHECK_LT(rw.snr, r.snr * 0.3); // wrong-sign window decorrelates -> no peak
}

// The sub-grid Doppler refine must beat the grid cell on an off-grid signal.
//
// The grid step must be the transform bin. peak_from_reduction inverts r = |g(1-d)|/|g(d)| with
// g(x) = sin(pi*x*u)/x and x in bins, so the +-1 grid neighbour must be the +-1 bin; on a sub-bin
// grid the ratio is ~1 whatever the offset and the estimate saturates at half a step.
// GnssChannelizedSearch warns on such a grid and the CUDA acquire refuses it; production uses a
// 62.5 Hz step on a 62.5 Hz bin. Here bin = fs / (Mp * fft_len) = 20e6 / (1000 * 20) = 1000 Hz.
// Truth is the analytic Doppler; the channelization bias, ~0.01 bin, is inside the tolerance.
BOOST_AUTO_TEST_CASE(doppler_refine_beats_grid) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    constexpr double FS_L5 = 20.0e6, FOFF_L5 = 5.0e6;
    constexpr int N_L5 = 10;
    gnss::ChannelizedReplicaBank bank(*sig, FS_L5, FOFF_L5, N_L5, 4, dsp::Window::Hamming, {1});
    const int fft_len = 2 * N_L5;
    const int Mp = bank.repl_period_hops();
    const long long anchor = (long long)Mp * fft_len;
    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp);
    const auto cov = energy_covering_n(repl0, N_L5);

    // Truth 0.3 bin off a grid point: far enough that a saturating or inert refine fails, close
    // enough to stay in the estimator's interpolating regime.
    const double bin_hz = FS_L5 / ((double)Mp * (double)fft_len); // 1000 Hz here
    const double true_cp = 137.0, true_dop = 3000.0 + 0.3 * bin_hz;
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);
    auto acq = [&](const std::vector<double>& g) {
        return channelized_acquire(data, repl0, cov, g, FS_L5, sig->chip_rate_hz, N_L5,
                                   sig->code_length, cov, fft_len);
    };
    auto grid_around = [&](double c, double margin, double step) {
        std::vector<double> g;
        for (double f = c - margin; f <= c + margin + 1e-6; f += step)
            g.push_back(f);
        return g;
    };
    const double ref = -true_dop; // the search reports the conjugate sign; analytic truth
    // Bin-spaced grid, wide enough that the winning cell is strictly interior (the refine is
    // skipped at the edges).
    const auto coarse = grid_around(-3000.0, 3.0 * bin_hz, bin_hz);
    const double refined = acq(coarse).doppler_hz;
    // the grid cell nearest truth, 0.3 bin away: the answer without the refine
    double cell = coarse[0];
    for (double f : coarse)
        if (std::abs(f - ref) < std::abs(cell - ref))
            cell = f;
    const double err_grid = std::abs(cell - ref), err_ref = std::abs(refined - ref);
    BOOST_TEST_MESSAGE("refine: fine-ref=" << ref << " coarse-cell=" << cell
                                           << " refined=" << refined << " | grid_err=" << err_grid
                                           << " refined_err=" << err_ref);
    BOOST_CHECK_LT(err_ref, err_grid);     // closer to truth than the raw cell (0.3 bin)
    BOOST_CHECK_LT(err_ref, 0.1 * bin_hz); // and inside a tenth of a bin
}

// Galileo E1-C at 5 MSPS, N = 20: the BOC(1,1) pilot as an expanded code (2 half-cycle slots per
// chip at 2.046 Mcps, the same comb_mult expansion as L2C TDM, which gives the hop-rate path and
// the GPU despread BOC support). Runs the real bank through channelized_acquire and the
// stage's exact-despread refine.
BOOST_AUTO_TEST_CASE(gal_e1c_single_window_acquire_recovers) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GAL_E1C");
    BOOST_REQUIRE(sig != nullptr);
    constexpr double FS = 5.0e6; // 5 MSPS real
    constexpr double FOFF = 1.25e6;
    constexpr int N = 20;
    gnss::ChannelizedReplicaBank bank(*sig, FS, FOFF, N, 4, dsp::Window::Hamming, {11});
    BOOST_REQUIRE_EQUAL(bank.comb_mult(), 2);          // BOC(1,1) -> expanded x2
    BOOST_REQUIRE_EQUAL(bank.eff_code_length(), 8184); // 4092 * 2
    const int fft_len = 2 * N;
    const int Mp = bank.repl_period_hops(); // 20000 samples / gcd(40, 20000) = 500 hops = 4 ms
    BOOST_REQUIRE_EQUAL(Mp, 500); // one 4 ms E1 period is an integer number of hops at N=20
    const long long anchor = (long long)Mp * fft_len;

    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp);
    const auto cov = energy_covering_n(repl0, N);
    BOOST_REQUIRE(!cov.empty());

    const double true_cp = 1234.25; // E1 chips (4.89 samples/chip)
    const double true_dop = 300.0;
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);

    const std::vector<double> grid = {-600, -300, 0, 300, 600};
    auto r = channelized_acquire(data, repl0, cov, grid, FS, sig->chip_rate_hz, N, sig->code_length,
                                 cov, fft_len);
    BOOST_TEST_MESSAGE("E1C acquire: snr=" << r.snr << " dop=" << r.doppler_hz
                                           << " cp=" << r.code_phase_chips << " (true " << true_cp
                                           << ")  cov=" << cov.size() << " chans");
    BOOST_CHECK_LT(std::abs(std::abs(r.doppler_hz) - true_dop), 150.0);
    BOOST_CHECK_GT(r.snr, 15.0);

    // Stage-style exact-despread refine around the coarse lag -> sub-chip cp.
    const double cps = sig->chip_rate_hz / FS;
    const double dop = -r.doppler_hz; // r2c sign flip, as the stage applies
    double best_cp = r.code_phase_chips, best_pw = -1.0;
    for (int off = -fft_len; off <= fft_len; ++off) {
        const double cp = r.code_phase_chips + off * cps;
        const auto repl = bank.channels(0, anchor, cp, dop, Mp);
        std::vector<std::vector<cf>> d, rr;
        for (int c : cov) {
            d.push_back(data[c]);
            rr.push_back(repl[c]);
        }
        const double pw = std::norm(gnss::channelized_despread(d, rr).amplitude);
        if (pw > best_pw) {
            best_pw = pw;
            best_cp = cp;
        }
    }
    double cp_err = std::fabs(best_cp - true_cp);
    cp_err = std::min(cp_err, (double)sig->code_length - cp_err);
    BOOST_TEST_MESSAGE("E1C refined cp=" << best_cp << " err " << cp_err << " chips");
    BOOST_CHECK_LT(cp_err, 0.5); // the BOC correlation peak is sharper than BPSK's

    // Cross-PRN rejection: despreading PRN 11's signal with PRN 12's replica finds no peak.
    gnss::ChannelizedReplicaBank bank12(*sig, FS, FOFF, N, 4, dsp::Window::Hamming, {12});
    auto repl12 = bank12.channels(0, anchor, 0.0, 0.0, Mp);
    auto r12 = channelized_acquire(data, repl12, cov, grid, FS, sig->chip_rate_hz, N,
                                   sig->code_length, cov, fft_len);
    BOOST_TEST_MESSAGE("E1C cross-PRN snr=" << r12.snr);
    BOOST_CHECK_LT(r12.snr, r.snr * 0.25);
}

// BeiDou-3 B1C pilot at 20 MSPS, N = 10: Weil-code primary, BOC(1,1) expanded, 10 ms period =
// 10000 hops. Same acquire and refine as the E1C case.
BOOST_AUTO_TEST_CASE(bds_b1c_single_window_acquire_recovers) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("BDS_B1C_P");
    BOOST_REQUIRE(sig != nullptr);
    constexpr double FS = 20.0e6;
    constexpr double FOFF = 5.0e6;
    constexpr int N = 10;
    gnss::ChannelizedReplicaBank bank(*sig, FS, FOFF, N, 4, dsp::Window::Hamming, {19});
    BOOST_REQUIRE_EQUAL(bank.comb_mult(), 2);
    const int fft_len = 2 * N;
    const int Mp = bank.repl_period_hops();
    BOOST_REQUIRE_EQUAL(Mp, 10000); // 10 ms = 200000 samples / 20 = integer hops
    const long long anchor = (long long)Mp * fft_len;

    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp);
    const auto cov = energy_covering_n(repl0, N);
    BOOST_REQUIRE(!cov.empty());

    const double true_cp = 5678.5;
    const double true_dop = -150.0;
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);
    const std::vector<double> grid = {-300, -150, 0, 150, 300};
    auto r = channelized_acquire(data, repl0, cov, grid, FS, sig->chip_rate_hz, N, sig->code_length,
                                 cov, fft_len);
    BOOST_TEST_MESSAGE("B1C acquire: snr=" << r.snr << " dop=" << r.doppler_hz
                                           << " cp=" << r.code_phase_chips);
    BOOST_CHECK_GT(r.snr, 15.0);
    BOOST_CHECK_LT(std::abs(std::abs(r.doppler_hz) - 150.0), 80.0);

    const double cps = sig->chip_rate_hz / FS;
    const double dop = -r.doppler_hz;
    double best_cp = r.code_phase_chips, best_pw = -1.0;
    for (int off = -fft_len; off <= fft_len; ++off) {
        const double cp = r.code_phase_chips + off * cps;
        const auto repl = bank.channels(0, anchor, cp, dop, Mp);
        std::vector<std::vector<cf>> d, rr;
        for (int c : cov) {
            d.push_back(data[c]);
            rr.push_back(repl[c]);
        }
        const double pw = std::norm(gnss::channelized_despread(d, rr).amplitude);
        if (pw > best_pw) {
            best_pw = pw;
            best_cp = cp;
        }
    }
    double cp_err = std::fabs(best_cp - true_cp);
    cp_err = std::min(cp_err, (double)sig->code_length - cp_err);
    BOOST_TEST_MESSAGE("B1C refined cp=" << best_cp << " err " << cp_err);
    BOOST_CHECK_LT(cp_err, 0.5);
}
