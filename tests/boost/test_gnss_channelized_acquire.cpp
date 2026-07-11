#define BOOST_TEST_MODULE "test_gnss_channelized_acquire"

#include "gnssChannelizedAcquire.hpp"  // for channelized_acquire
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

// Reproducible standard-normal sample via Box-Muller on the engine's raw output.
// (std::normal_distribution is NOT portable across stdlib implementations, so a
// seeded test would diverge between macOS/libc++ and cobalt/libstdc++.)
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

// Analyze with fftwEngine's exact PFB convention: newest-first pfb_push,
// pfb_fold, forward DFT, then fftshift. Warmed from the (periodic) tail so the
// first hop is circular-correct, matching the core's circular correlation.
std::vector<std::vector<cf>> analyze_fftwe(const std::vector<cf>& x, int Nc,
                                           const std::vector<float>& proto) {
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
            ch[j][m] = spec[(j + Nc / 2) % Nc]; // fftshift: DC at j = Nc/2
    };
    for (int w = 0; w < P - 1; ++w)
        do_hop(Mh - (P - 1) + w, false); // warm history from the tail
    for (int m = 0; m < Mh; ++m)
        do_hop(m, true);
    return ch;
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

} // namespace

BOOST_AUTO_TEST_CASE(recovers_with_fftwengine_convention) {
    // Even Nc dividing one code period (4092 = 2^2*3*11*31): Nc=12, DC at bin 6.
    constexpr int Nc = 12;
    const auto proto = dsp::pfb_prototype(Nc, P, dsp::Window::Hamming);
    const long true_tau = 20; // q=1, s=8 -> exercises fine lag
    const double true_dop = 200.0;

    auto data = analyze_fftwe(gen(5, true_tau * CHIP_RATE / FS, true_dop, cf(1.0f, 0.0f)), Nc, proto);
    auto repl0 = analyze_fftwe(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), Nc, proto);
    const auto cov = energy_covering_n(repl0, Nc);

    // fftshifted bank: channel j <-> frequency (j - Nc/2).
    std::vector<int> chan_freq;
    for (int c : cov)
        chan_freq.push_back(c - Nc / 2);

    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    auto r = gnss::channelized_acquire(data, repl0, cov, grid, FS, CHIP_RATE, Nc, CODE_LEN,
                                       chan_freq);
    BOOST_CHECK_LT(std::abs(r.doppler_hz - true_dop), 100.0); // sub-grid refine: within half a 200 Hz cell
    BOOST_CHECK_LE(std::abs(r.peak_tau_samples - true_tau), (long)SP);
    BOOST_CHECK_GT(r.snr, 15.0);
}

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

    BOOST_CHECK_LT(std::abs(r.doppler_hz - true_dop), 100.0); // sub-grid refine: within half a 200 Hz cell
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
    BOOST_CHECK_LT(std::abs(r.doppler_hz), 100.0); // sub-grid refine: ~0 within half a cell
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

// channelized_acquire is just channelized_accumulate(one window) + channelized_peak;
// exercising the split directly (with an explicit AcquireWorkspace) must reproduce
// the wrapper exactly, and report the expected surface dimensions.
BOOST_AUTO_TEST_CASE(accumulate_peak_matches_wrapper) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 50;
    const double true_dop = 200.0;
    auto data = stft(gen(5, true_tau * CHIP_RATE / FS, true_dop, cf(1.0f, 0.0f)), proto);
    auto repl0 = stft(gen(5, 0.0, 0.0, cf(1.0f, 0.0f)), proto);
    const auto cov = energy_covering(repl0);
    const std::vector<double> grid = {-400, -200, 0, 200, 400};

    auto rw = gnss::channelized_acquire(data, repl0, cov, grid, FS, CHIP_RATE, N, CODE_LEN);

    gnss::AcquireWorkspace ws;
    std::vector<double> surf;
    const auto dims = gnss::channelized_accumulate(data, repl0, cov, grid, FS, N, surf, ws);
    auto rs = gnss::channelized_peak(surf, dims, grid, FS, CHIP_RATE, CODE_LEN);

    BOOST_CHECK_EQUAL(dims.n_dop, (int)grid.size());
    BOOST_CHECK_EQUAL(dims.Mp, M); // repl0 hop-period
    BOOST_CHECK_EQUAL(dims.sph, N); // complex bank: hop == channel count (samples_per_hop=0)
    BOOST_CHECK_EQUAL(rs.peak_tau_samples, rw.peak_tau_samples);
    BOOST_CHECK_EQUAL(rs.doppler_hz, rw.doppler_hz);
    BOOST_CHECK_CLOSE(rs.snr, rw.snr, 1e-2);
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
    BOOST_CHECK_CLOSE(rK.snr, r1.snr, 1e-2);        // ratio invariant
}

// A signal too weak to acquire in one window (a noise spike wins the surface) is
// recovered once enough independent-noise windows are integrated incoherently --
// the sensitivity mechanism the accumulation exists for.
BOOST_AUTO_TEST_CASE(accumulation_recovers_weak_signal_under_noise) {
    const auto proto = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    const long true_tau = 50;
    const double true_dop = 200.0;
    const double cp = true_tau * CHIP_RATE / FS;
    constexpr float AMP = 0.30f;   // per-sample signal amplitude
    constexpr float SIGMA = 6.0f;  // per-I/Q noise sigma (signal well below noise)
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

    BOOST_CHECK_GT(std::abs(tau1 - true_tau), (long)SP);      // single window: misses
    BOOST_CHECK_LT(std::abs(rK.doppler_hz - true_dop), 100.0); // K windows: Doppler right (sub-grid refine)
    BOOST_CHECK_LE(std::abs(rK.peak_tau_samples - true_tau), (long)SP); // ...and code phase
}

// --------------------------------------------------------------------------------------------
// L2C bug hunt: the channelized live_l2c.yaml detects nothing while the monolithic correlator
// gets snr ~133 on the same sky -> an algorithmic bug in the CHANNELIZED path. These two cases
// drive the REAL gnss::ChannelizedReplicaBank (the time-multiplexed CM/CL combined code) through
// the REAL gnss::channelized_acquire, at the production geometry (Fs=5 MHz, N=12, f_off=1.25 MHz),
// to localize: does acquire+replica work for L2C (-> bug is in the search STAGE wrapper), and is
// the live hops_per_record window a legal incoherent window?
namespace {
constexpr double FS_L2C = 5.0e6;   // airspy 5 MSPS real
constexpr double FOFF_L2C = 1.25e6; // L1/L2 IF = Fs/4
constexpr int N_L2C = 12;           // live_l2c.yaml spectrum_length
constexpr int TAPS_L2C = 4;
} // namespace

// Localizer: a known L2C code phase + Doppler, generated and matched through the production
// replica bank over ONE replica period (an integer number of code periods), is recovered with a
// sharp peak. If this passes, channelized_acquire + the time-mux replica are sound for L2C, so a
// live no-detect lives in the STAGE wrapper (snapshot windowing / covering-channel slice / etc).
BOOST_AUTO_TEST_CASE(l2c_single_window_acquire_recovers) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L2C_CM");
    BOOST_REQUIRE(sig != nullptr);
    gnss::ChannelizedReplicaBank bank(*sig, FS_L2C, FOFF_L2C, N_L2C, TAPS_L2C, dsp::Window::Hamming,
                                      {1});
    const int fft_len = 2 * N_L2C;            // 24 real samples / hop (r2c)
    const int Mp = bank.repl_period_hops();   // 12500 hops = 3 code periods (24 does not divide 100000)
    const long long anchor = (long long)Mp * fft_len; // warm-up reads the periodic code

    // repl0 (code 0, Doppler 0) over one replica period; covering = channels carrying energy.
    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp); // [N][Mp], natural-order r2c
    const auto cov = energy_covering_n(repl0, N_L2C);
    BOOST_REQUIRE(!cov.empty());

    const double true_cp = 137.0;  // CM chips (sub-chip / sub-hop lag exercised)
    const double true_dop = 200.0; // Hz
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);

    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    // Mirror the stage: samples_per_hop = fft_len (r2c), chan_freq = covering (natural order),
    // peak reported in COMPONENT CM chips (sig->chip_rate_hz / code_length).
    auto r = gnss::channelized_acquire(data, repl0, cov, grid, FS_L2C, sig->chip_rate_hz, N_L2C,
                                       sig->code_length, cov, fft_len);

    BOOST_TEST_MESSAGE("L2C single-window: snr=" << r.snr << " dop=" << r.doppler_hz
                                                 << " cp=" << r.code_phase_chips << " (true "
                                                 << true_cp << ")");
    BOOST_CHECK_LT(std::abs(std::abs(r.doppler_hz) - true_dop), 100.0); // grid value (sign per bank convention)
    BOOST_CHECK_GT(r.snr, 15.0);                         // sharp coherent peak
    double cp_err = std::fabs(r.code_phase_chips - true_cp);
    cp_err = std::min(cp_err, (double)sig->code_length - cp_err);
    BOOST_CHECK_LT(cp_err, 2.0); // within ~1 CM chip
}

// The live-config failure mode. live_l2c.yaml sets hops_per_record: 4166 (~1 code period). But at
// N=12 / Fs=5 MHz one L2C code period is 100000 samples = 4166.67 hops -- NOT an integer number of
// hops. The incoherent search slices CONTIGUOUS windows of hops_per_record, so a 4166-hop stride
// advances the code phase by 99984 mod 100000 = -16 samples each window: the |D|^2 peak walks and
// the incoherent sum smears it away (-> no detection). An integer number of code periods (here the
// 12500-hop replica period = 3 periods) keeps the peak stationary so the sum builds. This is the
// invariant the stage must honour: the incoherent window must span an integer number of code
// periods. (For L1 it does -- 5000 samples, repl_period_hops=625=3 periods, all integer.)
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

    // Slice CONTIGUOUS windows of `hpr` hops out of the snapshot and integrate the |D|^2 surface,
    // exactly as GnssChannelizedSearch::search_snapshot does. Returns the surface peak/mean SNR.
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
    const double snr_live = run(4166); // live_l2c.yaml value (~1 period, NON-integer) -> smears

    BOOST_TEST_MESSAGE("L2C incoherent snr: integer-period(12500)=" << snr_int << "  live(4166)="
                                                                    << snr_live);
    BOOST_CHECK_GT(snr_int, 15.0);           // legal window detects
    BOOST_CHECK_LT(snr_live, snr_int * 0.5); // non-integer window smears the peak away
}

// L5 (live_l5.yaml): the wide signal -- 10.23 Mcps at the 20 MSPS front end. At Fs=20 MHz we
// capture the central ~10 MHz (carrier +-5 MHz) of L5's ~20 MHz main lobe -- about half, but the
// ChannelizedReplicaBank band-limits the replica identically (same r2c PFB / Fs), so the despread
// stays matched and recovers the code phase + Doppler. N=10 -> fft_len 20 | 20000 = 1 code period
// (1 ms), so the search window is exactly 1 period (stationary + within one NH-overlay chip). We
// target the Q5 PILOT (dataless). Confirms the channelized acquire path works for the L5 geometry.
BOOST_AUTO_TEST_CASE(l5_single_window_acquire_recovers) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    BOOST_REQUIRE(sig != nullptr);
    constexpr double FS_L5 = 20.0e6;  // 20 MSPS wide front end (airspy sample_bw 10)
    constexpr double FOFF_L5 = 5.0e6; // = Fs/4
    constexpr int N_L5 = 10;          // live_l5.yaml spectrum_length
    gnss::ChannelizedReplicaBank bank(*sig, FS_L5, FOFF_L5, N_L5, 4, dsp::Window::Hamming, {1});
    const int fft_len = 2 * N_L5;            // 20
    const int Mp = bank.repl_period_hops();  // 20000 / gcd(20,20000) = 1000 hops = 1 ms (1 period)
    const long long anchor = (long long)Mp * fft_len;
    BOOST_REQUIRE_EQUAL(Mp, 1000); // one code period is an integer number of hops at N=10

    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp);
    const auto cov = energy_covering_n(repl0, N_L5);
    BOOST_REQUIRE(!cov.empty());

    const double true_cp = 137.0;  // L5 chips (~1.96 samples/chip -> sub-hop lag exercised)
    const double true_dop = 200.0;
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);

    const std::vector<double> grid = {-400, -200, 0, 200, 400};
    auto r = gnss::channelized_acquire(data, repl0, cov, grid, FS_L5, sig->chip_rate_hz, N_L5,
                                       sig->code_length, cov, fft_len);

    BOOST_TEST_MESSAGE("L5 acquire: snr=" << r.snr << " dop=" << r.doppler_hz
                                          << " cp=" << r.code_phase_chips << " (true " << true_cp
                                          << ")  cov=" << cov.size() << " chans");
    BOOST_CHECK_LT(std::abs(std::abs(r.doppler_hz) - true_dop), 100.0); // detects with the right Doppler
    BOOST_CHECK_GT(r.snr, 15.0);                          // strong peak
    // Acquire localizes the coarse lag to ~1 hop: at 20 MSPS we hold only ~half L5's main lobe, so
    // the autocorrelation peak is ~2x broader and the raw cp is hop-coarse (here ~0.8 hop off).
    const long true_tau = (long)std::llround(true_cp * FS_L5 / sig->chip_rate_hz);
    BOOST_CHECK_LE(std::abs(r.peak_tau_samples - true_tau), (long)(2 * fft_len));

    // The stage sharpens cp after acquire with an exact-despread +-1-hop scan (search_snapshot).
    // Replicate it: the despread |A| peaks at the true code phase -> sub-chip cp. This validates
    // the full L5 acquire->refine path the pipeline actually runs.
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
    BOOST_CHECK_LT(cp_err, 1.0); // refine nails it to sub-chip
}

// Almanac-narrowed search: the broker pushes the PHYSICAL predicted Doppler; GnssChannelizedSearch
// scans a tight grid centred at -hint (the REAL bank's r2c flip: reported = -grid value, as the
// L5/L2C cases above show). This checks, through the real ChannelizedReplicaBank, that a narrow
// grid around -true_dop still recovers the sat (narrowing doesn't break detection) AND pins the
// sign -- a sign error would search the wrong half and find nothing, the failure the broker<->search
// handoff must avoid. Uses a large Doppler so the wrong-sign window (2*dop away) is a clean miss,
// not just coarsely off (at small Doppler the 1 ms coherent resolution forgives a sign slip).
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

    const double true_cp = 137.0, true_dop = 3000.0; // PHYSICAL Doppler (the broker's prediction)
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);

    const double hint = true_dop, margin = 100.0, step = 50.0;
    auto grid_around = [&](double centre) {
        std::vector<double> g;
        for (double f = centre - margin; f <= centre + margin + 1e-6; f += step)
            g.push_back(f);
        return g;
    };
    // Right: centre at -hint (the convention the search uses) -> recovers, reports the grid value.
    const auto narrow = grid_around(-hint);
    auto r = gnss::channelized_acquire(data, repl0, cov, narrow, FS_L5, sig->chip_rate_hz, N_L5,
                                       sig->code_length, cov, fft_len);
    BOOST_TEST_MESSAGE("narrow(-hint): snr=" << r.snr << " dop=" << r.doppler_hz);
    BOOST_CHECK_LT(std::abs(r.doppler_hz + true_dop), 25.0); // ~-physical within half a 50 Hz cell (r2c + sub-grid refine)
    BOOST_CHECK_GT(r.snr, 15.0);                // narrowing kept the detection

    // Wrong sign: centre at +hint (2*dop from the true peak) -> a clean miss.
    const auto wrong = grid_around(hint);
    auto rw = gnss::channelized_acquire(data, repl0, cov, wrong, FS_L5, sig->chip_rate_hz, N_L5,
                                        sig->code_length, cov, fft_len);
    BOOST_TEST_MESSAGE("narrow(+hint, wrong): snr=" << rw.snr);
    BOOST_CHECK_LT(rw.snr, r.snr * 0.3); // wrong-sign window decorrelates -> no peak
}

// The sub-grid Doppler parabola refine must BEAT the coarse-grid cell on an OFF-grid signal.
// Reference truth = a fine-grid (step 5) max in the SAME channelization frame (so any small
// channelization Doppler bias cancels); a coarse grid (step 100) + refine should land closer.
BOOST_AUTO_TEST_CASE(doppler_parabola_refine_beats_grid) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    constexpr double FS_L5 = 20.0e6, FOFF_L5 = 5.0e6;
    constexpr int N_L5 = 10;
    gnss::ChannelizedReplicaBank bank(*sig, FS_L5, FOFF_L5, N_L5, 4, dsp::Window::Hamming, {1});
    const int fft_len = 2 * N_L5;
    const int Mp = bank.repl_period_hops();
    const long long anchor = (long long)Mp * fft_len;
    auto repl0 = bank.channels(0, anchor, 0.0, 0.0, Mp);
    const auto cov = energy_covering_n(repl0, N_L5);

    const double true_cp = 137.0, true_dop = 3037.0; // OFF the coarse grid (-3000 step 100)
    auto data = bank.channels(0, anchor, true_cp, true_dop, Mp);
    auto acq = [&](const std::vector<double>& g) {
        return gnss::channelized_acquire(data, repl0, cov, g, FS_L5, sig->chip_rate_hz, N_L5,
                                         sig->code_length, cov, fft_len);
    };
    auto grid_around = [&](double c, double margin, double step) {
        std::vector<double> g;
        for (double f = c - margin; f <= c + margin + 1e-6; f += step)
            g.push_back(f);
        return g;
    };
    const double ref = acq(grid_around(-3037.0, 60.0, 5.0)).doppler_hz; // fine-grid "truth"
    const auto coarse = grid_around(-3000.0, 250.0, 100.0);
    const double refined = acq(coarse).doppler_hz;
    // the nearest coarse cell to ref (what we'd have reported WITHOUT the refine)
    double cell = coarse[0];
    for (double f : coarse)
        if (std::abs(f - ref) < std::abs(cell - ref))
            cell = f;
    const double err_grid = std::abs(cell - ref), err_ref = std::abs(refined - ref);
    BOOST_TEST_MESSAGE("refine: fine-ref=" << ref << " coarse-cell=" << cell << " refined="
                                           << refined << " | grid_err=" << err_grid
                                           << " refined_err=" << err_ref);
    BOOST_CHECK_LT(err_ref, err_grid);  // the refine is closer to truth than the raw cell
    BOOST_CHECK_LT(err_ref, 25.0);      // and within ~step/4
}

// Galileo E1-C at the L1 front-end geometry (5 MSPS / N=20, the live_l1 configs): the BOC(1,1)
// pilot as an EXPANDED code (2 half-cycle slots per chip at 2.046 Mcps -- the same comb
// machinery as L2C TDM, which is what gives the hoprate path AND the GPU despread kernel BOC
// support). Drives the real ChannelizedReplicaBank(GAL_E1C) + channelized_acquire, then the
// stage's exact-despread refine, at production geometry -- the whole Galileo acquire path.
BOOST_AUTO_TEST_CASE(gal_e1c_single_window_acquire_recovers) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GAL_E1C");
    BOOST_REQUIRE(sig != nullptr);
    constexpr double FS = 5.0e6;   // live L1 front end
    constexpr double FOFF = 1.25e6;
    constexpr int N = 20;          // live_l1 spectrum_length
    gnss::ChannelizedReplicaBank bank(*sig, FS, FOFF, N, 4, dsp::Window::Hamming, {11});
    BOOST_REQUIRE_EQUAL(bank.comb_mult(), 2);             // BOC(1,1) -> expanded x2
    BOOST_REQUIRE_EQUAL(bank.eff_code_length(), 8184);    // 4092 * 2
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
    auto r = gnss::channelized_acquire(data, repl0, cov, grid, FS, sig->chip_rate_hz, N,
                                       sig->code_length, cov, fft_len);
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
    BOOST_CHECK_LT(cp_err, 0.5); // BOC correlation peak is SHARPER than BPSK -- sub-half-chip

    // Cross-PRN rejection: despreading PRN 11's signal with PRN 12's replica finds no peak.
    gnss::ChannelizedReplicaBank bank12(*sig, FS, FOFF, N, 4, dsp::Window::Hamming, {12});
    auto repl12 = bank12.channels(0, anchor, 0.0, 0.0, Mp);
    auto r12 = gnss::channelized_acquire(data, repl12, cov, grid, FS, sig->chip_rate_hz, N,
                                         sig->code_length, cov, fft_len);
    BOOST_TEST_MESSAGE("E1C cross-PRN snr=" << r12.snr);
    BOOST_CHECK_LT(r12.snr, r.snr * 0.25);
}
