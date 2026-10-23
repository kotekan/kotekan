// Validation and timing of the code-phase-table (Psi) replica synthesis prototype
// (cudaGnssPsiSynth.cu) against the production synthesis (gnss_waveform_kernel) at CHORD's
// production geometry: fs 3.2 GS/s, N 8192 (fft_len 16384), 4-tap Hamming PFB, GPS L5 Q at
// 1176.45 MHz on the live cx51 comb {5975 .. 6071, stride 16}, 2048-hop records, 12 PRNs each at
// its own code phase and Doppler, at the deep absolute anchor of cudaGnssDespreadTest.
//
// Replica sets (all E/P/L rows, [3*n_job][n_chan][n_hops]):
//   FULL  launch_waveform, fp32 Phi, the full 212-chip span, three exact trials
//   NOA   FULL with PhiA zeroed: 0.5 conj(pa) S_B in the rect model, the Psi path's reference
//   PROD  launch_waveform as production runs it: fp16 Phi, 80 centred chips, E/L from the
//         rotated prompt
//   PSI   launch_psi_waveform, G = 1 tables (fractional-edge model), three exact trials, fp64
//         per-hop phases; plus variants: rotated-prompt E/L, fixed-point phases, fp16 tables,
//         and a table built with the production's sample-snapped chip edges (RECT)
// Per-sample comparisons are per channel over every job and hop; the physically meaningful
// test is the correlation of a synthetic sky (the bank's full rect-model replica, both images)
// against each set: the lobe-coherent E-L discriminator and the prompt SNR, normalised as
// gnssFleetDll does.
//
//   ./cuda_gnss_psi_bench [all|bench|tiled|fused|fusedprof] [n_prn]
//
// "bench" skips the CPU references (for ncu); "tiled" checks the tiled-code split on GPS_L5_Q_NH.
// "fused" validates launch_psi_pack (synthesis + 4+4b pack in one kernel) byte for byte against
// launch_psi_waveform + launch_pack44 on the stage's synth-array geometry, runs the sky test on
// the dequantised lanes, and times it against the production pair; "fusedprof" launches each
// fused configuration once, in a fixed order, for ncu.
// Never run beside a live node: the bench saturates the GPU.

#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "cudaGnssPsiSynth.hpp"
#include "dspPfbPrototype.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <vector>

using cf = std::complex<float>;
using cd = std::complex<double>;

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,           \
                    __LINE__);                                                                     \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

namespace {

constexpr long double TWO_PI_L = 6.283185307179586476925286766559005768L;

/// Relative rms of (a - b) over a row set, per channel: sqrt(sum|a-b|^2 / sum|b|^2).
struct RowStats {
    std::vector<double> d2, b2; // [3][n_chan]
    int n_chan;
    explicit RowStats(int nc) : d2(3 * (size_t)nc, 0.0), b2(3 * (size_t)nc, 0.0), n_chan(nc) {}
    void add(int t, int c, const float2& a, const float2& b) {
        const double dx = (double)a.x - b.x, dy = (double)a.y - b.y;
        d2[(size_t)t * n_chan + c] += dx * dx + dy * dy;
        b2[(size_t)t * n_chan + c] += (double)b.x * b.x + (double)b.y * b.y;
    }
    double rel(int t, int c) const {
        return std::sqrt(d2[(size_t)t * n_chan + c] / std::max(1e-300, b2[(size_t)t * n_chan + c]));
    }
};

RowStats compare_sets(const std::vector<float2>& a, const std::vector<float2>& b, int n_job,
                      int n_chan, int n_hops) {
    RowStats s(n_chan);
    for (int j = 0; j < n_job; ++j)
        for (int t = 0; t < 3; ++t)
            for (int c = 0; c < n_chan; ++c) {
                const size_t o = ((size_t)(3 * j + t) * n_chan + c) * n_hops;
                for (int m = 0; m < n_hops; ++m)
                    s.add(t, c, a[o + m], b[o + m]);
            }
    return s;
}

void print_stats(const char* title, const RowStats& s, const std::vector<int>& chans) {
    printf("%s\n   chan        E          P          L\n", title);
    for (int c = 0; c < (int)chans.size(); ++c)
        printf("   %4d  %9.2e  %9.2e  %9.2e\n", chans[(size_t)c], s.rel(0, c), s.rel(1, c),
               s.rel(2, c));
}

double max_abs_diff(const std::vector<float2>& a, const std::vector<float2>& b, double* rms_b) {
    double mx = 0.0, s2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        mx = std::max({mx, std::fabs((double)a[i].x - b[i].x), std::fabs((double)a[i].y - b[i].y)});
        s2 += (double)b[i].x * b[i].x + (double)b[i].y * b[i].y;
    }
    if (rms_b)
        *rms_b = std::sqrt(s2 / std::max<size_t>(1, a.size()));
    return mx;
}

/// min / median of per-launch cudaEvent timings, microseconds.
template<class F>
void time_launch(const char* name, F&& launch, int n_it, double* out_min, double* out_med) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int i = 0; i < 3; ++i) // warm-up
        CK(launch());
    CK(cudaDeviceSynchronize());
    std::vector<double> us;
    for (int i = 0; i < n_it; ++i) {
        CK(cudaEventRecord(e0, 0));
        CK(launch());
        CK(cudaEventRecord(e1, 0));
        CK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        us.push_back(1000.0 * ms);
    }
    std::sort(us.begin(), us.end());
    *out_min = us.front();
    *out_med = us[us.size() / 2];
    printf("   %-34s min %8.1f us  median %8.1f us  (n=%d)\n", name, *out_min, *out_med, n_it);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

/// Tiled-code check on GPS_L5_Q_NH (L = 20 x 10230): the kernel's Psi_plain / Psi_prev / sec
/// split against a table built over the whole 204600-chip code, and both against NOA on the
/// NH bank. Two PRNs; the jobs are replicated to @c n_job for the timing.
// build_psi_split by FFT (the production builder) against the direct sum, per PRN: the build
// times, the agreement of the double accumulators (the FFT's roundoff, ~1e-13 relative) and
// how many float2 entries round differently (a handful of ties at most).
void psi_split_methods(const gnss::ChannelizedReplicaBank& bank, const std::vector<int>& chans,
                       const std::vector<double>& dop, double fs, int n_chips) {
    const int n_prn = (int)dop.size();
    const long L = bank.eff_code_length();
    double t_fft = 0.0, t_dir = 0.0, t_fft_max = 0.0, t_dir_max = 0.0;
    double rel_max = 0.0, abs_max = 0.0;
    size_t n_diff = 0, n_tot = 0;
    for (int b = 0; b < n_prn; ++b) {
        const int8_t* code = bank.full_code(b).data();
        const int P = gnss_cuda::psi_primary_period(code, L, n_chips);
        const gnss_cuda::PsiCodeSplit cs = gnss_cuda::psi_split_code(code, L, P);
        const gnss_cuda::PsiSplit F = gnss_cuda::build_psi_split_from(
            bank, b, chans, dop[(size_t)b], fs, 1, cs, true, gnss_cuda::PsiBuildMethod::FFT, true);
        const gnss_cuda::PsiSplit D =
            gnss_cuda::build_psi_split_from(bank, b, chans, dop[(size_t)b], fs, 1, cs, true,
                                            gnss_cuda::PsiBuildMethod::DIRECT, true);
        t_fft += F.build_seconds;
        t_dir += D.build_seconds;
        t_fft_max = std::max(t_fft_max, F.build_seconds);
        t_dir_max = std::max(t_dir_max, D.build_seconds);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < F.plain_d.size(); ++i) {
            const double d = std::abs(F.plain_d[i] - D.plain_d[i]);
            num += d * d;
            den += std::norm(D.plain_d[i]);
            abs_max = std::max(abs_max, d);
        }
        rel_max = std::max(rel_max, std::sqrt(num / std::max(den, 1e-300)));
        for (size_t i = 0; i < F.psi_plain.size(); ++i)
            n_diff +=
                (F.psi_plain[i].x != D.psi_plain[i].x) || (F.psi_plain[i].y != D.psi_plain[i].y);
        n_tot += F.psi_plain.size();
        // The lookback is the direct sum on both paths and must be identical.
        for (size_t i = 0; i < F.prev_d.size(); ++i)
            if (F.prev_d[i] != D.prev_d[i]) {
                printf("   PRN %d: psi_prev differs between the builders at %zu\n", b + 1, i);
                break;
            }
    }
    printf("build_psi_split (%d chans x %d chips, P from psi_primary_period): direct mean %.1f "
           "max %.1f ms/PRN, FFT mean %.1f max %.1f ms/PRN (x%.1f); FFT vs direct in double: "
           "rel rms max %.2e, abs max %.2e; float2 entries differing %zu / %zu %s\n",
           (int)chans.size(), n_chips, 1e3 * t_dir / n_prn, 1e3 * t_dir_max, 1e3 * t_fft / n_prn,
           1e3 * t_fft_max, t_dir / std::max(t_fft, 1e-12), rel_max, abs_max, n_diff, n_tot,
           rel_max < 1e-11 ? "OK" : "DISAGREE");
}

int run_tiled(int n_job) {
    const double fs = 3.2e9, f_off = 1176.45e6;
    const int N = 8192, taps = 4, fft_len = 2 * N, n_hops = 2048;
    const double spacing = 0.5;
    const std::vector<int> chans = {5975, 5991, 6007, 6023, 6039, 6055, 6071};
    const int n_chan = (int)chans.size();
    const long long window_start = 114436200145LL * 16384LL;
    const long long n0 = window_start + fft_len - 1;
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q_NH");
    if (!sig)
        return 2;
    const int n_prn = 2;
    gnss::ChannelizedReplicaBank bank(*sig, fs, f_off, N, taps, dsp::window_from_string("hamming"),
                                      {1, 2});
    const int L = (int)bank.eff_code_length();
    const int P = 10230;
    const int Lf = fft_len * taps;
    printf("tiled check: %s L %d = %d x %d, %d jobs\n", sig->name, L, L / P, P, n_job);
    const double dop[2] = {-3000.0, 2500.0};
    const double cp_true[2] = {123456.789, 204599.5};
    std::vector<int8_t> code8;
    for (int b = 0; b < n_prn; ++b) {
        const auto& fc = bank.full_code(b);
        code8.insert(code8.end(), fc.begin(), fc.end());
    }
    int8_t* d_code = nullptr;
    CK(cudaMalloc(&d_code, code8.size()));
    CK(cudaMemcpy(d_code, code8.data(), code8.size(), cudaMemcpyHostToDevice));

    const size_t phi_n = (size_t)n_chan * (Lf + 1);
    float2* d_zeroA = nullptr;
    CK(cudaMalloc(&d_zeroA, phi_n * sizeof(float2)));
    CK(cudaMemset(d_zeroA, 0, phi_n * sizeof(float2)));
    std::vector<float2*> d_phiB(n_prn), d_full(n_prn), d_plain(n_prn), d_prev(n_prn);
    std::vector<int8_t*> d_sec(n_prn);
    std::vector<gnss_cuda::PsiTable> T(n_prn);
    std::vector<gnss_cuda::PsiSplit> S(n_prn);
    int n_chips = 0;
    double cps[2], wc[2];
    for (int b = 0; b < n_prn; ++b) {
        cps[b] =
            bank.eff_chip_rate() / fs * (1.0 + bank.code_doppler_sign * dop[b] / sig->carrier_hz);
        wc[b] = 2.0 * M_PI * (f_off + dop[b]) / fs;
        const auto filt = bank.hoprate_filter(chans, dop[b], b);
        n_chips = filt.n_chips;
        std::vector<float2> hB(phi_n);
        for (int c = 0; c < n_chan; ++c)
            for (int k = 0; k <= Lf; ++k)
                hB[(size_t)c * (Lf + 1) + k] =
                    make_float2((float)filt.PhiB[(size_t)c][(size_t)k].real(),
                                (float)filt.PhiB[(size_t)c][(size_t)k].imag());
        CK(cudaMalloc(&d_phiB[(size_t)b], phi_n * sizeof(float2)));
        CK(cudaMemcpy(d_phiB[(size_t)b], hB.data(), phi_n * sizeof(float2),
                      cudaMemcpyHostToDevice));
        T[(size_t)b] = gnss_cuda::build_psi_table(bank, b, chans, dop[b], fs, 1, true);
        S[(size_t)b] = gnss_cuda::build_psi_split(bank, b, chans, dop[b], fs, 1, P, true);
        printf("   PRN %d: full table %.3f s (%zu entries/chan), split (FFT) %.3f s (%d + %d "
               "entries/chan, Ns %d, sec[0..3] %d %d %d %d)\n",
               b + 1, T[(size_t)b].build_seconds, T[(size_t)b].psi.size() / n_chan,
               S[(size_t)b].build_seconds, P, S[(size_t)b].n_prev, S[(size_t)b].Ns,
               S[(size_t)b].sec[0], S[(size_t)b].sec[1], S[(size_t)b].sec[2], S[(size_t)b].sec[3]);
        const size_t nf = T[(size_t)b].psi.size(), np = S[(size_t)b].psi_plain.size(),
                     nv = S[(size_t)b].psi_prev.size();
        CK(cudaMalloc(&d_full[(size_t)b], nf * sizeof(float2)));
        CK(cudaMemcpy(d_full[(size_t)b], T[(size_t)b].psi.data(), nf * sizeof(float2),
                      cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_plain[(size_t)b], np * sizeof(float2)));
        CK(cudaMemcpy(d_plain[(size_t)b], S[(size_t)b].psi_plain.data(), np * sizeof(float2),
                      cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_prev[(size_t)b], nv * sizeof(float2)));
        CK(cudaMemcpy(d_prev[(size_t)b], S[(size_t)b].psi_prev.data(), nv * sizeof(float2),
                      cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_sec[(size_t)b], S[(size_t)b].sec.size()));
        CK(cudaMemcpy(d_sec[(size_t)b], S[(size_t)b].sec.data(), S[(size_t)b].sec.size(),
                      cudaMemcpyHostToDevice));
    }

    psi_split_methods(bank, chans, std::vector<double>(dop, dop + n_prn), fs, n_chips);

    std::vector<gnss_cuda::DespreadJob> jn((size_t)n_job);
    std::vector<gnss_cuda::PsiJob> pf((size_t)n_job), ps((size_t)n_job);
    std::vector<float> theta((size_t)n_job * n_chan);
    for (int j = 0; j < n_job; ++j) {
        const int b = j % n_prn;
        const double cp0 = cp_true[b] + 0.1 * (j / n_prn);
        long double c = (long double)cp0 + (long double)n0 * (long double)cps[b];
        c = fmodl(c, (long double)L);
        if (c < 0.0L)
            c += (long double)L;
        long double fr = fmodl(
            ((long double)f_off + (long double)dop[b]) / (long double)fs * (long double)n0, 1.0L);
        if (fr < 0.0L)
            fr += 1.0L;
        const double C0 = cp0 + (double)n0 * cps[b];
        const double next_b = (std::floor(C0 / L) + 1.0) * L;
        int m_head = (int)std::ceil((next_b - C0) / ((double)fft_len * cps[b]));
        m_head = std::max(0, std::min(n_hops, m_head));
        jn[(size_t)j] = {spacing,
                         (double)c,
                         cps[b],
                         1.0 / cps[b],
                         wc[b],
                         (double)(TWO_PI_L * fr),
                         b * L,
                         L,
                         (1ULL << n_chan) - 1,
                         d_zeroA,
                         d_phiB[(size_t)b],
                         n_chips,
                         m_head,
                         0};
        gnss_cuda::PsiJob F{};
        F.psi = d_full[(size_t)b];
        F.P = L;
        F.Ns = 1;
        F.G = 1;
        gnss_cuda::psi_fixed_setup(jn[(size_t)j], F, fft_len, n_hops);
        pf[(size_t)j] = F;
        gnss_cuda::PsiJob Q = F;
        Q.psi = d_plain[(size_t)b];
        Q.psi_prev = d_prev[(size_t)b];
        Q.sec = d_sec[(size_t)b];
        Q.P = P;
        Q.Ns = L / P;
        Q.n_prev = S[(size_t)b].n_prev;
        ps[(size_t)j] = Q;
        for (int c2 = 0; c2 < n_chan; ++c2)
            theta[(size_t)j * n_chan + c2] =
                (float)(gnss_cuda::psi_theta(chans[(size_t)c2], fft_len, wc[b]) / cps[b]);
    }
    gnss_cuda::DespreadJob* d_jn = nullptr;
    gnss_cuda::PsiJob *d_pf = nullptr, *d_ps = nullptr;
    float* d_theta = nullptr;
    CK(cudaMalloc(&d_jn, jn.size() * sizeof(gnss_cuda::DespreadJob)));
    CK(cudaMalloc(&d_pf, pf.size() * sizeof(gnss_cuda::PsiJob)));
    CK(cudaMalloc(&d_ps, ps.size() * sizeof(gnss_cuda::PsiJob)));
    CK(cudaMalloc(&d_theta, theta.size() * sizeof(float)));
    CK(cudaMemcpy(d_jn, jn.data(), jn.size() * sizeof(gnss_cuda::DespreadJob),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pf, pf.data(), pf.size() * sizeof(gnss_cuda::PsiJob), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ps, ps.data(), ps.size() * sizeof(gnss_cuda::PsiJob), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_theta, theta.data(), theta.size() * sizeof(float), cudaMemcpyHostToDevice));

    gnss_cuda::DespreadParams p{};
    p.n0 = n0;
    p.fft_len = fft_len;
    p.n_hops = n_hops;
    p.Lf = Lf;
    p.data_stride = n_hops;
    const size_t wn = (size_t)3 * n_job * n_chan * n_hops, en = (size_t)4 * n_job * n_chan;
    float2* d_wave = nullptr;
    double* d_energy = nullptr;
    CK(cudaMalloc(&d_wave, wn * sizeof(float2)));
    CK(cudaMalloc(&d_energy, en * sizeof(double)));
    std::vector<float2> w_noa(wn), w_full(wn), w_split(wn);
    std::vector<double> e_full(en), e_split(en);
    CK(gnss_cuda::launch_waveform(d_code, d_jn, n_job, n_chan, p, nullptr, d_wave, d_energy, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(w_noa.data(), d_wave, wn * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(gnss_cuda::launch_psi_waveform(d_jn, d_pf, d_theta, n_job, n_chan, p, nullptr, 1, d_wave,
                                      d_energy, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(w_full.data(), d_wave, wn * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(e_full.data(), d_energy, en * sizeof(double), cudaMemcpyDeviceToHost));
    CK(gnss_cuda::launch_psi_waveform(d_jn, d_ps, d_theta, n_job, n_chan, p, nullptr, 1, d_wave,
                                      d_energy, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(w_split.data(), d_wave, wn * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(e_split.data(), d_energy, en * sizeof(double), cudaMemcpyDeviceToHost));
    print_stats("PSI split (Psi_plain/Psi_prev/sec) vs PSI full-length table (NH, fixed, 3 "
                "trials)",
                compare_sets(w_split, w_full, n_job, n_chan, n_hops), chans);
    double emax = 0.0;
    for (size_t i = 0; i < en; ++i)
        emax = std::max(emax, std::fabs(e_split[i] - e_full[i]) / std::max(1e-300, e_full[i]));
    printf("   energy rows split vs full: max rel %.3e\n", emax);
    print_stats("PSI full-length table vs NOA (NH bank)",
                compare_sets(w_full, w_noa, n_job, n_chan, n_hops), chans);
    double mn, md;
    time_launch(
        "NH NOA (fp32, 212 chips, 3 trials)",
        [&] {
            return gnss_cuda::launch_waveform(d_code, d_jn, n_job, n_chan, p, nullptr, d_wave,
                                              d_energy, 0);
        },
        20, &mn, &md);
    time_launch(
        "NH PSI full table (1.6 MB/chan)",
        [&] {
            return gnss_cuda::launch_psi_waveform(d_jn, d_pf, d_theta, n_job, n_chan, p, nullptr, 1,
                                                  d_wave, d_energy, 0);
        },
        20, &mn, &md);
    time_launch(
        "NH PSI split (80 KB + 1.7 KB/chan)",
        [&] {
            return gnss_cuda::launch_psi_waveform(d_jn, d_ps, d_theta, n_job, n_chan, p, nullptr, 1,
                                                  d_wave, d_energy, 0);
        },
        20, &mn, &md);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    const bool bench_only = (mode == "bench");
    const bool fused_mode = (mode == "fused" || mode == "fusedprof");
    const int n_prn = argc > 2 ? atoi(argv[2]) : 12;
    if (mode == "tiled")
        return run_tiled(n_prn);

    // ---- production geometry ----
    const double fs = 3.2e9, f_off = 1176.45e6;
    const int N = 8192, taps = 4;
    const int fft_len = 2 * N;
    const int n_hops = 2048;
    const double spacing = 0.5;
    const std::vector<int> chans = {5975, 5991, 6007, 6023, 6039, 6055, 6071};
    const int n_chan = (int)chans.size();
    long long window_start = 114436200145LL * 16384LL;
    if (const char* e = getenv("GNSS_TEST_HOP0"))
        window_start = atoll(e) * 16384LL;
    const long long n0 = window_start + fft_len - 1;

    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    if (!sig) {
        fprintf(stderr, "GPS_L5_Q unknown\n");
        return 2;
    }
    std::vector<int> prns;
    for (int q = 1; q <= n_prn; ++q)
        prns.push_back(q);
    gnss::ChannelizedReplicaBank bank(*sig, fs, f_off, N, taps, dsp::window_from_string("hamming"),
                                      prns);
    const int L = (int)bank.eff_code_length();
    const int comb = bank.comb_mult();
    const int Lf = fft_len * taps;
    printf("Psi synthesis bench: %s fs %.3g N %d fft_len %d taps %d Lf %d L %d comb %d n_hops %d "
           "n_prn %d chans",
           sig->name, fs, N, fft_len, taps, Lf, L, comb, n_hops, n_prn);
    for (int c : chans)
        printf(" %d", c);
    printf("\n");

    // Per-PRN code phase (argument, referred to sample 0) and Doppler.
    unsigned seed = 1234567u;
    auto frand = [&seed]() { // deterministic LCG, [0, 1)
        seed = seed * 1664525u + 1013904223u;
        return (double)(seed >> 8) / (double)(1 << 24);
    };
    std::vector<double> cp_true(n_prn), dop(n_prn), cps(n_prn), wc(n_prn);
    for (int b = 0; b < n_prn; ++b) {
        cp_true[(size_t)b] = frand() * L;
        dop[(size_t)b] = -5000.0 + 10000.0 * (n_prn > 1 ? (double)b / (n_prn - 1) : 0.5);
        cps[(size_t)b] = bank.eff_chip_rate() / fs
                         * (1.0 + bank.code_doppler_sign * dop[(size_t)b] / sig->carrier_hz);
        wc[(size_t)b] = 2.0 * M_PI * (f_off + dop[(size_t)b]) / fs;
    }
    // GnssCudaDespread::Impl::ang0_for / cp_ref_for / m_head_for, duplicated as the despread
    // test does (this is the independent check).
    auto ang0_of = [&](int b) {
        long double fr = fmodl(((long double)f_off + (long double)dop[(size_t)b]) / (long double)fs
                                   * (long double)n0,
                               1.0L);
        if (fr < 0.0L)
            fr += 1.0L;
        return (double)(TWO_PI_L * fr);
    };
    auto cp_ref_of = [&](double cp0, int b) {
        long double c = (long double)cp0 + (long double)n0 * (long double)cps[(size_t)b];
        c = fmodl(c, (long double)L);
        if (c < 0.0L)
            c += (long double)L;
        return (double)c;
    };
    auto m_head_of = [&](double cp0, int b) {
        const double C0 = cp0 + (double)n0 * cps[(size_t)b];
        const double next_b = (std::floor(C0 / L) + 1.0) * L;
        int m = (int)std::ceil((next_b - C0) / ((double)fft_len * cps[(size_t)b]));
        return std::max(0, std::min(n_hops, m));
    };

    // ---- code table ----
    std::vector<int8_t> code8;
    for (int b = 0; b < n_prn; ++b) {
        const auto& fc = bank.full_code(b);
        code8.insert(code8.end(), fc.begin(), fc.end());
    }
    int8_t* d_code = nullptr;
    CK(cudaMalloc(&d_code, code8.size()));
    CK(cudaMemcpy(d_code, code8.data(), code8.size(), cudaMemcpyHostToDevice));

    // ---- Phi tables per PRN (own Doppler): fp32, fp16, and one zero PhiA ----
    const size_t phi_n = (size_t)n_chan * (Lf + 1);
    std::vector<float2*> d_phiA(n_prn), d_phiB(n_prn);
    std::vector<void*> d_hA(n_prn), d_hB(n_prn);
    int n_chips_full = 0;
    {
        std::vector<float2> hA(phi_n), hB(phi_n);
        std::vector<unsigned short> hh(2 * phi_n); // __half2 storage, 4 bytes each
        for (int b = 0; b < n_prn; ++b) {
            const auto filt = bank.hoprate_filter(chans, dop[(size_t)b], b);
            n_chips_full = filt.n_chips;
            if (filt.d_first != 0) {
                fprintf(stderr, "bank truncation set\n");
                return 2;
            }
            for (int c = 0; c < n_chan; ++c)
                for (int k = 0; k <= Lf; ++k) {
                    hA[(size_t)c * (Lf + 1) + k] =
                        make_float2((float)filt.PhiA[(size_t)c][(size_t)k].real(),
                                    (float)filt.PhiA[(size_t)c][(size_t)k].imag());
                    hB[(size_t)c * (Lf + 1) + k] =
                        make_float2((float)filt.PhiB[(size_t)c][(size_t)k].real(),
                                    (float)filt.PhiB[(size_t)c][(size_t)k].imag());
                }
            CK(cudaMalloc(&d_phiA[(size_t)b], phi_n * sizeof(float2)));
            CK(cudaMalloc(&d_phiB[(size_t)b], phi_n * sizeof(float2)));
            CK(cudaMemcpy(d_phiA[(size_t)b], hA.data(), phi_n * sizeof(float2),
                          cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_phiB[(size_t)b], hB.data(), phi_n * sizeof(float2),
                          cudaMemcpyHostToDevice));
            CK(cudaMalloc(&d_hA[(size_t)b], phi_n * sizeof(float2) / 2));
            CK(cudaMalloc(&d_hB[(size_t)b], phi_n * sizeof(float2) / 2));
            gnss_cuda::phi_to_half(hA.data(), hh.data(), phi_n);
            CK(cudaMemcpy(d_hA[(size_t)b], hh.data(), phi_n * sizeof(float2) / 2,
                          cudaMemcpyHostToDevice));
            gnss_cuda::phi_to_half(hB.data(), hh.data(), phi_n);
            CK(cudaMemcpy(d_hB[(size_t)b], hh.data(), phi_n * sizeof(float2) / 2,
                          cudaMemcpyHostToDevice));
        }
    }
    float2* d_zeroA = nullptr;
    CK(cudaMalloc(&d_zeroA, phi_n * sizeof(float2)));
    CK(cudaMemset(d_zeroA, 0, phi_n * sizeof(float2)));
    const int prod_max_chips = 80;
    const int prod_d_first = (n_chips_full - prod_max_chips) / 2;
    const int prod_n_chips = prod_d_first + prod_max_chips;
    printf("PFB span n_chips %d (full, d_first 0); production window d_first %d n_chips %d\n",
           n_chips_full, prod_d_first, prod_n_chips);

    // ---- Psi tables per PRN: fractional-edge (the prototype) and rect (production edges) ----
    std::vector<gnss_cuda::PsiTable> T(n_prn), Trect(n_prn);
    std::vector<float2*> d_psi(n_prn), d_psi_rect(n_prn);
    std::vector<void*> d_psi_h(n_prn);
    double build_s = 0.0, build_min = 1e9, build_max = 0.0;
    for (int b = 0; b < n_prn; ++b) {
        T[(size_t)b] = gnss_cuda::build_psi_table(bank, b, chans, dop[(size_t)b], fs, 1, true);
        Trect[(size_t)b] = gnss_cuda::build_psi_table(bank, b, chans, dop[(size_t)b], fs, 1, false);
        build_s += T[(size_t)b].build_seconds;
        build_min = std::min(build_min, T[(size_t)b].build_seconds);
        build_max = std::max(build_max, T[(size_t)b].build_seconds);
        const size_t n = T[(size_t)b].psi.size();
        CK(cudaMalloc(&d_psi[(size_t)b], n * sizeof(float2)));
        CK(cudaMemcpy(d_psi[(size_t)b], T[(size_t)b].psi.data(), n * sizeof(float2),
                      cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_psi_rect[(size_t)b], n * sizeof(float2)));
        CK(cudaMemcpy(d_psi_rect[(size_t)b], Trect[(size_t)b].psi.data(), n * sizeof(float2),
                      cudaMemcpyHostToDevice));
        std::vector<unsigned short> hh(2 * n);
        gnss_cuda::psi_to_half(T[(size_t)b].psi.data(), hh.data(), n);
        CK(cudaMalloc(&d_psi_h[(size_t)b], n * sizeof(float2) / 2));
        CK(cudaMemcpy(d_psi_h[(size_t)b], hh.data(), n * sizeof(float2) / 2,
                      cudaMemcpyHostToDevice));
    }
    printf("build_psi_table (frac, G=1, %d chans x %d chips x L %d): total %.3f s for %d PRNs, "
           "per PRN min %.3f mean %.3f max %.3f s; table %.1f KB fp32 per (PRN, channel)\n",
           n_chan, n_chips_full, L, build_s, n_prn, build_min, build_s / n_prn, build_max,
           (double)L * sizeof(float2) / 1024.0);
    psi_split_methods(bank, chans, dop, fs, n_chips_full);
    {
        // The far image and the near-image choice per channel at this chain (the skeptic's
        // guard: |theta_far| * inv_cps < 50 would be a > 2e-2 far image at the centre channel).
        printf("theta per channel (PRN 0, rad/sample): chan theta_B theta_far |theta_far|*inv\n");
        for (int c = 0; c < n_chan; ++c) {
            double far = 0.0;
            const double thB = gnss_cuda::psi_theta(chans[(size_t)c], fft_len, wc[0], &far);
            printf("   %4d  %+.6f  %.4f  %.0f\n", chans[(size_t)c], thB, far, far / cps[0]);
        }
    }

    // ---- jobs: n_prn at the true phase, n_prn at +0.1 chip (validation); replicas for the
    // bench reuse PRN b = j mod n_prn ----
    auto make_jobs =
        [&](int n_job, double off_step, std::vector<gnss_cuda::DespreadJob>& jf,
            std::vector<gnss_cuda::DespreadJob>& jn, std::vector<gnss_cuda::DespreadJob>& jp,
            std::vector<gnss_cuda::PsiJob>& pj, std::vector<gnss_cuda::PsiJob>& pj_rect,
            std::vector<gnss_cuda::PsiJob>& pj_h, std::vector<float>& rot,
            std::vector<float>& theta) {
            jf.resize((size_t)n_job);
            jn.resize((size_t)n_job);
            jp.resize((size_t)n_job);
            pj.resize((size_t)n_job);
            pj_rect.resize((size_t)n_job);
            pj_h.resize((size_t)n_job);
            rot.resize((size_t)n_job * n_chan);
            theta.resize((size_t)n_job * n_chan);
            const uint64_t all_mask = (1ULL << n_chan) - 1;
            for (int j = 0; j < n_job; ++j) {
                const int b = j % n_prn;
                const double cp0 = (double)comb * (cp_true[(size_t)b] + off_step * (j / n_prn));
                gnss_cuda::DespreadJob J = {(double)comb * spacing,
                                            cp_ref_of(cp0, b),
                                            cps[(size_t)b],
                                            1.0 / cps[(size_t)b],
                                            wc[(size_t)b],
                                            ang0_of(b),
                                            b * L,
                                            L,
                                            all_mask,
                                            d_phiA[(size_t)b],
                                            d_phiB[(size_t)b],
                                            n_chips_full,
                                            m_head_of(cp0, b),
                                            0};
                jf[(size_t)j] = J;
                jn[(size_t)j] = J;
                jn[(size_t)j].phiA = d_zeroA;
                jp[(size_t)j] = J;
                jp[(size_t)j].phiA = (const float2*)d_hA[(size_t)b];
                jp[(size_t)j].phiB = (const float2*)d_hB[(size_t)b];
                jp[(size_t)j].n_chips = prod_n_chips;
                jp[(size_t)j].d_first = prod_d_first;
                gnss_cuda::PsiJob P{};
                P.psi = d_psi[(size_t)b];
                P.psi_prev = nullptr;
                P.sec = nullptr;
                P.P = L;
                P.Ns = 1;
                P.G = 1;
                P.n_prev = 0;
                gnss_cuda::psi_fixed_setup(J, P, fft_len, n_hops);
                pj[(size_t)j] = P;
                pj_rect[(size_t)j] = P;
                pj_rect[(size_t)j].psi = d_psi_rect[(size_t)b];
                pj_h[(size_t)j] = P;
                pj_h[(size_t)j].psi = d_psi_h[(size_t)b];
                // GnssCudaDespread.cpp's epl_rot: +2 pi (f_c - f_car) ds/cps/fs for row 0.
                const double delay_s = J.ds / J.cps / fs;
                const double f_car = J.wc * fs / (2.0 * M_PI);
                for (int c = 0; c < n_chan; ++c) {
                    const double f_c = (double)chans[(size_t)c] * fs / (double)fft_len;
                    rot[(size_t)j * n_chan + c] = (float)(2.0 * M_PI * (f_c - f_car) * delay_s);
                    theta[(size_t)j * n_chan + c] =
                        (float)(gnss_cuda::psi_theta(chans[(size_t)c], fft_len, J.wc) * J.inv_cps
                                / P.G);
                }
            }
        };

    struct DevJobs {
        gnss_cuda::DespreadJob *full = nullptr, *noa = nullptr, *prod = nullptr;
        gnss_cuda::PsiJob *psi = nullptr, *rect = nullptr, *half = nullptr;
        float *rot = nullptr, *theta = nullptr;
        int n_job = 0;
    };
    auto upload_jobs = [&](int n_job, double off_step) {
        std::vector<gnss_cuda::DespreadJob> jf, jn, jp;
        std::vector<gnss_cuda::PsiJob> pj, pr, ph;
        std::vector<float> rot, theta;
        make_jobs(n_job, off_step, jf, jn, jp, pj, pr, ph, rot, theta);
        DevJobs d;
        d.n_job = n_job;
        const size_t jb = (size_t)n_job * sizeof(gnss_cuda::DespreadJob);
        const size_t pb = (size_t)n_job * sizeof(gnss_cuda::PsiJob);
        CK(cudaMalloc(&d.full, jb));
        CK(cudaMalloc(&d.noa, jb));
        CK(cudaMalloc(&d.prod, jb));
        CK(cudaMalloc(&d.psi, pb));
        CK(cudaMalloc(&d.rect, pb));
        CK(cudaMalloc(&d.half, pb));
        CK(cudaMalloc(&d.rot, rot.size() * sizeof(float)));
        CK(cudaMalloc(&d.theta, theta.size() * sizeof(float)));
        CK(cudaMemcpy(d.full, jf.data(), jb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.noa, jn.data(), jb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.prod, jp.data(), jb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.psi, pj.data(), pb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.rect, pr.data(), pb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.half, ph.data(), pb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.rot, rot.data(), rot.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.theta, theta.data(), theta.size() * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    };

    gnss_cuda::DespreadParams p32{};
    p32.n0 = n0;
    p32.fft_len = fft_len;
    p32.n_hops = n_hops;
    p32.Lf = Lf;
    p32.data_stride = n_hops;
    p32.phi_half = 0;
    gnss_cuda::DespreadParams p16 = p32;
    p16.phi_half = 1;

    // ---- the synthetic sky (bank.channels_hoprate: full span, rect model, both images; x3 +
    // noise) for PRN slot b, and the correlation-level observables of a replica set against
    // it: e = |sum_c G_E|^2 / (sum_c E_E)^2 (l likewise), disc = (e-l)/(e+l), prompt =
    // |sum_c G_P|^2 / sum_c E_P, as gnssFleetDll normalises ----
    auto make_data = [&](int b) {
        const auto sky = bank.channels_hoprate(b, window_start, cp_true[(size_t)b], dop[(size_t)b],
                                               n_hops, chans);
        std::vector<std::vector<cf>> data((size_t)n_chan, std::vector<cf>((size_t)n_hops));
        unsigned s2 = 777u + (unsigned)b;
        auto nz = [&s2]() {
            s2 = s2 * 1664525u + 1013904223u;
            return ((double)(s2 >> 8) / (double)(1 << 24)) - 0.5;
        };
        for (int c = 0; c < n_chan; ++c)
            for (int m = 0; m < n_hops; ++m)
                data[(size_t)c][(size_t)m] =
                    sky[(size_t)c][(size_t)m] * 3.0f + cf((float)nz(), (float)nz());
        return data;
    };
    struct Obs {
        double disc, prompt;
    };
    // From per-channel correlations g[t][c] and energy rows.
    auto obs_of = [&](const std::vector<cd>& g, const std::vector<double>& energy, int j) {
        cd gs[3] = {0.0, 0.0, 0.0};
        double e[3] = {0.0, 0.0, 0.0};
        for (int t = 0; t < 3; ++t)
            for (int c = 0; c < n_chan; ++c) {
                gs[t] += g[(size_t)t * n_chan + c];
                e[t] += energy[(size_t)(4 * j + t) * n_chan + c];
            }
        const double ee = std::norm(gs[0]) / (e[0] * e[0]);
        const double ll = std::norm(gs[2]) / (e[2] * e[2]);
        Obs o;
        o.disc = (ee - ll) / (ee + ll);
        o.prompt = std::norm(gs[1]) / e[1];
        return o;
    };
    auto observe_wave = [&](const std::vector<float2>& wave, const std::vector<double>& energy,
                            int j, const std::vector<std::vector<cf>>& data) {
        std::vector<cd> g((size_t)3 * n_chan, cd(0.0, 0.0));
        for (int t = 0; t < 3; ++t)
            for (int c = 0; c < n_chan; ++c) {
                const size_t o = ((size_t)(3 * j + t) * n_chan + c) * n_hops;
                cd gc(0.0, 0.0);
                for (int m = 0; m < n_hops; ++m)
                    gc += cd(data[(size_t)c][(size_t)m])
                          * std::conj(cd(wave[o + m].x, wave[o + m].y));
                g[(size_t)t * n_chan + c] = gc;
            }
        return obs_of(g, energy, j);
    };

    // =========================== validation ===========================
    if (!bench_only && !fused_mode) {
        const int n_job = 2 * n_prn; // true phase, then +0.1 chip
        const DevJobs dj = upload_jobs(n_job, 0.1);
        const size_t wn = (size_t)3 * n_job * n_chan * n_hops, en = (size_t)4 * n_job * n_chan;
        float2* d_wave = nullptr;
        double* d_energy = nullptr;
        CK(cudaMalloc(&d_wave, wn * sizeof(float2)));
        CK(cudaMalloc(&d_energy, en * sizeof(double)));
        struct Set {
            const char* name;
            std::vector<float2> wave;
            std::vector<double> energy;
        };
        auto run = [&](const char* name, std::function<cudaError_t()> launch) {
            Set s;
            s.name = name;
            CK(cudaMemset(d_wave, 0, wn * sizeof(float2)));
            CK(cudaMemset(d_energy, 0, en * sizeof(double)));
            CK(launch());
            CK(cudaDeviceSynchronize());
            s.wave.resize(wn);
            s.energy.resize(en);
            CK(cudaMemcpy(s.wave.data(), d_wave, wn * sizeof(float2), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(s.energy.data(), d_energy, en * sizeof(double), cudaMemcpyDeviceToHost));
            return s;
        };
        const Set FULL = run("FULL", [&] {
            return gnss_cuda::launch_waveform(d_code, dj.full, n_job, n_chan, p32, nullptr, d_wave,
                                              d_energy, 0);
        });
        const Set NOA = run("NOA", [&] {
            return gnss_cuda::launch_waveform(d_code, dj.noa, n_job, n_chan, p32, nullptr, d_wave,
                                              d_energy, 0);
        });
        const Set PROD = run("PROD", [&] {
            return gnss_cuda::launch_waveform(d_code, dj.prod, n_job, n_chan, p16, dj.rot, d_wave,
                                              d_energy, 0);
        });
        const Set PSI = run("PSI", [&] {
            return gnss_cuda::launch_psi_waveform(dj.full, dj.psi, dj.theta, n_job, n_chan, p32,
                                                  nullptr, 0, d_wave, d_energy, 0);
        });
        const Set PSI_ROT = run("PSI-ROT", [&] {
            return gnss_cuda::launch_psi_waveform(dj.full, dj.psi, dj.theta, n_job, n_chan, p32,
                                                  dj.rot, 0, d_wave, d_energy, 0);
        });
        const Set PSI_FIX = run("PSI-FIXED", [&] {
            return gnss_cuda::launch_psi_waveform(dj.full, dj.psi, dj.theta, n_job, n_chan, p32,
                                                  nullptr, 1, d_wave, d_energy, 0);
        });
        const Set PSI_H16 = run("PSI-H16", [&] {
            return gnss_cuda::launch_psi_waveform(dj.full, dj.half, dj.theta, n_job, n_chan, p16,
                                                  nullptr, 0, d_wave, d_energy, 0);
        });
        const Set PSI_RECT = run("PSI-RECT", [&] {
            return gnss_cuda::launch_psi_waveform(dj.full, dj.rect, dj.theta, n_job, n_chan, p32,
                                                  nullptr, 0, d_wave, d_energy, 0);
        });

        // ---- (a) CPU double reference of the Psi formula from the float2 table ----
        std::vector<float2> REF(wn);
        {
            std::vector<gnss_cuda::DespreadJob> jf, jn, jp;
            std::vector<gnss_cuda::PsiJob> pj, pr, ph;
            std::vector<float> rot, theta;
            make_jobs(n_job, 0.1, jf, jn, jp, pj, pr, ph, rot, theta);
            for (int j = 0; j < n_job; ++j) {
                const gnss_cuda::DespreadJob& J = jf[(size_t)j];
                const gnss_cuda::PsiTable& tb = T[(size_t)j % n_prn];
                for (int c = 0; c < n_chan; ++c) {
                    const double th = tb.theta[(size_t)c] * J.inv_cps; // G = 1
                    const float2* psi = tb.psi.data() + (size_t)c * L;
                    for (int m = 0; m < n_hops; ++m) {
                        const double dn = (double)m * fft_len;
                        const double C_P = J.cp_ref + dn * J.cps;
                        const double ang = J.ang0 + std::fmod(J.wc * dn, 2.0 * M_PI);
                        const cd pb = std::conj(std::polar(1.0, ang));
                        for (int t = 0; t < 3; ++t) {
                            const double C = C_P + (t - 1) * J.ds;
                            const double qd = std::floor(C);
                            const double f = C - qd;
                            long long q = (long long)qd % L;
                            if (q < 0)
                                q += L;
                            const double w[4] = {
                                -f * (f - 1) * (f - 2) / 6.0, (f + 1) * (f - 1) * (f - 2) / 2.0,
                                -(f + 1) * f * (f - 2) / 2.0, (f + 1) * f * (f - 1) / 6.0};
                            cd S(0.0, 0.0);
                            for (int jj = 0; jj < 4; ++jj) {
                                long long a = (q + jj - 1) % L;
                                if (a < 0)
                                    a += L;
                                const cd v(psi[a].x, psi[a].y);
                                S += w[jj] * std::polar(1.0, th * (jj - 1)) * v;
                            }
                            S *= std::polar(1.0, -th * f);
                            const cd r = 0.5 * pb * S;
                            REF[((size_t)(3 * j + t) * n_chan + c) * n_hops + m] =
                                make_float2((float)r.real(), (float)r.imag());
                        }
                    }
                }
            }
        }

        printf("\n=== per-sample comparisons: rel rms per channel over %d jobs x %d hops, rows "
               "E/P/L ===\n",
               n_job, n_hops);
        print_stats("PSI vs CPU double reference of the Psi formula (kernel arithmetic check)",
                    compare_sets(PSI.wave, REF, n_job, n_chan, n_hops), chans);
        print_stats("PSI vs NOA (rect model, B image only: the production edge staircase)",
                    compare_sets(PSI.wave, NOA.wave, n_job, n_chan, n_hops), chans);
        print_stats("PSI-RECT (table with snapped edges) vs NOA",
                    compare_sets(PSI_RECT.wave, NOA.wave, n_job, n_chan, n_hops), chans);
        print_stats("PSI vs FULL (adds the far image A)",
                    compare_sets(PSI.wave, FULL.wave, n_job, n_chan, n_hops), chans);
        print_stats("PROD vs FULL (production: fp16, 80 chips, rotated E/L; reference only)",
                    compare_sets(PROD.wave, FULL.wave, n_job, n_chan, n_hops), chans);
        print_stats("PSI-FIXED (int64 phases) vs PSI (fp64 phases)",
                    compare_sets(PSI_FIX.wave, PSI.wave, n_job, n_chan, n_hops), chans);
        {
            double rms = 0.0;
            const double mx = max_abs_diff(PSI_FIX.wave, PSI.wave, &rms);
            printf("   PSI-FIXED vs PSI max abs diff %.3e (sample rms %.3e)\n", mx, rms);
        }
        print_stats("PSI-H16 (fp16 tables) vs PSI",
                    compare_sets(PSI_H16.wave, PSI.wave, n_job, n_chan, n_hops), chans);
        print_stats("PSI-ROT (E/L = rotated prompt) vs PSI (P row must be exact)",
                    compare_sets(PSI_ROT.wave, PSI.wave, n_job, n_chan, n_hops), chans);
        print_stats("PSI-ROT vs PROD (both rotated-prompt E/L)",
                    compare_sets(PSI_ROT.wave, PROD.wave, n_job, n_chan, n_hops), chans);

        printf("\n=== energy rows: relative difference of the per-channel sums over jobs, rows "
               "E/P/L/P_HEAD ===\n");
        auto energy_cmp = [&](const char* title, const Set& A, const Set& B) {
            printf("%s\n   chan        E          P          L     P_HEAD   (max over jobs, P)\n",
                   title);
            for (int c = 0; c < n_chan; ++c) {
                double rel[4];
                double mxp = 0.0;
                for (int t = 0; t < 4; ++t) {
                    double sa = 0.0, sb = 0.0;
                    for (int j = 0; j < n_job; ++j) {
                        const double a = A.energy[(size_t)(4 * j + t) * n_chan + c];
                        const double b = B.energy[(size_t)(4 * j + t) * n_chan + c];
                        sa += a;
                        sb += b;
                        if (t == 1)
                            mxp = std::max(mxp, std::fabs(a - b) / std::max(1e-300, b));
                    }
                    rel[t] = (sa - sb) / std::max(1e-300, sb);
                }
                printf("   %4d  %+9.2e  %+9.2e  %+9.2e  %+9.2e   %9.2e\n", chans[(size_t)c], rel[0],
                       rel[1], rel[2], rel[3], mxp);
            }
        };
        energy_cmp("PSI vs NOA", PSI, NOA);
        energy_cmp("PSI vs FULL", PSI, FULL);
        energy_cmp("PROD vs FULL", PROD, FULL);
        energy_cmp("PSI-FIXED vs PSI", PSI_FIX, PSI);
        energy_cmp("PSI-H16 vs PSI", PSI_H16, PSI);
        {
            // m_head per job, for the record
            printf("   m_head per job:");
            std::vector<gnss_cuda::DespreadJob> jf, jn, jp;
            std::vector<gnss_cuda::PsiJob> pj, pr, ph;
            std::vector<float> rot, theta;
            make_jobs(n_job, 0.1, jf, jn, jp, pj, pr, ph, rot, theta);
            for (int j = 0; j < n_job; ++j)
                printf(" %d", jf[(size_t)j].m_head);
            printf("\n");
        }

        // ---- (b) the synthetic sky: correlation-level test ----
        printf("\n=== synthetic sky (bank.channels_hoprate: full span, rect model, both images; "
               "x3 + noise) correlated against each replica set ===\n");
        printf("   e = |sum_c G_E|^2 / (sum_c E_E)^2 (l likewise), disc = (e-l)/(e+l); "
               "prompt = |sum_c G_P|^2 / sum_c E_P\n");
        auto observe = [&](const Set& S, int j, const std::vector<std::vector<cf>>& data) {
            return observe_wave(S.wave, S.energy, j, data);
        };
        const Set* sets[] = {&FULL, &NOA, &PROD, &PSI, &PSI_ROT, &PSI_FIX, &PSI_H16};
        const int n_sets = 7;
        printf("   job PRN  offset    set        disc          prompt      d_disc(vs FULL)  "
               "d_prompt/prompt   d_disc(vs NOA)  d_prompt/prompt\n");
        // PSI vs NOA is the acceptance: both are the B image alone. FULL (and the sky) carry the
        // rect model's far image A, which moves the discriminator by a few 1e-3 per PRN on its
        // own (NOA vs FULL shows it), so PSI vs FULL is reported, not judged.
        double worst_disc = 0.0, worst_prompt = 0.0, worst_disc_off = 0.0, worst_prompt_off = 0.0;
        double worst_disc_n = 0.0, worst_prompt_n = 0.0, worst_disc_noff = 0.0,
               worst_prompt_noff = 0.0;
        for (int j = 0; j < n_job; ++j) {
            const int b = j % n_prn;
            const std::vector<std::vector<cf>> data = make_data(b);
            Obs ref{0.0, 0.0}, refn{0.0, 0.0};
            for (int s = 0; s < n_sets; ++s) {
                const Obs o = observe(*sets[s], j, data);
                if (s == 0)
                    ref = o;
                if (s == 1)
                    refn = o;
                const double dd = o.disc - ref.disc, dp = (o.prompt - ref.prompt) / ref.prompt;
                const double ddn = o.disc - refn.disc, dpn = (o.prompt - refn.prompt) / refn.prompt;
                printf("   %3d %3d  %+.1f  %-9s  %+.6e  %.6e  %+.3e  %+.3e   %+.3e  %+.3e\n", j,
                       prns[(size_t)b], 0.1 * (j / n_prn), sets[s]->name, o.disc, o.prompt, dd, dp,
                       ddn, dpn);
                if (sets[s] == &PSI) {
                    if (j < n_prn) {
                        worst_disc = std::max(worst_disc, std::fabs(dd));
                        worst_prompt = std::max(worst_prompt, std::fabs(dp));
                        worst_disc_n = std::max(worst_disc_n, std::fabs(ddn));
                        worst_prompt_n = std::max(worst_prompt_n, std::fabs(dpn));
                    } else {
                        worst_disc_off = std::max(worst_disc_off, std::fabs(dd));
                        worst_prompt_off = std::max(worst_prompt_off, std::fabs(dp));
                        worst_disc_noff = std::max(worst_disc_noff, std::fabs(ddn));
                        worst_prompt_noff = std::max(worst_prompt_noff, std::fabs(dpn));
                    }
                }
            }
        }
        printf("   PSI vs NOA  at the true phase: max |d_disc| %.3e, max |d_prompt/prompt| %.3e "
               "%s\n",
               worst_disc_n, worst_prompt_n,
               (worst_disc_n < 1e-3 && worst_prompt_n < 1e-3) ? "OK" : "FAIL");
        printf("   PSI vs NOA  at +0.1 chip:      max |d_disc| %.3e, max |d_prompt/prompt| %.3e\n",
               worst_disc_noff, worst_prompt_noff);
        printf("   PSI vs FULL at the true phase: max |d_disc| %.3e, max |d_prompt/prompt| %.3e "
               "(the sky's far image; NOA vs FULL shows the same)\n",
               worst_disc, worst_prompt);
        printf("   PSI vs FULL at +0.1 chip:      max |d_disc| %.3e, max |d_prompt/prompt| %.3e\n",
               worst_disc_off, worst_prompt_off);

        cudaFree(d_wave);
        cudaFree(d_energy);
    }

    // =========================== fused synthesis + pack ===========================
    if (fused_mode) {
        const bool prof = (mode == "fusedprof");
        // The stage's synth array (cudaGnssInject): [n_hops][frame_chan_stride][lane_pitch]
        // bytes, this chain at lane_base with its covering channels at chan_map.
        const int fcs = 16, lane_pitch = 512, lane_base = 128;
        const std::vector<int> h_chan_map = {2, 4, 6, 8, 10, 12, 14};
        int* d_chan_map = nullptr;
        CK(cudaMalloc(&d_chan_map, (size_t)n_chan * sizeof(int)));
        CK(cudaMemcpy(d_chan_map, h_chan_map.data(), (size_t)n_chan * sizeof(int),
                      cudaMemcpyHostToDevice));
        const size_t synth_len = (size_t)n_hops * fcs * lane_pitch;
        unsigned char *d_synth_a = nullptr, *d_synth_b = nullptr;
        CK(cudaMalloc(&d_synth_a, synth_len));
        CK(cudaMalloc(&d_synth_b, synth_len));
        auto n_lanes_for = [](int lps, int n_slot) {
            const int v = (lps * n_slot + 15) / 16 * 16;
            return v < 64 ? 64 : v; // the chain's num_synth at 12 PRNs is 64
        };
        auto byte_at = [&](const std::vector<unsigned char>& a, int m, int c, int lane) {
            return a[((size_t)m * fcs + h_chan_map[(size_t)c]) * lane_pitch + lane_base + lane];
        };

        // Jobs: specs 0..n_prn-1 at the true phase ("this record"), n_prn..2n_prn-1 at +0.1 chip
        // (a different launch, whose energies stand in for the frame's record-0 rows). Case B
        // clears channel UNCOV of every job's chan_mask and leaves slots 3 and 7 inactive.
        const int n_job = 2 * n_prn;
        const DevJobs dj = upload_jobs(n_job, 0.1);
        const int UNCOV = 3;
        gnss_cuda::DespreadJob* d_jobs_b = nullptr;
        {
            std::vector<gnss_cuda::DespreadJob> jf, jn, jp;
            std::vector<gnss_cuda::PsiJob> pj, pr, ph;
            std::vector<float> rot, theta;
            make_jobs(n_job, 0.1, jf, jn, jp, pj, pr, ph, rot, theta);
            for (auto& J : jf)
                J.chan_mask &= ~(1ULL << UNCOV);
            CK(cudaMalloc(&d_jobs_b, jf.size() * sizeof(gnss_cuda::DespreadJob)));
            CK(cudaMemcpy(d_jobs_b, jf.data(), jf.size() * sizeof(gnss_cuda::DespreadJob),
                          cudaMemcpyHostToDevice));
        }
        std::vector<int> m_head(n_job);
        std::vector<float> rot_host((size_t)n_job * n_chan);
        {
            std::vector<gnss_cuda::DespreadJob> jf, jn, jp;
            std::vector<gnss_cuda::PsiJob> pj, pr, ph;
            std::vector<float> rot, theta;
            make_jobs(n_job, 0.1, jf, jn, jp, pj, pr, ph, rot, theta);
            for (int j = 0; j < n_job; ++j)
                m_head[(size_t)j] = jf[(size_t)j].m_head;
            rot_host = rot;
        }
        const int n_slot = n_prn;
        std::vector<int> s2s_a(n_slot), s2s_b(n_slot);
        for (int q = 0; q < n_slot; ++q)
            s2s_a[(size_t)q] = q;
        {
            int sp = 0;
            for (int q = 0; q < n_slot; ++q)
                s2s_b[(size_t)q] = (q == 3 || q == 7) ? -1 : sp++;
        }
        const std::vector<int> s2s_c((size_t)n_slot, -1);
        int *d_s2s_a = nullptr, *d_s2s_b = nullptr, *d_s2s_c = nullptr;
        CK(cudaMalloc(&d_s2s_a, (size_t)n_slot * sizeof(int)));
        CK(cudaMalloc(&d_s2s_b, (size_t)n_slot * sizeof(int)));
        CK(cudaMalloc(&d_s2s_c, (size_t)n_slot * sizeof(int)));
        CK(cudaMemcpy(d_s2s_a, s2s_a.data(), (size_t)n_slot * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_s2s_b, s2s_b.data(), (size_t)n_slot * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_s2s_c, s2s_c.data(), (size_t)n_slot * sizeof(int), cudaMemcpyHostToDevice));

        const size_t wn = (size_t)3 * n_job * n_chan * n_hops, en = (size_t)4 * n_job * n_chan;
        float2* d_wave = nullptr;
        double *d_energy = nullptr, *d_energy_f = nullptr;
        CK(cudaMalloc(&d_wave, wn * sizeof(float2)));
        CK(cudaMalloc(&d_energy, en * sizeof(double)));
        CK(cudaMalloc(&d_energy_f, en * sizeof(double)));

        struct FusedCase {
            const char* name;
            int lps;
            bool rot, half;
            int fixed;
            bool caseB, other;
            int opts = 0;
            bool none = false; ///< case C: every slot inactive, jobs/pjobs/theta/rot nullptr
        };
        std::vector<unsigned char> A(synth_len), Bv(synth_len);
        std::vector<double> e_ref(en), e_fus(en);
        int n_fail = 0;
        // Reference (launch_psi_waveform -> launch_pack44) and fused (launch_psi_pack) for one
        // configuration; every byte of the synth array and the active specs' energy rows.
        auto run_case = [&](const FusedCase& fc) {
            const gnss_cuda::DespreadParams& pp = fc.half ? p16 : p32;
            const gnss_cuda::PsiJob* pjobs = fc.half ? dj.half : dj.psi;
            const float* rot = fc.rot ? dj.rot : nullptr;
            const gnss_cuda::DespreadJob* jobs = fc.caseB ? d_jobs_b : dj.full;
            const int* s2s = fc.none ? d_s2s_c : (fc.caseB ? d_s2s_b : d_s2s_a);
            const std::vector<int>& s2s_h = fc.none ? s2s_c : (fc.caseB ? s2s_b : s2s_a);
            const int n_lanes = n_lanes_for(fc.lps, n_slot);
            CK(cudaMemset(d_energy, 0, en * sizeof(double)));
            CK(cudaMemset(d_energy_f, 0xff, en * sizeof(double))); // NaN: untouched rows show
            CK(cudaMemset(d_synth_a, 0x5a, synth_len));
            CK(cudaMemset(d_synth_b, 0x5a, synth_len));
            const double* escale = fc.other ? d_energy + (size_t)4 * n_prn * n_chan : d_energy;
            if (fc.none) {
                // A record with no active PRN, as the stage launches it: slot2spec all -1 and
                // the job arrays unwritten (nullptr here, so a read would fault).
                CK(gnss_cuda::launch_pack44(d_wave, d_energy, nullptr, s2s, n_slot, fc.lps, n_chan,
                                            n_hops, d_chan_map, fcs, n_lanes, lane_base, lane_pitch,
                                            d_synth_a, 0));
                CK(gnss_cuda::launch_psi_pack_opts(nullptr, nullptr, nullptr, nullptr, fc.fixed,
                                                   s2s, n_slot, fc.lps, n_chan, pp, d_chan_map, fcs,
                                                   n_lanes, lane_base, lane_pitch, nullptr,
                                                   d_energy_f, d_synth_b, fc.opts, 0));
            } else {
                CK(gnss_cuda::launch_psi_waveform(jobs, pjobs, dj.theta, n_job, n_chan, pp, rot,
                                                  fc.fixed, d_wave, d_energy, 0));
                CK(gnss_cuda::launch_pack44(d_wave, escale, jobs, s2s, n_slot, fc.lps, n_chan,
                                            n_hops, d_chan_map, fcs, n_lanes, lane_base, lane_pitch,
                                            d_synth_a, 0));
                CK(gnss_cuda::launch_psi_pack_opts(
                    jobs, pjobs, dj.theta, rot, fc.fixed, s2s, n_slot, fc.lps, n_chan, pp,
                    d_chan_map, fcs, n_lanes, lane_base, lane_pitch, fc.other ? escale : nullptr,
                    d_energy_f, d_synth_b, fc.opts, 0));
            }
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(A.data(), d_synth_a, synth_len, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(Bv.data(), d_synth_b, synth_len, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(e_ref.data(), d_energy, en * sizeof(double), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(e_fus.data(), d_energy_f, en * sizeof(double), cudaMemcpyDeviceToHost));

            // Bytes: the whole array (a fused write outside the chain's lanes would show as a
            // difference against the reference's untouched 0x5a).
            size_t n_diff = 0, first = synth_len;
            for (size_t i = 0; i < synth_len; ++i)
                if (A[i] != Bv[i]) {
                    if (n_diff == 0)
                        first = i;
                    ++n_diff;
                }
            // 0x88 by region over the chain's bytes: lanes beyond n_slot, inactive slots, the
            // uncovered channel, the head lane past m_head, and the live rest (natural zeros).
            size_t z[5] = {0, 0, 0, 0, 0}, tot[5] = {0, 0, 0, 0, 0};
            for (int m = 0; m < n_hops; ++m)
                for (int c = 0; c < n_chan; ++c)
                    for (int lane = 0; lane < n_lanes; ++lane) {
                        const int slot = lane / fc.lps, tl = lane - slot * fc.lps;
                        const int t = (fc.lps == 4) ? tl : (tl == 0 ? 1 : 3);
                        int region = 4;
                        if (slot >= n_slot)
                            region = 0;
                        else if (s2s_h[(size_t)slot] < 0)
                            region = 1;
                        else if (fc.caseB && c == UNCOV)
                            region = 2;
                        else if (t == 3 && m >= m_head[(size_t)s2s_h[(size_t)slot]])
                            region = 3;
                        ++tot[region];
                        if (byte_at(Bv, m, c, lane) == 0x88)
                            ++z[region];
                    }
            // Energy rows of the active specs.
            int n_spec = 0;
            for (int q = 0; q < n_slot; ++q)
                n_spec = std::max(n_spec, s2s_h[(size_t)q] + 1);
            size_t e_same = 0, e_n = 0;
            double e_rel = 0.0;
            for (int j = 0; j < n_spec; ++j)
                for (int t = 0; t < 4; ++t)
                    for (int c = 0; c < n_chan; ++c) {
                        const size_t i = (size_t)(4 * j + t) * n_chan + c;
                        ++e_n;
                        if (std::memcmp(&e_ref[i], &e_fus[i], sizeof(double)) == 0)
                            ++e_same;
                        else
                            e_rel = std::max(e_rel, std::fabs(e_fus[i] - e_ref[i])
                                                        / std::max(1e-300, std::fabs(e_ref[i])));
                    }
            const bool ok = (n_diff == 0) && (e_same == e_n) && z[0] == tot[0] && z[1] == tot[1]
                            && z[2] == tot[2] && z[3] == tot[3];
            if (!ok)
                ++n_fail;
            printf("   %-44s bytes differ %zu / %zu", fc.name, n_diff, synth_len);
            if (n_diff)
                printf(" (first at byte %zu: hop %zu chan-pos %zu lane-pos %zu, ref 0x%02x fused "
                       "0x%02x)",
                       first, first / ((size_t)fcs * lane_pitch), (first / lane_pitch) % fcs,
                       first % lane_pitch, A[first], Bv[first]);
            printf("; energy rows %zu/%zu bit-identical (max rel %.3e); 0x88: beyond-n_slot "
                   "%zu/%zu inactive %zu/%zu uncovered %zu/%zu head-gated %zu/%zu live %zu/%zu "
                   "%s\n",
                   e_same, e_n, e_rel, z[0], tot[0], z[1], tot[1], z[2], tot[2], z[3], tot[3], z[4],
                   tot[4], ok ? "OK" : "FAIL");
        };

        if (!prof) {
            printf("\n=== fused: launch_psi_pack vs launch_psi_waveform + launch_pack44, synth "
                   "[%d][%d][%d] bytes, lane_base %d, n_lanes 64, chan_map 2,4,..,14; %d slots; "
                   "case B = slots 3,7 inactive + channel %d uncovered; case C = no active PRN "
                   "(slot2spec all -1, job pointers nullptr); 'own' = record-0 scale "
                   "(this launch's energies), 'other' = scale from the +0.1-chip jobs' energies "
                   "===\n",
                   n_hops, fcs, lane_pitch, lane_base, n_slot, chans[(size_t)UNCOV]);
            const FusedCase cases[] = {
                {"lanes 2 rot  fp16 fixed  A own", 2, true, true, 1, false, false},
                {"lanes 2 rot  fp16 fixed  A other", 2, true, true, 1, false, true},
                {"lanes 2 rot  fp16 fixed  B own", 2, true, true, 1, true, false},
                {"lanes 2 rot  fp16 fixed  B other", 2, true, true, 1, true, true},
                {"lanes 2 rot  fp32 fixed  A own", 2, true, false, 1, false, false},
                {"lanes 2 rot  fp32 fixed  B other", 2, true, false, 1, true, true},
                {"lanes 2 rot  fp32 fp64   A own", 2, true, false, 0, false, false},
                {"lanes 2 rot  fp16 fp64   B other", 2, true, true, 0, true, true},
                {"lanes 4 rot  fp16 fixed  A own", 4, true, true, 1, false, false},
                {"lanes 4 rot  fp16 fixed  A other", 4, true, true, 1, false, true},
                {"lanes 4 rot  fp16 fixed  B own", 4, true, true, 1, true, false},
                {"lanes 4 rot  fp16 fixed  B other", 4, true, true, 1, true, true},
                {"lanes 4 rot  fp32 fixed  A own", 4, true, false, 1, false, false},
                {"lanes 4 rot  fp32 fp64   B other", 4, true, false, 0, true, true},
                {"lanes 4 exact fp16 fixed A own", 4, false, true, 1, false, false},
                {"lanes 4 exact fp16 fixed A other", 4, false, true, 1, false, true},
                {"lanes 4 exact fp16 fixed B own", 4, false, true, 1, true, false},
                {"lanes 4 exact fp16 fixed B other", 4, false, true, 1, true, true},
                {"lanes 4 exact fp32 fixed A own", 4, false, false, 1, false, false},
                {"lanes 4 exact fp32 fp64  B other", 4, false, false, 0, true, true},
                {"lanes 2 rot  fp16 fixed  A own   T512", 2, true, true, 1, false, false, 1},
                {"lanes 2 rot  fp16 fixed  A other T512", 2, true, true, 1, false, true, 1},
                {"lanes 2 rot  fp16 fixed  B own   T512", 2, true, true, 1, true, false, 1},
                {"lanes 2 rot  fp16 fixed  B other T512", 2, true, true, 1, true, true, 1},
                {"lanes 2 rot  fp32 fixed  B other T512", 2, true, false, 1, true, true, 1},
                {"lanes 4 rot  fp16 fixed  A own   T512", 4, true, true, 1, false, false, 1},
                {"lanes 4 rot  fp16 fixed  B other T512", 4, true, true, 1, true, true, 1},
                {"lanes 4 exact fp16 fixed A own   T512", 4, false, true, 1, false, false, 1},
                {"lanes 4 exact fp16 fixed B other T512", 4, false, true, 1, true, true, 1},
                {"lanes 2 rot  fp16 fixed  C (no PRN, null jobs)", 2, true, true, 1, false, false,
                 0, true},
                {"lanes 2 rot  fp16 fixed  C T512", 2, true, true, 1, false, false, 1, true},
                {"lanes 4 rot  fp16 fixed  C T512", 4, true, true, 1, false, false, 1, true},
            };
            for (const auto& fc : cases)
                run_case(fc);
            printf("   byte-compare: %d of %zu cases FAIL\n", n_fail,
                   sizeof(cases) / sizeof(cases[0]));

            // ---- (c) sky test on the dequantised lanes ----
            // Each lane is undone with the pack's scale from the record-0 energy rows as
            // GnssN2RecordAssemble does (s = 7/(3 sqrt(e0/n_hops)), the conjugate undone), and
            // under lanes 2 the E/L correlations are the prompt's rotated by the kernel's rot
            // (corr_E = corr_P e^{-i rot}, corr_L = corr_P e^{+i rot}).
            auto observe_lanes = [&](const std::vector<unsigned char>& synth,
                                     const std::vector<double>& energy, int lps, int j,
                                     const std::vector<std::vector<cf>>& data) {
                std::vector<cd> g((size_t)3 * n_chan, cd(0.0, 0.0));
                for (int c = 0; c < n_chan; ++c) {
                    const double e0 = energy[(size_t)(4 * j + 1) * n_chan + c];
                    const double sP = 7.0 / (3.0 * std::sqrt(e0 / n_hops));
                    cd gP(0.0, 0.0);
                    for (int m = 0; m < n_hops; ++m) {
                        const unsigned char bt = byte_at(synth, m, c, lps * j + (lps == 4 ? 1 : 0));
                        const cd w(((double)((bt >> 4) & 0xf) - 8.0) / sP,
                                   -(((double)(bt & 0xf) - 8.0) / sP));
                        gP += cd(data[(size_t)c][(size_t)m]) * std::conj(w);
                    }
                    g[(size_t)1 * n_chan + c] = gP;
                    for (int t = 0; t < 3; t += 2) {
                        if (lps == 2) {
                            const double sgn = (t == 0) ? -1.0 : 1.0;
                            g[(size_t)t * n_chan + c] =
                                gP
                                * std::polar(1.0, sgn * (double)rot_host[(size_t)j * n_chan + c]);
                            continue;
                        }
                        const double et = energy[(size_t)(4 * j + t) * n_chan + c];
                        const double st = 7.0 / (3.0 * std::sqrt(et / n_hops));
                        cd gt(0.0, 0.0);
                        for (int m = 0; m < n_hops; ++m) {
                            const unsigned char bt = byte_at(synth, m, c, lps * j + t);
                            const cd w(((double)((bt >> 4) & 0xf) - 8.0) / st,
                                       -(((double)(bt & 0xf) - 8.0) / st));
                            gt += cd(data[(size_t)c][(size_t)m]) * std::conj(w);
                        }
                        g[(size_t)t * n_chan + c] = gt;
                    }
                }
                return obs_of(g, energy, j);
            };
            struct LaneSet {
                const char* name;
                std::vector<unsigned char> synth;
                std::vector<double> energy;
                int lps;
            };
            struct WaveSet {
                const char* name;
                std::vector<float2> wave;
                std::vector<double> energy;
            };
            auto grab_wave = [&](const char* name, std::function<cudaError_t()> launch) {
                WaveSet w;
                w.name = name;
                CK(cudaMemset(d_energy, 0, en * sizeof(double)));
                CK(launch());
                CK(cudaDeviceSynchronize());
                w.wave.resize(wn);
                w.energy.resize(en);
                CK(cudaMemcpy(w.wave.data(), d_wave, wn * sizeof(float2), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(w.energy.data(), d_energy, en * sizeof(double),
                              cudaMemcpyDeviceToHost));
                return w;
            };
            // The float parents and their quantised children, record-0 scale (own energies).
            const WaveSet PSIW = grab_wave("PSI f16 rot", [&] {
                return gnss_cuda::launch_psi_waveform(dj.full, dj.half, dj.theta, n_job, n_chan,
                                                      p16, dj.rot, 1, d_wave, d_energy, 0);
            });
            const WaveSet PSIX = grab_wave("PSI f16 exact", [&] {
                return gnss_cuda::launch_psi_waveform(dj.full, dj.half, dj.theta, n_job, n_chan,
                                                      p16, nullptr, 1, d_wave, d_energy, 0);
            });
            const WaveSet PRODW = grab_wave("PROD", [&] {
                return gnss_cuda::launch_waveform(d_code, dj.prod, n_job, n_chan, p16, dj.rot,
                                                  d_wave, d_energy, 0);
            });
            auto grab_fused = [&](const char* name, int lps, bool rot_on) {
                LaneSet L;
                L.name = name;
                L.lps = lps;
                CK(cudaMemset(d_energy_f, 0, en * sizeof(double)));
                CK(cudaMemset(d_synth_b, 0x5a, synth_len));
                CK(gnss_cuda::launch_psi_pack(dj.full, dj.half, dj.theta, rot_on ? dj.rot : nullptr,
                                              1, d_s2s_a, n_slot, lps, n_chan, p16, d_chan_map, fcs,
                                              n_lanes_for(lps, n_slot), lane_base, lane_pitch,
                                              nullptr, d_energy_f, d_synth_b, 0));
                CK(cudaDeviceSynchronize());
                L.synth.resize(synth_len);
                L.energy.resize(en);
                CK(cudaMemcpy(L.synth.data(), d_synth_b, synth_len, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(L.energy.data(), d_energy_f, en * sizeof(double),
                              cudaMemcpyDeviceToHost));
                return L;
            };
            auto grab_prodq = [&](const char* name, int lps) {
                // PROD's wave is in d_wave only until the next launch: re-run it, then pack.
                LaneSet L;
                L.name = name;
                L.lps = lps;
                CK(cudaMemset(d_energy, 0, en * sizeof(double)));
                CK(cudaMemset(d_synth_a, 0x5a, synth_len));
                CK(gnss_cuda::launch_waveform(d_code, dj.prod, n_job, n_chan, p16, dj.rot, d_wave,
                                              d_energy, 0));
                CK(gnss_cuda::launch_pack44(d_wave, d_energy, dj.full, d_s2s_a, n_slot, lps, n_chan,
                                            n_hops, d_chan_map, fcs, n_lanes_for(lps, n_slot),
                                            lane_base, lane_pitch, d_synth_a, 0));
                CK(cudaDeviceSynchronize());
                L.synth.resize(synth_len);
                L.energy.resize(en);
                CK(cudaMemcpy(L.synth.data(), d_synth_a, synth_len, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(L.energy.data(), d_energy, en * sizeof(double),
                              cudaMemcpyDeviceToHost));
                return L;
            };
            const LaneSet F2 = grab_fused("FUSED l2 rot", 2, true);
            const LaneSet F4 = grab_fused("FUSED l4 rot", 4, true);
            const LaneSet F4X = grab_fused("FUSED l4 exact", 4, false);
            const LaneSet Q2 = grab_prodq("PRODq l2", 2);
            const LaneSet Q4 = grab_prodq("PRODq l4", 4);

            printf("\n=== sky test on the dequantised lanes (fp16 tables, fixed phases; %d jobs "
                   "at the true phase): disc and prompt per set, d = quantised - its float "
                   "parent ===\n",
                   n_prn);
            printf("   job PRN  set             disc          prompt       d_disc      "
                   "d_prompt/prompt\n");
            struct Worst {
                double disc = 0.0, prompt = 0.0;
            } wF2, wF4, wF4X, wQ2, wQ4;
            auto line = [&](int j, const char* name, const Obs& o, const Obs& parent, Worst* w) {
                const double dd = o.disc - parent.disc;
                const double dp = (o.prompt - parent.prompt) / parent.prompt;
                printf("   %3d %3d  %-14s  %+.6e  %.6e  %+.3e  %+.3e\n", j, prns[(size_t)j % n_prn],
                       name, o.disc, o.prompt, dd, dp);
                if (w) {
                    w->disc = std::max(w->disc, std::fabs(dd));
                    w->prompt = std::max(w->prompt, std::fabs(dp));
                }
            };
            for (int j = 0; j < n_prn; ++j) {
                const std::vector<std::vector<cf>> data = make_data(j);
                const Obs oPSI = observe_wave(PSIW.wave, PSIW.energy, j, data);
                const Obs oPSIX = observe_wave(PSIX.wave, PSIX.energy, j, data);
                const Obs oPROD = observe_wave(PRODW.wave, PRODW.energy, j, data);
                line(j, "PSI float rot", oPSI, oPSI, nullptr);
                line(j, "FUSED l2 rot", observe_lanes(F2.synth, F2.energy, 2, j, data), oPSI, &wF2);
                line(j, "FUSED l4 rot", observe_lanes(F4.synth, F4.energy, 4, j, data), oPSI, &wF4);
                line(j, "PSI float exact", oPSIX, oPSIX, nullptr);
                line(j, "FUSED l4 exact", observe_lanes(F4X.synth, F4X.energy, 4, j, data), oPSIX,
                     &wF4X);
                line(j, "PROD float", oPROD, oPROD, nullptr);
                line(j, "PRODq l2", observe_lanes(Q2.synth, Q2.energy, 2, j, data), oPROD, &wQ2);
                line(j, "PRODq l4", observe_lanes(Q4.synth, Q4.energy, 4, j, data), oPROD, &wQ4);
            }
            printf("   quantisation effect, max over jobs |d_disc| / |d_prompt/prompt|: FUSED l2 "
                   "rot %.3e / %.3e, FUSED l4 rot %.3e / %.3e, FUSED l4 exact %.3e / %.3e, PRODq "
                   "l2 %.3e / %.3e, PRODq l4 %.3e / %.3e\n",
                   wF2.disc, wF2.prompt, wF4.disc, wF4.prompt, wF4X.disc, wF4X.prompt, wQ2.disc,
                   wQ2.prompt, wQ4.disc, wQ4.prompt);
        }

        // ---- (d) bench ----
        cudaFree(d_wave);
        cudaFree(d_energy);
        cudaFree(d_energy_f);
        if (!prof)
            printf("\n=== fused bench: cudaEvent per launch (pairs timed as two launches), one "
                   "stream, warm-up 3, n=20 ===\n");
        for (int ns : {n_prn, 2 * n_prn, 3 * n_prn}) {
            if (prof && ns != n_prn)
                break;
            const DevJobs db = upload_jobs(ns, 0.1);
            const size_t wnb = (size_t)3 * ns * n_chan * n_hops, enb = (size_t)4 * ns * n_chan;
            float2* d_wb = nullptr;
            double *d_eb = nullptr, *d_e0 = nullptr;
            CK(cudaMalloc(&d_wb, wnb * sizeof(float2)));
            CK(cudaMalloc(&d_eb, enb * sizeof(double)));
            CK(cudaMalloc(&d_e0, enb * sizeof(double)));
            std::vector<int> ident(ns), none(ns, -1);
            for (int q = 0; q < ns; ++q)
                ident[(size_t)q] = q;
            int *d_ident = nullptr, *d_none = nullptr;
            CK(cudaMalloc(&d_ident, (size_t)ns * sizeof(int)));
            CK(cudaMalloc(&d_none, (size_t)ns * sizeof(int)));
            CK(cudaMemcpy(d_ident, ident.data(), (size_t)ns * sizeof(int), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_none, none.data(), (size_t)ns * sizeof(int), cudaMemcpyHostToDevice));
            // A record-1 launch: the scale from d_e0 (filled by one record-0 launch first).
            CK(gnss_cuda::launch_psi_pack(db.full, db.half, db.theta, db.rot, 1, d_ident, ns, 2,
                                          n_chan, p16, d_chan_map, fcs, n_lanes_for(2, ns),
                                          lane_base, lane_pitch, nullptr, d_e0, d_synth_b, 0));
            CK(cudaDeviceSynchronize());
            auto fused = [&](const gnss_cuda::PsiJob* pj, const gnss_cuda::DespreadParams& pp,
                             const float* rot, int lps, const int* s2s, const double* es,
                             int opts = 0) {
                return gnss_cuda::launch_psi_pack_opts(
                    db.full, pj, db.theta, rot, 1, s2s, ns, lps, n_chan, pp, d_chan_map, fcs,
                    n_lanes_for(lps, ns), lane_base, lane_pitch, es, d_eb, d_synth_b, opts, 0);
            };
            if (prof) {
                // One launch each, in this order, for ncu -k regex:psi_pack: 1 fp16 l2 rot,
                // 2 fp32 l2 rot, 3 fp16 l4 rot, 4 fp16 l4 exact, 5 fp16 l2 rot all-inactive,
                // 6 fp16 l2 rot record-1 (energy_scale); then regex:pack44: one pack44 l2 after
                // a PSI fp16 rot launch; then regex:psi_waveform: that PSI launch.
                CK(fused(db.half, p16, db.rot, 2, d_ident, nullptr));
                CK(fused(db.psi, p32, db.rot, 2, d_ident, nullptr));
                CK(fused(db.half, p16, db.rot, 4, d_ident, nullptr));
                CK(fused(db.half, p16, nullptr, 4, d_ident, nullptr));
                CK(fused(db.half, p16, db.rot, 2, d_none, nullptr));
                CK(fused(db.half, p16, db.rot, 2, d_ident, d_e0));
                // 7 fp16 l2 rot T512, 8 T512 record-1, 9 lanes 4 rot T512, 10 all-inactive
                // T512 (the sweeper + inactive blocks), 11 fp32 l2 rot T512.
                CK(fused(db.half, p16, db.rot, 2, d_ident, nullptr, 1));
                CK(fused(db.half, p16, db.rot, 2, d_ident, d_e0, 1));
                CK(fused(db.half, p16, db.rot, 4, d_ident, nullptr, 1));
                CK(fused(db.half, p16, db.rot, 2, d_none, nullptr, 1));
                CK(fused(db.psi, p32, db.rot, 2, d_ident, nullptr, 1));
                CK(gnss_cuda::launch_psi_waveform(db.full, db.half, db.theta, ns, n_chan, p16,
                                                  db.rot, 1, d_wb, d_eb, 0));
                CK(gnss_cuda::launch_pack44(d_wb, d_eb, db.full, d_ident, ns, 2, n_chan, n_hops,
                                            d_chan_map, fcs, n_lanes_for(2, ns), lane_base,
                                            lane_pitch, d_synth_b, 0));
                CK(cudaDeviceSynchronize());
                printf("fusedprof: %d slots, launches done\n", ns);
                break;
            }
            printf("-- %d slots x %d channels, %d hops; n_lanes %d (lanes 2) / %d (lanes 4)\n", ns,
                   n_chan, n_hops, n_lanes_for(2, ns), n_lanes_for(4, ns));
            double mn, md;
            time_launch(
                "FUSED fp16 fixed lanes2 rot (rec0)",
                [&] { return fused(db.half, p16, db.rot, 2, d_ident, nullptr); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 fixed lanes2 rot (rec1)",
                [&] { return fused(db.half, p16, db.rot, 2, d_ident, d_e0); }, 20, &mn, &md);
            time_launch(
                "FUSED fp32 fixed lanes2 rot (rec0)",
                [&] { return fused(db.psi, p32, db.rot, 2, d_ident, nullptr); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 fixed lanes4 rot (rec0)",
                [&] { return fused(db.half, p16, db.rot, 4, d_ident, nullptr); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 fixed lanes4 exact (rec0)",
                [&] { return fused(db.half, p16, nullptr, 4, d_ident, nullptr); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 lanes2, all slots inactive",
                [&] { return fused(db.half, p16, db.rot, 2, d_none, nullptr); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 lanes2 rot (rec0) T512",
                [&] { return fused(db.half, p16, db.rot, 2, d_ident, nullptr, 1); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 lanes2 rot (rec1) T512",
                [&] { return fused(db.half, p16, db.rot, 2, d_ident, d_e0, 1); }, 20, &mn, &md);
            time_launch(
                "FUSED fp32 lanes2 rot (rec0) T512",
                [&] { return fused(db.psi, p32, db.rot, 2, d_ident, nullptr, 1); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 lanes4 rot (rec0) T512",
                [&] { return fused(db.half, p16, db.rot, 4, d_ident, nullptr, 1); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 lanes4 rot (rec1) T512",
                [&] { return fused(db.half, p16, db.rot, 4, d_ident, d_e0, 1); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 lanes4 exact (rec0) T512",
                [&] { return fused(db.half, p16, nullptr, 4, d_ident, nullptr, 1); }, 20, &mn, &md);
            time_launch(
                "FUSED fp16 lanes2 all inactive T512",
                [&] { return fused(db.half, p16, db.rot, 2, d_none, nullptr, 1); }, 20, &mn, &md);
            time_launch(
                "PSI fp16 fixed rot + pack44 lanes2",
                [&] {
                    cudaError_t e = gnss_cuda::launch_psi_waveform(
                        db.full, db.half, db.theta, ns, n_chan, p16, db.rot, 1, d_wb, d_eb, 0);
                    if (e != cudaSuccess)
                        return e;
                    return gnss_cuda::launch_pack44(d_wb, d_eb, db.full, d_ident, ns, 2, n_chan,
                                                    n_hops, d_chan_map, fcs, n_lanes_for(2, ns),
                                                    lane_base, lane_pitch, d_synth_b, 0);
                },
                20, &mn, &md);
            time_launch(
                "PROD (fp16, 80 chips, rot) + pack44 lanes2",
                [&] {
                    cudaError_t e = gnss_cuda::launch_waveform(d_code, db.prod, ns, n_chan, p16,
                                                               db.rot, d_wb, d_eb, 0);
                    if (e != cudaSuccess)
                        return e;
                    return gnss_cuda::launch_pack44(d_wb, d_eb, db.full, d_ident, ns, 2, n_chan,
                                                    n_hops, d_chan_map, fcs, n_lanes_for(2, ns),
                                                    lane_base, lane_pitch, d_synth_b, 0);
                },
                20, &mn, &md);
            time_launch(
                "PROD + pack44 lanes4",
                [&] {
                    cudaError_t e = gnss_cuda::launch_waveform(d_code, db.prod, ns, n_chan, p16,
                                                               db.rot, d_wb, d_eb, 0);
                    if (e != cudaSuccess)
                        return e;
                    return gnss_cuda::launch_pack44(d_wb, d_eb, db.full, d_ident, ns, 4, n_chan,
                                                    n_hops, d_chan_map, fcs, n_lanes_for(4, ns),
                                                    lane_base, lane_pitch, d_synth_b, 0);
                },
                20, &mn, &md);
            time_launch(
                "pack44 lanes2 alone",
                [&] {
                    return gnss_cuda::launch_pack44(d_wb, d_eb, db.full, d_ident, ns, 2, n_chan,
                                                    n_hops, d_chan_map, fcs, n_lanes_for(2, ns),
                                                    lane_base, lane_pitch, d_synth_b, 0);
                },
                20, &mn, &md);
            cudaFree(d_wb);
            cudaFree(d_eb);
            cudaFree(d_e0);
            cudaFree(d_ident);
            cudaFree(d_none);
        }
        printf("done\n");
        return n_fail ? 1 : 0;
    }

    // =========================== bench ===========================
    if (!fused_mode)
        printf("\n=== bench: cudaEvent per launch, one stream, warm-up 3, n=20 ===\n");
    for (int n_job : {n_prn, 2 * n_prn, 3 * n_prn}) {
        if (fused_mode)
            break;
        const DevJobs dj = upload_jobs(n_job, 0.1);
        const size_t wn = (size_t)3 * n_job * n_chan * n_hops, en = (size_t)4 * n_job * n_chan;
        float2* d_wave = nullptr;
        double* d_energy = nullptr;
        CK(cudaMalloc(&d_wave, wn * sizeof(float2)));
        CK(cudaMalloc(&d_energy, en * sizeof(double)));
        printf("-- %d jobs x %d channels = %d blocks, %d hops\n", n_job, n_chan, n_job * n_chan,
               n_hops);
        double mn, md;
        if (n_job == n_prn) {
            time_launch(
                "FULL  (fp32, 212 chips, 3 trials)",
                [&] {
                    return gnss_cuda::launch_waveform(d_code, dj.full, n_job, n_chan, p32, nullptr,
                                                      d_wave, d_energy, 0);
                },
                20, &mn, &md);
        }
        time_launch(
            "PROD  (fp16, 80 chips, rot)",
            [&] {
                return gnss_cuda::launch_waveform(d_code, dj.prod, n_job, n_chan, p16, dj.rot,
                                                  d_wave, d_energy, 0);
            },
            20, &mn, &md);
        time_launch(
            "PSI   (fp32 tab, fp64 phase, 3 trials)",
            [&] {
                return gnss_cuda::launch_psi_waveform(dj.full, dj.psi, dj.theta, n_job, n_chan, p32,
                                                      nullptr, 0, d_wave, d_energy, 0);
            },
            20, &mn, &md);
        time_launch(
            "PSI   (fp32 tab, fp64 phase, rot)",
            [&] {
                return gnss_cuda::launch_psi_waveform(dj.full, dj.psi, dj.theta, n_job, n_chan, p32,
                                                      dj.rot, 0, d_wave, d_energy, 0);
            },
            20, &mn, &md);
        time_launch(
            "PSI   (fp32 tab, fixed phase, 3 trials)",
            [&] {
                return gnss_cuda::launch_psi_waveform(dj.full, dj.psi, dj.theta, n_job, n_chan, p32,
                                                      nullptr, 1, d_wave, d_energy, 0);
            },
            20, &mn, &md);
        time_launch(
            "PSI   (fp32 tab, fixed phase, rot)",
            [&] {
                return gnss_cuda::launch_psi_waveform(dj.full, dj.psi, dj.theta, n_job, n_chan, p32,
                                                      dj.rot, 1, d_wave, d_energy, 0);
            },
            20, &mn, &md);
        time_launch(
            "PSI   (fp16 tab, fp64 phase, 3 trials)",
            [&] {
                return gnss_cuda::launch_psi_waveform(dj.full, dj.half, dj.theta, n_job, n_chan,
                                                      p16, nullptr, 0, d_wave, d_energy, 0);
            },
            20, &mn, &md);
        time_launch(
            "PSI   (fp16 tab, fixed phase, rot)",
            [&] {
                return gnss_cuda::launch_psi_waveform(dj.full, dj.half, dj.theta, n_job, n_chan,
                                                      p16, dj.rot, 1, d_wave, d_energy, 0);
            },
            20, &mn, &md);
        cudaFree(d_wave);
        cudaFree(d_energy);
    }
    printf("done\n");
    return 0;
}
