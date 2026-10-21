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
//   ./cuda_gnss_psi_bench [all|bench] [n_prn]
//
// "bench" skips the CPU references (for ncu). Never run beside a live node: the bench
// saturates the GPU.

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

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    const bool bench_only = (mode == "bench");
    const int n_prn = argc > 2 ? atoi(argv[2]) : 12;

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

    // =========================== validation ===========================
    if (!bench_only) {
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
        struct Obs {
            double disc, prompt;
        };
        auto observe = [&](const Set& S, int j, const std::vector<std::vector<cf>>& data) {
            cd g[3] = {0.0, 0.0, 0.0};
            double e[3] = {0.0, 0.0, 0.0};
            for (int t = 0; t < 3; ++t)
                for (int c = 0; c < n_chan; ++c) {
                    cd gc(0.0, 0.0);
                    const size_t o = ((size_t)(3 * j + t) * n_chan + c) * n_hops;
                    for (int m = 0; m < n_hops; ++m)
                        gc += cd(data[(size_t)c][(size_t)m])
                              * std::conj(cd(S.wave[o + m].x, S.wave[o + m].y));
                    g[t] += gc;
                    e[t] += S.energy[(size_t)(4 * j + t) * n_chan + c];
                }
            const double ee = std::norm(g[0]) / (e[0] * e[0]);
            const double ll = std::norm(g[2]) / (e[2] * e[2]);
            Obs o;
            o.disc = (ee - ll) / (ee + ll);
            o.prompt = std::norm(g[1]) / e[1];
            return o;
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
            const auto sky = bank.channels_hoprate(b, window_start, cp_true[(size_t)b],
                                                   dop[(size_t)b], n_hops, chans);
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

    // =========================== bench ===========================
    printf("\n=== bench: cudaEvent per launch, one stream, warm-up 3, n=20 ===\n");
    for (int n_job : {n_prn, 2 * n_prn, 3 * n_prn}) {
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
