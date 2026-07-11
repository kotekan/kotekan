// Standalone CPU-vs-GPU validation for the fused GNSS despread kernel (G1a gate).
//
// Builds the CPU reference with the real pipeline code (ChannelizedReplicaBank::hoprate_filter/
// hoprate_stream + gnss::channelized_despread), synthesizes a data window (replica-as-signal at a
// known cp/doppler + deterministic pseudo-noise), then runs gnss_cuda::launch_despread on the same
// inputs and compares per-channel correlation + energy. Pass: <=1e-5 relative on every output.
//
//   ./cuda_gnss_despread_test            (prints PASS/FAIL per trial + max relative error)

#include "cudaGnssDespreadKernel.hpp"
#include "GnssCudaDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssChannelizedDespread.hpp"
#include "gnssSignal.hpp"
#include "gpsCACode.hpp"
#include "pfbPrototype.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using cf = std::complex<float>;

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,           \
                    __LINE__);                                                                     \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

int main() {
    // L1 C/A at the GX10 front-end geometry (5 MSPS real, N=20 channels, 4-tap hamming PFB).
    const double fs = 5.0e6, f_off = 1.25e6;
    const int N = 20, taps = 4;
    const int prn = 19;
    const double dop = -2900.0, cp_true = 517.25;
    const long long window_start = 40LL * 125 * 3600; // an "aged" absolute anchor (~1 hr of hops)

    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L1CA");
    gnss::ChannelizedReplicaBank bank(*sig, fs, f_off, N, taps, dsp::window_from_string("hamming"),
                                      {prn});
    const int n_hops = bank.repl_period_hops(); // 125 @ L1/5MSPS
    const int fft_len = 2 * N;

    // Covering channels for this Doppler.
    const auto cover = bank.covering_bins(dop, 250000.0);
    const int n_chan = (int)cover.size();

    // "Sky": the replica itself at (cp_true, dop) + deterministic noise, on the covering channels.
    const auto sky = bank.channels_hoprate(0, window_start, cp_true, dop, n_hops, cover);
    std::vector<std::vector<cf>> data_ch(n_chan, std::vector<cf>(n_hops));
    unsigned seed = 1234567u;
    auto frand = [&seed]() { // deterministic LCG noise, +-0.5
        seed = seed * 1664525u + 1013904223u;
        return ((double)(seed >> 8) / (double)(1 << 24)) - 0.5;
    };
    for (int c = 0; c < n_chan; ++c)
        for (int m = 0; m < n_hops; ++m)
            data_ch[c][m] = sky[c][m] * 3.0f + cf((float)frand(), (float)frand());

    // Trials: E/P/L around cp_true plus an off-peak control (4 jobs, one PRN).
    const double spacing = 0.5;
    const std::vector<double> cps_trials = {cp_true - spacing, cp_true, cp_true + spacing,
                                            cp_true + 137.0};

    // --- CPU reference: hoprate replica per trial + channelized_despread ---
    const auto filt = bank.hoprate_filter(cover, dop);
    std::vector<std::complex<double>> cpu_corr;  // [trial][chan]
    std::vector<double> cpu_energy;
    for (double cp : cps_trials) {
        const auto repl = bank.hoprate_stream(filt, 0, window_start, cp, dop, n_hops);
        for (int c = 0; c < n_chan; ++c) {
            std::complex<double> g(0.0, 0.0);
            double e = 0.0;
            for (int m = 0; m < n_hops; ++m) {
                g += std::complex<double>(data_ch[c][m]) * std::conj(std::complex<double>(repl[c][m]));
                e += std::norm(std::complex<double>(repl[c][m]));
            }
            cpu_corr.push_back(g);
            cpu_energy.push_back(e);
        }
    }

    // --- GPU: same inputs through the fused kernel ---
    const int Lf = fft_len * taps;
    const int n_batch = (int)cps_trials.size();
    const double cps = sig->chip_rate_hz / fs * (1.0 + dop / sig->carrier_hz); // comb_mult=1 (L1CA)
    const double wc = 2.0 * M_PI * (f_off + dop) / fs;

    auto code = gps::generate_ca_code(prn); // +-1, 1023 chips
    std::vector<int8_t> code8(code.begin(), code.end());

    std::vector<double2> phiA((size_t)n_chan * (Lf + 1)), phiB((size_t)n_chan * (Lf + 1));
    for (int c = 0; c < n_chan; ++c)
        for (int k = 0; k <= Lf; ++k) {
            phiA[(size_t)c * (Lf + 1) + k] =
                make_double2(filt.PhiA[c][k].real(), filt.PhiA[c][k].imag());
            phiB[(size_t)c * (Lf + 1) + k] =
                make_double2(filt.PhiB[c][k].real(), filt.PhiB[c][k].imag());
        }
    std::vector<float2> dataf((size_t)n_chan * n_hops);
    for (int c = 0; c < n_chan; ++c)
        for (int m = 0; m < n_hops; ++m)
            dataf[(size_t)c * n_hops + m] = make_float2(data_ch[c][m].real(), data_ch[c][m].imag());
    std::vector<gnss_cuda::DespreadJob> jobs(n_batch);
    const uint64_t all_mask = (n_chan >= 64) ? ~0ULL : ((1ULL << n_chan) - 1);
    // (d_phiA/d_phiB are set below once allocated; per-job pointers = the batch API shape)

    gnss_cuda::DespreadParams p;
    p.n0 = window_start + fft_len - 1; // hoprate_stream's per-hop reference sample
    p.fft_len = fft_len;
    p.n_hops = n_hops;
    p.Lf = Lf;

    float2* d_data;
    int8_t* d_code;
    double2 *d_phiA, *d_phiB, *d_corr;
    gnss_cuda::DespreadJob* d_jobs;
    double* d_energy;
    CK(cudaMalloc(&d_data, dataf.size() * sizeof(float2)));
    CK(cudaMalloc(&d_code, code8.size()));
    CK(cudaMalloc(&d_phiA, phiA.size() * sizeof(double2)));
    CK(cudaMalloc(&d_phiB, phiB.size() * sizeof(double2)));
    CK(cudaMalloc(&d_jobs, jobs.size() * sizeof(gnss_cuda::DespreadJob)));
    CK(cudaMalloc(&d_corr, (size_t)n_batch * n_chan * sizeof(double2)));
    CK(cudaMalloc(&d_energy, (size_t)n_batch * n_chan * sizeof(double)));
    CK(cudaMemcpy(d_data, dataf.data(), dataf.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_code, code8.data(), code8.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_phiA, phiA.data(), phiA.size() * sizeof(double2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_phiB, phiB.data(), phiB.size() * sizeof(double2), cudaMemcpyHostToDevice));
    for (int b = 0; b < n_batch; ++b)
        jobs[b] = {cps_trials[b], cps, wc, 0, (int)code8.size(), all_mask,
                   d_phiA, d_phiB, filt.n_chips};
    CK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(gnss_cuda::DespreadJob),
                  cudaMemcpyHostToDevice));

    CK(gnss_cuda::launch_despread(d_data, d_code, d_jobs, n_batch, n_chan, p, d_corr, d_energy,
                                  0));
    CK(cudaDeviceSynchronize());

    std::vector<double2> g_corr((size_t)n_batch * n_chan);
    std::vector<double> g_energy((size_t)n_batch * n_chan);
    CK(cudaMemcpy(g_corr.data(), d_corr, g_corr.size() * sizeof(double2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g_energy.data(), d_energy, g_energy.size() * sizeof(double),
                  cudaMemcpyDeviceToHost));

    // --- Compare ---
    double max_rel = 0.0;
    bool pass = true;
    for (int b = 0; b < n_batch; ++b) {
        std::complex<double> cc(0, 0), gc(0, 0);
        double ce = 0, ge = 0;
        for (int c = 0; c < n_chan; ++c) {
            cc += cpu_corr[(size_t)b * n_chan + c];
            gc += std::complex<double>(g_corr[(size_t)b * n_chan + c].x,
                                       g_corr[(size_t)b * n_chan + c].y);
            ce += cpu_energy[(size_t)b * n_chan + c];
            ge += g_energy[(size_t)b * n_chan + c];
        }
        const double rc = std::abs(gc - cc) / std::max(1e-30, std::abs(cc));
        const double re = std::fabs(ge - ce) / std::max(1e-30, ce);
        max_rel = std::max({max_rel, rc, re});
        const bool ok = rc < 1e-5 && re < 1e-5;
        pass = pass && ok;
        printf("trial cp=%9.2f: CPU |G|=%12.4f E=%12.1f | GPU |G|=%12.4f E=%12.1f | rel %.2e/%.2e %s\n",
               cps_trials[b], std::abs(cc), ce, std::abs(gc), ge, rc, re, ok ? "OK" : "FAIL");
    }
    printf("%s (max relative error %.3e; %d trials x %d channels x %d hops)\n",
           pass ? "PASS" : "FAIL", max_rel, n_batch, n_chan, n_hops);

    // --- Throughput bench: the REAL per-record path (GnssCudaDespread::despread_batch --
    // upload_window + one batched launch + one sync per record), full-constellation batch.
    // Reports records/s and the realtime margin vs the tracker's 1 kHz record cadence.
    for (int bench_prn : {12, 32}) {
        std::vector<int> prns;
        for (int q = 1; q <= bench_prn; ++q)
            prns.push_back(q);
        gnss::ChannelizedReplicaBank bbank(*sig, fs, f_off, N, taps,
                                           dsp::window_from_string("hamming"), prns);
        GnssCudaDespread gpu(bbank, bench_prn, N, 0, n_hops, fs, f_off);
        std::vector<cf> win((size_t)n_hops * N);
        for (auto& v : win)
            v = cf((float)frand(), (float)frand());
        std::vector<GnssCudaDespread::Spec> specs(bench_prn);
        std::vector<int> allch(N);
        for (int c = 0; c < N; ++c)
            allch[c] = c;
        for (int q = 0; q < bench_prn; ++q)
            specs[q] = {q, 100.0 + 7.0 * q, 0.5, -4000.0 + 250.0 * q, allch};
        gpu.upload_window(win.data(), window_start);
        (void)gpu.despread_batch(specs); // warm-up (Phi builds + JIT)
        const int NREC = 2000;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < NREC; ++r) {
            gpu.upload_window(win.data(), window_start + (long long)r * n_hops * fft_len);
            (void)gpu.despread_batch(specs);
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                              .count();
        printf("BENCH %2d PRN x E/P/L x %d chan x %d hops: %6.0f rec/s (%.1fx realtime @1kHz, "
               "%.0f PRN-despreads/s)\n",
               bench_prn, N, n_hops, NREC / dt, NREC / dt / 1000.0, 3.0 * bench_prn * NREC / dt);
    }
    return pass ? 0 : 1;
}
