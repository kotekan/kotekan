/**
 * @file
 * @brief THE KERNEL GATE for fp16 Phi tables (docs/CHORD_GPU_TODO.md item 3).
 *
 * phibits answered the STORAGE error (3.3e-4 worst case on a chip step); wavebench answered
 * the TIMING (1.27-1.37x). Neither drives the shipped engine, and #71 is the monument to why
 * that is not enough. So this drives GnssCudaDespread through enqueue_waveform -- THE call
 * cudaGnssInject makes in production -- twice:
 *
 *   fp32   set_phi_fp16(false) -- today's path, float2 tables
 *   fp16   set_phi_fp16(true)  -- __half2 tables, half the resident bytes
 *
 * and compares the engine's own outputs (wave + energy), plus a host-side correlation of both
 * waves against one shared random voltage so the number that matters downstream -- the
 * CORRELATION error the tracker's disc actually eats -- is measured, not inferred.
 *
 * It also asserts the guardrails that make the change safe rather than merely accurate:
 *   - fp32 flag-off is DETERMINISTIC (two runs byte-equal) and round-trips through an
 *     arm/disarm cycle byte-for-byte;
 *   - despread_batch THROWS while fp16 is armed (launch_despread reads raw float2);
 *   - set_shared_phi is REFUSED while fp16 is armed (that gather is fp32-only).
 *
 * Usage: ./phi16gpu [n_prn]      (run on an IDLE A40 node for the timing to mean anything)
 */
#include "GnssCudaDespread.hpp"
#include "cudaGnssDespreadKernel.hpp" // sizeof(DespreadJob) for the device arena
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { printf("CUDA %s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

int main(int argc, char** argv) {
    const int n_prn = (argc > 1) ? atoi(argv[1]) : 11; // production-class job count
    const double FS = 3.2e9, F_OFF = 1176.45e6;
    // SPECTRUM length (r2c), not fft_len -- see phisharegpu's warning. Production geometry:
    // 2048-hop records so the timing below is the node's own shape, not a toy.
    const int SPEC_LEN = 8192, NTAPS = 4, N_HOPS = 2048;

    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    if (!sig) { printf("no GPS_L5_Q\n"); return 2; }
    std::vector<int> prns;
    for (int i = 0; i < n_prn; ++i) prns.push_back(1 + i);
    gnss::ChannelizedReplicaBank bank(*sig, FS, F_OFF, SPEC_LEN, NTAPS, dsp::Window::Hamming,
                                      prns);
    const std::vector<int> chans{5972, 5988, 6004, 6020, 6036, 6052, 6068};
    const int n_chan = (int)chans.size();

    GnssCudaDespread eng(bank, n_prn, chans, N_HOPS, FS, F_OFF, /*refresh_hz=*/1e9);

    const double dops[] = {0.0, 137.0, -900.0, 2500.0, -5000.0, 4321.0, -137.0, 3000.0};
    std::vector<GnssCudaDespread::Spec> specs;
    for (int i = 0; i < n_prn; ++i) {
        GnssCudaDespread::Spec sp;
        sp.p = i;
        sp.doppler_hz = dops[i % 8];
        sp.cp_seed = 100.0 + 37.0 * i;
        sp.spacing_chips = 0.5;
        sp.ctrim_hz = 0.0;
        sp.covering = {0, 1, 2, 3, 4, 5, 6};
        specs.push_back(sp);
    }

    // Device arena, exactly the shipped call's shape.
    const size_t nwave = (size_t)3 * n_prn * n_chan * N_HOPS;
    const size_t nen = (size_t)4 * n_prn * n_chan;
    void *d_jobs, *d_wave, *d_energy;
    ck(cudaMalloc(&d_jobs, (size_t)n_prn * sizeof(gnss_cuda::DespreadJob)), "jobs");
    ck(cudaMalloc(&d_wave, nwave * sizeof(float2)), "wave");
    ck(cudaMalloc(&d_energy, nen * sizeof(double)), "energy");

    auto run = [&](std::vector<std::complex<float>>& wave, std::vector<double>& en) {
        eng.enqueue_waveform(0, specs, d_jobs, d_wave, d_energy, nullptr);
        wave.resize(nwave);
        en.resize(nen);
        ck(cudaMemcpy(wave.data(), d_wave, nwave * sizeof(float2), cudaMemcpyDeviceToHost), "dw");
        ck(cudaMemcpy(en.data(), d_energy, nen * sizeof(double), cudaMemcpyDeviceToHost), "de");
    };

    printf("phi16gpu: %d PRNs x %d channels x %d hops (production record shape)\n", n_prn,
           n_chan, N_HOPS);
    std::vector<std::complex<float>> w32, w32b, w16;
    std::vector<double> e32, e32b, e16;
    eng.set_phi_fp16(false);
    run(w32, e32);
    run(w32b, e32b); // determinism: same call twice must be byte-equal
    const bool det = !std::memcmp(w32.data(), w32b.data(), nwave * sizeof(float2))
                     && !std::memcmp(e32.data(), e32b.data(), nen * sizeof(double));
    printf("  fp32 determinism (two runs byte-equal): %s\n", det ? "OK" : "FAIL");

    const bool took = eng.set_phi_fp16(true);
    printf("  set_phi_fp16(true) -> %s\n", took ? "IN EFFECT" : "REFUSED");
    if (!took) { printf("  cannot gate what did not arm\n"); return 2; }

    // Guardrails while armed.
    bool threw = false;
    try { (void)eng.despread_batch(specs); } catch (const std::exception&) { threw = true; }
    printf("  despread_batch under fp16: %s\n", threw ? "THREW (correct)" : "RAN <-- FAIL");
    const bool sh = eng.set_shared_phi(true);
    printf("  set_shared_phi under fp16: %s\n", sh ? "TOOK <-- FAIL" : "REFUSED (correct)");

    run(w16, e16);

    // One shared random voltage; correlate BOTH waves against it on the host. The correlation
    // is what the tracker integrates, so its relative error is the number the disc eats.
    std::mt19937 rng(9876);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<std::complex<float>> data((size_t)n_chan * N_HOPS);
    for (auto& v : data) v = std::complex<float>(g(rng), g(rng));

    printf("\n  %-4s %9s   %11s %11s %11s\n", "PRN", "doppler", "wave rel", "energy rel",
           "corr rel");
    int bad = 0;
    double worst_w = 0, worst_e = 0, worst_c = 0;
    for (int i = 0; i < n_prn; ++i) {
        double wmax = 0, dwmax = 0, erel = 0, crel = 0;
        for (int t = 0; t < 3; ++t) {
            std::complex<double> c32(0, 0), c16(0, 0);
            for (int c = 0; c < n_chan; ++c) {
                const size_t row = ((size_t)(3 * i + t) * n_chan + c) * N_HOPS;
                for (int m = 0; m < N_HOPS; ++m) {
                    const auto a = w32[row + m], b = w16[row + m];
                    wmax = std::max(wmax, (double)std::abs(a));
                    dwmax = std::max(dwmax, (double)std::abs(b - a));
                    const std::complex<double> d(data[(size_t)c * N_HOPS + m]);
                    c32 += d * std::conj(std::complex<double>(a));
                    c16 += d * std::conj(std::complex<double>(b));
                }
            }
            const size_t er = (size_t)(4 * i + t) * n_chan;
            for (int c = 0; c < n_chan; ++c)
                if (e32[er + c] > 0)
                    erel = std::max(erel, std::fabs(e16[er + c] - e32[er + c]) / e32[er + c]);
            if (std::abs(c32) > 0)
                crel = std::max(crel, std::abs(c16 - c32) / std::abs(c32));
        }
        const double wrel = wmax > 0 ? dwmax / wmax : 0;
        // Bars: wave/energy at 2e-3 (storage is 3.3e-4; the gather sums ~52 signed chip terms,
        // so a few-x accumulation over storage is expected and anything past ~6x is a bug).
        // corr on RANDOM data is a noise-noise quotient and can only be looser; 1e-2 catches a
        // wrong-table/wrong-branch failure (those read ~1e0) without flagging statistics.
        const bool fail = wrel > 2e-3 || erel > 2e-3 || crel > 1e-2;
        if (fail) bad++;
        worst_w = std::max(worst_w, wrel);
        worst_e = std::max(worst_e, erel);
        worst_c = std::max(worst_c, crel);
        printf("  %-4d %+9.0f   %11.3e %11.3e %11.3e%s\n", prns[(size_t)i],
               specs[(size_t)i].doppler_hz, wrel, erel, crel, fail ? "   <-- FAIL" : "");
    }
    printf("  worst: wave %.3e  energy %.3e  corr %.3e  (storage floor 3.3e-4)\n", worst_w,
           worst_e, worst_c);

    // Round-trip: disarm and the fp32 answer must come back byte-for-byte.
    eng.set_phi_fp16(false);
    std::vector<std::complex<float>> w32c;
    std::vector<double> e32c;
    run(w32c, e32c);
    const bool rt = !std::memcmp(w32.data(), w32c.data(), nwave * sizeof(float2))
                    && !std::memcmp(e32.data(), e32c.data(), nen * sizeof(double));
    printf("  fp32 after disarm byte-equal to before: %s\n", rt ? "OK" : "FAIL");

    // ---- TIMING (synthesis kernel via the shipped call; run on an idle A40) --------------
    {
        const int REP = 40, WARM = 6;
        auto bench = [&](bool half) {
            eng.set_phi_fp16(half);
            for (int r = 0; r < WARM; ++r)
                eng.enqueue_waveform(0, specs, d_jobs, d_wave, d_energy, nullptr);
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < REP; ++r)
                eng.enqueue_waveform(0, specs, d_jobs, d_wave, d_energy, nullptr);
            cudaDeviceSynchronize();
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / REP;
        };
        const double t32 = bench(false), t16 = bench(true);
        printf("\n  TIMING (%d jobs, enqueue_waveform):  fp32 %.3f ms   fp16 %.3f ms   %.2fx\n",
               n_prn, t32, t16, t32 / t16);
    }

    const bool ok = det && threw && !sh && rt && !bad;
    printf("\n  %s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
