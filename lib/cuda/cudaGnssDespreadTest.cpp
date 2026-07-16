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

    // Two fused jobs, each emitting the full quad (E, P, L, P_HEAD): one at the true code phase,
    // one an off-peak control. m_head cuts P_HEAD mid-record (40 of 125 hops) on the first job and
    // disables it (0 -> an all-zero row) on the second, so both branches are covered.
    const double spacing = 0.5;
    struct FusedTrial {
        double cp_prompt;
        int m_head;
    };
    const std::vector<FusedTrial> ftrials = {{cp_true, 40}, {cp_true + 137.0, 0}};
    const char* row_name[4] = {"E", "P", "L", "HEAD"};
    // Row -> (code phase, hop range) the CPU must reproduce; must mirror the kernel's contract.
    auto row_cp = [&](const FusedTrial& ft, int t) {
        return ft.cp_prompt + (t == 3 ? 0.0 : (t - 1) * spacing);
    };
    auto row_hops = [&](const FusedTrial& ft, int t) { return (t == 3) ? ft.m_head : n_hops; };

    // --- CPU reference: hoprate replica per row + channelized_despread ---
    const auto filt = bank.hoprate_filter(cover, dop);
    std::vector<std::complex<double>> cpu_corr; // [row][chan]
    std::vector<double> cpu_energy;
    for (const auto& ft : ftrials)
        for (int t = 0; t < 4; ++t) {
            const auto repl = bank.hoprate_stream(filt, 0, window_start, row_cp(ft, t), dop, n_hops);
            const int hi = row_hops(ft, t);
            for (int c = 0; c < n_chan; ++c) {
                std::complex<double> g(0.0, 0.0);
                double e = 0.0;
                for (int m = 0; m < hi; ++m) {
                    g += std::complex<double>(data_ch[c][m])
                         * std::conj(std::complex<double>(repl[c][m]));
                    e += std::norm(std::complex<double>(repl[c][m]));
                }
                cpu_corr.push_back(g);
                cpu_energy.push_back(e);
            }
        }

    // --- GPU: same inputs through the fused kernel ---
    const int Lf = fft_len * taps;
    const int n_spec = (int)ftrials.size(); // kernel jobs
    const int n_out = 4 * n_spec;           // output rows (E, P, L, P_HEAD per job)
    const double cps = sig->chip_rate_hz / fs * (1.0 + dop / sig->carrier_hz); // comb_mult=1 (L1CA)
    const double wc = 2.0 * M_PI * (f_off + dop) / fs;

    auto code = gps::generate_ca_code(prn); // +-1, 1023 chips
    std::vector<int8_t> code8(code.begin(), code.end());

    std::vector<float2> phiA((size_t)n_chan * (Lf + 1)), phiB((size_t)n_chan * (Lf + 1));
    for (int c = 0; c < n_chan; ++c)
        for (int k = 0; k <= Lf; ++k) {
            phiA[(size_t)c * (Lf + 1) + k] =
                make_float2((float)filt.PhiA[c][k].real(), (float)filt.PhiA[c][k].imag());
            phiB[(size_t)c * (Lf + 1) + k] =
                make_float2((float)filt.PhiB[c][k].real(), (float)filt.PhiB[c][k].imag());
        }
    std::vector<float2> dataf((size_t)n_chan * n_hops);
    for (int c = 0; c < n_chan; ++c)
        for (int m = 0; m < n_hops; ++m)
            dataf[(size_t)c * n_hops + m] = make_float2(data_ch[c][m].real(), data_ch[c][m].imag());
    std::vector<gnss_cuda::DespreadJob> jobs(n_spec);
    const uint64_t all_mask = (n_chan >= 64) ? ~0ULL : ((1ULL << n_chan) - 1);
    // (d_phiA/d_phiB are set below once allocated; per-job pointers = the batch API shape)

    gnss_cuda::DespreadParams p;
    p.n0 = window_start + fft_len - 1; // hoprate_stream's per-hop reference sample
    p.fft_len = fft_len;
    p.n_hops = n_hops;
    p.Lf = Lf;
    p.data_stride = n_hops;

    float2* d_data;
    int8_t* d_code;
    float2 *d_phiA, *d_phiB;
    double2* d_corr;
    gnss_cuda::DespreadJob* d_jobs;
    double* d_energy;
    CK(cudaMalloc(&d_data, dataf.size() * sizeof(float2)));
    CK(cudaMalloc(&d_code, code8.size()));
    CK(cudaMalloc(&d_phiA, phiA.size() * sizeof(float2)));
    CK(cudaMalloc(&d_phiB, phiB.size() * sizeof(float2)));
    CK(cudaMalloc(&d_jobs, jobs.size() * sizeof(gnss_cuda::DespreadJob)));
    CK(cudaMalloc(&d_corr, (size_t)n_out * n_chan * sizeof(double2)));
    CK(cudaMalloc(&d_energy, (size_t)n_out * n_chan * sizeof(double)));
    CK(cudaMemcpy(d_data, dataf.data(), dataf.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_code, code8.data(), code8.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_phiA, phiA.data(), phiA.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_phiB, phiB.data(), phiB.size() * sizeof(float2), cudaMemcpyHostToDevice));
    for (int b = 0; b < n_spec; ++b) // comb_mult = 1 for L1CA, so ds = spacing directly
        jobs[b] = {ftrials[b].cp_prompt, spacing,   cps,           1.0 / cps,
                   wc,                   0,         (int)code8.size(), all_mask,
                   d_phiA,               d_phiB,    filt.n_chips,  ftrials[b].m_head};
    CK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(gnss_cuda::DespreadJob),
                  cudaMemcpyHostToDevice));

    CK(gnss_cuda::launch_despread(d_data, d_code, d_jobs, n_spec, n_chan, p, d_corr, d_energy, 0));
    CK(cudaDeviceSynchronize());

    std::vector<double2> g_corr((size_t)n_out * n_chan);
    std::vector<double> g_energy((size_t)n_out * n_chan);
    CK(cudaMemcpy(g_corr.data(), d_corr, g_corr.size() * sizeof(double2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g_energy.data(), d_energy, g_energy.size() * sizeof(double),
                  cudaMemcpyDeviceToHost));

    // --- Compare ---
    double max_rel = 0.0;
    bool pass = true;
    for (int b = 0; b < n_out; ++b) {
        const FusedTrial& ft = ftrials[b / 4];
        const int t = b % 4;
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
        printf("%-4s cp=%9.2f hops=%4d: CPU |G|=%12.4f E=%12.1f | GPU |G|=%12.4f E=%12.1f | "
               "rel %.2e/%.2e %s\n",
               row_name[t], row_cp(ft, t), row_hops(ft, t), std::abs(cc), ce, std::abs(gc), ge, rc,
               re, ok ? "OK" : "FAIL");
    }
    printf("%s (max relative error %.3e; %d jobs x 4 rows x %d channels x %d hops)\n",
           pass ? "PASS" : "FAIL", max_rel, n_spec, n_chan, n_hops);

    // --- CHORD 4+4b codec + quantized despread path ---
    // TWO separate claims, tested separately:
    //  (1) our ENCODER matches kotekan's reference DECODE. In CHORD the PFB + 4b quantizer live in
    //      the RFSoC F-engine, not here, so the only in-tree spec is the consumer:
    //      lib/testing/gpuSimulate.cpp does `(b & 0x0f) - 8` for imag, `((b & 0xf0) >> 4) - 8` for
    //      real. Re-implement that decode INDEPENDENTLY (not by calling unpack_44) and require it
    //      to invert pack_44 exactly over all 256 codes + a voltage sweep. If this drifts, our
    //      bytes stop being CHORD bytes and nothing else here would notice.
    //  (2) the quantized DESPREAD equals the float despread on the SAME dequantized voltages, to
    //      float precision -- i.e. launch_despread_q's only difference is the load. (The C/N0 cost
    //      of quantizing is a separate, already-answered question: < 0.2 dB at 2 lsb on real sky,
    //      scratchpad/quant_test.py. It is NOT testable on this synthetic window, whose signal sits
    //      3x ABOVE the noise and would simply rail.)
    {
        bool qpass = true;
        // (1) encoder vs the reference decode, over every representable code.
        int codec_bad = 0;
        for (int r = -8; r <= 7; ++r)
            for (int i = -8; i <= 7; ++i) {
                int railed = 0;
                const unsigned char b =
                    gnss_cuda::pack_44(make_float2((float)r, (float)i), 1.0f, &railed);
                const int ref_i = (b & 0x0f) - 8;        // gpuSimulate.cpp, verbatim
                const int ref_r = ((b & 0xf0) >> 4) - 8; // gpuSimulate.cpp, verbatim
                if (ref_r != r || ref_i != i || railed)
                    codec_bad++;
            }
        // Rails must clamp, not wrap: +-100 lsb has to land on the rail codes, never fold over.
        {
            int railed = 0;
            const unsigned char hi = gnss_cuda::pack_44(make_float2(100.f, 100.f), 1.0f, &railed);
            const unsigned char lo = gnss_cuda::pack_44(make_float2(-100.f, -100.f), 1.0f, &railed);
            if (((hi & 0xf0) >> 4) - 8 != 7 || (hi & 0x0f) - 8 != 7)
                codec_bad++;
            if (((lo & 0xf0) >> 4) - 8 != -8 || (lo & 0x0f) - 8 != -8)
                codec_bad++;
            if (!railed)
                codec_bad++; // the rail flag must fire -- it is the RFI watchdog
        }
        printf("4+4b codec vs gpuSimulate reference decode (256 codes + rails): %s\n",
               codec_bad ? "FAIL" : "OK");
        qpass = qpass && (codec_bad == 0);

        // (2) quantized despread == float despread on the same dequantized voltages.
        // Scale each channel so its rms lands at ~3 lsb, exactly as the ingest will.
        std::vector<float> h_scale(n_chan), h_inv(n_chan);
        for (int c = 0; c < n_chan; ++c) {
            double ss = 0.0;
            for (int m = 0; m < n_hops; ++m)
                ss += std::norm(std::complex<double>(data_ch[c][m]));
            const double rms = std::sqrt(ss / (2.0 * n_hops)); // per-component rms
            h_inv[c] = (float)(3.0 / std::max(1e-12, rms));    // volts -> lsb
            h_scale[c] = 1.0f / h_inv[c];                      // lsb -> volts
        }
        std::vector<unsigned char> q((size_t)n_chan * n_hops);
        std::vector<float2> deq((size_t)n_chan * n_hops);
        for (int c = 0; c < n_chan; ++c)
            for (int m = 0; m < n_hops; ++m) {
                const size_t k = (size_t)c * n_hops + m;
                int railed = 0;
                q[k] = gnss_cuda::pack_44(
                    make_float2(data_ch[c][m].real(), data_ch[c][m].imag()), h_inv[c], &railed);
                deq[k] = gnss_cuda::unpack_44(q[k], h_scale[c]); // what the q kernel will see
            }
        unsigned char* d_q;
        float2* d_deq;
        float* d_scale;
        double2* d_qc;
        double* d_qe;
        CK(cudaMalloc(&d_q, q.size()));
        CK(cudaMalloc(&d_deq, deq.size() * sizeof(float2)));
        CK(cudaMalloc(&d_scale, h_scale.size() * sizeof(float)));
        CK(cudaMalloc(&d_qc, (size_t)n_out * n_chan * sizeof(double2)));
        CK(cudaMalloc(&d_qe, (size_t)n_out * n_chan * sizeof(double)));
        CK(cudaMemcpy(d_q, q.data(), q.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_deq, deq.data(), deq.size() * sizeof(float2), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_scale, h_scale.data(), h_scale.size() * sizeof(float),
                      cudaMemcpyHostToDevice));
        // float kernel on the DEQUANTIZED data = the reference for the q kernel on the bytes.
        CK(gnss_cuda::launch_despread(d_deq, d_code, d_jobs, n_spec, n_chan, p, d_corr, d_energy,
                                      0));
        CK(gnss_cuda::launch_despread_q(d_q, d_scale, d_code, d_jobs, n_spec, n_chan, p, d_qc,
                                        d_qe, 0));
        CK(cudaDeviceSynchronize());
        std::vector<double2> fc((size_t)n_out * n_chan), qc((size_t)n_out * n_chan);
        CK(cudaMemcpy(fc.data(), d_corr, fc.size() * sizeof(double2), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(qc.data(), d_qc, qc.size() * sizeof(double2), cudaMemcpyDeviceToHost));
        double qmax = 0.0;
        for (size_t k = 0; k < fc.size(); ++k) {
            const std::complex<double> a(fc[k].x, fc[k].y), b(qc[k].x, qc[k].y);
            qmax = std::max(qmax, std::abs(b - a) / std::max(1e-9, std::abs(a)));
        }
        printf("4+4b despread vs float despread on identical dequantized voltages: max rel %.3e "
               "%s\n",
               qmax, (qmax < 1e-6) ? "OK" : "FAIL");
        qpass = qpass && (qmax < 1e-6);
        pass = pass && qpass;
        cudaFree(d_q);
        cudaFree(d_deq);
        cudaFree(d_scale);
        cudaFree(d_qc);
        cudaFree(d_qe);
    }

    // --- Throughput bench: the REAL per-record path (GnssCudaDespread::despread_batch --
    // upload_window + one batched launch + one sync per record), full-constellation batch.
    // Reports records/s and the realtime margin vs the tracker's 1 kHz record cadence.
    //
    // ⚠️ BENCH AT THE DEPLOYED GEOMETRY, NOT THE CORRECTNESS ONE ABOVE. This kernel's speed is a
    // strong function of HOPS PER THREAD (= n_hops / min(pow2>=n_hops, 256)), because that sets
    // whether it is latency-bound (occupancy rules) or throughput-bound (replica work rules).
    // The 5 MSPS / 125-hop geometry used for the correctness check sits at ~1 hop/thread and
    // ranks kernels the OPPOSITE way from every band we actually fly -- it cost real debugging
    // time on 2026-07-16. Every deployed band is far from it:
    //     GPS L1CA / L5 / E5a / B2a  20 MSPS, N=10  -> 1000 hops  =  3.9 hops/thread
    //     GAL E1C  (4 ms)            20 MSPS, N=10  -> 4000 hops  = 15.6
    //     BDS B1C  (10 ms)           20 MSPS, N=10  -> 10000 hops = 39.1
    //     GPS L2C  (20 ms)            5 MSPS, N=10  -> 5000 hops  = 19.5
    // So bench the wide front end: config/live_l1_dual20.yaml (run_band.sh l1) = 20 MSPS, N=10,
    // 1000-hop GPS records -- the LEAST favourable of the deployed set, i.e. a lower bound.
    const double bfs = 20.0e6, bf_off = 5.0e6; // 20 MSPS-mode L1 IF = Fs/4
    const int bN = 10;
    for (int bench_prn : {12, 32}) {
        std::vector<int> prns;
        for (int q = 1; q <= bench_prn; ++q)
            prns.push_back(q);
        gnss::ChannelizedReplicaBank bbank(*sig, bfs, bf_off, bN, taps,
                                           dsp::window_from_string("hamming"), prns);
        const int bn_hops = bbank.repl_period_hops(); // 1000 @ L1/20MSPS
        const int bfft_len = 2 * bN;
        GnssCudaDespread gpu(bbank, bench_prn, bN, 0, bn_hops, bfs, bf_off);
        std::vector<cf> win((size_t)bn_hops * bN);
        for (auto& v : win)
            v = cf((float)frand(), (float)frand());
        std::vector<GnssCudaDespread::Spec> specs(bench_prn);
        std::vector<int> allch(bN);
        for (int c = 0; c < bN; ++c)
            allch[c] = c;
        for (int q = 0; q < bench_prn; ++q)
            specs[q] = {q, 100.0 + 7.0 * q, 0.5, -4000.0 + 250.0 * q, allch};
        gpu.upload_window(win.data(), window_start);
        (void)gpu.despread_batch(specs); // warm-up (Phi builds + JIT)
        const int NREC = 2000;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < NREC; ++r) {
            gpu.upload_window(win.data(), window_start + (long long)r * bn_hops * bfft_len);
            (void)gpu.despread_batch(specs);
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                              .count();
        printf("BENCH(live 20MSPS) %2d PRN x E/P/L x %2d chan x %d hops (%.1f hops/thread): "
               "%6.0f rec/s (%.1fx realtime @1kHz, %.0f PRN-despreads/s)\n",
               bench_prn, bN, bn_hops, (double)bn_hops / (bn_hops < 256 ? bn_hops : 256),
               NREC / dt, NREC / dt / 1000.0, 3.0 * bench_prn * NREC / dt);

        // ⚠️ AND the PRODUCTION path. despread_batch above is the CPU-tracker fallback, whose
        // 3-trial contract carries NO head job -- so it benches a job mix no band flies, and it
        // ranks kernel changes DIFFERENTLY from the GPU chain (it called the 2026-07-16 fusion a
        // 5% regression while the chain got 14-22% faster). The GPU chain runs
        // enqueue_batch_device: 4 rows/spec including P_HEAD, m_head > 0 on every record. Bench
        // what ships.
        {
            float2* d_win = nullptr;
            gnss_cuda::DespreadJob* d_j = nullptr;
            double2* d_c = nullptr;
            double* d_e = nullptr;
            cudaStream_t st;
            CK(cudaStreamCreate(&st));
            CK(cudaMalloc(&d_win, (size_t)bN * bn_hops * sizeof(float2)));
            CK(cudaMemset(d_win, 0, (size_t)bN * bn_hops * sizeof(float2)));
            CK(cudaMalloc(&d_j, (size_t)bench_prn * sizeof(gnss_cuda::DespreadJob)));
            CK(cudaMalloc(&d_c, (size_t)4 * bench_prn * bN * sizeof(double2)));
            CK(cudaMalloc(&d_e, (size_t)4 * bench_prn * bN * sizeof(double)));
            (void)gpu.enqueue_batch_device(d_win, bn_hops, window_start, specs, d_j, d_c, d_e, st);
            CK(cudaStreamSynchronize(st)); // warm-up
            const auto p0 = std::chrono::steady_clock::now();
            for (int r = 0; r < NREC; ++r) {
                (void)gpu.enqueue_batch_device(d_win, bn_hops,
                                               window_start + (long long)r * bn_hops * bfft_len,
                                               specs, d_j, d_c, d_e, st);
                CK(cudaStreamSynchronize(st));
            }
            const double pdt =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
            printf("   ^ PRODUCTION path (E/P/L/P_HEAD, 4 rows/spec, the GPU chain's mix): "
                   "%6.0f rec/s (%.1fx realtime @1kHz)\n",
                   NREC / pdt, NREC / pdt / 1000.0);
            cudaFree(d_win);
            cudaFree(d_j);
            cudaFree(d_c);
            cudaFree(d_e);
            cudaStreamDestroy(st);
        }
    }
    return pass ? 0 : 1;
}
