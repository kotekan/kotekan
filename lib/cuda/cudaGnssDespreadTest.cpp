// Standalone GPU checks for the GNSS despread kernels, on a synthetic window (the replica at a
// known code phase and Doppler plus deterministic noise):
//  - the fused despread (launch_despread) against a CPU reference built with
//    ChannelizedReplicaBank::hoprate_filter/hoprate_stream: correlation and energy per output
//    row, summed over channels, within 1e-5 relative plus an anchor-scaled term
//  - the waveform + N x M correlator split on CHORD 4+4b data against the fused kernel, with
//    the split's kernel timings
//  - the despread as an int8 GEMM
// followed by a throughput bench. Exits 0 when every check passes.
//
//   ./cuda_gnss_despread_test [signal_name] [N] [n_hops] [cp_true]

#include "GnssCudaDespread.hpp"
#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "gnssChannelizedDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"
#include "gpsCACode.hpp"
#include "pfbPrototype.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <numeric>
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

int main(int argc, char** argv) {
    // Default: GPS L1 C/A at 5 MSPS real, N=20 channels, 4-tap Hamming PFB. Examples:
    //   `cuda_gnss_despread_test GPS_L2C_CM 10`: the TDM signal (comb_mult 2), 5000-hop (20 ms)
    //   records.
    //   `cuda_gnss_despread_test GPS_L2C_CL 10 5000`: 20 ms records despread against a segment
    //   of the 1.5 s CL code. Without the n_hops override a record is one code period, 375,000
    //   hops here.
    // cp_true is the code-phase argument, referred to absolute sample 0 (see
    // ChannelizedReplicaBank::phase_from_arg), so where a record falls in the code also depends
    // on the anchor below; a record across the CL code wrap needs the two chosen together.
    // L1 C/A and CM runs take seconds; CL runs take minutes, because the CPU reference and the
    // bench walk CL's 1,534,500-chip combined code (75x CM's). The bench saturates the GPU, so
    // never run this beside a live node.
    const double fs = 5.0e6, f_off = 1.25e6;
    const int N = (argc > 2) ? atoi(argv[2]) : 20;
    const int taps = 4;
    const int prn = 19;
    const double dop = -2900.0;
    // Absolute anchor at CHORD scale: hop 114436200145 of 16384 samples (n ~ 1.9e15, 6.8 days
    // of uptime, scripts/gnss/e2e's --hop0 default). Errors from a small quantity times a huge
    // absolute index (wc*n, n*cps) are invisible at small anchors, and this test is the
    // independent CPU check on the GPU replica, so it runs where they show.
    long long window_start = 114436200145LL * 16384LL;
    // GNSS_TEST_HOP0 overrides the anchor hop, to scan it: a GPU-vs-CPU error that grows with
    // the anchor is precision lost in an absolute-index product.
    if (const char* e = getenv("GNSS_TEST_HOP0"))
        window_start = atoll(e) * 16384LL;

    const gnss::SignalDescriptor* sig = gnss::signal_by_name(argc > 1 ? argv[1] : "GPS_L1CA");
    if (!sig) {
        fprintf(stderr, "unknown signal %s\n", argv[1]);
        return 2;
    }
    const double cp_true =
        (argc > 4) ? atof(argv[4]) : ((sig->code_length > 1023) ? 5170.25 : 517.25);
    gnss::ChannelizedReplicaBank bank(*sig, fs, f_off, N, taps, dsp::window_from_string("hamming"),
                                      {prn});
    // Window length: one code period by default; override for sub-period records (L2C CL).
    const int n_hops = (argc > 3) ? atoi(argv[3]) : bank.repl_period_hops(); // 125 @ L1/5MSPS
    const int fft_len = 2 * N;

    const auto cover = bank.covering_bins(dop, 250000.0);
    const int n_chan = (int)cover.size();

    // "Sky": the replica at (cp_true, dop), scaled by 3, plus deterministic noise.
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

    // Two fused jobs, each emitting E, P, L and P_HEAD: one at the true code phase, one an
    // off-peak control. The first cuts P_HEAD at hop 40; the second sets m_head = 0, which
    // gives an all-zero P_HEAD row, so both branches are covered.
    const double spacing = 0.5;
    struct FusedTrial {
        double cp_prompt;
        int m_head;
    };
    const std::vector<FusedTrial> ftrials = {{cp_true, 40}, {cp_true + 137.0, 0}};
    const char* row_name[4] = {"E", "P", "L", "HEAD"};
    // Row -> (code phase, hop count) as the kernel defines them; the CPU must reproduce both.
    auto row_cp = [&](const FusedTrial& ft, int t) {
        return ft.cp_prompt + (t == 3 ? 0.0 : (t - 1) * spacing);
    };
    auto row_hops = [&](const FusedTrial& ft, int t) { return (t == 3) ? ft.m_head : n_hops; };

    // --- CPU reference: a hoprate replica per row, correlated in double ---
    const auto filt = bank.hoprate_filter(cover, dop);
    std::vector<std::complex<double>> cpu_corr; // [row][chan]
    std::vector<double> cpu_energy;
    for (const auto& ft : ftrials)
        for (int t = 0; t < 4; ++t) {
            const auto repl =
                bank.hoprate_stream(filt, 0, window_start, row_cp(ft, t), dop, n_hops);
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
    // As GnssCudaDespread builds its jobs: combined-stream chip rate and code table (TDM/BOC
    // expanded), code phase and spacing scaled by comb_mult below.
    const int comb = bank.comb_mult();
    const double cps =
        bank.eff_chip_rate() / fs * (1.0 + bank.code_doppler_sign * dop / sig->carrier_hz);
    const double wc = 2.0 * M_PI * (f_off + dop) / fs;
    // Anchor-scaled tolerance. The kernel reduces the code phase mod code_len once at n0 (on
    // the host, in long double) and advances by an intra-record offset; the reference evaluates
    // cp0 + n_m*cps in long double every hop. They differ by O(1) long-double ULPs of the anchor
    // product n0*cps, which grows with the anchor (2^-15 chips at a week here). A constant bound
    // would be either red at a deep anchor or meaningless at a shallow one, so the bound scales
    // with that ULP. The two sides are deliberately independent: giving the reference the
    // kernel's anchor convention would leave this term unable to fail. 32 ULP is margin over the
    // few ULP seen, and a double-precision reference still fails it at a deep anchor.
    const long double anchor_chips =
        std::fabs((long double)(window_start + fft_len - 1) * (long double)cps);
    const double ulp_anchor = // long double ULP at the anchor product, in chips (64-bit mantissa)
        (anchor_chips > 0.0L) ? std::ldexp(1.0, std::ilogbl(anchor_chips) - 63) : 0.0;
    const double TOL = 1e-5 + 32.0 * ulp_anchor;
    printf("anchor n0*cps = %.4Le chips -> long double ULP %.3e chips -> tolerance %.3e\n",
           anchor_chips, ulp_anchor, TOL);
    // Carrier phase at the hop reference sample, reduced in long double as
    // GnssCudaDespread::Impl::ang0_for does, so the kernel's wc multiplies only the intra-record
    // offset. Duplicated rather than shared because this test is the independent check on the
    // kernel; keep the two expressions equivalent by hand.
    const auto ang0_of = [&](long long n0) {
        constexpr long double TWO_PI_L = 6.283185307179586476925286766559005768L;
        long double fr = fmodl(
            ((long double)f_off + (long double)dop) / (long double)fs * (long double)n0, 1.0L);
        if (fr < 0.0L)
            fr += 1.0L;
        return (double)(TWO_PI_L * fr);
    };

    const auto& fcode = bank.full_code(0); // slot 0 == our PRN
    std::vector<int8_t> code8(fcode.begin(), fcode.end());

    // Mirrors GnssCudaDespread::Impl::cp_ref_for, duplicated for the same reason as ang0_of.
    const auto cp_ref_of = [&](double cp0, long long n0) {
        long double c = (long double)cp0 + (long double)n0 * (long double)cps;
        const long double L = (long double)code8.size();
        c = fmodl(c, L);
        if (c < 0.0L)
            c += L;
        return (double)c;
    };

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
    // Each job carries its own Phi pointers (d_phiA/d_phiB), filled in below once allocated.

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
    for (int b = 0; b < n_spec; ++b) // component -> combined chips, as production does
        jobs[b] = {(double)comb * ftrials[b].cp_prompt,
                   (double)comb * spacing,
                   cp_ref_of((double)comb * ftrials[b].cp_prompt, window_start + fft_len - 1),
                   cps,
                   1.0 / cps,
                   wc,
                   ang0_of(window_start + fft_len - 1),
                   0,
                   (int)code8.size(),
                   all_mask,
                   d_phiA,
                   d_phiB,
                   filt.n_chips,
                   ftrials[b].m_head};
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
        const bool ok = rc < TOL && re < TOL;
        pass = pass && ok;
        printf("%-4s cp=%9.2f hops=%4d: CPU |G|=%12.4f E=%12.1f | GPU |G|=%12.4f E=%12.1f | "
               "rel %.2e/%.2e %s\n",
               row_name[t], row_cp(ft, t), row_hops(ft, t), std::abs(cc), ce, std::abs(gc), ge, rc,
               re, ok ? "OK" : "FAIL");
    }
    printf("%s (max relative error %.3e; %d jobs x 4 rows x %d channels x %d hops)\n",
           pass ? "PASS" : "FAIL", max_rel, n_spec, n_chan, n_hops);

    // --- CHORD 4+4b data ---
    // The voltage quantized to 4+4b bytes (gnss44.hpp), each channel scaled to ~3 lsb rms per
    // component. The fused kernel despreads the same voltages dequantized to float, and that is
    // the reference for the split below.
    {
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
                q[k] = gnss44::pack(data_ch[c][m].real(), data_ch[c][m].imag(), h_inv[c], &railed);
                deq[k] = gnss_cuda::unpack_44(q[k], h_scale[c]); // what the split will see
            }
        float2* d_deq;
        float* d_scale;
        CK(cudaMalloc(&d_deq, deq.size() * sizeof(float2)));
        CK(cudaMalloc(&d_scale, h_scale.size() * sizeof(float)));
        CK(cudaMemcpy(d_deq, deq.data(), deq.size() * sizeof(float2), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_scale, h_scale.data(), h_scale.size() * sizeof(float),
                      cudaMemcpyHostToDevice));
        CK(gnss_cuda::launch_despread(d_deq, d_code, d_jobs, n_spec, n_chan, p, d_corr, d_energy,
                                      0));
        CK(cudaDeviceSynchronize());
        std::vector<double2> fc((size_t)n_out * n_chan);
        std::vector<double> fe((size_t)n_out * n_chan);
        CK(cudaMemcpy(fc.data(), d_corr, fc.size() * sizeof(double2), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fe.data(), d_energy, fe.size() * sizeof(double), cudaMemcpyDeviceToHost));

        // --- CHORD split (waveform generation + N x M correlation) vs the fused kernel ---
        // The split generates each replica once and correlates it against N antennas instead of
        // regenerating it inside the MAC. At N=1 both paths do the same arithmetic in the same
        // order (gather, carrier, combination, accumulation), so they must agree exactly.
        {
            const int n_elem = 1;
            // The fused kernel read the dequantized voltage as [chan][hop] floats; the CHORD
            // correlator reads the bytes in its native frame order [hop][chan][elem] and
            // dequantizes on load. Same voltages, transposed.
            std::vector<unsigned char> qt((size_t)n_hops * n_chan * n_elem);
            for (int c = 0; c < n_chan; ++c)
                for (int m = 0; m < n_hops; ++m)
                    qt[((size_t)m * n_chan + c) * n_elem + 0] = q[(size_t)c * n_hops + m];
            std::vector<int> h_chan_ids(n_chan);
            for (int c = 0; c < n_chan; ++c)
                h_chan_ids[c] = c;

            unsigned char* d_qt = nullptr;
            int* d_chan_ids = nullptr;
            float2* d_wave = nullptr;
            double2* d_ncorr = nullptr;
            double* d_nenergy = nullptr;
            CK(cudaMalloc(&d_qt, qt.size()));
            CK(cudaMalloc(&d_chan_ids, h_chan_ids.size() * sizeof(int)));
            CK(cudaMalloc(&d_wave, (size_t)3 * n_spec * n_chan * n_hops * sizeof(float2)));
            CK(cudaMalloc(&d_ncorr, (size_t)4 * n_spec * n_chan * n_elem * sizeof(double2)));
            CK(cudaMalloc(&d_nenergy, (size_t)4 * n_spec * n_chan * sizeof(double)));
            CK(cudaMemcpy(d_qt, qt.data(), qt.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_chan_ids, h_chan_ids.data(), h_chan_ids.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

            CK(gnss_cuda::launch_waveform(d_code, d_jobs, n_spec, n_chan, p, d_wave, d_nenergy, 0));
            CK(gnss_cuda::launch_correlate_nm(d_qt, d_scale, d_chan_ids, d_wave, d_jobs, n_spec,
                                              n_chan, n_elem, n_elem, n_chan, p, d_ncorr, 0));
            CK(cudaDeviceSynchronize());

            std::vector<double2> nc((size_t)4 * n_spec * n_chan * n_elem);
            std::vector<double> ne((size_t)4 * n_spec * n_chan);
            CK(cudaMemcpy(nc.data(), d_ncorr, nc.size() * sizeof(double2), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ne.data(), d_nenergy, ne.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));

            double cmax = 0.0, emax = 0.0;
            for (int b = 0; b < n_spec; ++b)
                for (int j = 0; j < 4; ++j)
                    for (int c = 0; c < n_chan; ++c) {
                        const size_t kf = (size_t)(4 * b + j) * n_chan + c;
                        const size_t ks = ((size_t)(4 * b + j) * n_chan + c) * n_elem + 0;
                        const std::complex<double> a(fc[kf].x, fc[kf].y);
                        const std::complex<double> bb(nc[ks].x, nc[ks].y);
                        cmax = std::max(cmax, std::abs(bb - a) / std::max(1e-9, std::abs(a)));
                        emax = std::max(emax, std::abs(ne[kf] - fe[kf])
                                                  / std::max(1e-9, std::abs(fe[kf])));
                    }
            const bool split_ok = (cmax == 0.0) && (emax == 0.0);
            printf("CHORD split (waveform + NxM) vs fused despread, N=1: corr %.3e energy %.3e "
                   "%s\n",
                   cmax, emax, split_ok ? "OK (exact)" : "FAIL");

            // ---- Kernel split timing ----
            // enqueue_batch_nm launches launch_waveform (replica synthesis) then
            // launch_correlate_nm (the correlation over elements). Time them separately: the
            // correlation is the part CHORD's N^2 correlator can absorb (as it does for
            // cudaGnssInject's waveforms); the synthesis it cannot.
            {
                const int NIT = 200;
                cudaEvent_t e0, e1, e2;
                CK(cudaEventCreate(&e0));
                CK(cudaEventCreate(&e1));
                CK(cudaEventCreate(&e2));
                // warm-up
                CK(gnss_cuda::launch_waveform(d_code, d_jobs, n_spec, n_chan, p, d_wave, d_nenergy,
                                              0));
                CK(gnss_cuda::launch_correlate_nm(d_qt, d_scale, d_chan_ids, d_wave, d_jobs, n_spec,
                                                  n_chan, n_elem, n_elem, n_chan, p, d_ncorr, 0));
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0, 0));
                for (int it = 0; it < NIT; ++it)
                    CK(gnss_cuda::launch_waveform(d_code, d_jobs, n_spec, n_chan, p, d_wave,
                                                  d_nenergy, 0));
                CK(cudaEventRecord(e1, 0));
                for (int it = 0; it < NIT; ++it)
                    CK(gnss_cuda::launch_correlate_nm(d_qt, d_scale, d_chan_ids, d_wave, d_jobs,
                                                      n_spec, n_chan, n_elem, n_elem, n_chan, p,
                                                      d_ncorr, 0));
                CK(cudaEventRecord(e2, 0));
                CK(cudaDeviceSynchronize());
                float ms_syn = 0.f, ms_cor = 0.f;
                CK(cudaEventElapsedTime(&ms_syn, e0, e1));
                CK(cudaEventElapsedTime(&ms_cor, e1, e2));
                ms_syn /= NIT;
                ms_cor /= NIT;
                printf(
                    "KERNEL SPLIT (%d spec x %d chan x %d elem x %d hops):\n"
                    "   synthesis  (waveform)     %8.4f ms  %5.1f%%   <- N^2 CANNOT absorb this\n"
                    "   correlation(nm, %2d elem) %8.4f ms  %5.1f%%   <- this is the N^2 "
                    "candidate\n",
                    n_spec, n_chan, n_elem, p.n_hops, ms_syn, 100.0 * ms_syn / (ms_syn + ms_cor),
                    n_elem, ms_cor, 100.0 * ms_cor / (ms_syn + ms_cor));
                // Element sweep: synthesis does not depend on n_elem (one replica per PRN/chan/hop
                // serves every element) while the correlation scales with it, so a 1-element timing
                // understates the correlation's share.
                for (int ne : {1, 8, 32}) {
                    unsigned char* d_q2 = nullptr;
                    double2* d_c2 = nullptr;
                    CK(cudaMalloc(&d_q2, (size_t)n_hops * n_chan * ne));
                    CK(cudaMemset(d_q2, 0, (size_t)n_hops * n_chan * ne));
                    CK(cudaMalloc(&d_c2, (size_t)4 * n_spec * n_chan * ne * sizeof(double2)));
                    CK(gnss_cuda::launch_correlate_nm(d_q2, d_scale, d_chan_ids, d_wave, d_jobs,
                                                      n_spec, n_chan, ne, ne, n_chan, p, d_c2, 0));
                    CK(cudaDeviceSynchronize());
                    cudaEvent_t a0, a1;
                    CK(cudaEventCreate(&a0));
                    CK(cudaEventCreate(&a1));
                    CK(cudaEventRecord(a0, 0));
                    for (int it = 0; it < NIT; ++it)
                        CK(gnss_cuda::launch_correlate_nm(d_q2, d_scale, d_chan_ids, d_wave, d_jobs,
                                                          n_spec, n_chan, ne, ne, n_chan, p, d_c2,
                                                          0));
                    CK(cudaEventRecord(a1, 0));
                    CK(cudaDeviceSynchronize());
                    float ms = 0.f;
                    CK(cudaEventElapsedTime(&ms, a0, a1));
                    ms /= NIT;
                    printf("   n_elem %2d: synthesis %8.4f ms (%5.1f%%) | correlation %8.4f ms "
                           "(%5.1f%%)\n",
                           ne, ms_syn, 100.0 * ms_syn / (ms_syn + ms), ms,
                           100.0 * ms / (ms_syn + ms));
                    cudaEventDestroy(a0);
                    cudaEventDestroy(a1);
                    cudaFree(d_q2);
                    cudaFree(d_c2);
                }
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                cudaEventDestroy(e2);
            }
            pass = pass && split_ok;

            cudaFree(d_qt);
            cudaFree(d_ncorr);

            // --- Element axis: N>1 indexing and stride ---
            // N=1 above checks the arithmetic, not the element indexing. Replicate the voltage into
            // every element, pad the element axis with a poison byte (0x00 = (-8,-8), the loudest
            // code; 0x88 decodes to zero and would hide a stride error), and require every element
            // to match the fused answer exactly. n_elem = 5 with elem_stride = 7 is not a power of
            // two and is less than the stride. It is below the 32-wide element block, so
            // launch_correlate_nm makes the block 5 wide and the ragged-tail predication (n_elem
            // above 32, not a multiple of it) is not exercised.
            const int nE = 5, strideE = 7;
            std::vector<unsigned char> qm((size_t)n_hops * n_chan * strideE, 0x00); // poison
            for (int c = 0; c < n_chan; ++c)
                for (int m = 0; m < n_hops; ++m)
                    for (int e = 0; e < nE; ++e)
                        qm[((size_t)m * n_chan + c) * strideE + e] = q[(size_t)c * n_hops + m];

            unsigned char* d_qm = nullptr;
            double2* d_mcorr = nullptr;
            CK(cudaMalloc(&d_qm, qm.size()));
            CK(cudaMalloc(&d_mcorr, (size_t)4 * n_spec * n_chan * nE * sizeof(double2)));
            CK(cudaMemcpy(d_qm, qm.data(), qm.size(), cudaMemcpyHostToDevice));
            CK(gnss_cuda::launch_correlate_nm(d_qm, d_scale, d_chan_ids, d_wave, d_jobs, n_spec,
                                              n_chan, nE, strideE, n_chan, p, d_mcorr, 0));
            CK(cudaDeviceSynchronize());
            std::vector<double2> mc((size_t)4 * n_spec * n_chan * nE);
            CK(cudaMemcpy(mc.data(), d_mcorr, mc.size() * sizeof(double2), cudaMemcpyDeviceToHost));

            double mmax = 0.0;
            for (int b = 0; b < n_spec; ++b)
                for (int j = 0; j < 4; ++j)
                    for (int c = 0; c < n_chan; ++c) {
                        const size_t kf = (size_t)(4 * b + j) * n_chan + c;
                        const std::complex<double> a(fc[kf].x, fc[kf].y);
                        for (int e = 0; e < nE; ++e) {
                            const size_t km = ((size_t)(4 * b + j) * n_chan + c) * nE + e;
                            const std::complex<double> bb(mc[km].x, mc[km].y);
                            mmax = std::max(mmax, std::abs(bb - a) / std::max(1e-9, std::abs(a)));
                        }
                    }
            printf("CHORD NxM element axis, N=%d stride=%d (poison-padded): max rel %.3e %s\n", nE,
                   strideE, mmax, (mmax == 0.0) ? "OK (exact)" : "FAIL");
            pass = pass && (mmax == 0.0);

            cudaFree(d_qm);
            cudaFree(d_mcorr);
            cudaFree(d_chan_ids);
            cudaFree(d_wave);
            cudaFree(d_nenergy);
        }

        cudaFree(d_deq);
        cudaFree(d_scale);
    }

    // --- The despread as an int8 GEMM (no stage uses this form) ---
    // A = D x R^H per covering channel: a [1 x T] data row against [T x M] replica columns, as four
    // real int8 tensor GEMMs batched over channels. The gated claim is that the GEMM computes the
    // same object as the despread, so the reference uses the same dequantized integers the GEMM
    // multiplies (data nibbles x int8 replicas); int8 products sum exactly in int32, so the two
    // agree to rounding (gate 1e-6). The difference from the float CPU reference (cpu_corr, on
    // unquantized data) adds the 4+4b data and int8 replica quantization; it is reported, not
    // gated.
    {
        cublasHandle_t bh;
        if (cublasCreate(&bh) != CUBLAS_STATUS_SUCCESS) {
            printf("GEMM xval: no cublas handle -- SKIP\n");
        } else {
            const int M = n_out;              // 8 replica columns (2 jobs x E/P/L/HEAD)
            const int TP = (n_hops + 3) & ~3; // int8 gemm wants k%4: zero-pad (sums unchanged)
            // Data rows: int8 nibbles from the 4+4b bytes (reference decode), padded to [4 x TP]
            // per channel (m=1 padded to 4 for the alignment rules; rows 1-3 zero).
            std::vector<int8_t> hDr((size_t)n_chan * 4 * TP, 0), hDi((size_t)n_chan * 4 * TP, 0);
            std::vector<unsigned char> qbytes((size_t)n_chan * n_hops);
            std::vector<float> dsc(n_chan);
            for (int c = 0; c < n_chan; ++c) {
                double ss = 0.0;
                for (int m = 0; m < n_hops; ++m)
                    ss += std::norm(std::complex<double>(data_ch[c][m]));
                const float inv = (float)(2.1 / std::sqrt(ss / n_hops)); // ~2 lsb rms (complex)
                dsc[c] = 1.0f / inv;
                for (int m = 0; m < n_hops; ++m) {
                    int railed = 0;
                    const unsigned char b =
                        gnss44::pack(data_ch[c][m].real(), data_ch[c][m].imag(), inv, &railed);
                    qbytes[(size_t)c * n_hops + m] = b;
                    // col-major [4 x TP], row 0: element (0,t) at [t*4]
                    hDr[((size_t)c * TP + m) * 4] = (int8_t)(((b >> 4) & 0x0f) - 8);
                    hDi[((size_t)c * TP + m) * 4] = (int8_t)((b & 0x0f) - 8);
                }
            }
            // Replica columns: same hoprate generator as the CPU reference, quantized to int8,
            // HEAD columns zeroed past their m_head (the segmented despread as a column shape).
            std::vector<int8_t> hRr((size_t)n_chan * M * TP, 0), hRi((size_t)n_chan * M * TP, 0);
            std::vector<float> rsc(M);
            for (int col = 0; col < M; ++col) {
                const FusedTrial& ft = ftrials[col / 4];
                const int t4 = col % 4;
                const auto repl =
                    bank.hoprate_stream(filt, 0, window_start, row_cp(ft, t4), dop, n_hops);
                const int hi = row_hops(ft, t4);
                float mx = 1e-12f;
                for (int c = 0; c < n_chan; ++c)
                    for (int m = 0; m < hi; ++m)
                        mx = std::max(
                            {mx, std::fabs(repl[c][m].real()), std::fabs(repl[c][m].imag())});
                rsc[col] = mx / 127.0f;
                for (int c = 0; c < n_chan; ++c)
                    for (int m = 0; m < hi; ++m) {
                        const size_t o = ((size_t)c * M + col) * TP + m; // [T x M] k-major
                        hRr[o] = (int8_t)std::lrint(repl[c][m].real() / rsc[col]);
                        hRi[o] = (int8_t)std::lrint(repl[c][m].imag() / rsc[col]);
                    }
            }
            int8_t *dDr, *dDi, *dRr, *dRi;
            int32_t* dC;
            CK(cudaMalloc(&dDr, hDr.size()));
            CK(cudaMalloc(&dDi, hDi.size()));
            CK(cudaMalloc(&dRr, hRr.size()));
            CK(cudaMalloc(&dRi, hRi.size()));
            CK(cudaMalloc(&dC, (size_t)4 * n_chan * 4 * M * sizeof(int32_t)));
            CK(cudaMemcpy(dDr, hDr.data(), hDr.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dDi, hDi.data(), hDi.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dRr, hRr.data(), hRr.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dRi, hRi.data(), hRi.size(), cudaMemcpyHostToDevice));
            const int32_t one = 1, zero = 0;
            const int8_t* As[4] = {dDr, dDi, dDi, dDr};
            const int8_t* Bs[4] = {dRr, dRi, dRr, dRi};
            bool gemm_ok = true;
            for (int quad = 0; quad < 4; ++quad) {
                auto st = cublasGemmStridedBatchedEx(
                    bh, CUBLAS_OP_N, CUBLAS_OP_N, 4, M, TP, &one, As[quad], CUDA_R_8I, 4,
                    (long long)4 * TP, Bs[quad], CUDA_R_8I, TP, (long long)M * TP, &zero,
                    dC + (size_t)quad * n_chan * 4 * M, CUDA_R_32I, 4, (long long)4 * M, n_chan,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
                if (st != CUBLAS_STATUS_SUCCESS) {
                    printf("GEMM xval: int8 gemm unsupported here (%d) -- FAIL\n", (int)st);
                    gemm_ok = false;
                    break;
                }
            }
            if (gemm_ok) {
                CK(cudaDeviceSynchronize());
                std::vector<int32_t> hC((size_t)4 * n_chan * 4 * M);
                CK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                auto cint = [&](int quad, int c, int col) {
                    return hC[(size_t)quad * n_chan * 4 * M + ((size_t)c * M + col) * 4];
                };
                double max_exact = 0.0, phys_p = 0.0;
                for (int col = 0; col < M; ++col) {
                    std::complex<double> gg(0, 0), ee(0, 0);
                    for (int c = 0; c < n_chan; ++c) {
                        // GEMM result, rescaled: corr = Sum d*conj(r) * dsc[c] * rsc[col]
                        const double s = (double)dsc[c] * rsc[col];
                        gg += std::complex<double>(
                            (cint(0, c, col) + cint(1, c, col)) * s,  // dr*rr + di*ri
                            (cint(2, c, col) - cint(3, c, col)) * s); // di*rr - dr*ri
                        // Reference from the same dequantized integers
                        std::complex<double> e(0, 0);
                        const int hi = row_hops(ftrials[col / 4], col % 4);
                        for (int m = 0; m < hi; ++m) {
                            const unsigned char b = qbytes[(size_t)c * n_hops + m];
                            const std::complex<double> dv((((b >> 4) & 0x0f) - 8) * (double)dsc[c],
                                                          ((b & 0x0f) - 8) * (double)dsc[c]);
                            const size_t o = ((size_t)c * M + col) * TP + m;
                            const std::complex<double> rv(hRr[o] * (double)rsc[col],
                                                          hRi[o] * (double)rsc[col]);
                            e += dv * std::conj(rv);
                        }
                        ee += e;
                    }
                    const double rel = std::abs(gg - ee) / std::max(1e-30, std::abs(ee));
                    max_exact = std::max(max_exact, rel);
                    if (col == 1) // strong prompt row: difference from the float CPU reference
                        phys_p = std::abs(gg
                                          - std::accumulate(cpu_corr.begin() + (size_t)col * n_chan,
                                                            cpu_corr.begin()
                                                                + (size_t)(col + 1) * n_chan,
                                                            std::complex<double>(0, 0)))
                                 / std::abs(gg);
                }
                const bool ok = max_exact < 1e-6;
                printf("GEMM despread vs exact same-integer reference: max rel %.3e %s "
                       "(physics delta vs float ref, strong P row: %.2e)\n",
                       max_exact, ok ? "OK" : "FAIL", phys_p);
                pass = pass && ok;
            } else {
                pass = false;
            }
            cudaFree(dDr);
            cudaFree(dDi);
            cudaFree(dRr);
            cudaFree(dRi);
            cudaFree(dC);
            cublasDestroy(bh);
        }
    }

    // --- Throughput bench: GnssCudaDespread::despread_batch (upload_window, one batched launch and
    // one sync per record), every PRN slot in one batch. Reports records/s, and records/s over
    // 1 kHz as the realtime factor, which assumes L1 C/A's 1 ms records.
    //
    // The bench uses a 20 MSPS, N=10 front end (1000-hop records for L1 C/A, ~3.9 hops per thread),
    // not the correctness geometry above. The kernel's speed depends on hops per thread
    // (n_hops / min(pow2 >= n_hops, 256)): near 1 it is latency-bound and occupancy dominates; at
    // several it is throughput-bound and replica work dominates, and the two regimes rank kernel
    // changes differently. The 125-hop correctness geometry sits at ~1 hop per thread.
    const double bfs = 20.0e6, bf_off = 5.0e6; // IF = Fs/4
    const int bN = 10;
    for (int bench_prn : {12, 32}) {
        std::vector<int> prns;
        for (int q = 1; q <= bench_prn; ++q)
            prns.push_back(q);
        gnss::ChannelizedReplicaBank bbank(*sig, bfs, bf_off, bN, taps,
                                           dsp::window_from_string("hamming"), prns);
        const int bn_hops = bbank.repl_period_hops(); // 1000 @ L1/20MSPS
        const int bfft_len = 2 * bN;
        std::vector<int> bench_ids((size_t)bN);
        std::iota(bench_ids.begin(), bench_ids.end(), 0); // contiguous 0..bN-1 for the bench
        GnssCudaDespread gpu(bbank, bench_prn, bench_ids, bn_hops, bfs, bf_off);
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
        const double dt =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("BENCH(live 20MSPS) %2d PRN x E/P/L x %2d chan x %d hops (%.1f hops/thread): "
               "%6.0f rec/s (%.1fx realtime @1kHz, %.0f PRN-despreads/s)\n",
               bench_prn, bN, bn_hops, (double)bn_hops / (bn_hops < 256 ? bn_hops : 256), NREC / dt,
               NREC / dt / 1000.0, 3.0 * bench_prn * NREC / dt);
    }
    return pass ? 0 : 1;
}
