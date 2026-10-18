// Standalone GPU checks for the GNSS despread kernels, on a synthetic window (the replica at a
// known code phase and Doppler plus deterministic noise):
//  - the fused despread (launch_despread) against a CPU reference built with
//    ChannelizedReplicaBank::hoprate_filter/hoprate_stream: correlation and energy per output
//    row, summed over channels, within 1e-5 relative plus an anchor-scaled term
//  - the waveform + N x M correlator split on CHORD 4+4b data against the fused kernel, with
//    the split's kernel timings
// followed by a throughput bench. Exits 0 when every check passes.
//
//   ./cuda_gnss_despread_test [signal_name] [N] [n_hops] [cp_true]

#include "GnssCudaDespread.hpp"
#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "dspPfbPrototype.hpp"
#include "gnssChannelizedDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
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

    // Two jobs: one at the true code phase, one an off-peak control. The fused kernel returns
    // E, P and L for each and an all-zero P_HEAD row. m_head matters to the waveform split
    // below: the first job cuts P_HEAD at hop 40, the second sets m_head = 0, which gives an
    // all-zero P_HEAD row, so both branches are covered.
    const double spacing = 0.5;
    struct FusedTrial {
        double cp_prompt;
        int m_head;
    };
    const std::vector<FusedTrial> ftrials = {{cp_true, 40}, {cp_true + 137.0, 0}};
    const char* row_name[3] = {"E", "P", "L"};
    // Row -> code phase as the kernel defines it; the CPU must reproduce it.
    auto row_cp = [&](const FusedTrial& ft, int t) { return ft.cp_prompt + (t - 1) * spacing; };

    // --- CPU reference: a hoprate replica per row, correlated in double ---
    const auto filt = bank.hoprate_filter(cover, dop);
    std::vector<std::complex<double>> cpu_corr; // [job][E/P/L][chan]
    std::vector<double> cpu_energy;
    for (const auto& ft : ftrials)
        for (int t = 0; t < 3; ++t) {
            const auto repl =
                bank.hoprate_stream(filt, 0, window_start, row_cp(ft, t), dop, n_hops);
            for (int c = 0; c < n_chan; ++c) {
                std::complex<double> g(0.0, 0.0);
                double e = 0.0;
                for (int m = 0; m < n_hops; ++m) {
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
        jobs[b] = {(double)comb * spacing,
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
            gc += std::complex<double>(g_corr[(size_t)b * n_chan + c].x,
                                       g_corr[(size_t)b * n_chan + c].y);
            ge += g_energy[(size_t)b * n_chan + c];
        }
        if (t == 3) { // the P_HEAD slot: the fused kernel writes zeros
            const bool ok = gc == std::complex<double>(0.0, 0.0) && ge == 0.0;
            pass = pass && ok;
            printf("HEAD (zero in the fused kernel): GPU |G|=%12.4f E=%12.1f %s\n", std::abs(gc),
                   ge, ok ? "OK" : "FAIL");
            continue;
        }
        const size_t crow = (size_t)(3 * (b / 4) + t) * n_chan; // CPU rows are [job][E/P/L]
        // The correlation is compared as the vector of per-channel values, not their sum: an
        // E or L row's channel sum can be a near-null (L2C CM over its covering set, twice the
        // CM rate since the time-multiplexed rule: |sum| is 0.2% of the prompt's), where the
        // GPU's float tables against the CPU's double replica read 1e-5 of each term as 1e-2
        // of the sum. The vector norm has no cancellation and the same ULP floor.
        double d2 = 0.0, c2 = 0.0;
        for (int c = 0; c < n_chan; ++c) {
            const std::complex<double> g(g_corr[(size_t)b * n_chan + c].x,
                                         g_corr[(size_t)b * n_chan + c].y);
            const std::complex<double> r = cpu_corr[crow + c];
            d2 += std::norm(g - r);
            c2 += std::norm(r);
            cc += r;
            ce += cpu_energy[crow + c];
        }
        const double rc = std::sqrt(d2 / std::max(1e-60, c2));
        const double re = std::fabs(ge - ce) / std::max(1e-30, ce);
        max_rel = std::max({max_rel, rc, re});
        const bool ok = rc < TOL && re < TOL;
        pass = pass && ok;
        printf("%-4s cp=%9.2f hops=%4d: CPU |G|=%12.4f E=%12.1f | GPU |G|=%12.4f E=%12.1f | "
               "rel %.2e/%.2e %s\n",
               row_name[t], row_cp(ft, t), n_hops, std::abs(cc), ce, std::abs(gc), ge, rc, re,
               ok ? "OK" : "FAIL");
    }
    printf("%s (max relative error %.3e; %d jobs x 4 rows x %d channels x %d hops)\n",
           pass ? "PASS" : "FAIL", max_rel, n_spec, n_chan, n_hops);

    // --- CHORD 4+4b data ---
    // The voltage quantized to 4+4b bytes (gnss44.hpp), each channel scaled to ~3 lsb rms per
    // component. The fused kernel despreads the same voltages dequantized to float, and that is
    // the reference for the split below: rows E/P/L from the record, and the P_HEAD row from the
    // fused prompt over the head hops alone.
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
        // P_HEAD reference: the fused prompt row of a launch over hops [0, m_head) only. Here
        // every lane of either path holds at most one head hop and the reductions are the same
        // power-of-two trees (the split's extra lanes add exact zeros), so it is exact. A job
        // with m_head = 0 keeps the fused kernel's zero row.
        for (int b = 0; b < n_spec; ++b) {
            if (ftrials[b].m_head == 0)
                continue;
            gnss_cuda::DespreadParams ph = p;
            ph.n_hops = ftrials[b].m_head;
            CK(gnss_cuda::launch_despread(d_deq, d_code, d_jobs, n_spec, n_chan, ph, d_corr,
                                          d_energy, 0));
            CK(cudaDeviceSynchronize());
            const size_t src = (size_t)(4 * b + 1) * n_chan, dst = (size_t)(4 * b + 3) * n_chan;
            CK(cudaMemcpy(fc.data() + dst, d_corr + src, n_chan * sizeof(double2),
                          cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(fe.data() + dst, d_energy + src, n_chan * sizeof(double),
                          cudaMemcpyDeviceToHost));
        }

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

            CK(gnss_cuda::launch_waveform(d_code, d_jobs, n_spec, n_chan, p, nullptr, d_wave,
                                          d_nenergy, 0));
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
            bool split_ok = (cmax == 0.0) && (emax == 0.0);
            printf("CHORD split (waveform + NxM) vs fused despread, N=1: corr %.3e energy %.3e "
                   "%s\n",
                   cmax, emax, split_ok ? "OK (exact)" : "FAIL");

            // --- E/L from the prompt (launch_waveform's rot) vs the fused kernel's own E/L ---
            // Row 0 (the prompt - ds trial) is the prompt rotated by +2 pi (f_c - f_car)
            // ds/cps/fs on the channel at f_c; row 2 the opposite (the sign is empirical). The
            // rotation ignores the delay's phase ramp across one channel (0.06 rad at CHORD,
            // 0.4 rad at this test's 125 kHz channels), so the checks are: P and P_HEAD exact
            // as before, and the rotated prompt much closer to the fused E than the opposite
            // rotation (which is what the L row carries) is.
            {
                std::vector<float> h_rot((size_t)n_spec * n_chan);
                const double delay_s = (double)comb * spacing / cps / fs;
                const double f_car = wc * fs / (2.0 * M_PI);
                for (int b = 0; b < n_spec; ++b)
                    for (int c = 0; c < n_chan; ++c) {
                        const double f_c = (double)cover[c] * fs / (double)fft_len;
                        h_rot[(size_t)b * n_chan + c] =
                            (float)(2.0 * M_PI * (f_c - f_car) * delay_s);
                    }
                float* d_rot = nullptr;
                CK(cudaMalloc(&d_rot, h_rot.size() * sizeof(float)));
                CK(cudaMemcpy(d_rot, h_rot.data(), h_rot.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
                CK(gnss_cuda::launch_waveform(d_code, d_jobs, n_spec, n_chan, p, d_rot, d_wave,
                                              d_nenergy, 0));
                CK(gnss_cuda::launch_correlate_nm(d_qt, d_scale, d_chan_ids, d_wave, d_jobs, n_spec,
                                                  n_chan, n_elem, n_elem, n_chan, p, d_ncorr, 0));
                CK(cudaDeviceSynchronize());
                std::vector<double2> rc((size_t)4 * n_spec * n_chan * n_elem);
                std::vector<double> re((size_t)4 * n_spec * n_chan);
                CK(cudaMemcpy(rc.data(), d_ncorr, rc.size() * sizeof(double2),
                              cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(re.data(), d_nenergy, re.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
                double pmax = 0.0, same = 0.0, flip = 0.0, den = 0.0, emaxr = 0.0;
                for (int b = 0; b < n_spec; ++b)
                    for (int c = 0; c < n_chan; ++c) {
                        auto at = [&](int j) {
                            const size_t k = ((size_t)(4 * b + j) * n_chan + c) * n_elem;
                            return std::complex<double>(rc[k].x, rc[k].y);
                        };
                        auto fused = [&](int j) {
                            const size_t k = (size_t)(4 * b + j) * n_chan + c;
                            return std::complex<double>(fc[k].x, fc[k].y);
                        };
                        for (int j = 1; j < 4; j += 2) { // P, P_HEAD: exact
                            const size_t k = (size_t)(4 * b + j) * n_chan + c;
                            pmax = std::max(pmax, std::abs(at(j) - fused(j))
                                                      / std::max(1e-9, std::abs(fused(j))));
                            pmax = std::max(pmax, std::fabs(re[k] - fe[k])
                                                      / std::max(1e-9, std::fabs(fe[k])));
                        }
                        // E and L energy rows must equal the prompt's
                        const size_t kp = (size_t)(4 * b + 1) * n_chan + c;
                        for (int j = 0; j < 3; j += 2)
                            emaxr = std::max(emaxr, std::fabs(re[kp + (j - 1) * n_chan] - re[kp])
                                                        / std::max(1e-9, re[kp]));
                        same += std::norm(at(0) - fused(0)) + std::norm(at(2) - fused(2));
                        flip += std::norm(at(2) - fused(0)) + std::norm(at(0) - fused(2));
                        den += std::norm(fused(0)) + std::norm(fused(2));
                    }
                const double r_same = std::sqrt(same / std::max(1e-30, den));
                const double r_flip = std::sqrt(flip / std::max(1e-30, den));
                const bool rot_ok =
                    (pmax == 0.0) && (emaxr == 0.0) && (r_same < 0.25) && (r_same < 0.25 * r_flip);
                printf("E/L from the rotated prompt vs fused E/L: P,P_HEAD %.3e (exact), E/L "
                       "energy = P %.3e, E/L residual %.3e (opposite rotation %.3e) %s\n",
                       pmax, emaxr, r_same, r_flip, rot_ok ? "OK" : "FAIL");
                split_ok = split_ok && rot_ok;

                // --- The pack's lane contract: two lanes per slot (P, P_HEAD) must hold the
                // bytes the four-lane layout holds at its P and P_HEAD lanes, head gate and
                // scale included (job 0 gates the head at hop 40, job 1 has none); lanes the
                // chain does not own are untouched. Written at lane_base 16, a chain sharing
                // the axis with another. ---
                {
                    const int n_slot = n_spec, pitch = 48;
                    std::vector<int> h_s2s(n_slot), h_cmap(n_chan);
                    for (int s = 0; s < n_slot; ++s)
                        h_s2s[s] = s;
                    for (int c = 0; c < n_chan; ++c)
                        h_cmap[c] = c;
                    int *d_s2s = nullptr, *d_cmap = nullptr;
                    unsigned char *d_s4 = nullptr, *d_s2 = nullptr;
                    const size_t sbytes = (size_t)n_hops * n_chan * pitch;
                    CK(cudaMalloc(&d_s2s, h_s2s.size() * sizeof(int)));
                    CK(cudaMalloc(&d_cmap, h_cmap.size() * sizeof(int)));
                    CK(cudaMalloc(&d_s4, sbytes));
                    CK(cudaMalloc(&d_s2, sbytes));
                    CK(cudaMemcpy(d_s2s, h_s2s.data(), h_s2s.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(d_cmap, h_cmap.data(), h_cmap.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
                    CK(cudaMemset(d_s4, 0, sbytes));
                    CK(cudaMemset(d_s2, 0, sbytes));
                    CK(gnss_cuda::launch_pack44(d_wave, d_nenergy, d_jobs, d_s2s, n_slot, 4, n_chan,
                                                n_hops, d_cmap, n_chan, 4 * n_slot, 0, pitch, d_s4,
                                                0));
                    CK(gnss_cuda::launch_pack44(d_wave, d_nenergy, d_jobs, d_s2s, n_slot, 2, n_chan,
                                                n_hops, d_cmap, n_chan, 2 * n_slot, 16, pitch, d_s2,
                                                0));
                    CK(cudaDeviceSynchronize());
                    std::vector<unsigned char> h4(sbytes), h2(sbytes);
                    CK(cudaMemcpy(h4.data(), d_s4, sbytes, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(h2.data(), d_s2, sbytes, cudaMemcpyDeviceToHost));
                    long bad = 0, live = 0, stray = 0;
                    for (int m = 0; m < n_hops; ++m)
                        for (int c = 0; c < n_chan; ++c) {
                            const unsigned char* a4 = h4.data() + ((size_t)m * n_chan + c) * pitch;
                            const unsigned char* a2 = h2.data() + ((size_t)m * n_chan + c) * pitch;
                            for (int s = 0; s < n_slot; ++s) {
                                bad += (a2[16 + 2 * s] != a4[4 * s + 1]);     // P
                                bad += (a2[16 + 2 * s + 1] != a4[4 * s + 3]); // P_HEAD
                                live += (a4[4 * s + 1] != 0x88) + (a4[4 * s + 3] != 0x88);
                            }
                            for (int l = 0; l < pitch; ++l)
                                if ((l < 16 || l >= 16 + 2 * n_slot) && a2[l] != 0)
                                    ++stray;
                        }
                    const bool pack_ok = bad == 0 && live > 0 && stray == 0;
                    printf("pack lanes: two per slot = the four-lane P/P_HEAD bytes, %ld "
                           "mismatches over %ld cells (%ld live), %ld bytes outside the "
                           "chain's lanes %s\n",
                           bad, (long)n_hops * n_chan * n_slot * 2, live, stray,
                           pack_ok ? "OK" : "FAIL");
                    split_ok = split_ok && pack_ok;
                    CK(cudaFree(d_s2s));
                    CK(cudaFree(d_cmap));
                    CK(cudaFree(d_s4));
                    CK(cudaFree(d_s2));
                }
                CK(cudaFree(d_rot));
            }

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
                CK(gnss_cuda::launch_waveform(d_code, d_jobs, n_spec, n_chan, p, nullptr, d_wave,
                                              d_nenergy, 0));
                CK(gnss_cuda::launch_correlate_nm(d_qt, d_scale, d_chan_ids, d_wave, d_jobs, n_spec,
                                                  n_chan, n_elem, n_elem, n_chan, p, d_ncorr, 0));
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0, 0));
                for (int it = 0; it < NIT; ++it)
                    CK(gnss_cuda::launch_waveform(d_code, d_jobs, n_spec, n_chan, p, nullptr,
                                                  d_wave, d_nenergy, 0));
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
                    "   synthesis  (waveform)     %8.4f ms  %5.1f%%   <- N^2 cannot absorb this\n"
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
        GnssCudaDespread gpu(bbank, bench_prn, bench_ids, bn_hops, bfs);
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
