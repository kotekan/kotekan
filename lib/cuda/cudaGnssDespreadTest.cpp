// Standalone CPU-vs-GPU validation for the fused GNSS despread kernel (G1a gate).
//
// Builds the CPU reference with the real pipeline code (ChannelizedReplicaBank::hoprate_filter/
// hoprate_stream + gnss::channelized_despread), synthesizes a data window (replica-as-signal at a
// known cp/doppler + deterministic pseudo-noise), then runs gnss_cuda::launch_despread on the same
// inputs and compares per-channel correlation + energy. Pass: <=1e-5 relative on every output.
//
//   ./cuda_gnss_despread_test            (prints PASS/FAIL per trial + max relative error)

#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "GnssCudaDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssChannelizedDespread.hpp"
#include "gnssSignal.hpp"
#include "gpsCACode.hpp"
#include "pfbPrototype.hpp"

#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <complex>
#include <cstdio>
#include <cstdlib>
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
    // Default: L1 C/A at the GX10 front-end geometry (5 MSPS real, N=20 channels, 4-tap hamming
    // PFB). Optional argv: [signal_name] [N] [n_hops] [cp_true] -- e.g.
    //   `cuda_gnss_despread_test GPS_L2C_CM 10` runs the TDM (zero-stuffed comb) signal at the
    //   live L2C geometry (5000-hop records), the case that validated the GPU despread during
    //   the 2026-07-18 L2C zero-cert hunt.
    //   `cuda_gnss_despread_test GPS_L2C_CL 10 5000` is the L2C CL PILOT at the live record
    //   shape: 5000-hop (20 ms) records despread against a rolling SEGMENT of the 1.5 s code
    //   (record != code period -- the default n_hops = repl_period_hops would be 375,000 here,
    //   a whole 1.5 s window, which is not what production runs). The n_hops override exists
    //   for exactly this partial-period case.
    //   `cuda_gnss_despread_test GPS_L2C_CL 10 5000 765749.75` starts the prompt 1500.25
    //   component chips before the CL code END, so the record CROSSES the 767250->0 wrap
    //   mid-window -- the code-index wrap path a 20 ms sub-period record hits once per 1.5 s.
    // EXPECTED DURATION: seconds for L1CA/CM, but ~13-16 min per CL invocation (measured
    // twice, 2026-07-28) -- the CPU reference regenerates a hoprate replica per row plus the
    // 32-PRN bench tail, and every code-table walk touches CL's 1,534,500-comb-chip code
    // (75x CM's). Not a hang; it exits 0. Budget accordingly, and NEVER run any of these
    // against a live node -- the bench tail saturates the GPU and the soak drops frames
    // (measured 2026-07-28: 135k L5 frames in ~10 min; see data_quality_windows.md).
    // The GPU side below is fully signal-generic (bank-driven), mirroring GnssCudaDespread's
    // job construction -- it was L1CA-hardcoded before, which made non-L1 runs vacuous.
    const double fs = 5.0e6, f_off = 1.25e6;
    const int N = (argc > 2) ? atoi(argv[2]) : 20;
    const int taps = 4;
    const int prn = 19;
    const double dop = -2900.0;
    const long long window_start = 40LL * 125 * 3600; // an "aged" absolute anchor (~1 hr of hops)

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
    // Signal-generic, mirroring GnssCudaDespread's job construction: COMBINED-stream chip rate
    // and code table (zero-stuffed for TDM), cp/spacing scaled by comb_mult below.
    const int comb = bank.comb_mult();
    const double cps =
        bank.eff_chip_rate() / fs * (1.0 + bank.code_doppler_sign * dop / sig->carrier_hz);
    const double wc = 2.0 * M_PI * (f_off + dop) / fs;

    const auto& fcode = bank.full_code(0); // slot 0 == our PRN
    std::vector<int8_t> code8(fcode.begin(), fcode.end());

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
    for (int b = 0; b < n_spec; ++b) // component -> combined chips, as production does
        jobs[b] = {(double)comb * ftrials[b].cp_prompt, (double)comb * spacing,
                   cps,           1.0 / cps,
                   wc,            0,         (int)code8.size(), all_mask,
                   d_phiA,        d_phiB,    filt.n_chips,  ftrials[b].m_head};
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

    // ===================== VOLTAGE PEEL + ANALYTIC ADD-BACK (path B) =====================
    // docs/gnss_voltage_peel_live.md. Three claims, checked in order, all against the SAME
    // replica the despread above correlates with -- which is the point of testing them here
    // rather than in a separate program: the add-back is only exact if the waveform the peel
    // subtracts and the reference the despread correlates are the same object.
    //
    //   (1) turning the cross-terms ON does not perturb corr/energy. The kernel had to reorder
    //       its trials to {P, E, L} so the prompt replica is live when E/L are formed; that
    //       reorder must be a numerical no-op.
    //   (2) the cross-terms <R_P, R_E> / <R_P, R_L> (whole and head-restricted) match a CPU
    //       reference. These are the PFB's true code-triangle values -- the reason the add-back
    //       must not use the textbook 0.5.
    //   (3) THE IDENTITY: despread the peeled voltage, add back a*<R_P,R_k>, and recover the
    //       un-peeled correlation -- for a gain that is deliberately WRONG (and sign-flipped
    //       across the code-period boundary), because the identity holds for any gain.
    {
        printf("\n-- voltage peel (path B) --\n");
        const int m_head_peel = ftrials[0].m_head; // 40 of 125: a real boundary mid-record
        const int n_x = 4 * n_spec;                // xcorr rows per job: E, L, E|head, L|head

        // ---- (1)+(2): re-run the despread with cross-terms enabled ----
        double2* d_xcorr;
        CK(cudaMalloc(&d_xcorr, (size_t)n_x * n_chan * sizeof(double2)));
        CK(gnss_cuda::launch_despread(d_data, d_code, d_jobs, n_spec, n_chan, p, d_corr, d_energy,
                                      0, d_xcorr));
        CK(cudaDeviceSynchronize());
        std::vector<double2> g_corr_xc((size_t)n_out * n_chan), g_x((size_t)n_x * n_chan);
        std::vector<double> g_energy_xc((size_t)n_out * n_chan);
        CK(cudaMemcpy(g_corr_xc.data(), d_corr, g_corr_xc.size() * sizeof(double2),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g_energy_xc.data(), d_energy, g_energy_xc.size() * sizeof(double),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g_x.data(), d_xcorr, g_x.size() * sizeof(double2), cudaMemcpyDeviceToHost));

        // Does enabling the cross-terms (XC=off vs on, the two launch_despread calls above)
        // leave corr/energy unchanged? It is a MATH no-op -- acc[t]/e[t] sum the same terms
        // in the same order either way -- but NOT bitwise: with XC on the kernel keeps r_P
        // live and the xc MACs raise register pressure, so nvcc contracts/schedules the
        // identical corr/energy float MACs differently, shifting them ~1 ULP (~1e-7 rel,
        // and DATA-DEPENDENT: seen 1.9e-7..2.7e-7 across runs/machines). Gate at the suite's
        // own float floor (the plain despread-vs-CPU check above lands ~6e-7), NOT at
        // bit-identity -- 1e-12 was wrong for a float quantity. A real leak from the xc path
        // into acc/e would be orders larger. STATIC PROOF it is codegen, not a leak: in the
        // kernel's accumulation region xc[] is never written into acc[]/e[] on any path, so
        // no cross-term value can reach corr/energy -- only the compiler's float choices,
        // which differ because XC changes register liveness, can move them.
        const double XC_INVARIANCE_TOL = 1e-6;
        double max_reorder = 0.0;
        for (size_t i = 0; i < g_corr_xc.size(); ++i) {
            const std::complex<double> a(g_corr[i].x, g_corr[i].y), b(g_corr_xc[i].x,
                                                                     g_corr_xc[i].y);
            max_reorder = std::max(max_reorder, std::abs(b - a) / std::max(1e-30, std::abs(a)));
            max_reorder = std::max(max_reorder, std::fabs(g_energy_xc[i] - g_energy[i])
                                                    / std::max(1e-30, g_energy[i]));
        }
        printf("cross-terms off->on perturb corr/energy by %.3e (float codegen; tol %.0e) %s\n",
               max_reorder, XC_INVARIANCE_TOL, (max_reorder < XC_INVARIANCE_TOL) ? "OK" : "FAIL");
        pass = pass && (max_reorder < XC_INVARIANCE_TOL);

        // CPU cross-terms: <R_P, R_k> = sum_hop r_P conj(r_k), whole window and head-restricted.
        double max_xc = 0.0;
        for (int b = 0; b < n_spec; ++b) {
            const auto rP = bank.hoprate_stream(filt, 0, window_start, ftrials[b].cp_prompt, dop,
                                                n_hops);
            for (int k = 0; k < 2; ++k) { // 0 = Early, 1 = Late
                const double cpk = ftrials[b].cp_prompt + (k == 0 ? -spacing : spacing);
                const auto rk = bank.hoprate_stream(filt, 0, window_start, cpk, dop, n_hops);
                for (int half = 0; half < 2; ++half) { // 0 = whole window, 1 = head only
                    const int hi = half ? m_head_peel : n_hops;
                    std::complex<double> cx(0.0, 0.0), gx(0.0, 0.0);
                    for (int c = 0; c < n_chan; ++c) {
                        for (int m = 0; m < hi; ++m)
                            cx += std::complex<double>(rP[c][m])
                                  * std::conj(std::complex<double>(rk[c][m]));
                        const size_t row = (size_t)(4 * b + k + 2 * half) * n_chan + c;
                        gx += std::complex<double>(g_x[row].x, g_x[row].y);
                    }
                    // job 1 has m_head 0, so its head rows are legitimately zero
                    if (half && ftrials[b].m_head == 0)
                        continue;
                    const double rel = std::abs(gx - cx) / std::max(1e-30, std::abs(cx));
                    max_xc = std::max(max_xc, rel);
                }
            }
        }
        printf("reference cross-terms <R_P,R_E>/<R_P,R_L> (whole + head): max rel %.3e %s\n",
               max_xc, (max_xc < 1e-5) ? "OK" : "FAIL");
        pass = pass && (max_xc < 1e-5);

        // ---- the peel itself ----
        // A deliberately WRONG gain, sign-FLIPPED across the boundary: |a| is not the true
        // amplitude and its phase is arbitrary. Both are on purpose -- a feed-forward gain is
        // never exactly right, and the add-back must not care. The flip exercises the two-segment
        // path (one gain across a sign flip is the 2026-07-12 "peels 0.9 dB" failure).
        std::vector<float2> gains((size_t)2 * n_chan);
        for (int c = 0; c < n_chan; ++c) {
            const double ph = 0.37 * c + 0.11;
            const float re = (float)(2.4 * std::cos(ph)), im = (float)(2.4 * std::sin(ph));
            gains[c] = make_float2(re, im);                    // head
            gains[n_chan + c] = make_float2(-re, -im);         // tail: the overlay flipped
        }
        float2* d_gain;
        float2* d_resid;
        gnss_cuda::PeelJob* d_pjobs;
        CK(cudaMalloc(&d_gain, gains.size() * sizeof(float2)));
        CK(cudaMalloc(&d_resid, (size_t)n_chan * n_hops * sizeof(float2)));
        CK(cudaMalloc(&d_pjobs, sizeof(gnss_cuda::PeelJob)));
        CK(cudaMemcpy(d_gain, gains.data(), gains.size() * sizeof(float2),
                      cudaMemcpyHostToDevice));
        gnss_cuda::PeelJob pj{(double)comb * ftrials[0].cp_prompt,
                              cps,
                              1.0 / cps,
                              wc,
                              0,
                              (int)code8.size(),
                              all_mask,
                              d_phiA,
                              d_phiB,
                              filt.n_chips,
                              m_head_peel,
                              d_gain,
                              d_gain + n_chan};
        CK(cudaMemcpy(d_pjobs, &pj, sizeof(pj), cudaMemcpyHostToDevice));
        CK(gnss_cuda::launch_peel(d_data, d_code, d_pjobs, 1, n_chan, p, d_resid, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float2> g_resid((size_t)n_chan * n_hops);
        CK(cudaMemcpy(g_resid.data(), d_resid, g_resid.size() * sizeof(float2),
                      cudaMemcpyDeviceToHost));

        // CPU residual: data - a_seg * R_P, sample by sample.
        const auto rP0 =
            bank.hoprate_stream(filt, 0, window_start, ftrials[0].cp_prompt, dop, n_hops);
        double max_res = 0.0, res_scale = 0.0;
        for (int c = 0; c < n_chan; ++c)
            for (int m = 0; m < n_hops; ++m) {
                const std::complex<double> a =
                    (m < m_head_peel)
                        ? std::complex<double>(gains[c].x, gains[c].y)
                        : std::complex<double>(gains[n_chan + c].x, gains[n_chan + c].y);
                const std::complex<double> want =
                    std::complex<double>(data_ch[c][m]) - a * std::complex<double>(rP0[c][m]);
                const std::complex<double> got(g_resid[(size_t)c * n_hops + m].x,
                                               g_resid[(size_t)c * n_hops + m].y);
                max_res = std::max(max_res, std::abs(got - want));
                res_scale = std::max(res_scale, std::abs(want));
            }
        const double res_rel = max_res / std::max(1e-30, res_scale);
        printf("peel residual vs CPU (per-sample, %d chan x %d hops, 2 gain segments): "
               "max rel %.3e %s\n",
               n_chan, n_hops, res_rel, (res_rel < 1e-5) ? "OK" : "FAIL");
        pass = pass && (res_rel < 1e-5);

        // ---- THE ADD-BACK IDENTITY ----
        // Despread the residual with the SAME job, then restore the peel's known contribution:
        //     V[k] = V'[k] + a_head*<R_P 1_head, R_k> + a_tail*(<R_P,R_k> - <R_P 1_head,R_k>)
        // and compare against the un-peeled correlation. This is the whole design in one number.
        gnss_cuda::DespreadParams pr = p;
        pr.data_stride = n_hops; // the residual is packed
        double2* d_corr_r;
        double* d_energy_r;
        CK(cudaMalloc(&d_corr_r, (size_t)4 * n_chan * sizeof(double2)));
        CK(cudaMalloc(&d_energy_r, (size_t)4 * n_chan * sizeof(double)));
        CK(gnss_cuda::launch_despread(d_resid, d_code, d_jobs, 1, n_chan, pr, d_corr_r,
                                      d_energy_r, 0));
        CK(cudaDeviceSynchronize());
        std::vector<double2> r_corr((size_t)4 * n_chan);
        CK(cudaMemcpy(r_corr.data(), d_corr_r, r_corr.size() * sizeof(double2),
                      cudaMemcpyDeviceToHost));

        double max_ab = 0.0;
        for (int t = 0; t < 4; ++t) { // E, P, L, P_HEAD
            std::complex<double> vfull(0.0, 0.0), vtrue(0.0, 0.0);
            for (int c = 0; c < n_chan; ++c) {
                const std::complex<double> ah(gains[c].x, gains[c].y),
                    at(gains[n_chan + c].x, gains[n_chan + c].y);
                std::complex<double> xw(0.0, 0.0), xh(0.0, 0.0); // <R_P,R_t> whole / head
                if (t == 0 || t == 2) {
                    const int k = (t == 0) ? 0 : 1;
                    const size_t rw = (size_t)k * n_chan + c, rh = (size_t)(k + 2) * n_chan + c;
                    xw = std::complex<double>(g_x[rw].x, g_x[rw].y);
                    xh = std::complex<double>(g_x[rh].x, g_x[rh].y);
                } else if (t == 1) { // <R_P,R_P> = energy rows 1 (whole) and 3 (head)
                    xw = g_energy_xc[(size_t)1 * n_chan + c];
                    xh = g_energy_xc[(size_t)3 * n_chan + c];
                } else { // R_PH is R_P on the head: whole == head == E_head, tail contributes 0
                    xw = g_energy_xc[(size_t)3 * n_chan + c];
                    xh = xw;
                }
                const std::complex<double> vres(r_corr[(size_t)t * n_chan + c].x,
                                                r_corr[(size_t)t * n_chan + c].y);
                vfull += vres + ah * xh + at * (xw - xh);
                vtrue += std::complex<double>(g_corr_xc[(size_t)t * n_chan + c].x,
                                              g_corr_xc[(size_t)t * n_chan + c].y);
            }
            const double rel = std::abs(vfull - vtrue) / std::max(1e-30, std::abs(vtrue));
            max_ab = std::max(max_ab, rel);
            printf("  add-back %-4s: |V_full|=%12.4f  |V_unpeeled|=%12.4f  rel %.3e %s\n",
                   row_name[t], std::abs(vfull), std::abs(vtrue), rel, (rel < 1e-5) ? "OK" : "FAIL");
        }
        printf("ADD-BACK IDENTITY: max rel %.3e %s  (gain deliberately wrong + sign-flipped -- "
               "the identity does not depend on it)\n",
               max_ab, (max_ab < 1e-5) ? "OK" : "FAIL");
        pass = pass && (max_ab < 1e-5);

        CK(cudaFree(d_xcorr));
        CK(cudaFree(d_gain));
        CK(cudaFree(d_resid));
        CK(cudaFree(d_pjobs));
        CK(cudaFree(d_corr_r));
        CK(cudaFree(d_energy_r));
        // restore the XC-off correlations for anything downstream that reuses d_corr/d_energy
        CK(cudaMemcpy(d_corr, g_corr.data(), g_corr.size() * sizeof(double2),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_energy, g_energy.data(), g_energy.size() * sizeof(double),
                      cudaMemcpyHostToDevice));
    }

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

        // --- CHORD split (waveform generation + N x M correlation) vs the FUSED kernel ---
        // The CHORD path generates the replica once and correlates it against N antennas, instead
        // of regenerating it inside the MAC. At N=1 it must reproduce the fused kernel EXACTLY:
        // same gather, same carrier, same combination, same accumulation order. Not "close" --
        // the two paths do identical arithmetic on identical inputs, so anything but 0.0 here
        // means the split has drifted from the fused path, which is the failure mode that shows
        // up downstream only as an unexplained correlation-amplitude discrepancy.
        {
            const int n_elem = 1;
            // The fused kernel reads [chan][hop]; the CHORD correlator reads the voltage in its
            // NATIVE frame order [hop][chan][elem]. Same bytes, transposed.
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
            std::vector<double> ne((size_t)4 * n_spec * n_chan), qe((size_t)n_out * n_chan);
            CK(cudaMemcpy(nc.data(), d_ncorr, nc.size() * sizeof(double2), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ne.data(), d_nenergy, ne.size() * sizeof(double), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(qe.data(), d_qe, qe.size() * sizeof(double), cudaMemcpyDeviceToHost));

            double cmax = 0.0, emax = 0.0;
            for (int b = 0; b < n_spec; ++b)
                for (int j = 0; j < 4; ++j)
                    for (int c = 0; c < n_chan; ++c) {
                        const size_t kf = (size_t)(p.out_rows_spec * b + j) * n_chan + c;
                        const size_t ks = ((size_t)(4 * b + j) * n_chan + c) * n_elem + 0;
                        const std::complex<double> a(qc[kf].x, qc[kf].y);
                        const std::complex<double> bb(nc[ks].x, nc[ks].y);
                        cmax = std::max(cmax, std::abs(bb - a) / std::max(1e-9, std::abs(a)));
                        const size_t ke = (size_t)(4 * b + j) * n_chan + c;
                        emax = std::max(emax, std::abs(ne[ke] - qe[kf])
                                                  / std::max(1e-9, std::abs(qe[kf])));
                    }
            const bool split_ok = (cmax == 0.0) && (emax == 0.0);
            printf("CHORD split (waveform + NxM) vs fused despread, N=1: corr %.3e energy %.3e "
                   "%s\n",
                   cmax, emax, split_ok ? "OK (exact)" : "FAIL");

            // ---- KERNEL SPLIT AT CHORD GEOMETRY (2026-08-05) ----------------------------
            // The tracker runs THIS path, not the fused kernel: enqueue_batch_device launches
            // waveform (replica SYNTHESIS) then correlate_nm (the CORRELATION over elements).
            // Time them separately, because the two answer different questions:
            //   * correlation is what eventually folds into CHORD's N^2
            //   * synthesis NEVER can -- N^2 knows nothing about GNSS codes
            // If synthesis dominates, folding the correlation into N^2 buys ~nothing on the GPU,
            // and that is a planning input worth having measured rather than assumed.
            {
                const int NIT = 200;
                cudaEvent_t e0, e1, e2;
                CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
                // warm-up
                CK(gnss_cuda::launch_waveform(d_code, d_jobs, n_spec, n_chan, p, d_wave,
                                              d_nenergy, 0));
                CK(gnss_cuda::launch_correlate_nm(d_qt, d_scale, d_chan_ids, d_wave, d_jobs,
                                                  n_spec, n_chan, n_elem, n_elem, n_chan, p,
                                                  d_ncorr, 0));
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
                ms_syn /= NIT; ms_cor /= NIT;
                printf("KERNEL SPLIT (%d spec x %d chan x %d elem x %d hops):\n"
                       "   synthesis  (waveform)     %8.4f ms  %5.1f%%   <- N^2 CANNOT absorb this\n"
                       "   correlation(nm, %2d elem) %8.4f ms  %5.1f%%   <- this is the N^2 candidate\n",
                       n_spec, n_chan, n_elem, p.n_hops, ms_syn,
                       100.0 * ms_syn / (ms_syn + ms_cor), n_elem, ms_cor,
                       100.0 * ms_cor / (ms_syn + ms_cor));
                // ELEMENT SWEEP. Synthesis is n_elem-INDEPENDENT (one replica per PRN/chan/hop,
                // broadcast across element lanes); the correlation scales with it. So a
                // 1-element measurement flatters synthesis, and any claim that "the MAC is 0% of
                // the kernel" measured at 1 element does NOT transfer to CHORD's 32. Sweep it.
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
                        CK(gnss_cuda::launch_correlate_nm(d_q2, d_scale, d_chan_ids, d_wave,
                                                          d_jobs, n_spec, n_chan, ne, ne, n_chan,
                                                          p, d_c2, 0));
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
                cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
            }
            pass = pass && split_ok;

            cudaFree(d_qt);
            cudaFree(d_ncorr);

            // --- element axis: N>1 indexing, stride and tail handling ---
            // N=1 above proves the ARITHMETIC matches; it says nothing about whether the element
            // axis is indexed correctly. So: replicate the same voltage into every element, pad
            // the frame's element axis with a POISON byte (0x00 -> (-8,-8), the loudest value the
            // codec can produce -- deliberately not 0x88, which decodes to zero and would hide a
            // stride error), and require every element to reproduce the fused answer exactly.
            //
            // n_elem = 5 with elem_stride = 7 is chosen to be awkward: not a power of two, not a
            // multiple of the 32-wide element block, and strictly less than the stride. That is
            // the shape that catches both a stride bug and the divergent-__syncthreads() tail bug
            // (the block must run a whole number of element passes, predicating the ragged tail
            // rather than letting threads drop out at different barriers).
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
                        const size_t kf = (size_t)(p.out_rows_spec * b + j) * n_chan + c;
                        const std::complex<double> a(qc[kf].x, qc[kf].y);
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

        cudaFree(d_q);
        cudaFree(d_deq);
        cudaFree(d_scale);
        cudaFree(d_qc);
        cudaFree(d_qe);
    }

    // --- GEMM-despread cross-validation (the CHORD-shape path) ---
    // The despread as A = D x R^H: per covering channel, a [1 x T] data row against [T x M]
    // replica columns, int8 tensor GEMM (4 real gemms), batched over channels -- the exact shape
    // chordShapeBench times at Pathfinder/Full scale. THE CLAIM GATED HERE: the GEMM computes the
    // IDENTICAL mathematical object to the despread. So the reference is built from the SAME
    // dequantized integers the GEMM multiplies (data nibbles x int8 replicas): int8 x int8 sums
    // are int32-EXACT, so GEMM-vs-reference must agree to float rounding (~1e-7), gate 1e-6.
    // The separate physics question (int8 REPLICA quantization -- deterministic, no noise to
    // dither it) is reported against the true float reference, not gated: expect ~1e-2 relative
    // on strong rows (~-50 dB power), well under every peel depth we use.
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
                const float inv = (float)(2.1 / std::sqrt(ss / n_hops)); // CHORD loading
                dsc[c] = 1.0f / inv;
                for (int m = 0; m < n_hops; ++m) {
                    int railed = 0;
                    const unsigned char b = gnss_cuda::pack_44(
                        make_float2(data_ch[c][m].real(), data_ch[c][m].imag()), inv, &railed);
                    qbytes[(size_t)c * n_hops + m] = b;
                    // col-major [4 x TP], row 0: element (0,t) at [t*4]
                    hDr[((size_t)c * TP + m) * 4] = (int8_t)(((b >> 4) & 0x0f) - 8);
                    hDi[((size_t)c * TP + m) * 4] = (int8_t)((b & 0x0f) - 8);
                }
            }
            // Replica columns: same hoprate generator as the CPU reference, quantized to int8,
            // HEAD columns zeroed past their m_head (the segmented despread as a COLUMN SHAPE).
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
                        mx = std::max({mx, std::fabs(repl[c][m].real()),
                                       std::fabs(repl[c][m].imag())});
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
                CK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
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
                            (cint(0, c, col) + cint(1, c, col)) * s,   // dr*rr + di*ri
                            (cint(2, c, col) - cint(3, c, col)) * s);  // di*rr - dr*ri
                        // EXACT reference from the same dequantized integers
                        std::complex<double> e(0, 0);
                        const int hi = row_hops(ftrials[col / 4], col % 4);
                        for (int m = 0; m < hi; ++m) {
                            const unsigned char b = qbytes[(size_t)c * n_hops + m];
                            const std::complex<double> dv(
                                (((b >> 4) & 0x0f) - 8) * (double)dsc[c],
                                ((b & 0x0f) - 8) * (double)dsc[c]);
                            const size_t o = ((size_t)c * M + col) * TP + m;
                            const std::complex<double> rv(hRr[o] * (double)rsc[col],
                                                          hRi[o] * (double)rsc[col]);
                            e += dv * std::conj(rv);
                        }
                        ee += e;
                    }
                    const double rel =
                        std::abs(gg - ee) / std::max(1e-30, std::abs(ee));
                    max_exact = std::max(max_exact, rel);
                    if (col == 1) // strong prompt: physics delta vs the true float reference
                        phys_p = std::abs(gg
                                          - std::accumulate(
                                              cpu_corr.begin() + (size_t)col * n_chan,
                                              cpu_corr.begin() + (size_t)(col + 1) * n_chan,
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

            // ...and the SAME path with the voltage peel in front of it: peel -> despread the
            // residual (+ the add-back cross-terms). This is what the peeled chain actually
            // costs per record, and it is the number that gates enabling it live.
            {
                std::vector<GnssCudaDespread::PeelSpec> pspecs(bench_prn);
                for (int q = 0; q < bench_prn; ++q) {
                    pspecs[q] = {q, specs[q].cp_seed, specs[q].doppler_hz, allch,
                                 std::vector<std::complex<float>>((size_t)bN, {0.3f, 0.1f}),
                                 std::vector<std::complex<float>>((size_t)bN, {-0.3f, -0.1f})};
                }
                gnss_cuda::PeelJob* d_pj = nullptr;
                float2 *d_gain = nullptr, *d_res = nullptr;
                double2* d_xc = nullptr;
                CK(cudaMalloc(&d_pj, (size_t)bench_prn * sizeof(gnss_cuda::PeelJob)));
                CK(cudaMalloc(&d_gain, (size_t)2 * bench_prn * bN * sizeof(float2)));
                CK(cudaMalloc(&d_res, (size_t)bN * bn_hops * sizeof(float2)));
                CK(cudaMalloc(&d_xc, (size_t)4 * bench_prn * bN * sizeof(double2)));
                (void)gpu.enqueue_peel_device(d_win, bn_hops, window_start, pspecs, d_pj, d_gain,
                                              d_res, st);
                (void)gpu.enqueue_batch_device(d_res, bn_hops, window_start, specs, d_j, d_c, d_e,
                                               st, nullptr, d_xc);
                CK(cudaStreamSynchronize(st)); // warm-up
                const auto q0 = std::chrono::steady_clock::now();
                for (int r = 0; r < NREC; ++r) {
                    const long long ws = window_start + (long long)r * bn_hops * bfft_len;
                    (void)gpu.enqueue_peel_device(d_win, bn_hops, ws, pspecs, d_pj, d_gain, d_res,
                                                  st);
                    (void)gpu.enqueue_batch_device(d_res, bn_hops, ws, specs, d_j, d_c, d_e, st,
                                                   nullptr, d_xc);
                    CK(cudaStreamSynchronize(st));
                }
                const double qdt =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
                printf("   ^ PEELED path (peel -> despread residual + add-back cross-terms): "
                       "%6.0f rec/s (%.1fx realtime @1kHz, %.2fx the un-peeled cost)\n",
                       NREC / qdt, NREC / qdt / 1000.0, qdt / pdt);
                cudaFree(d_pj);
                cudaFree(d_gain);
                cudaFree(d_res);
                cudaFree(d_xc);
            }
            cudaFree(d_win);
            cudaFree(d_j);
            cudaFree(d_c);
            cudaFree(d_e);
            cudaStreamDestroy(st);
        }
    }
    return pass ? 0 : 1;
}
