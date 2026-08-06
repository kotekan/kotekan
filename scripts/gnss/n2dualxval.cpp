/**
 * @file
 * @brief M3: cross-validate path B (n2k_dual N x M tiles) against path A (launch_correlate_nm)
 *        at deployed CHORD L5 geometry, with REAL replicas. GPU required.
 *
 * Both paths consume the IDENTICAL 4+4b voltage frame, so the data (and its noise) cancels
 * EXACTLY in the comparison: the A-vs-B difference isolates the ONE thing path B changes --
 * the replica is quantized to 4+4b synthetic lanes instead of consumed as float2. Antenna 0
 * carries a noiseless strong signal for a clean quantization-loss figure; antennas 1..31 are
 * realistic (noise rms 2 lsb + signal 1 lsb).
 *
 * Frame: [n_hops=2048][F=8][E=128] native CHORD layout == the dual correlator's e_in_a
 * verbatim (that identity is the point of path B). Synthetic input: lane 4*i+t = spec i's
 * E/P/L/P_HEAD replica, quantized with per-lane scale, P_HEAD = P gated to hops [0, m_head)
 * (zero = offset 0x88 beyond), unused lanes/channels 0x88.
 *
 * The conjugation orientation of the mixed tiles vs path A's convention is detected
 * EMPIRICALLY and printed -- it feeds the M5 consumer, and conjugation errors on this
 * instrument have a history (F-engine conjugation; validate numerically, never by eye).
 *
 * Output: per row-type (E/P/L/PH) relative error stats, amplitude loss in dB, phase rms;
 * plus the M^2 replica cross-terms vs a CPU float reference. Exit 1 on structural failure
 * (NaN, empty, orientation ambiguous); the loss figures are printed for judgment.
 */

#include "GnssCudaDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"

#include <n2k_dual/DualCorrelator.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using cf = std::complex<float>;
using cd = std::complex<double>;

#define CK(x)                                                                                     \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);       \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)

// Deployed CHORD L5 geometry.
static constexpr double FS = 3.2e9, F_OFF = 1176450000.0;
static constexpr int NCH = 8192, TAPS = 4, FFT = 16384;
static constexpr int N_HOPS = 2048; // one tracker record
static constexpr int NFRAME = 8;    // frame channel axis (7 comb channels + 1 spare)
static constexpr int NSA = 128, NSB = 128, N_ELEM = 32;
static constexpr int N_SPEC = 4; // 16 lanes of 128

int main(int argc, char** argv) {
    const unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 4242u;
    // --conj: simulate the CHORD F-ENGINE CONJUGATION (the frame carries conj(sky), measured
    // 2026-07-30) and exercise both sides' compensation -- path A's DespreadParams::conj_data
    // and path B's conj_replica in the pack. Without this the harness only ever proved the
    // unconjugated case, which is why the live tiles read noise on 2026-08-06.
    const bool CONJ = (argc > 2 && std::string(argv[2]) == "--conj");
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // ---- bank + despread (the shipped path-A machinery) ------------------------------------
    const auto* sig = gnss::signal_by_name("GPS_L5_Q_NH");
    if (!sig) {
        printf("no signal descriptor\n");
        return 1;
    }
    std::vector<int> prns{3, 7, 13, 25};
    gnss::ChannelizedReplicaBank bank(*sig, FS, F_OFF, NCH, TAPS, dsp::Window::Hamming, prns);

    std::vector<int> chan_ids_global; // stride-16 comb, as deployed on cx19 GPU0
    for (int k = 0; k < 7; k++)
        chan_ids_global.push_back(5972 + 16 * k);
    const int n_chan = (int)chan_ids_global.size();

    GnssCudaDespread despr(bank, N_SPEC, chan_ids_global, N_HOPS, FS, F_OFF);

    const long long window_start = 1000000LL * FFT; // hop-aligned absolute sample
    std::vector<GnssCudaDespread::Spec> specs(N_SPEC);
    std::vector<int> covering(n_chan);
    for (int c = 0; c < n_chan; c++)
        covering[c] = c;
    for (int i = 0; i < N_SPEC; i++) {
        specs[i].p = i;
        specs[i].cp_seed = 1234.567 + 2111.3 * i;
        specs[i].spacing_chips = 0.5;
        specs[i].doppler_hz = -2500.0 + 1300.0 * i;
        specs[i].covering = covering;
    }
    // Park spec 1 just below a combined-code-period boundary so the boundary lands mid-window
    // and the P_HEAD gate (m_head strictly inside (0, n_hops)) actually gets exercised: the
    // window spans ~107k of the 204.6k-chip combined period, so arbitrary seeds miss it half
    // the time (and all four defaults above happen to miss it).
    specs[1].cp_seed = 160000.0;

    // ---- device buffers for path A ---------------------------------------------------------
    const size_t frame_bytes = (size_t)N_HOPS * NFRAME * NSA;
    const size_t wave_n = (size_t)3 * N_SPEC * n_chan * N_HOPS;
    const size_t corr_n = (size_t)4 * N_SPEC * n_chan * N_ELEM;
    const size_t energy_n = (size_t)4 * N_SPEC * n_chan;

    unsigned char* d_frame;
    float2* d_wave;
    double2* d_corr;
    double* d_energy;
    gnss_cuda::DespreadJob* d_jobs;
    float* d_chan_scale;
    int* d_chan_idx;
    CK(cudaMalloc(&d_frame, frame_bytes));
    CK(cudaMalloc(&d_wave, wave_n * sizeof(float2)));
    CK(cudaMalloc(&d_corr, corr_n * sizeof(double2)));
    CK(cudaMalloc(&d_energy, energy_n * sizeof(double)));
    CK(cudaMalloc(&d_jobs, N_SPEC * sizeof(gnss_cuda::DespreadJob)));
    CK(cudaMalloc(&d_chan_scale, n_chan * sizeof(float)));
    CK(cudaMalloc(&d_chan_idx, n_chan * sizeof(int)));

    std::vector<float> h_scale(n_chan, 1.0f); // lsb units straight through
    std::vector<int> h_chan_idx(covering);    // frame channels 0..6 (LOCAL indices)
    CK(cudaMemcpy(d_chan_scale, h_scale.data(), n_chan * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_chan_idx, h_chan_idx.data(), n_chan * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_frame, 0x88, frame_bytes)); // zero frame for the wave-materialising pass

    // ---- pass 1: materialise the replicas (wave) and the jobs (m_head) ---------------------
    despr.enqueue_batch_nm(d_frame, d_chan_scale, d_chan_idx, d_wave, N_ELEM, NSA, NFRAME,
                           window_start, specs, d_jobs, d_corr, d_energy, nullptr, false);
    CK(cudaDeviceSynchronize());

    std::vector<cf> wave(wave_n);
    CK(cudaMemcpy(wave.data(), d_wave, wave_n * sizeof(float2), cudaMemcpyDeviceToHost));
    std::vector<gnss_cuda::DespreadJob> jobs(N_SPEC);
    CK(cudaMemcpy(jobs.data(), d_jobs, N_SPEC * sizeof(gnss_cuda::DespreadJob),
                  cudaMemcpyDeviceToHost));

    auto wat = [&](int spec, int trial, int c, int m) -> cf {
        return wave[((size_t)(3 * spec + trial) * n_chan + c) * N_HOPS + m];
    };

    printf("n2dualxval: %d specs x %d chan x %d hops, seed %u\n", N_SPEC, n_chan, N_HOPS, seed);
    for (int i = 0; i < N_SPEC; i++)
        printf("  spec %d: PRN slot %d dop %+.0f Hz cp %.3f m_head %d n_chips %d\n", i,
               specs[i].p, specs[i].doppler_hz, specs[i].cp_seed, jobs[i].m_head, jobs[i].n_chips);

    // ---- build the REAL frame: noise + per-antenna signal from the P replica ---------------
    // Antenna 0: signal only, amplitude 5 lsb (clean quantization-loss channel).
    // Antennas 1..31: noise rms 2 lsb + signal amplitude 1 lsb, random per-(spec,antenna) phase.
    std::vector<cd> gains((size_t)N_SPEC * N_ELEM);
    for (auto& g : gains) {
        double ph = 2.0 * M_PI * uni(rng);
        g = cd(cos(ph), sin(ph));
    }
    std::vector<uint8_t> h_frame(frame_bytes, 0x88);
    for (int m = 0; m < N_HOPS; m++)
        for (int c = 0; c < n_chan; c++)
            for (int e = 0; e < N_ELEM; e++) {
                cd v(0.0, 0.0);
                if (e > 0) {
                    v = cd(2.0 * gauss(rng), 2.0 * gauss(rng));
                }
                const double amp = (e == 0) ? 5.0 : 1.0;
                for (int i = 0; i < N_SPEC; i++) {
                    cf w = wat(i, 1, c, m); // PROMPT replica
                    const cd s = amp * gains[(size_t)i * N_ELEM + e] * cd(w.real(), w.imag());
                    // The F-engine emits conj(sky); path A undoes it with conj_data, path B
                    // by injecting a conjugated replica.
                    v += CONJ ? std::conj(s) : s;
                }
                auto q = [](double x) {
                    int r = (int)lrint(x);
                    return r < -7 ? -7 : (r > 7 ? 7 : r);
                };
                h_frame[((size_t)m * NFRAME + c) * NSA + e] =
                    (uint8_t)(((q(v.real()) + 8) << 4) | ((q(v.imag()) + 8) & 0xf));
            }
    CK(cudaMemcpy(d_frame, h_frame.data(), frame_bytes, cudaMemcpyHostToDevice));

    // ---- path A on the real frame ----------------------------------------------------------
    despr.enqueue_batch_nm(d_frame, d_chan_scale, d_chan_idx, d_wave, N_ELEM, NSA, NFRAME,
                           window_start, specs, d_jobs, d_corr, d_energy, nullptr, CONJ);
    CK(cudaDeviceSynchronize());
    std::vector<double2> corr_a(corr_n);
    CK(cudaMemcpy(corr_a.data(), d_corr, corr_n * sizeof(double2), cudaMemcpyDeviceToHost));
    auto ca = [&](int spec, int row, int c, int e) -> cd {
        const double2& v = corr_a[((size_t)(4 * spec + row) * n_chan + c) * N_ELEM + e];
        return cd(v.x, v.y);
    };

    // ---- pack the synthetic lanes: quantize wave -> 4+4b, lane 4i+t ------------------------
    std::vector<uint8_t> h_synth((size_t)N_HOPS * NFRAME * NSB, 0x88);
    std::vector<double> lane_scale((size_t)4 * N_SPEC, 1.0);
    for (int i = 0; i < N_SPEC; i++)
        for (int t = 0; t < 4; t++) {
            const int wrow = (t == 3) ? 1 : t; // P_HEAD packs the PROMPT, gated below
            float peak = 0.0f;
            for (int c = 0; c < n_chan; c++)
                for (int m = 0; m < N_HOPS; m++) {
                    cf w = wat(i, wrow, c, m);
                    peak = std::max({peak, std::abs(w.real()), std::abs(w.imag())});
                }
            const double s = (peak > 0.0f) ? 7.0 / peak : 1.0;
            lane_scale[(size_t)4 * i + t] = s;
            const int mlim = (t == 3) ? jobs[i].m_head : N_HOPS;
            for (int c = 0; c < n_chan; c++)
                for (int m = 0; m < mlim; m++) {
                    cf w = wat(i, wrow, c, m);
                    auto q = [](double x) {
                        int r = (int)lrint(x);
                        return r < -7 ? -7 : (r > 7 ? 7 : r);
                    };
                    const double wi = CONJ ? -(double)w.imag() : (double)w.imag();
                    h_synth[((size_t)m * NFRAME + c) * NSB + (4 * i + t)] =
                        (uint8_t)(((q(s * w.real()) + 8) << 4) | ((q(s * wi) + 8) & 0xf));
                }
        }
    unsigned char* d_synth;
    CK(cudaMalloc(&d_synth, h_synth.size()));
    CK(cudaMemcpy(d_synth, h_synth.data(), h_synth.size(), cudaMemcpyHostToDevice));

    // ---- path B: dual correlator, mixed tiles ----------------------------------------------
    const int NS = NSA + NSB;
    auto ntiles = [](int ns) { return (ns / 16) * (ns / 16 + 1) / 2; };
    const size_t nvis = (size_t)NFRAME * ntiles(NS) * 512;
    int* d_vis;
    CK(cudaMalloc(&d_vis, nvis * 4));
    CK(cudaMemset(d_vis, 0xa5, nvis * 4));

    std::vector<uint32_t> h_rm((size_t)(N_HOPS / 1024) * NFRAME * 32, 0xffffffffu);
    uint* d_rm;
    CK(cudaMalloc(&d_rm, h_rm.size() * 4));
    CK(cudaMemcpy(d_rm, h_rm.data(), h_rm.size() * 4, cudaMemcpyHostToDevice));

    n2k_dual::DualCorrelator dual(NSA, NSB, NFRAME);
    dual.launch(d_vis, (const int8_t*)d_frame, (const int8_t*)d_synth, d_rm, 1, N_HOPS, nullptr,
                true);

    std::vector<int> vis(nvis);
    CK(cudaMemcpy(vis.data(), d_vis, nvis * 4, cudaMemcpyDeviceToHost));
    // V[i_global=128+lane][j=antenna] at freq f: tile (ihi=8+lane/16, jhi=j/16).
    auto vb = [&](int lane, int f, int e) -> cd {
        int gi = NSA + lane, ihi = gi >> 4, ilo = gi & 15, jhi = e >> 4, jlo = e & 15;
        size_t o = (size_t)f * ntiles(NS) * 512 + 512 * ((size_t)ihi * (ihi + 1) / 2 + jhi)
                   + 32 * ilo + 2 * jlo;
        return cd(vis[o], vis[o + 1]);
    };

    // ---- orientation: does path A match V/s or conj(V)/s? ----------------------------------
    // Decide on the strong clean entry (spec 0, P row, antenna 0), then validate globally.
    {
        cd A = ca(0, 1, 3, 0); // mid channel
        cd V = vb(1, 3, 0) / lane_scale[1];
        double e_plain = std::abs(V - A), e_conj = std::abs(std::conj(V) - A);
        printf("\norientation: |V/s - A| %.3g  |conj(V)/s - A| %.3g  -> path A == %s\n", e_plain,
               e_conj, e_conj < e_plain ? "conj(V_BA)/s" : "V_BA/s");
        if (std::min(e_plain, e_conj) > 0.5 * std::max(e_plain, e_conj)) {
            printf("*** orientation AMBIGUOUS -- something is structurally wrong\n");
            return 1;
        }
    }
    const bool use_conj = std::abs(std::conj(vb(1, 3, 0) / lane_scale[1]) - ca(0, 1, 3, 0))
                          < std::abs(vb(1, 3, 0) / lane_scale[1] - ca(0, 1, 3, 0));

    // ---- compare, per row-type -------------------------------------------------------------
    const char* row_name[4] = {"E ", "P ", "L ", "PH"};
    printf("\n%-4s %10s %10s %12s %12s   (entries with |A| above 10%% of row max)\n", "row",
           "med rel", "max rel", "amp dB", "phase mrad");
    int structural_fail = 0;
    for (int t = 0; t < 4; t++) {
        std::vector<double> rel, phase_mr;
        double amp_num = 0.0, amp_den = 0.0;
        double amax = 0.0;
        for (int i = 0; i < N_SPEC; i++)
            for (int c = 0; c < n_chan; c++)
                for (int e = 0; e < N_ELEM; e++)
                    amax = std::max(amax, std::abs(ca(i, t, c, e)));
        for (int i = 0; i < N_SPEC; i++)
            for (int c = 0; c < n_chan; c++)
                for (int e = 0; e < N_ELEM; e++) {
                    cd A = ca(i, t, c, e);
                    cd V = vb(4 * i + t, c, e) / lane_scale[(size_t)4 * i + t];
                    cd B = use_conj ? std::conj(V) : V;
                    if (std::isnan(std::abs(B)) || std::isnan(std::abs(A)))
                        structural_fail = 1;
                    if (std::abs(A) < 0.1 * amax)
                        continue;
                    rel.push_back(std::abs(B - A) / std::abs(A));
                    amp_num += std::abs(B);
                    amp_den += std::abs(A);
                    phase_mr.push_back(std::arg(B / A) * 1e3);
                }
        if (rel.empty()) {
            printf("%-4s   -- no entries above threshold (m_head=0 head row?)\n", row_name[t]);
            continue;
        }
        std::sort(rel.begin(), rel.end());
        double prms = 0.0;
        for (double p : phase_mr)
            prms += p * p;
        prms = sqrt(prms / phase_mr.size());
        printf("%-4s %10.4g %10.4g %+12.4f %12.3f\n", row_name[t], rel[rel.size() / 2],
               rel.back(), 20.0 * log10(amp_num / amp_den), prms);
    }

    // ---- the clean channel: pure quantization loss -----------------------------------------
    {
        cd A = ca(0, 1, 3, 0);
        cd V = vb(1, 3, 0) / lane_scale[1];
        cd B = use_conj ? std::conj(V) : V;
        printf("\nclean antenna 0 (no noise), spec 0 P mid-channel:\n"
               "  A = %+.6g%+.6gi   B = %+.6g%+.6gi   |B|/|A| = %.6f (%+.4f dB)  dphase %.3f "
               "mrad\n",
               A.real(), A.imag(), B.real(), B.imag(), std::abs(B) / std::abs(A),
               20.0 * log10(std::abs(B) / std::abs(A)), std::arg(B / A) * 1e3);
    }

    // ---- M^2 block: replica cross-terms vs CPU float reference -----------------------------
    // V[128+4i+ta][128+4i+tb] should equal s_a s_b sum_m,c w_ta conj(w_tb) (per channel f=c).
    {
        printf("\nM^2 replica cross-terms (spec 0, mid channel): tile vs float reference\n");
        int i = 0, c = 3;
        for (int ta = 1; ta < 3; ta++) { // <P,E> and <L,P>-style neighbours
            int tb = ta - 1;
            int gi = NSA + 4 * i + ta, gj = NSA + 4 * i + tb;
            int ihi = gi >> 4, ilo = gi & 15, jhi = gj >> 4, jlo = gj & 15;
            size_t o = (size_t)c * ntiles(NS) * 512 + 512 * ((size_t)ihi * (ihi + 1) / 2 + jhi)
                       + 32 * ilo + 2 * jlo;
            cd V = cd(vis[o], vis[o + 1])
                   / (lane_scale[(size_t)4 * i + ta] * lane_scale[(size_t)4 * i + tb]);
            // The tiles hold the PACKED replica, which is conj(R) when the F-engine
            // conjugation is being compensated -- so the reference must be conjugated too,
            // or this reads as a 0.4 relative error that is purely a convention mismatch.
            cd R(0.0, 0.0);
            for (int m = 0; m < N_HOPS; m++) {
                cf a = wat(i, ta, c, m), b = wat(i, tb, c, m);
                cd A = cd(a.real(), a.imag()), B = cd(b.real(), b.imag());
                if (CONJ) { A = std::conj(A); B = std::conj(B); }
                R += A * std::conj(B);
            }
            printf("  <%s,%s>: tile %+.5g%+.5gi  float %+.5g%+.5gi  rel %.3g\n", row_name[ta],
                   row_name[tb], V.real(), V.imag(), R.real(), R.imag(),
                   std::abs(V - R) / std::abs(R));
        }
    }

    printf(structural_fail ? "\n*** STRUCTURAL FAIL (NaN)\n" : "\ndone.\n");
    return structural_fail;
}
