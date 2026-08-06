/**
 * @file
 * @brief Bench the CHORD replica-synthesis kernel (launch_waveform) AT CHORD GEOMETRY.
 *
 * WHY A SYNTHETIC BENCH IS ALLOWED HERE, given that the last two attempts to reason about this
 * kernel died on exactly that mistake. The trap was never "synthetic" -- it was GEOMETRY. The
 * gather depth n_chips = ceil((Lf-1)*cps)+2 is 210 at CHORD, ~13 at airspy L5 and ~8 at the unit
 * test, and every number that transferred badly transferred across that. This bench therefore
 * hard-codes CHORD's numbers (below) and its FIRST OUTPUT is the baseline duration, which must
 * reproduce the 2.40 ms ncu measured on cx19 at grid (11,7)x256. If it does not, nothing else
 * printed here means anything and the run says so.
 *
 * The Phi CONTENTS do not affect timing -- only the addresses do, and those are fixed by
 * (inv_cps, n_chips, Lf, n_hops, cp0's fractional part). So the tables are filled with junk.
 *
 * usage: ./wavebench [--jobs N] [--chan N] [--chips N] [--iters N]
 */

#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// ---- CHORD geometry, from config/generated/chord_gnss_cx19.yaml + the L5 signal table --------
static constexpr double FS = 3.2e9;         // sample_rate
static constexpr double CHIP_RATE = 10.23e6;// GPS L5 chipping rate
static constexpr int FFT_LEN = 16384;       // fft_length (= samples per hop)
static constexpr int NUM_TAPS = 4;          // PFB taps -> Lf = 65536, Phi has Lf+1 entries
static constexpr int N_HOPS = 2048;         // hops_per_record
static constexpr int CODE_LEN = 204600;     // GPS_L5_Q_NH: NH baked in, so the period is 20 ms
static constexpr double F_OFFSET = 1176450000.0;

int main(int argc, char** argv) {
    int n_job = 11, n_chan = 7, n_chips = 140, iters = 50, share_phi = 0, phi16 = 0, lockstep = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return atoi(argv[++i]); };
        if (a == "--jobs")
            n_job = next();
        else if (a == "--chan")
            n_chan = next();
        else if (a == "--chips")
            n_chips = next();
        else if (a == "--iters")
            iters = next();
        else if (a == "--share-phi")
            // CEILING TEST for any Doppler-factored scheme: point every job at ONE table, so
            // the tables are perfectly shared and trivially L2-resident. Physically wrong (every
            // PRN gets PRN 0's filter) but the ADDRESSES are what set the time, and this is the
            // best case sharing could ever reach. If it does not win here, it cannot win.
            share_phi = 1;
        else if (a == "--phi16")
            phi16 = 1; // store Phi as __half2: half the bytes, SAME number of scattered requests
        else if (a == "--lockstep")
            // CEILING TEST for hop-ordering. Pick cps so a hop advances an INTEGER number of
            // chips: then frac(C_P) is identical for every hop, every lane in a warp gets the
            // same `base`, and the 32 lanes collapse from ~20 scattered sectors onto ~1. That is
            // the best any locality scheme (e.g. sorting hops by fractional phase) could reach.
            // Physically wrong -- it is not L5's chip rate -- but the ADDRESSES are the point.
            lockstep = 1;
    }

    const int Lf = FFT_LEN * NUM_TAPS;
    const double cps = CHIP_RATE / FS;
    const double inv_cps = 1.0 / cps;
    const int full_chips = (int)std::ceil((double)(Lf - 1) * cps) + 2;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  SMs %d  L2 %.2f MB  shmem/block %.0f KB\n", prop.name,
           prop.multiProcessorCount, prop.l2CacheSize / 1048576.0,
           prop.sharedMemPerBlockOptin / 1024.0);
    printf("geometry: Lf %d  inv_cps %.4f samples/chip  n_hops %d  full span %d chips (capped %d)\n",
           Lf, inv_cps, N_HOPS, full_chips, n_chips);

    gnss_cuda::DespreadParams p{};
    p.n0 = 2950000000000000LL; // ten days of F-engine uptime, as the kernel's fma note assumes
    p.fft_len = FFT_LEN;
    p.n_hops = N_HOPS;
    p.Lf = Lf;
    p.out_rows_spec = 4;
    p.conj_data = 1;
    p.data_stride = N_HOPS;

    // ---- Phi tables. ONE PER JOB, [n_chan][Lf+1], because the tables are Doppler-bucketed and
    // every PRN sits at its own Doppler (GnssCudaDespread::Impl::PhiCache is a per-PRN vector).
    // That is the whole point of this bench: blocks do NOT share Phi.
    const size_t phi_n = (size_t)n_chan * (Lf + 1);
    std::vector<float2> junk(phi_n);
    for (size_t k = 0; k < phi_n; ++k)
        junk[k] = make_float2(1e-3f * (float)(k % 977), 1e-3f * (float)(k % 613));
    std::vector<float2*> dA(n_job, nullptr), dB(n_job, nullptr);
    std::vector<__half2> junk16(phi_n);
    for (size_t k = 0; k < phi_n; ++k)
        junk16[k] = __floats2half2_rn(junk[k].x, junk[k].y);
    const size_t elem = phi16 ? sizeof(__half2) : sizeof(float2);
    const void* src = phi16 ? (const void*)junk16.data() : (const void*)junk.data();
    for (int b = 0; b < n_job; ++b) {
        CK(cudaMalloc(&dA[b], phi_n * elem));
        CK(cudaMalloc(&dB[b], phi_n * elem));
        CK(cudaMemcpy(dA[b], src, phi_n * elem, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB[b], src, phi_n * elem, cudaMemcpyHostToDevice));
    }
    if (lockstep)
        printf("*** --lockstep: cps forced to %.0f chips/hop exactly -> all lanes share one "
               "`base` (locality CEILING test)\n", std::floor(cps * FFT_LEN));
    if (share_phi)
        printf("*** --share-phi: ALL jobs point at job 0's tables (sharing CEILING test)\n");
    printf("Phi footprint: %.1f MB per job-channel slice x %d jobs x %d chan = %.1f MB live\n",
           2 * (Lf + 1) * sizeof(float2) / 1048576.0, n_job, n_chan,
           2.0 * n_job * phi_n * sizeof(float2) / 1048576.0);

    int8_t* d_code = nullptr;
    {
        std::vector<int8_t> code(CODE_LEN);
        for (int i = 0; i < CODE_LEN; ++i)
            code[i] = (int8_t)(((i * 2654435761u) >> 16) & 1 ? 1 : -1);
        CK(cudaMalloc(&d_code, CODE_LEN));
        CK(cudaMemcpy(d_code, code.data(), CODE_LEN, cudaMemcpyHostToDevice));
    }

    std::vector<gnss_cuda::DespreadJob> h_jobs(n_job);
    for (int b = 0; b < n_job; ++b) {
        auto& j = h_jobs[b];
        const double dop = -3500.0 + 700.0 * b; // realistic L5 Doppler spread
        j.cps = lockstep ? std::floor(cps * FFT_LEN) / FFT_LEN : cps * (1.0 - dop / F_OFFSET);
        j.inv_cps = 1.0 / j.cps;
        j.wc = 2.0 * M_PI * (F_OFFSET + dop) / FS;
        // cp0's FRACTIONAL part is what sets `base` and hence the address walk; spread it.
        j.cp0 = 1234.0 + 0.0937 * b;
        j.ds = 0.5; // DLL early/late spacing in chips -- sets how far the three trials' windows sit apart
        j.code_offset = 0;
        j.code_len = CODE_LEN;
        j.chan_mask = (n_chan >= 64) ? ~0ULL : ((1ULL << n_chan) - 1ULL);
        j.phiA = share_phi ? dA[0] : dA[b];
        j.phiB = share_phi ? dB[0] : dB[b];
        j.n_chips = n_chips;
        j.m_head = N_HOPS / 3;
    }
    gnss_cuda::DespreadJob* d_jobs = nullptr;
    CK(cudaMalloc(&d_jobs, n_job * sizeof(gnss_cuda::DespreadJob)));
    CK(cudaMemcpy(d_jobs, h_jobs.data(), n_job * sizeof(gnss_cuda::DespreadJob),
                  cudaMemcpyHostToDevice));

    float2* d_wave = nullptr;
    double* d_energy = nullptr;
    const size_t wave_n = (size_t)3 * n_job * n_chan * N_HOPS;
    CK(cudaMalloc(&d_wave, wave_n * sizeof(float2)));
    CK(cudaMalloc(&d_energy, (size_t)4 * n_job * n_chan * sizeof(double)));

    // ---- baseline -----------------------------------------------------------------------
    // The BASELINE is the 256-wide launch, i.e. what shipped when ncu profiled cx19 -- not
    // whatever WAVE_THREADS currently defaults to. Pinning it explicitly is what keeps the
    // faithfulness gate below meaningful after the default moves.
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    // --phi16 is TIMING ONLY. The accuracy of fp16 Phi is answered offline by
    // scripts/gnss/phibits.cpp (3.3e-04 worst case on a chip step); here the tables are half the
    // size, so the fp32 reference path would read them OUT OF BOUNDS -- do not try to compare
    // `wave` across the two. Report the time and stop.
    if (phi16) {
        CK(gnss_cuda::launch_waveform_tuned(d_code, d_jobs, n_job, n_chan, p, d_wave, d_energy,
                                            1024, 1, 1, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0, 0));
        for (int it = 0; it < iters; ++it)
            CK(gnss_cuda::launch_waveform_tuned(d_code, d_jobs, n_job, n_chan, p, d_wave, d_energy,
                                                1024, 1, 1, 0));
        CK(cudaEventRecord(e1, 0));
        CK(cudaDeviceSynchronize());
        float mh = 0.f;
        CK(cudaEventElapsedTime(&mh, e0, e1));
        mh /= iters;
        printf("\nFP16 Phi (timing only)  fused/1024 %8.4f ms   vs fp32 baseline below\n", mh);
        printf("   half the table bytes, the SAME number of scattered requests\n");
        return 0;
    }
    CK(gnss_cuda::launch_waveform_tuned(d_code, d_jobs, n_job, n_chan, p, d_wave, d_energy, 256, 0, 0, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0, 0));
    for (int it = 0; it < iters; ++it)
        CK(gnss_cuda::launch_waveform_tuned(d_code, d_jobs, n_job, n_chan, p, d_wave, d_energy, 256,
                                            0, 0, 0));
    CK(cudaEventRecord(e1, 0));
    CK(cudaDeviceSynchronize());
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= iters;

    // The ONLY number that matters on the first run: does this reproduce cx19's in-situ 2.40 ms?
    printf("\nBASELINE (256-wide, as profiled)  grid(%d,%d)   %8.4f ms\n", n_job, n_chan, ms);
    printf("   ncu on cx19 (A40, 11 jobs x 7 chan x 140 chips): 2.4022 ms\n");
    if (n_job == 11 && n_chan == 7 && n_chips == 140) {
        const double rel = (ms - 2.4022) / 2.4022;
        printf("   -> %+.1f%% vs in situ  %s\n", 100.0 * rel,
               (rel > -0.15 && rel < 0.15) ? "FAITHFUL" : "*** NOT FAITHFUL -- do not trust "
                                                          "anything else from this bench ***");
    }

    // ---- block-width sweep ---------------------------------------------------------------
    // Each block owns ONE (PRN, channel) Phi slice and re-walks it once per (hop pass x trial).
    // Widening the block cuts the hop passes, so the traffic should fall as 1/W until something
    // else becomes the limit. Also check the replica samples are unchanged: `wave` must be
    // bit-identical at every width (each hop's sample depends only on that hop), even though the
    // ENERGY reduction order legitimately moves.
    const size_t en_n = (size_t)4 * n_job * n_chan;
    std::vector<float2> ref(wave_n);
    std::vector<double> eref(en_n), egot(en_n);
    CK(gnss_cuda::launch_waveform_tuned(d_code, d_jobs, n_job, n_chan, p, d_wave, d_energy, 256, 0, 0, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), d_wave, wave_n * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(eref.data(), d_energy, en_n * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<float2> got(wave_n);
    // BOTH knobs, swept together. The block width and the trial fusion each cut the re-walk
    // count, but the fusion costs registers and registers are what cap the width, so the winner
    // is a corner of the 2D grid and not the product of two 1D wins.
    printf("\nre-walk sweep (grid %d x %d):  re-walks = trials x hop passes\n", n_job, n_chan);
    for (int fuse : {0, 1}) {
        for (int w : {256, 512, 1024}) {
            CK(gnss_cuda::launch_waveform_tuned(d_code, d_jobs, n_job, n_chan, p, d_wave,
                                                       d_energy, w, fuse, phi16, 0));
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(got.data(), d_wave, wave_n * sizeof(float2), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(egot.data(), d_energy, en_n * sizeof(double), cudaMemcpyDeviceToHost));
            size_t bad = 0;
            double wrel = 0.0; // magnitude of the difference, not just its presence -- with
                               // --phi16 `wave` is EXPECTED to move, and by how much is the point
            for (size_t k = 0; k < wave_n; ++k) {
                if (got[k].x != ref[k].x || got[k].y != ref[k].y)
                    ++bad;
                const double m = std::hypot((double)ref[k].x, (double)ref[k].y);
                if (m > 0.0)
                    wrel = std::max(wrel, std::hypot((double)got[k].x - ref[k].x,
                                                     (double)got[k].y - ref[k].y) / m);
            }
            double emax = 0.0;
            for (size_t k = 0; k < en_n; ++k)
                if (eref[k] != 0.0)
                    emax = std::max(emax, std::fabs(egot[k] - eref[k]) / std::fabs(eref[k]));
            CK(cudaEventRecord(e0, 0));
            for (int it = 0; it < iters; ++it)
                CK(gnss_cuda::launch_waveform_tuned(d_code, d_jobs, n_job, n_chan, p, d_wave,
                                                    d_energy, w, fuse, phi16, 0));
            CK(cudaEventRecord(e1, 0));
            CK(cudaDeviceSynchronize());
            float mw = 0.f;
            CK(cudaEventElapsedTime(&mw, e0, e1));
            mw /= iters;
            const int passes = (N_HOPS + w - 1) / w;
            // wave must be BIT-identical -- the width only moves which thread computes a hop, and
            // the fusion only interleaves three independent accumulations. Energy may move in the
            // last ulp under a width change, because the width IS the reduction's lane count.
            printf("   %s %4d threads  %2d re-walks  %8.4f ms  %5.2fx   wave %-13s rel %.2e\n",
                   fuse ? "fused  " : "3-gather", w, (fuse ? 1 : 3) * passes, mw, ms / mw,
                   bad ? (phi16 ? "fp16 (expected)" : "MISMATCH ***") : "bit-identical", wrel);
        }
    }

    // Traffic accounting, from the access pattern rather than from a counter: per block, per
    // chip, the 256 threads' t indices span ~inv_cps entries (base = phi*inv_cps is in
    // [0,inv_cps) for EVERY hop and every trial, so the window does not move with the hop), and
    // that window is re-walked once per (hop pass x trial).
    const int hop_passes = (N_HOPS + 255) / 256;
    const double win_bytes = (inv_cps + 2.0) * 8.0 * 2.0;   // A and B, float2
    const double ideal = (double)n_job * n_chan * n_chips * win_bytes;
    const double actual = ideal * hop_passes * 3.0;
    printf("\ntraffic model: %d hop passes x 3 trials = %d re-walks of each block's %.0f KB slice\n",
           hop_passes, hop_passes * 3, ideal / n_job / n_chan / 1024.0);
    printf("   touched once  %8.1f MB     re-walked %8.1f MB     measured %8.1f MB (%.0f GB/s)\n",
           ideal / 1048576.0, actual / 1048576.0, ms * 1e-3 * 669e9 / 1048576.0, 669.0);
    return 0;
}
