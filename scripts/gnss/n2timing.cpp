/**
 * @file
 * @brief What does path B actually COST against stock N^2? GPU required.
 *
 * The live comparison (cx19 51% vs cx27 8-12%) is misleading: the GNSS dev configs DROP
 * run_n2k entirely, so cx27 runs NO correlator at all and that 40-point gap bundles the whole
 * stock N^2 -- which production pays anyway -- with path B's marginal cost. This measures the
 * ratio directly at the deployed shape (nt_outer=1, nt_inner=8192, nfreq=384):
 *
 *   [1] stock    n2k::Correlator(128, 384)                  what production pays today
 *   [2] dual     n2k_dual(128+128, 384) all blocks          what cx19 runs now
 *   [3] dual     n2k_dual(128+128, 384) MIXED|BB only       block_class_mask, already built
 *   [4] proxy    n2k_dual(128+128, 8)   MIXED|BB only       the freq-map's projected cost
 *                                                            (8 freqs ~ the 7-channel comb)
 *
 * [4] is a PROXY, not a measurement of the real thing: the frequency map does not exist yet,
 * so this runs a small-nfreq instantiation whose per-block work is identical and whose block
 * count scales the same way. It projects what restricting MIXED|BB to the comb would cost.
 *
 * min-of-N: single-shot GPU timings on this node spread 1.4x under contention (measured
 * 2026-08-06 during the synthesis work). Run it on an idle GPU.
 */

#include <n2k/Correlator.hpp>
#include <n2k_dual/DualCorrelator.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                     \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);       \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)

static constexpr int NT_INNER = 8192; // = sub_integration_ntime, production
static constexpr int NSA = 128, NSB = 128;

static int ntiles(int ns) {
    const int m = ns / 16;
    return m * (m + 1) / 2;
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 5;
    const int NF = argc > 2 ? atoi(argv[2]) : 384;
    const int NF_SMALL = 8; // must have a compiled (256, NF_SMALL) instantiation

    printf("n2timing: nt_outer=1 nt_inner=%d nfreq=%d, min of %d\n\n", NT_INNER, NF, iters);

    // ---- inputs (content is irrelevant to timing; the kernel is data-independent) ----------
    const size_t ea = (size_t)NT_INNER * NF * NSA;
    const size_t eb = (size_t)NT_INNER * NF * NSB;
    int8_t *dA, *dB;
    uint* drm;
    CK(cudaMalloc(&dA, ea));
    CK(cudaMalloc(&dB, eb));
    CK(cudaMemset(dA, 0x11, ea)); // offset-encoded, never nibble 0 (= -8)
    CK(cudaMemset(dB, 0x11, eb));
    const size_t nrm = (size_t)NF * (NT_INNER / 32);
    CK(cudaMalloc(&drm, nrm * 4));
    CK(cudaMemset(drm, 0xFF, nrm * 4)); // all-pass, as deployed

    int *v_stock, *v_dual;
    CK(cudaMalloc(&v_stock, (size_t)NF * ntiles(NSA) * 512 * 4));
    CK(cudaMalloc(&v_dual, (size_t)NF * ntiles(NSA + NSB) * 512 * 4));

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    // Aborting check: the bench lambda returns double, so CK's `return 1` cannot live here.
    auto ck = [](cudaError_t e, const char* where) {
        if (e != cudaSuccess) {
            printf("CUDA error %s at %s\n", cudaGetErrorString(e), where);
            exit(1);
        }
    };

    auto bench = [&](const char* label, auto&& launch) -> double {
        launch(); // warm up (n2k notes the first few launches run slow)
        ck(cudaDeviceSynchronize(), "warmup");
        float best = 1e30f;
        for (int i = 0; i < iters; i++) {
            ck(cudaEventRecord(t0), "rec t0");
            launch();
            ck(cudaEventRecord(t1), "rec t1");
            ck(cudaEventSynchronize(t1), "sync");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
            best = std::min(best, ms);
        }
        printf("%-52s %8.3f ms\n", label, best);
        return (double)best;
    };

    n2k::Correlator stock(NSA, NF);
    const double t_stock = bench("[1] stock  n2k(128, 384)", [&] {
        stock.launch(v_stock, dA, drm, 1, NT_INNER, nullptr, false);
    });

    n2k_dual::DualCorrelator dual(NSA, NSB, NF);
    const double t_dual = bench("[2] dual   n2k_dual(128+128, 384) ALL blocks", [&] {
        dual.launch(v_dual, dA, dB, drm, 1, NT_INNER, nullptr, false);
    });

    n2k_dual::DualCorrelatorParams pm(NSA, NSB, NF,
                                      n2k_dual::BLOCK_MASK_MIXED | n2k_dual::BLOCK_MASK_BB);
    n2k_dual::DualCorrelator dmask(pm);
    const double t_mask = bench("[3] dual   MIXED|BB only, 384 freqs", [&] {
        dmask.launch(v_dual, dA, dB, drm, 1, NT_INNER, nullptr, false);
    });

    // [4] THE REAL FREQUENCY MAP: MIXED|BB over just the GNSS comb, at the deployed nfreq.
    // This is no longer a proxy -- it is the launch path B would actually use.
    {
        std::vector<int> comb;
        for (int k = 0; k < 7; k++)
            comb.push_back(277 + k); // cx19 GPU0's local comb indices
        n2k_dual::DualCorrelatorParams pc(NSA, NSB, NF,
                                          n2k_dual::BLOCK_MASK_MIXED | n2k_dual::BLOCK_MASK_BB,
                                          comb);
        n2k_dual::DualCorrelator dcomb(pc);
        int* v_comb;
        CK(cudaMalloc(&v_comb, (size_t)comb.size() * ntiles(NSA + NSB) * 512 * 4));
        const double t_comb = bench("[4] dual   MIXED|BB over the 7-channel COMB (freq map)",
                                    [&] {
            dcomb.launch(v_comb, dA, dB, drm, 1, NT_INNER, nullptr, false);
        });
        printf("\n    stock + comb-only path B : %.3f ms = %.2fx stock  (+%.3f ms marginal)\n",
               t_stock + t_comb, (t_stock + t_comb) / t_stock, t_comb);
    }

    // [5] old proxy, kept for continuity with the 2026-08-06 numbers.
    const size_t nrm_s = (size_t)NF_SMALL * (NT_INNER / 32);
    uint* drm_s;
    CK(cudaMalloc(&drm_s, nrm_s * 4));
    CK(cudaMemset(drm_s, 0xFF, nrm_s * 4));
    int* v_small;
    CK(cudaMalloc(&v_small, (size_t)NF_SMALL * ntiles(NSA + NSB) * 512 * 4));
    n2k_dual::DualCorrelatorParams ps(NSA, NSB, NF_SMALL,
                                      n2k_dual::BLOCK_MASK_MIXED | n2k_dual::BLOCK_MASK_BB);
    n2k_dual::DualCorrelator dsmall(ps);
    const double t_small = bench("[5] dual   MIXED|BB only, 8 freqs (old PROXY)", [&] {
        dsmall.launch(v_small, dA, dB, drm_s, 1, NT_INNER, nullptr, false);
    });

    printf("\nblocks/frame: stock %d, dual %d, MIXED|BB@384 %d, MIXED|BB@8 %d\n",
           stock.params.threadblocks_per_freq * NF, dual.params.threadblocks_per_freq * NF,
           dmask.params.threadblocks_per_freq * NF, dsmall.params.threadblocks_per_freq * NF_SMALL);
    printf("\nWHAT PATH B COSTS OVER STOCK N^2:\n");
    printf("  as deployed now  : %.3f ms vs %.3f ms = %.2fx stock  (+%.3f ms)\n", t_dual, t_stock,
           t_dual / t_stock, t_dual - t_stock);
    printf("  stock + MIXED|BB : %.3f ms = %.2fx stock  (+%.3f ms marginal)\n",
           t_stock + t_mask, (t_stock + t_mask) / t_stock, t_mask);
    printf("  stock + freq-map : %.3f ms = %.2fx stock  (+%.3f ms marginal)  <- the target\n",
           t_stock + t_small, (t_stock + t_small) / t_stock, t_small);
    return 0;
}
