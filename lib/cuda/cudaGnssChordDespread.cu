#include "cudaGnssChordDespread.hpp"

#include "cudaGnssReplicaDevice.cuh"

#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>

// The synthesis kernel is the fused kernel's replica path (cudaGnssDespreadKernel.cu) without the
// data load and MAC, and the correlation kernel is that MAC with an element axis added.
// cudaGnssDespreadTest requires the split and fused paths to agree exactly, so keep the
// arithmetic textually the same.

namespace {

using gnss_cuda::chip_gather;
using gnss_cuda::cmulf;

/// grid.x = job (PRN), grid.y = covering channel, block.x = hop lane (grid-stride), as in
/// gnss_despread_kernel, so the per-hop arithmetic is identical.
///
/// MAXT is the block width this instantiation is compiled for, via __launch_bounds__. Registers
/// cap the width (without the bound the gather's register use rules out a 1024-thread block),
/// and the width sets the DRAM traffic (see WAVE_THREADS), so the bound lets ptxas trade
/// registers for width. The 512 and 1024 instantiations both exist; scripts/gnss/wavebench.cpp
/// compares them.
/// FUSE3 walks the chip loop once for all three E/P/L trials (@ref gnss_cuda::chip_gather3)
/// instead of three times; bit-identical either way, at a cost in registers.
/// PHI = __half2 reads fp16 tables. ABL_NOLOAD (load-address ablation), ILV (float4 interleaved
/// PhiA/PhiB) and HOPPERM (lanes permuted by p.hop_perm) are bench variants. SHARED reconstructs
/// from the Doppler-free Phi/Psi tables.
template<int MAXT, bool FUSE3, class PHI = float2, bool ABL_NOLOAD = false, bool ILV = false,
         bool HOPPERM = false, bool SHARED = false>
__global__ __launch_bounds__(MAXT) void
gnss_waveform_kernel(const int8_t* __restrict__ code,
                     const gnss_cuda::DespreadJob* __restrict__ jobs, gnss_cuda::DespreadParams p,
                     float2* __restrict__ wave,     // [3*n_job][nchan][n_hops]
                     double* __restrict__ energy) { // [4*n_job][nchan]
    const int b = blockIdx.x;
    const int ci = blockIdx.y;
    const int m = threadIdx.x;
    const int n_chan = gridDim.y;
    const gnss_cuda::DespreadJob job = jobs[b];

    double e[4] = {0.0, 0.0, 0.0, 0.0}; // E, P, L, P_HEAD
    const bool covered = (job.chan_mask >> ci) & 1ULL;

    // Hoisted as in the fused kernel: invariant across hops and trials.
    const int ks = (int)job.inv_cps;
    const float kf = (float)(job.inv_cps - ks);
    // The channel stride is in elements of the table's own type: under ILV the table is float4,
    // so the offset must be taken on a float4 pointer.
    const PHI* phiA = ILV ? (const PHI*)((const float4*)job.phiA + (size_t)ci * (p.Lf + 1))
                          : (const PHI*)job.phiA + (size_t)ci * (p.Lf + 1);
    const PHI* phiB = (const PHI*)job.phiB + (size_t)ci * (p.Lf + 1);

    // Preload the permutation: read inside the loop, hop_perm[] is a dependent global load that
    // every hop's chip walk waits on, with nothing to hide it behind at ~2 hops per thread.
    // Issued up front, a thread's lookups overlap each other and the prologue. Only the first
    // MAX_HOPS_PER_THREAD hops per thread are preloaded, so HOPPERM needs
    // n_hops <= MAX_HOPS_PER_THREAD * blockDim.x.
    constexpr int MAX_HOPS_PER_THREAD = 8;
    int hop_id[MAX_HOPS_PER_THREAD];
    if (HOPPERM) {
#pragma unroll
        for (int k = 0; k < MAX_HOPS_PER_THREAD; ++k) {
            const int mh_i = m + k * (int)blockDim.x;
            hop_id[k] = (mh_i < p.n_hops) ? p.hop_perm[(size_t)b * p.n_hops + mh_i] : 0;
        }
    }
    int kk = 0;
    for (int mh_i = m; mh_i < p.n_hops; mh_i += blockDim.x, ++kk) {
        const int mh = HOPPERM ? hop_id[kk < MAX_HOPS_PER_THREAD ? kk : 0] : mh_i;
        // An uncovered channel still writes zero replica rows: launch_correlate_nm reads every
        // channel's rows, and zero is the correct replica for a channel this PRN does not occupy.
        if (!covered) {
#pragma unroll
            for (int t = 0; t < 3; ++t)
                wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] =
                    make_float2(0.f, 0.f);
            continue;
        }

        // Code and carrier phase both come from host references at p.n0 (job.cp_ref and
        // job.ang0, reduced in long double), so the per-sample rates multiply only the
        // intra-record offset dn, which is exact in double and at most ~3.4e7 samples at CHORD
        // geometry. Taken from the absolute sample instead (n ~ 2e15 after days of uptime),
        // n*cps reaches ~6e12 chips, where a double's ULP is ~1e-3 chips, and wc*n lands where
        // the ULP is ~1 rad. The carrier error is constant within a record but changes whenever
        // the Doppler is re-propagated, and per-record constants are what the cross-record
        // estimators (deep fold, ADR, spectrum window sum) integrate.
        const double dn = (double)((long long)mh * (long long)p.fft_len);
        const double C_P = job.cp_ref + dn * job.cps;

        // wc*dn <= ~7.8e7 rad has a 1.5e-8 rad ULP in double. Reduce before the float cast:
        // a float's ULP at 7.8e7 is 8 rad.
        const double ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        const float2 pa = make_float2(cn, sn);
        const float2 pb = make_float2(cn, -sn);
        const bool in_head = (mh < job.m_head);

        // Trial order {P, E, L}, as in the fused kernel, so the arithmetic matches.
        //
        // The two branches repeat the carrier, write and energy block deliberately: sharing it
        // through a float2 array of gather results changes the unfused branch's codegen enough
        // to break bit-identity with the fused kernel, which only the split-vs-fused test
        // detects.
        if (FUSE3) {
            // Same three code phases in the same order, one walk of the chip loop.
            const double Cs[3] = {C_P, C_P - job.ds, C_P + job.ds};
            float2 sA[3], sB[3];
            // SHARED is a template parameter so the per-PRN path does not carry the shared
            // path's per-chip sincos and extra loads in its instructions or registers.
            if (SHARED)
                gnss_cuda::chip_gather3<PHI, ABL_NOLOAD, ILV, true>(
                    job.inv_cps, job.code_offset, job.code_len, job.n_chips, job.d_first, p.Lf,
                    code, phiA, phiB, ks, kf, Cs, sA, sB,
                    (const PHI*)job.psiA + (size_t)ci * (p.Lf + 1),
                    (const PHI*)job.psiB + (size_t)ci * (p.Lf + 1), job.ddw);
            else
                gnss_cuda::chip_gather3<PHI, ABL_NOLOAD, ILV, false>(
                    job.inv_cps, job.code_offset, job.code_len, job.n_chips, job.d_first, p.Lf,
                    code, phiA, phiB, ks, kf, Cs, sA, sB);
#pragma unroll
            for (int tt = 0; tt < 3; ++tt) {
                const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2;
                const float2 t1 = cmulf(pa, sA[tt]);
                const float2 t2 = cmulf(pb, sB[tt]);
                const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
                wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] = r;

                const double ee = (double)(r.x * r.x + r.y * r.y);
                e[t] += ee;
                if (t == 1 && in_head)
                    e[3] += ee;
            }
        } else {
#pragma unroll
            for (int tt = 0; tt < 3; ++tt) {
                const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2; // -> cp0 / cp0-ds / cp0+ds
                float2 sA, sB;
                chip_gather(job.inv_cps, job.code_offset, job.code_len, job.n_chips,
                            job.d_first, p.Lf, code, (const float2*)phiA, (const float2*)phiB,
                            ks, kf, C_P + (double)(t - 1) * job.ds, sA, sB);
                const float2 t1 = cmulf(pa, sA);
                const float2 t2 = cmulf(pb, sB);
                const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
                wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] = r;

                const double ee = (double)(r.x * r.x + r.y * r.y);
                e[t] += ee;
                if (t == 1 && in_head)
                    e[3] += ee;
            }
        }
    }

    // Block reduction over hop lanes, one row at a time, as in the fused kernel. The tree halves
    // blockDim.x, so the width must be a power of two. sh_e is 8 KB at MAXT = 1024; registers,
    // not shared memory, limit occupancy.
    __shared__ double sh_e[MAXT];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j == 3 && job.m_head == 0) {
            if (m == 0)
                energy[(size_t)(4 * b + 3) * n_chan + ci] = 0.0;
            break;
        }
        __syncthreads();
        sh_e[m] = e[j];
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (m < off)
                sh_e[m] += sh_e[m + off];
            __syncthreads();
        }
        if (m == 0)
            energy[(size_t)(4 * b + j) * n_chan + ci] = sh_e[0];
    }
}


/// Tiled-streaming synthesis (bench only; see launch_waveform_tiled). The gather kernel's chip
/// walks issue scattered loads over the block's whole (PRN, channel) slice, which streams
/// through L1 once per hop pass. Here the slice crosses DRAM once, in coalesced tiles staged
/// through shared memory, and the scattered per-chip loads hit shared memory.
///
/// Per (hop, trial) the walk keeps chip_gather3's float accumulation (f += kf), floorf, clamps
/// and fmaf order, so the result is bit-identical; a tile boundary pauses a walk but never
/// reorders it. The prologue is chip_gather3's at d_first == 0: job.d_first is not honoured,
/// and neither are the shared Psi tables. Two differences, both exact:
///  * the previous boundary value (pA/pB) is reloaded from shared memory at tile entry and in
///    the checked tail instead of being carried. It is the same table entry, so the same bits,
///    and it saves registers, which lets a 1024-thread block hold 2 hops x 3 trials of walk
///    state. The previous tap sits at most ks+1 entries behind the current one, so each tile is
///    staged with a low-side halo of at least ks+2 entries (supplied by the caller).
///  * khi (= d*ks) doubles as the chip counter: d < n_chips  <=>  khi < n_chips*ks (ks > 0).
///
/// The energy epilogue accumulates hops in the same (m, m+W) order and reduces over the same
/// lane tree as the gather kernel, so at equal width the energy is bit-identical too.
template<int MAXT, class PHI>
__global__ __launch_bounds__(MAXT) void
gnss_waveform_tiled_kernel(const int8_t* __restrict__ code,
                           const gnss_cuda::DespreadJob* __restrict__ jobs,
                           gnss_cuda::DespreadParams p, int tile_n, int halo, int code_words,
                           float2* __restrict__ wave,     // [3*n_job][nchan][n_hops]
                           double* __restrict__ energy) { // [4*n_job][nchan]
    constexpr int MAX_HPT = 2; // launch_waveform_tiled refuses n_hops > 2*blockDim
    const int b = blockIdx.x;
    const int ci = blockIdx.y;
    const int m = threadIdx.x;
    const int n_chan = gridDim.y;
    const int W = blockDim.x;
    const gnss_cuda::DespreadJob job = jobs[b];

    // Uncovered channel: zero rows and zero energy, as in the gather kernel. chan_mask is
    // block-uniform, so returning before the tile loop cannot desynchronise the __syncthreads.
    if (!((job.chan_mask >> ci) & 1ULL)) {
        for (int mh = m; mh < p.n_hops; mh += W)
#pragma unroll
            for (int t = 0; t < 3; ++t)
                wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] = make_float2(0.f, 0.f);
        if (m == 0)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                energy[(size_t)(4 * b + j) * n_chan + ci] = 0.0;
        return;
    }

    const int ks = (int)job.inv_cps;
    const float kf = (float)(job.inv_cps - ks);
    const PHI* gA = (const PHI*)job.phiA + (size_t)ci * (p.Lf + 1);
    const PHI* gB = (const PHI*)job.phiB + (size_t)ci * (p.Lf + 1);
    const int khi_end = job.n_chips * ks;

    // Per-(hop, trial) walk state. The prologue follows chip_gather3's line for line (at
    // d_first == 0) so the two stay bit-identical.
    int khi[MAX_HPT][3], idx[MAX_HPT][3], tprev[MAX_HPT][3];
    float f[MAX_HPT][3];
    float2 sA[MAX_HPT][3], sB[MAX_HPT][3];
    // Optional hop-sorted lane mapping (p.hop_perm, null = identity). Sorting lanes by
    // fractional code phase puts a warp's 32 taps inside ~5 table entries, so the per-chip
    // shared loads are near-broadcast instead of bank-conflicted, and the code indices are
    // adjacent. The permuted load happens once per hop, here in the prologue. `wave` is
    // unaffected (each sample depends only on its own hop); the energy lane grouping moves, so
    // with a permutation the energy agrees to rounding, not bit-for-bit.
    int mh_p[MAX_HPT];
#pragma unroll
    for (int h = 0; h < MAX_HPT; ++h) {
        const int slot = m + h * W;
        const int mh = mh_p[h] =
            (slot < p.n_hops && p.hop_perm) ? p.hop_perm[(size_t)b * p.n_hops + slot] : slot;
        if (mh >= p.n_hops) {
#pragma unroll
            for (int u = 0; u < 3; ++u)
                khi[h][u] = khi_end; // inert slot: the walk below never runs
            continue;
        }
        const double dn = (double)((long long)mh * (long long)p.fft_len);
        const double C_P = job.cp_ref + dn * job.cps;
        const double Cs[3] = {C_P, C_P - job.ds, C_P + job.ds};
#pragma unroll
        for (int u = 0; u < 3; ++u) {
            const long long chip0 = (long long)floor(Cs[u]);
            const double phi = Cs[u] - (double)chip0;
            sA[h][u] = make_float2(0.f, 0.f);
            sB[h][u] = make_float2(0.f, 0.f);
            const double base = phi * job.inv_cps;
            long long i0 = chip0 % (long long)job.code_len;
            if (i0 < 0)
                i0 += job.code_len;
            idx[h][u] = (int)i0;
            const int prev_khi = -ks + (int)floorf((float)base - kf);
            f[h][u] = (float)base;
            khi[h][u] = 0;
            int t_prev = prev_khi + 1;
            tprev[h][u] = t_prev < 0 ? 0 : (t_prev > p.Lf ? p.Lf : t_prev);
        }
    }

    extern __shared__ unsigned char sh_raw[];
    PHI* shA = (PHI*)sh_raw; // [halo + tile_n]
    PHI* shB = shA + (size_t)(halo + tile_n);
    unsigned* sh_code = (unsigned*)(shB + (size_t)(halo + tile_n)); // [code_words] +-1 as bits

    // The code table is staged in shared memory as bits, so the walk makes no global loads: a
    // 204,600-chip code is a 25.6 KB bit table, built in one coalesced pass with
    // __ballot_sync, and cv = bit ? +1.0f : -1.0f equals (float)(int8 +-1) exactly. Every code
    // value must be +-1. A zero (the zero-stuffed TDM tables, L2C) reads as -1, and nothing
    // here checks for it.
    {
        const int n_words = (job.code_len + 31) / 32;
        const int lane = m & 31;
        const int nwrp = W >> 5;
        for (int w = m >> 5; w < n_words; w += nwrp) {
            const int bi = w * 32 + lane;
            const int8_t c = (bi < job.code_len) ? code[job.code_offset + bi] : (int8_t)1;
            const unsigned word = __ballot_sync(0xffffffffu, c > 0);
            if (lane == 0)
                sh_code[w] = word;
        }
        (void)code_words; // the launcher sizes the buffer for the batch's largest code_len
    }

    for (int T0 = 0; T0 <= p.Lf; T0 += tile_n) {
        const int lo = (T0 - halo) < 0 ? 0 : T0 - halo;
        const int hi = (T0 + tile_n) < (p.Lf + 1) ? (T0 + tile_n) : (p.Lf + 1); // exclusive
        __syncthreads(); // the previous tile's readers finish before this tile is staged
        for (int i = m; i < hi - lo; i += W) {
            shA[i] = gA[lo + i];
            shB[i] = gB[lo + i];
        }
        __syncthreads();
        // Two-phase walk, for pipelining rather than arithmetic. A per-chip tile-boundary test
        // makes every load wait on a compare, which leaves the kernel latency-bound. Phase 1
        // runs a fixed count of n_safe chips per trial, a conservative bound (each tap step is
        // at most ks+1 entries, plus a 2-entry margin for the float rounding of f) within which
        // the [0, Lf] clamp is inactive and no boundary test is needed; the compiler can
        // pipeline it like chip_gather3. Phase 2 finishes the few chips per trial that reach
        // the tile edge or the Lf clamp, fully checked. pA/pB are carried in registers within
        // phase 1.
#pragma unroll
        for (int h = 0; h < MAX_HPT; ++h) {
            // phase 1: joint fixed-count run for this hop's three trials
            int n_safe = INT_MAX;
#pragma unroll
            for (int u = 0; u < 3; ++u) {
                const int k = khi[h][u];
                const int rem = (khi_end - k) / ks; // chips left (k is a multiple of ks)
                const int tn = k + (int)floorf(f[h][u]) + 1;
                const int nt = (hi - 2 - tn) / (ks + 1); // in-tile bound, step <= ks+1
                const int n = rem < nt ? rem : nt;
                n_safe = n < n_safe ? n : n_safe;
            }
            if (n_safe > 0) {
                float2 pA[3], pB[3];
#pragma unroll
                for (int u = 0; u < 3; ++u) {
                    pA[u] = gnss_cuda::phi_ld(shA, tprev[h][u] - lo);
                    pB[u] = gnss_cuda::phi_ld(shB, tprev[h][u] - lo);
                }
                for (int n = 0; n < n_safe; ++n) {
#pragma unroll
                    for (int u = 0; u < 3; ++u) {
                        const int t = khi[h][u] + (int)floorf(f[h][u]) + 1; // clamp inactive here
                        f[h][u] += kf;
                        khi[h][u] += ks;
                        const float2 cA = gnss_cuda::phi_ld(shA, t - lo);
                        const float2 cB = gnss_cuda::phi_ld(shB, t - lo);
                        const int id = idx[h][u];
                        const float cv =
                            (sh_code[id >> 5] >> (id & 31)) & 1u ? 1.0f : -1.0f;
                        sA[h][u].x = fmaf(cv, cA.x - pA[u].x, sA[h][u].x);
                        sA[h][u].y = fmaf(cv, cA.y - pA[u].y, sA[h][u].y);
                        sB[h][u].x = fmaf(cv, cB.x - pB[u].x, sB[h][u].x);
                        sB[h][u].y = fmaf(cv, cB.y - pB[u].y, sB[h][u].y);
                        pA[u] = cA;
                        pB[u] = cB;
                        tprev[h][u] = t;
                        if (--idx[h][u] < 0)
                            idx[h][u] += job.code_len;
                    }
                }
            }
            // phase 2: the checked tail, up to the tile edge
            bool live = true;
            while (live) {
                live = false;
#pragma unroll
                for (int u = 0; u < 3; ++u) {
                    const int k = khi[h][u];
                    if (k >= khi_end)
                        continue;
                    int t = k + (int)floorf(f[h][u]) + 1;
                    t = t < 0 ? 0 : (t > p.Lf ? p.Lf : t);
                    if (t >= hi)
                        continue; // resumes in the next tile
                    live = true;
                    f[h][u] += kf;
                    khi[h][u] = k + ks;
                    const float2 cA = gnss_cuda::phi_ld(shA, t - lo);
                    const float2 cB = gnss_cuda::phi_ld(shB, t - lo);
                    const float2 pAv = gnss_cuda::phi_ld(shA, tprev[h][u] - lo);
                    const float2 pBv = gnss_cuda::phi_ld(shB, tprev[h][u] - lo);
                    const int id2 = idx[h][u];
                    const float cv =
                        (sh_code[id2 >> 5] >> (id2 & 31)) & 1u ? 1.0f : -1.0f;
                    sA[h][u].x = fmaf(cv, cA.x - pAv.x, sA[h][u].x);
                    sA[h][u].y = fmaf(cv, cA.y - pAv.y, sA[h][u].y);
                    sB[h][u].x = fmaf(cv, cB.x - pBv.x, sB[h][u].x);
                    sB[h][u].y = fmaf(cv, cB.y - pBv.y, sB[h][u].y);
                    tprev[h][u] = t;
                    if (--idx[h][u] < 0)
                        idx[h][u] += job.code_len;
                }
            }
        }
    }

    // Epilogue: carrier, write and energy as in the gather kernel, hop m then m+W.
    double e[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
    for (int h = 0; h < MAX_HPT; ++h) {
        const int mh = mh_p[h];
        if (mh >= p.n_hops)
            continue;
        const double dn = (double)((long long)mh * (long long)p.fft_len);
        const double ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        const float2 pa = make_float2(cn, sn);
        const float2 pb = make_float2(cn, -sn);
        const bool in_head = (mh < job.m_head);
#pragma unroll
        for (int tt = 0; tt < 3; ++tt) {
            const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2;
            const float2 t1 = cmulf(pa, sA[h][tt]);
            const float2 t2 = cmulf(pb, sB[h][tt]);
            const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
            wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] = r;
            const double ee = (double)(r.x * r.x + r.y * r.y);
            e[t] += ee;
            if (t == 1 && in_head)
                e[3] += ee;
        }
    }

    __shared__ double sh_e[MAXT];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j == 3 && job.m_head == 0) {
            if (m == 0)
                energy[(size_t)(4 * b + 3) * n_chan + ci] = 0.0;
            break;
        }
        __syncthreads();
        sh_e[m] = e[j];
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (m < off)
                sh_e[m] += sh_e[m + off];
            __syncthreads();
        }
        if (m == 0)
            energy[(size_t)(4 * b + j) * n_chan + ci] = sh_e[0];
    }
}

/// grid.x = job (PRN), grid.y = covering channel.
/// block.x = element lane (coalesced over the frame's fastest axis),
/// block.y = hop lane (reduced in shared memory at the end).
__global__ void gnss_correlate_nm_kernel(const unsigned char* __restrict__ data,
                                         const float* __restrict__ chan_scale,
                                         const int* __restrict__ chan_ids,
                                         const float2* __restrict__ wave, // [3*n_job][nchan][hops]
                                         const gnss_cuda::DespreadJob* __restrict__ jobs,
                                         gnss_cuda::DespreadParams p, int n_elem, int elem_stride,
                                         int frame_chan_stride,
                                         double2* __restrict__ corr) { // [4*n_job][nchan][n_elem]
    const int b = blockIdx.x;
    const int ci = blockIdx.y;
    const int n_chan = gridDim.y;
    const int fc = chan_ids[ci];
    const float scale = chan_scale[ci];
    // The only job field needed; everything else is already in `wave`.
    const int m_head = jobs[b].m_head;

    extern __shared__ double2 sh[]; // [blockDim.x * blockDim.y]

    // Round the element loop up to whole blockDim.x passes and predicate the memory accesses:
    // the reduction below calls __syncthreads(), so every thread must run the same number of
    // iterations, and n_elem is not a multiple of the block width in general.
    const int e_end = ((n_elem + blockDim.x - 1) / blockDim.x) * blockDim.x;

    for (int e0 = threadIdx.x; e0 < e_end; e0 += blockDim.x) {
        const bool active = (e0 < n_elem);
        double2 acc[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

        for (int mh = threadIdx.y; mh < p.n_hops; mh += blockDim.y) {
            // Coalesced across threadIdx.x: consecutive elements are adjacent bytes.
            float2 dd =
                active ? gnss_cuda::unpack_44(
                             data[((size_t)mh * frame_chan_stride + fc) * elem_stride + e0], scale)
                       : make_float2(0.f, 0.f);
            if (p.conj_data)
                dd.y = -dd.y; // F-engine conjugation -- see DespreadParams::conj_data
            const bool in_head = (mh < m_head);

#pragma unroll
            for (int t = 0; t < 3; ++t) {
                // Uniform across threadIdx.x, so a broadcast rather than a per-element load.
                const float2 r = wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh];
                const double re = (double)(dd.x * r.x + dd.y * r.y); // Re(d * conj(r))
                const double im = (double)(dd.y * r.x - dd.x * r.y); // Im(d * conj(r))
                acc[t].x += re;
                acc[t].y += im;
                if (t == 1 && in_head) {
                    acc[3].x += re;
                    acc[3].y += im;
                }
            }
        }

        // Reduce the hop lanes (threadIdx.y) for this element, one row at a time.
        const int slot = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __syncthreads();
            sh[slot] = acc[j];
            __syncthreads();
            for (int off = blockDim.y / 2; off > 0; off >>= 1) {
                if (threadIdx.y < off) {
                    sh[slot].x += sh[slot + off * blockDim.x].x;
                    sh[slot].y += sh[slot + off * blockDim.x].y;
                }
                __syncthreads();
            }
            if (threadIdx.y == 0 && active)
                corr[((size_t)(4 * b + j) * n_chan + ci) * n_elem + e0] = sh[threadIdx.x];
        }
    }
}

} // namespace

namespace gnss_cuda {

cudaError_t launch_waveform_tuned(const int8_t* code, const DespreadJob* jobs, int n_job,
                                  int n_chan, const DespreadParams& p, float2* wave, double* energy,
                                  int threads_hint, int fuse3, int phi16,
                                  cudaStream_t stream) {
    const bool fuse = fuse3 < 0 ? WAVE_FUSE3 : (fuse3 != 0);
    int cap = threads_hint > 0 ? threads_hint : WAVE_THREADS;
    if (cap > 1024)
        cap = 1024; // sh_e is sized for this
    int threads = p.n_hops < cap ? p.n_hops : cap;
    // The shared reduction tree halves blockDim.x, so it must be a power of two.
    int pow2 = 1;
    while (pow2 * 2 <= threads)
        pow2 *= 2;
    threads = pow2;
    // Clamp to each instantiation's maxThreadsPerBlock as the driver reports it: the ceiling
    // depends on the architecture and on ptxas's register allocation, and exceeding it fails
    // the launch ("too many resources requested") rather than degrading.
    auto ceiling = [](const void* fn) {
        cudaFuncAttributes at{};
        return cudaFuncGetAttributes(&at, fn) == cudaSuccess ? at.maxThreadsPerBlock : 256;
    };
    const dim3 grid(n_job, n_chan);
    // The table type follows phi16: __half2 for 1, 6 and any unlisted non-zero value (despite
    // job.phiA/phiB's float2 type), interleaved float4 for 4, float2 otherwise. launch_waveform
    // passes 0 or 1 (DespreadParams::phi_half); 2, 4, 5 and 6 are bench variants, launched
    // without the ceiling check.
    if (phi16 == 5) { // bench: hop-sorted lane->hop mapping (float2 tables)
        gnss_waveform_kernel<1024, true, float2, false, false, true>
            <<<grid, threads > 1024 ? 1024 : threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (phi16 == 6) { // bench: hop-sorted and fp16
        gnss_waveform_kernel<1024, true, __half2, false, false, true>
            <<<grid, threads > 1024 ? 1024 : threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (phi16 == 4) { // bench: interleaved float4 Phi, one 16 B load instead of two 8 B
        gnss_waveform_kernel<1024, true, float2, false, true>
            <<<grid, threads > 1024 ? 1024 : threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (phi16 == 2) { // bench: load-address ablation, all the index math and flops, no Phi traffic
        gnss_waveform_kernel<1024, true, float2, true>
            <<<grid, threads > 1024 ? 1024 : threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (phi16) {
        auto ceil16 = [](const void* fn) {
            cudaFuncAttributes at{};
            return cudaFuncGetAttributes(&at, fn) == cudaSuccess ? at.maxThreadsPerBlock : 256;
        };
        if (threads > 512) {
            static const int k16 = ceil16((const void*)gnss_waveform_kernel<1024, true, __half2>);
            if (threads <= k16) {
                gnss_waveform_kernel<1024, true, __half2>
                    <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
                return cudaGetLastError();
            }
            threads = 512;
        }
        gnss_waveform_kernel<512, true, __half2>
            <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    // Shared (Doppler-free) tables: a separate instantiation, so the per-PRN path does not carry
    // the shared gather's sincos and extra loads in its register budget (registers cap MAXT,
    // which sets the DRAM traffic; see WAVE_THREADS). Fused only, one MAXT = 1024 instantiation.
    if (p.shared) {
        static const int kmaxS = ceiling(
            (const void*)gnss_waveform_kernel<1024, true, float2, false, false, false, true>);
        const int th = threads > kmaxS ? kmaxS : threads;
        gnss_waveform_kernel<1024, true, float2, false, false, false, true>
            <<<grid, th, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (threads > 512) {
        static const int kmaxF = ceiling((const void*)gnss_waveform_kernel<1024, true>);
        static const int kmaxU = ceiling((const void*)gnss_waveform_kernel<1024, false>);
        if (threads <= (fuse ? kmaxF : kmaxU)) {
            if (fuse)
                gnss_waveform_kernel<1024, true>
                    <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
            else
                gnss_waveform_kernel<1024, false>
                    <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
            return cudaGetLastError();
        }
        threads = 512;
    }
    static const int k5F = ceiling((const void*)gnss_waveform_kernel<512, true>);
    static const int k5U = ceiling((const void*)gnss_waveform_kernel<512, false>);
    const int k5 = fuse ? k5F : k5U;
    if (threads > k5)
        threads = k5;
    if (fuse)
        gnss_waveform_kernel<512, true><<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
    else
        gnss_waveform_kernel<512, false><<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
    return cudaGetLastError();
}

cudaError_t launch_waveform(const int8_t* code, const DespreadJob* jobs, int n_job, int n_chan,
                            const DespreadParams& p, float2* wave, double* energy,
                            cudaStream_t stream) {
    // p.phi_half selects the __half2 gather; the tables the jobs point at must already be half
    // storage, which GnssCudaDespread::ensure_phi guarantees.
    return launch_waveform_tuned(code, jobs, n_job, n_chan, p, wave, energy, 0, -1,
                                 p.phi_half ? 1 : 0, stream);
}

void phi_to_half(const float2* src, void* dst, size_t n) {
    // Here rather than in GnssCudaDespread.cpp so __half2 stays out of g++-compiled code.
    __half2* d = (__half2*)dst;
    for (size_t i = 0; i < n; ++i)
        d[i] = __floats2half2_rn(src[i].x, src[i].y);
}

cudaError_t launch_waveform_tiled(const int8_t* code, const DespreadJob* jobs, int n_job,
                                  int n_chan, const DespreadParams& p, float2* wave,
                                  double* energy, int tile_entries, int halo_entries,
                                  int code_len_max, int phi16, cudaStream_t stream) {
    const int code_words = (code_len_max + 31) / 32;
    int threads = p.n_hops < 1024 ? p.n_hops : 1024;
    int pow2 = 1;
    while (pow2 * 2 <= threads)
        pow2 *= 2;
    threads = pow2;
    if (p.n_hops > 2 * threads)
        return cudaErrorInvalidValue; // MAX_HPT = 2 hops per thread, at most 1024 threads
    if (tile_entries < 256)
        tile_entries = 4096;
    if (halo_entries < 16)
        return cudaErrorInvalidValue; // must be >= max over jobs of (int)inv_cps + 2
    const size_t elem = phi16 ? sizeof(__half2) : sizeof(float2);
    const size_t shbytes =
        2 * (size_t)(tile_entries + halo_entries) * elem + (size_t)code_words * 4;
    const dim3 grid(n_job, n_chan);
    // Dynamic shared memory past 48 KB needs the opt-in, set once per instantiation to 90 KB:
    // about the per-block limit on sm_86/sm_89 less the 8 KB static sh_e. A tile that does not
    // fit fails the launch with an error the caller sees.
    if (phi16) {
        static bool armed16 = [] {
            cudaFuncSetAttribute((const void*)gnss_waveform_tiled_kernel<1024, __half2>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024);
            return true;
        }();
        (void)armed16;
        gnss_waveform_tiled_kernel<1024, __half2><<<grid, threads, shbytes, stream>>>(
            code, jobs, p, tile_entries, halo_entries, code_words, wave, energy);
        return cudaGetLastError();
    }
    static bool armed = [] {
        cudaFuncSetAttribute((const void*)gnss_waveform_tiled_kernel<1024, float2>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024);
        return true;
    }();
    (void)armed;
    gnss_waveform_tiled_kernel<1024, float2><<<grid, threads, shbytes, stream>>>(
        code, jobs, p, tile_entries, halo_entries, code_words, wave, energy);
    return cudaGetLastError();
}

cudaError_t launch_correlate_nm(const unsigned char* data, const float* chan_scale,
                                const int* chan_ids, const float2* wave, const DespreadJob* jobs,
                                int n_job, int n_chan, int n_elem, int elem_stride,
                                int frame_chan_stride, const DespreadParams& p, double2* corr,
                                cudaStream_t stream) {
    // Elements across x (coalescing), hops across y (parallelism): n_elem is small, so without
    // the hop lanes the launch would have too few threads to fill the device. blockDim.y is a
    // power of two for the reduction.
    int ex = n_elem < 32 ? n_elem : 32;
    int hy = 256 / ex;
    if (hy > p.n_hops)
        hy = p.n_hops;
    int pow2 = 1;
    while (pow2 * 2 <= hy)
        pow2 *= 2;
    hy = pow2 < 1 ? 1 : pow2;

    const dim3 block(ex, hy);
    const dim3 grid(n_job, n_chan);
    const size_t shmem = (size_t)ex * hy * sizeof(double2);
    gnss_correlate_nm_kernel<<<grid, block, shmem, stream>>>(
        data, chan_scale, chan_ids, wave, jobs, p, n_elem, elem_stride, frame_chan_stride, corr);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------
// Path B: quantize the replicas into synthetic 4+4b N^2 stations.
// ---------------------------------------------------------------------------------------------

namespace {

/// One thread per (hop, covering channel, lane); consecutive threads take consecutive lanes, so
/// each (hop, channel) row of lanes is one coalesced store. lane = 4*prn_slot + trial
/// (E/P/L/P_HEAD) is fixed per slot, so the consumer's lane -> PRN map does not change as
/// satellites rise and set; slots without a live spec this record write 0x88 (0+0j). The
/// P_HEAD lane is the prompt gated to hops [0, m_head): the N^2 integrates over its whole time
/// axis, so a time-gated replica needs its own lane.
///
/// Scale: s = 7 / (3 * rms) per (lane, channel), rms = sqrt(energy row / n_hops), with row 3
/// (the head energy) over the job's m_head hops instead. The path-B consumer recomputes s from
/// the energy rows and divides it out; it matches this s for E/P/L but not for P_HEAD (see
/// launch_pack44). Values clamp to [-7,7]: -8 corrupts the whole N^2 launch (n2k's
/// negate_4bit).
__global__ void gnss_pack44_kernel(const float2* __restrict__ wave,
                                   const double* __restrict__ energy,
                                   const gnss_cuda::DespreadJob* __restrict__ jobs,
                                   const int* __restrict__ slot2spec, int n_slot, int n_chan,
                                   int n_hops, const int* __restrict__ chan_map,
                                   int frame_chan_stride, int n_lanes, int lane_base,
                                   int lane_pitch, int conj_replica,
                                   unsigned char* __restrict__ synth) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = idx % n_lanes;
    const int c = (idx / n_lanes) % n_chan;
    const int m = idx / (n_lanes * n_chan);
    if (m >= n_hops)
        return;

    unsigned char out = 0x88; // offset-encoded 0+0j
    const int slot = lane >> 2, t = lane & 3;
    const int spec = (slot < n_slot) ? slot2spec[slot] : -1;
    if (spec >= 0) {
        const gnss_cuda::DespreadJob j = jobs[spec];
        const bool covered = (j.chan_mask >> c) & 1ULL;
        const bool live = (t != 3) || (m < j.m_head);
        if (covered && live) {
            const int wrow = (t == 3) ? 1 : t; // P_HEAD packs the prompt
            const float2 w = wave[((size_t)(3 * spec + wrow) * n_chan + c) * n_hops + m];
            const int nh = (t == 3) ? (j.m_head > 0 ? j.m_head : 1) : n_hops;
            const float rms = sqrtf((float)(energy[(size_t)(4 * spec + t) * n_chan + c] / nh));
            if (rms > 0.0f) {
                const float s = 7.0f / (3.0f * rms);
                int qr = __float2int_rn(s * w.x);
                // Conjugate the replica, not the data (see launch_pack44's conj_replica):
                // the mixed tile then equals the tracker's sum_t conj(D) conj(R) directly.
                int qi = __float2int_rn(conj_replica ? -s * w.y : s * w.y);
                qr = qr < -7 ? -7 : (qr > 7 ? 7 : qr);
                qi = qi < -7 ? -7 : (qi > 7 ? 7 : qi);
                out = (unsigned char)(((qr + 8) << 4) | ((qi + 8) & 0xf));
            }
        }
    }
    synth[((size_t)m * frame_chan_stride + chan_map[c]) * lane_pitch + lane_base + lane] = out;
}

} // namespace

cudaError_t launch_pack44(const float2* wave, const double* energy, const DespreadJob* jobs,
                          const int* slot2spec, int n_slot, int n_chan, int n_hops,
                          const int* chan_map, int frame_chan_stride, int n_lanes, int lane_base,
                          int lane_pitch, bool conj_replica, unsigned char* synth,
                          cudaStream_t stream) {
    const long total = (long)n_hops * n_chan * n_lanes;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    gnss_pack44_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        wave, energy, jobs, slot2spec, n_slot, n_chan, n_hops, chan_map, frame_chan_stride,
        n_lanes, lane_base, lane_pitch, conj_replica ? 1 : 0, synth);
    return cudaGetLastError();
}

} // namespace gnss_cuda
