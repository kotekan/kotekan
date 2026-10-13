/**
 * @file
 * @brief Device-side replica primitives shared by every GNSS kernel that synthesises a replica.
 *
 * Used by the fused despread (cudaGnssDespreadKernel.cu) and by the CHORD waveform generator
 * (cudaGnssChordDespread.cu), whose replicas must be bit-identical: the waveform generator feeds
 * the N x M correlator, where a divergence from the fused path would show only as a
 * correlation-amplitude error. One copy of the gather keeps them identical.
 */

#ifndef CUDA_GNSS_REPLICA_DEVICE_CUH
#define CUDA_GNSS_REPLICA_DEVICE_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace gnss_cuda {
/// Complex multiply with the multiply-adds pinned to fmaf. Written as `a.x*b.x - a.y*b.y`,
/// whether ptxas contracts it depends on the surrounding code, so two call sites that must
/// agree bit-for-bit (the split and fused despreads) could round differently.
__device__ inline float2 cmulf(float2 a, float2 b) {
    return make_float2(fmaf(a.x, b.x, -(a.y * b.y)), fmaf(a.x, b.y, a.y * b.x));
}

/// Phi element load: a float2 table, or a __half2 table (DespreadParams::phi_half) widened to
/// float2.
__device__ inline float2 phi_ld(const float2* p, int i) {
    return p[i];
}
__device__ inline float2 phi_ld(const __half2* p, int i) {
    return __half22float2(p[i]);
}

/// Chip gather over the filter span for three code phases (the E/P/L trials) in one walk of
/// the chip loop. For each trial u, at absolute code phase C = C[u]:
///   sA/sB = sum_d code[(chip0-d) mod code_len] * (Phi{A,B}[khi(d)+1] - Phi{A,B}[klo(d)])
/// with chip0 = floor(C), phi = C - chip0, khi(d) = floor((phi+d)*inv_cps),
/// klo(d) = khi(d-1)+1, d in [job_d_first, job_n_chips), and indices clamped to [0, Lf].
/// @c ks / @c kf are the integer/fraction split of job.inv_cps, hoisted by the caller (they are
/// invariant across hops and across the E/P/L trials, which share a Doppler).
/// @c job_d_first is the first chip of the walk (DespreadJob::d_first, centered truncation);
/// the walk state is seeded at that chip in closed form. It has no default, so a call site
/// that omits it fails to compile rather than silently walking from chip 0.
///
/// The trials' accumulations over @c d never interact; only the walk is shared. At chip @c d
/// they read `t(u,d) = d*ks + floor(base_u + d*kf) + 1` with every `base_u` in [0, inv_cps), so
/// the three indices lie within about inv_cps entries of one another and loads issued together
/// share L1 lines, which three separate walks of the Phi slice cannot. The cost is registers:
/// three walk states instead of one, and registers cap the waveform kernel's block width
/// (WAVE_THREADS in cudaGnssChordDespread.hpp).
///
/// Template options:
///  - @c PHI: float2 or __half2 tables (see phi_ld).
///  - @c SHARED: reconstruct each chip window from a Doppler-free (Phi, Psi) pair instead of
///    reading a table built at this PRN's Doppler. The gather reads only the difference over one
///    chip window, and a carrier offset ddw factors out of it as
///
///        dPhi(w0+ddw) ~ exp(-i*ddw*t0) * [ dPhi_0 - i*ddw*( dPsi - t0*dPhi_0 ) ]
///
///    so one pair per channel serves every PRN and Doppler, and the table footprint does not
///    grow with the PRN count. The B image is e^{-i(off-wc)k} and carries the offset with the
///    opposite sign, hence its flipped correction and conjugated rotor. t0 is the window's low
///    index, which the loop already carries; the midpoint would be more accurate but steps by
///    ks, ks+1/2 or ks+1. At CHORD geometry the error is 7.5e-6 at +-5 kHz (1.2e-6 with the
///    midpoint), below fp16's 3.3e-4 storage error (scripts/gnss/phibits.cpp).
///    The rotor is evaluated per chip with __sincosf from the integer t0 rather than advanced
///    by multiplication: t0 steps by ks or ks+1 only away from the index clamps, and an
///    advanced rotor would fall out of step with t0 at a clamp.
template<class PHI, bool SHARED = false>
__device__ inline void chip_gather3(double job_inv_cps, int job_code_offset, int job_code_len,
                                    int job_n_chips, int job_d_first, int Lf,
                                    const int8_t* __restrict__ code,
                                    const PHI* __restrict__ phiA, const PHI* __restrict__ phiB,
                                    int ks, float kf, const double C[3], float2 sA[3],
                                    float2 sB[3], const PHI* __restrict__ psiA = nullptr,
                                    const PHI* __restrict__ psiB = nullptr, float ddw = 0.0f) {
    int idx[3], khi_base[3];
    float f[3];
    float2 pA[3], pB[3];
    // SHARED-only state; dead, and dropped by the compiler, when SHARED is false.
    int tp[3] = {0, 0, 0};
    float2 pQA[3], pQB[3];
    // The chip loop does no fp64 and no 64-bit modulo: that work (C's floor/frac split, base,
    // the code-index seed) is done once before it, and the loop works from phi in [0,1).
    //  (a) klo(d) = floor((phi+d-1)*inv_cps)+1 and khi(d-1) = floor((phi+d-1)*inv_cps) are the
    //      same expression, so klo(d) == khi(d-1) + 1 and one boundary per chip suffices.
    //  (b) the code index chip0-d decrements by one per chip, so the modulo becomes a decrement
    //      and wrap.
    //  (c) with inv_cps = ks + kf (ks integer, kf in [0,1)) and base = phi*inv_cps,
    //      khi(d) = d*ks + floor(base + d*kf). The d*ks part is exact integer arithmetic; the
    //      remainder f = base + d*kf stays below inv_cps + n_chips and is carried in float, one
    //      add per chip. Where f lands within rounding of an integer the boundary can differ by
    //      one from the double form, moving one prototype tap between adjacent chips.
    //  (d) with t(d) = clamp(khi(d)+1, 0, Lf), the sum telescopes over the prefix table,
    //      s = sum_d code_d * (Phi[t(d)] - Phi[t(d-1)]), so each boundary entry is loaded once
    //      and carried to the next chip. This is an exact rewrite of the clamped [klo, khi]
    //      difference, not an approximation: an empty or fully clamped span gives
    //      t(d) == t(d-1), a zero difference, so empty spans need no branch.
#pragma unroll
    for (int u = 0; u < 3; ++u) {
        const long long chip0 = (long long)floor(C[u]);
        const double phi = C[u] - (double)chip0;
        sA[u] = make_float2(0.f, 0.f);
        sB[u] = make_float2(0.f, 0.f);
        const double base = phi * job_inv_cps;
        const int d0 = job_d_first;
        long long i0 = (chip0 - (long long)d0) % (long long)job_code_len;
        if (i0 < 0)
            i0 += job_code_len;
        idx[u] = (int)i0;                                      // (b) walks with d
        const float fstart = fmaf((float)d0, kf, (float)base); // (c) base + d0*kf, one rounding
        const int prev_khi = (d0 - 1) * ks + (int)floorf(fstart - kf); // (a) seed: khi(d0-1)
        f[u] = fstart;                                                 // (c) base + d*kf
        khi_base[u] = d0 * ks;                                         // (c) d*ks
        // (d) seed the carry with t(d0-1); the table has Lf+1 entries, indices 0..Lf.
        int t_prev = prev_khi + 1;
        t_prev = t_prev < 0 ? 0 : (t_prev > Lf ? Lf : t_prev);
        if (SHARED) {
            tp[u] = t_prev;
            pQA[u] = phi_ld(psiA, t_prev);
            pQB[u] = phi_ld(psiB, t_prev);
        }
        pA[u] = phi_ld(phiA, t_prev);
        pB[u] = phi_ld(phiB, t_prev);
    }
    for (int d = job_d_first; d < job_n_chips; ++d) {
#pragma unroll
        for (int u = 0; u < 3; ++u) {
            int t = khi_base[u] + (int)floorf(f[u]) + 1;
            t = t < 0 ? 0 : (t > Lf ? Lf : t);
            f[u] += kf;
            khi_base[u] += ks;
            const float2 cA = phi_ld(phiA, t);
            const float2 cB = phi_ld(phiB, t);
            const float cv = (float)code[job_code_offset + idx[u]];
            float2 dA = make_float2(cA.x - pA[u].x, cA.y - pA[u].y);
            float2 dB = make_float2(cB.x - pB[u].x, cB.y - pB[u].y);
            if (SHARED) {
                const float2 qA = phi_ld(psiA, t);
                const float2 qB = phi_ld(psiB, t);
                const float t0 = (float)tp[u];
                // g = dPsi - t0*dPhi ; corr_A = dPhi - i*ddw*g ; corr_B = dPhi + i*ddw*g
                const float2 gA = make_float2(qA.x - pQA[u].x - t0 * dA.x,
                                              qA.y - pQA[u].y - t0 * dA.y);
                const float2 gB = make_float2(qB.x - pQB[u].x - t0 * dB.x,
                                              qB.y - pQB[u].y - t0 * dB.y);
                const float2 rA = make_float2(dA.x + ddw * gA.y, dA.y - ddw * gA.x);
                const float2 rB = make_float2(dB.x - ddw * gB.y, dB.y + ddw * gB.x);
                float sn, cs;
                __sincosf(-ddw * t0, &sn, &cs); // exp(-i*ddw*t0); B takes its conjugate
                dA = make_float2(rA.x * cs - rA.y * sn, rA.x * sn + rA.y * cs);
                dB = make_float2(rB.x * cs + rB.y * sn, rB.y * cs - rB.x * sn);
                pQA[u] = qA;
                pQB[u] = qB;
                tp[u] = t;
            }
            // fmaf explicitly, not `s += cv * d`: whether ptxas contracts the plain expression
            // depends on the surrounding code, and the despread and waveform kernels, which both
            // call this, must produce the same bits.
            sA[u].x = fmaf(cv, dA.x, sA[u].x);
            sA[u].y = fmaf(cv, dA.y, sA[u].y);
            sB[u].x = fmaf(cv, dB.x, sB[u].x);
            sB[u].y = fmaf(cv, dB.y, sB[u].y);
            pA[u] = cA;
            pB[u] = cB;
            if (--idx[u] < 0)
                idx[u] += job_code_len;
        }
    }
}
} // namespace gnss_cuda

#endif // CUDA_GNSS_REPLICA_DEVICE_CUH
