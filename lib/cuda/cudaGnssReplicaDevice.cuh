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

#include "cudaGnssDespreadKernel.hpp" // for DespreadJob

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

/// A job's code phase and carrier in fixed point, so that the per-hop loop does no fp64: the
/// code phase in 2^-shift chips, the carrier in 2^-64 turns, as the FIXED path of
/// gnss_psi_waveform_kernel (cudaGnssPsiSynth.cu). On GB10, whose fp64 rate is 1/64 of fp32,
/// the per-hop double code phase, its floor and fraction per trial and the carrier's fmod kept
/// the fp64 pipe busy for ~40% of the waveform kernel. Built once per block from the job's
/// references at p.n0 (@ref hop_phase_setup).
struct HopPhase {
    long long cf_ref;           ///< cp_ref
    long long dcf;              ///< code advance per hop, fft_len * cps
    long long dsf;              ///< Early/Late displacement ds
    long long Lfx;              ///< code_len
    float cf_r, dcf_r;          ///< cf_ref and dcf in code periods, for the wrap estimate
    int shift;                  ///< fraction bits of the code phase
    unsigned long long ang0_t;  ///< ang0
    unsigned long long dturn_t; ///< carrier advance per hop, wc * fft_len
};

__device__ inline HopPhase hop_phase_setup(const DespreadJob& job, int fft_len, int n_hops) {
    HopPhase h;
    const double adv = (double)fft_len * job.cps;
    // The largest shift that keeps the record's code-phase range below 2^62.
    const double max_c = (double)job.code_len + fabs(job.ds) + 2.0 + (double)n_hops * adv;
    h.shift = min(60, 61 - ilogb(max_c));
    const double sc = ldexp(1.0, h.shift);
    h.cf_ref = llrint(job.cp_ref * sc);
    h.dcf = llrint(adv * sc);
    h.dsf = llrint(job.ds * sc);
    h.Lfx = (long long)job.code_len << h.shift;
    h.cf_r = (float)(job.cp_ref / job.code_len);
    h.dcf_r = (float)(adv / job.code_len);
    const double t0 = job.ang0 * (0.5 * M_1_PI);
    const double td = job.wc * fft_len * (0.5 * M_1_PI);
    const double two64 = 18446744073709551616.0;
    h.ang0_t = (unsigned long long)((t0 - floor(t0)) * two64);
    h.dturn_t = (unsigned long long)((td - floor(td)) * two64);
    return h;
}

/// Hop @c mh of the record: the carrier phasor, and the code phase of trial u (0 = prompt,
/// 1 = early, 2 = late) as chip index chip0[u] in [0, code_len) and fraction phi[u] in [0, 1).
template<int NT>
__device__ inline float2 hop_phase(const HopPhase& h, int mh, int* chip0, float* phi) {
    // The code-period wraps since cp_ref (<= ~11 per record at CHORD) from a float estimate,
    // then one conditional fix-up each way: a 64-bit modulo is emulated through fp64.
    const int q = (int)fmaf((float)mh, h.dcf_r, h.cf_r);
    long long c = h.cf_ref + (long long)mh * h.dcf - (long long)q * h.Lfx;
    c += c < 0 ? h.Lfx : 0;
    c -= c >= h.Lfx ? h.Lfx : 0;
#pragma unroll
    for (int u = 0; u < NT; ++u) {
        long long cu = c + (u == 0 ? 0 : (u == 1 ? -h.dsf : h.dsf));
        cu += cu < 0 ? h.Lfx : 0;
        cu -= cu >= h.Lfx ? h.Lfx : 0;
        chip0[u] = (int)(cu >> h.shift);
        // The fraction's top 24 bits: a 64-bit int to float conversion also runs on the fp64
        // pipe.
        phi[u] = (float)((unsigned)(cu >> (h.shift - 24)) & 0xffffffu) * (1.f / 16777216.f);
    }
    const unsigned long long at = h.ang0_t + (unsigned long long)mh * h.dturn_t;
    // The turn's top 24 bits, as [0, 2) half-turns.
    float sn, cn;
    sincospif((float)(unsigned)(at >> 40) * (2.f / 16777216.f), &sn, &cn);
    return make_float2(cn, sn);
}

/// Chip gather over the filter span for NT code phases (3 = the E/P/L trials, 1 = the prompt
/// alone) in one walk of the chip loop. For each trial u, at code phase chip0[u] + phi[u]
/// (@ref hop_phase):
///   sA/sB = sum_d code[(chip0-d) mod code_len] * (Phi{A,B}[khi(d)+1] - Phi{A,B}[klo(d)])
/// with khi(d) = floor((phi+d)*inv_cps),
/// klo(d) = khi(d-1)+1, d in [job_d_first, job_n_chips), and indices clamped to [0, Lf].
/// @c ks / @c kf are the integer/fraction split of job.inv_cps and @c inv_cps its float, hoisted
/// by the caller (they are invariant across hops and across the E/P/L trials, which share a
/// Doppler).
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
/// @c PHI is the table type, float2 or __half2 (see phi_ld). @c chip0, @c phi, @c sA and @c sB
/// hold NT entries. Trial u's arithmetic does not depend on NT, so the prompt of a one-trial walk is
/// bit-identical to the prompt of the three-trial walk. With @c IMAGE_A false (I/Q input, whose
/// phiA is zero; DespreadParams::complex_input) phiA is not read and sA is zero, as the walk
/// over a zero table gives.
template<class PHI, int NT = 3, bool IMAGE_A = true>
__device__ inline void chip_gather3(float inv_cps, int job_code_offset, int job_code_len,
                                    int job_n_chips, int job_d_first, int Lf,
                                    const int8_t* __restrict__ code,
                                    const PHI* __restrict__ phiA, const PHI* __restrict__ phiB,
                                    int ks, float kf, const int* chip0, const float* phi,
                                    float2* sA, float2* sB) {
    int idx[NT], khi_base[NT];
    float f[NT];
    float2 pA[NT], pB[NT];
    // The chip loop works from phi in [0,1), with base and the code-index seed set once before
    // it.
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
    for (int u = 0; u < NT; ++u) {
        sA[u] = make_float2(0.f, 0.f);
        sB[u] = make_float2(0.f, 0.f);
        const float base = phi[u] * inv_cps;
        const int d0 = job_d_first;
        int i0 = chip0[u] - d0; // chip0 is in [0, code_len) and d0 is within the filter span
        if (i0 < 0)
            i0 += job_code_len;
        idx[u] = i0;                                    // (b) walks with d
        const float fstart = fmaf((float)d0, kf, base); // (c) base + d0*kf, one rounding
        const int prev_khi = (d0 - 1) * ks + (int)floorf(fstart - kf); // (a) seed: khi(d0-1)
        f[u] = fstart;                                                 // (c) base + d*kf
        khi_base[u] = d0 * ks;                                         // (c) d*ks
        // (d) seed the carry with t(d0-1); the table has Lf+1 entries, indices 0..Lf.
        int t_prev = prev_khi + 1;
        t_prev = t_prev < 0 ? 0 : (t_prev > Lf ? Lf : t_prev);
        if (IMAGE_A)
            pA[u] = phi_ld(phiA, t_prev);
        pB[u] = phi_ld(phiB, t_prev);
    }
    for (int d = job_d_first; d < job_n_chips; ++d) {
#pragma unroll
        for (int u = 0; u < NT; ++u) {
            int t = khi_base[u] + (int)floorf(f[u]) + 1;
            t = t < 0 ? 0 : (t > Lf ? Lf : t);
            f[u] += kf;
            khi_base[u] += ks;
            const float cv = (float)code[job_code_offset + idx[u]];
            // fmaf explicitly, not `s += cv * d`: whether ptxas contracts the plain expression
            // depends on the surrounding code, and the despread and waveform kernels, which both
            // call this, must produce the same bits.
            if (IMAGE_A) {
                const float2 cA = phi_ld(phiA, t);
                const float2 dA = make_float2(cA.x - pA[u].x, cA.y - pA[u].y);
                sA[u].x = fmaf(cv, dA.x, sA[u].x);
                sA[u].y = fmaf(cv, dA.y, sA[u].y);
                pA[u] = cA;
            }
            const float2 cB = phi_ld(phiB, t);
            const float2 dB = make_float2(cB.x - pB[u].x, cB.y - pB[u].y);
            sB[u].x = fmaf(cv, dB.x, sB[u].x);
            sB[u].y = fmaf(cv, dB.y, sB[u].y);
            pB[u] = cB;
            if (--idx[u] < 0)
                idx[u] += job_code_len;
        }
    }
}
} // namespace gnss_cuda

#endif // CUDA_GNSS_REPLICA_DEVICE_CUH
