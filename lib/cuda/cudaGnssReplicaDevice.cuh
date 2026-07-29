/**
 * @file
 * @brief Device-side replica primitives shared by every GNSS kernel that synthesises a replica.
 *
 * Extracted verbatim from cudaGnssDespreadKernel.cu so the FUSED despread, the peel, and the
 * standalone CHORD waveform generator all call the SAME chip gather.
 *
 * WHY THIS IS A HEADER AND NOT THREE COPIES. The peel's analytic add-back is exact only if the
 * peel and the despread generate BIT-IDENTICAL replicas (docs/gnss_voltage_peel_live.md), and
 * the CHORD split has the same requirement for a different reason: the waveform generator now
 * produces the replica that the N x M correlator consumes, so any divergence from the fused
 * path shows up as a correlation-amplitude discrepancy with no other symptom. The gather is
 * also delicately tuned -- see the exactness argument in chip_gather -- and is exactly the kind
 * of code that rots when duplicated.
 */

#ifndef CUDA_GNSS_REPLICA_DEVICE_CUH
#define CUDA_GNSS_REPLICA_DEVICE_CUH

#include "cudaGnssDespreadKernel.hpp" // for gnss_cuda::unpack_44

#include <cuda_runtime.h>
#include <math.h>

namespace gnss_cuda {
__device__ inline float2 cmulf(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// Chip gather over the filter span at absolute code phase @c C:
///   sA/sB = sum_d code[chip0-d] * (Phi{A,B}[khi+1] - Phi{A,B}[klo]),  chip0 = floor(C).
/// @c ks / @c kf are the integer/fraction split of job.inv_cps, hoisted by the caller (they are
/// invariant across hops AND across the E/P/L trials, which share a Doppler).
/// Takes the four code-table scalars directly rather than a job struct, so the despread
/// (DespreadJob) and the peel (PeelJob) can share it without one pretending to be the other.
__device__ inline void chip_gather(double job_inv_cps, int job_code_offset, int job_code_len,
                                   int job_n_chips, int Lf, const int8_t* __restrict__ code,
                                   const float2* __restrict__ phiA,
                                   const float2* __restrict__ phiB, int ks, float kf, double C,
                                   float2& sA, float2& sB) {
    const long long chip0 = (long long)floor(C);
    const double phi = C - (double)chip0;
    sA = make_float2(0.f, 0.f);
    sB = make_float2(0.f, 0.f);
    // THIS LOOP IS THE KERNEL: an ablation (2026-07-16) put it at 74% of the runtime, scaling
    // linearly with n_chips, while the despread MAC measured at 0% (deleting it changed
    // nothing). It used to spend 2 double multiplies + 2 double floors + an int64 modulo PER
    // CHIP -- and fp64 is heavily rate-limited on the GB10. Three exact-or-near-exact
    // reformulations remove essentially all of that; measured 0.332 -> 0.134 ms/record (2.5x,
    // 3.4x at n_chips=8). None of them touches the mandatory-double part: C = cp0 + n_m*cps
    // (n_m ~1e10 blows float's 2^24) and its floor/frac reduction still happen above, and
    // everything below rides on `phi`, which is ALREADY reduced to [0,1).
    //
    //  (a) klo(d) == khi(d-1) + 1, identically: klo(d) = floor((phi+d-1)*inv_cps)+1 and
    //      khi(d-1) = floor((phi+d-1)*inv_cps) are the same expression. So carry the previous
    //      khi instead of recomputing a second boundary. EXACT, halves the boundary math.
    //  (b) the code index walks: cidx = chip0-d decrements by one per chip, so the int64
    //      modulo becomes a decrement + wrap. EXACT.
    //  (c) integer/fraction split of the step. With inv_cps = ks + kf (ks integer, kf in
    //      [0,1)):  khi(d) = floor(base + d*inv_cps) = d*ks + floor(base + d*kf),
    //      base = phi*inv_cps. The d*ks part is exact integer; the remainder (base + d*kf)
    //      stays in ~[0,31) -- small enough that float carries it, accumulated with one add
    //      per chip. Verified over 16e6 (phi,d) pairs across 5 geometries: 9 boundary
    //      disagreements vs the double form (5.6e-7), and only where the argument lands
    //      EXACTLY on an integer, i.e. where the tap sits on the chip edge and "correct" is
    //      arbitrary anyway. A disagreement moves ONE prototype tap between adjacent chips.
    //  (d) CARRY THE BOUNDARY VALUE (2026-07-27). (a) already says klo(d) == khi(d-1)+1
    //      identically -- so `phiA[kl]` on chip d is the SAME ARRAY ELEMENT as `phiA[kh+1]`
    //      on chip d-1, and the loop was loading each boundary TWICE. Fold both clamps into
    //      one index t(d) = clamp(khi(d)+1, 0, Lf) and the sum is a plain telescope over the
    //      prefix table:  s = sum_d code_d * (Phi[t(d)] - Phi[t(d-1)]).
    //
    //      This is EXACT, not an approximation -- the same two floats are subtracted, so the
    //      results are bit-identical. Each of the original's four cases maps over: khi+1 < 0
    //      gave kh < kl (skipped) and now gives t(d) == t(d-1) == 0; klo > Lf-1 likewise
    //      skipped and now gives t == t == Lf; the two half-clamped cases reduce to the same
    //      pair of indices the clamps produced. The `kh >= kl` guard therefore disappears --
    //      an empty span is a zero difference, so the branch is replaced by four flops that
    //      only ever occur when khi fails to advance (inv_cps > 1 on every deployed band, so
    //      essentially never).
    //
    //      Cost: 4 float2 loads per chip -> 2, plus one fewer branch. The gather was measured
    //      at 74% of kernel runtime and scales with n_chips, which is worst exactly where the
    //      slack is thinnest: L5 gathers 13 chips per hop (inv_cps 1.955) against L1's 4.
    //      Registers move by ~0 (two float2 of carry in, four ints out) -- this kernel's
    //      fusion win is register-sensitive, so that matters.
    const double base = phi * job_inv_cps;
    long long i0 = chip0 % (long long)job_code_len;
    if (i0 < 0)
        i0 += job_code_len;
    int idx = (int)i0;                                  // (b) walks with d
    int prev_khi = -ks + (int)floorf((float)base - kf); // (a) seed: khi(-1)
    float f = (float)base;                              // (c) base + d*kf
    int khi_base = 0;                                   // (c) d*ks
    // (d) seed the carry with t(-1); the table has Lf+1 entries, indices 0..Lf.
    int t_prev = prev_khi + 1;
    t_prev = t_prev < 0 ? 0 : (t_prev > Lf ? Lf : t_prev);
    float2 pA = phiA[t_prev];
    float2 pB = phiB[t_prev];
    for (int d = 0; d < job_n_chips; ++d) {
        int t = khi_base + (int)floorf(f) + 1;
        t = t < 0 ? 0 : (t > Lf ? Lf : t);
        f += kf;
        khi_base += ks;
        const float2 cA = phiA[t];
        const float2 cB = phiB[t];
        const float cv = (float)code[job_code_offset + idx];
        sA.x += cv * (cA.x - pA.x);
        sA.y += cv * (cA.y - pA.y);
        sB.x += cv * (cB.x - pB.x);
        sB.y += cv * (cB.y - pB.y);
        pA = cA;
        pB = cB;
        if (--idx < 0)
            idx += job_code_len;
    }
}

// Voltage load, overloaded on the ring's sample type: fp32 complex, or the CHORD 4+4b byte with
// its per-channel scale. Everything downstream of this line is identical, which is the point --
// the demotion must not fork the despread.
__device__ inline float2 load_v(const float2* __restrict__ d, size_t i,
                                const float* __restrict__ /*chan_scale*/, int /*ci*/) {
    return d[i];
}
__device__ inline float2 load_v(const unsigned char* __restrict__ d, size_t i,
                                const float* __restrict__ chan_scale, int ci) {
    return gnss_cuda::unpack_44(d[i], chan_scale[ci]);
}
} // namespace gnss_cuda

#endif // CUDA_GNSS_REPLICA_DEVICE_CUH
