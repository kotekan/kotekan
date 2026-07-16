#include "cudaGnssDespreadKernel.hpp"

#include <cuda_runtime.h>
#include <math.h>

// GPU port of gnss::ChannelizedReplicaBank::hoprate_stream + gnss::channelized_despread,
// FUSED: each thread generates one (channel, hop) replica sample from the precomputed
// cumulative filter tables (PhiA/PhiB, Doppler-bucketed, built on CPU) and immediately
// multiplies it against the data -- the correlate-at-data despread as one batched kernel.
//
// Geometry (one launch = one record window of one signal band):
//   grid.x  = spec index b -- ONE PRN's WHOLE CORRELATOR QUAD (E, P, L, P_HEAD)
//   grid.y  = covering-channel index ci
//   block.x = hops (m), one thread per hop (125 @ L1/5MSPS; <=1024 always for our bands)
// Each block computes partial sums over its hops via shared-memory reduction -> rows 4b+0..4b+3
// of corr/energy; the tiny cross-channel sum happens on the host (or a follow-up 1-block kernel
// when batches grow).
//
// WHY THE QUAD IS ONE BLOCK AND NOT FOUR (2026-07-16): E/P/L differ ONLY in code phase -- they
// share a Doppler, hence one carrier phasor, and they hit the same data sample. Run as four
// independent jobs, every one of them re-derived the carrier (a double fmod over a ~1e10-radian
// argument + a sincosf) and re-loaded the data, for a per-hop cost of 4x on both. Fused, the
// carrier and the load happen ONCE per (PRN, channel, hop) and only the chip gather -- the part
// that genuinely differs -- runs three times. P_HEAD rides along for free: it is the prompt's own
// MAC gated on `mh < m_head`, and the MAC measured at 0% of the runtime (see the gather note).
// Measured on the live job mix: 1.14x (1000-hop L1) to 1.22x (10000-hop B1C) kernel-only, 1.18x
// end-to-end through enqueue_batch_device (7244 -> 8568 rec/s, 32 PRN). ⚠️ The win is ALL from
// sharing the carrier/load: an A/B that only deleted the redundant P_HEAD job -- keeping E/P/L as
// three -- measured 0.98-1.00x, because the half-record it saves is paid straight back by the
// extra reduction pass that then has to run inside the prompt's block.
//
// ⚠️ THE WIN IS A FUNCTION OF HOPS PER THREAD -- it inverts below ~2. Fusing costs registers
// (56 -> 78, no spills), which at 128-256 threads/block is roughly a third of the occupancy.
// Where each thread handles ~1 hop the kernel is LATENCY-bound (one dependent fmod -> sincosf ->
// gather chain per thread, nothing to interleave), occupancy is what hides that latency, and the
// fused kernel LOSES ~10%. Where a thread walks several hops it is THROUGHPUT-bound, the carrier
// and load savings dominate, and it wins. Every deployed band is at 3.9-39 hops/thread (20 MSPS
// N=10: 1000-hop GPS, 4000-hop E1C, 10000-hop B1C; 5 MSPS L2C: 5000-hop) -- see the bench note in
// cudaGnssDespreadTest.cpp, which is why that bench runs the wide front end and not the 125-hop
// geometry the correctness check uses.
//
// Numerics (MIXED PRECISION -- the GB10's FP64 throughput is a small fraction of FP32, and the
// FP64 chip-gather was the measured ceiling at ~75k PRN-despreads/s):
//   double, mandatory: the absolute-anchored code phase C = cp0 + n*cps (n ~1e10 exceeds
//     float's 2^24 integer range), its floor/frac reduction, the tap-boundary indices
//     (floor((phi+d)*inv_cps) -- a boundary flip moves a prototype tap between adjacent chips,
//     so it is kept bit-comparable to the CPU division), and the carrier range reduction
//     fmod(wc*n, 2pi) (~1e10 rad).
//   float, everything per-chip after reduction: Phi tables + telescoped gather (values O(1);
//     cancellation error ~2e-6 relative per chip-delta), carrier phasor (sincosf on the reduced
//     angle, 1e-7 rad), replica sample, data MAC, energy.
//   double again for the cross-hop block reduction (125 signed terms; keeps the sums
//     well-conditioned and the outputs' types unchanged).
// Measured against the all-double CPU reference: ~1e-6 relative (gate 1e-5).

namespace {

__device__ inline float2 cmulf(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// Chip gather over the filter span at absolute code phase @c C:
///   sA/sB = sum_d code[chip0-d] * (Phi{A,B}[khi+1] - Phi{A,B}[klo]),  chip0 = floor(C).
/// @c ks / @c kf are the integer/fraction split of job.inv_cps, hoisted by the caller (they are
/// invariant across hops AND across the E/P/L trials, which share a Doppler).
__device__ inline void chip_gather(const gnss_cuda::DespreadJob& job, int Lf,
                                   const int8_t* __restrict__ code,
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
    const double base = phi * job.inv_cps;
    long long i0 = chip0 % (long long)job.code_len;
    if (i0 < 0)
        i0 += job.code_len;
    int idx = (int)i0;                                  // (b) walks with d
    int prev_khi = -ks + (int)floorf((float)base - kf); // (a) seed: khi(-1)
    float f = (float)base;                              // (c) base + d*kf
    int khi_base = 0;                                   // (c) d*ks
    for (int d = 0; d < job.n_chips; ++d) {
        const int khi = khi_base + (int)floorf(f);
        const int klo = prev_khi + 1;
        prev_khi = khi;
        f += kf;
        khi_base += ks;
        int kh = khi, kl = klo;
        if (kl < 0)
            kl = 0;
        if (kh > Lf - 1)
            kh = Lf - 1;
        if (kh >= kl) {
            const float cv = (float)code[job.code_offset + idx];
            sA.x += cv * (phiA[kh + 1].x - phiA[kl].x);
            sA.y += cv * (phiA[kh + 1].y - phiA[kl].y);
            sB.x += cv * (phiB[kh + 1].x - phiB[kl].x);
            sB.y += cv * (phiB[kh + 1].y - phiB[kl].y);
        }
        if (--idx < 0)
            idx += job.code_len;
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

template <typename T>
__global__ void gnss_despread_kernel(const T* __restrict__ data,          // [nchan][n_hops]
                                     const float* __restrict__ chan_scale, // [nchan], 4+4b only
                                     const int8_t* __restrict__ code, // [batch-shared code table]
                                     const gnss_cuda::DespreadJob* __restrict__ jobs, // [n_spec]
                                     gnss_cuda::DespreadParams p,
                                     double2* __restrict__ corr,    // [4*n_spec][nchan]
                                     double* __restrict__ energy) { // [4*n_spec][nchan]
    const int b = blockIdx.x;  // spec: one PRN's E/P/L/P_HEAD quad
    const int ci = blockIdx.y; // covering channel
    const int m = threadIdx.x; // hop lane: handles hops m, m+blockDim, ... (grid-stride over
                               // the record, so records LONGER than the block work -- 1000-hop
                               // L1 / 4000-hop E1C records at the 20 MSPS wide front end)
    const gnss_cuda::DespreadJob job = jobs[b];

    // acc/e rows: 0 = Early, 1 = Prompt, 2 = Late, 3 = P_HEAD (the prompt over [0, m_head)).
    double2 acc[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    double e[4] = {0.0, 0.0, 0.0, 0.0};
    const bool covered = (job.chan_mask >> ci) & 1ULL; // channel in this PRN's covering set?

    // Hoisted out of the hop loop: invariant in BOTH mh and the trial index.
    const int ks = (int)job.inv_cps;
    const float kf = (float)(job.inv_cps - ks);
    const float2* phiA = job.phiA + (size_t)ci * (p.Lf + 1);
    const float2* phiB = job.phiB + (size_t)ci * (p.Lf + 1);

    for (int mh = m; covered && mh < p.n_hops; mh += blockDim.x) {
        // Per-hop PROMPT code phase at the hop's reference sample (absolute anchoring): DOUBLE.
        // Early/Late are this +- ds, a double add each -- NOT three independent cp0 + n_m*cps
        // products (n_m ~1e10, so that multiply is the expensive fp64 one). The E/L rounding
        // differs from the unfused form by ~1 ulp of C (~1e-7 chips = 0.03 mm), far below the
        // gather's own float error.
        const long long n_m = p.n0 + (long long)mh * p.fft_len;
        const double C_P = job.cp0 + (double)n_m * job.cps;

        // Carrier phasor pa = e^{i wc n_m}: range-reduce in DOUBLE, trig in float. ONE per hop
        // for the whole quad -- E/P/L share a Doppler, and P_HEAD *is* the prompt.
        const double ang = fmod(job.wc * (double)n_m, 2.0 * M_PI);
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        const float2 pa = make_float2(cn, sn);
        const float2 pb = make_float2(cn, -sn);

        // ONE load for the quad (fp32 complex, or a 4+4b byte unpacked with this channel's scale)
        const float2 dd = load_v(data, (size_t)ci * p.data_stride + mh, chan_scale, ci);
        const bool in_head = (mh < job.m_head);

#pragma unroll
        for (int t = 0; t < 3; ++t) { // t = 0/1/2 -> cp0-ds / cp0 / cp0+ds
            float2 sA, sB;
            chip_gather(job, p.Lf, code, phiA, phiB, ks, kf, C_P + (double)(t - 1) * job.ds, sA,
                        sB);
            // Replica channel sample r = 0.5 (pa sA + conj(pa) sB); then the despread MAC:
            // acc += data * conj(r), e += |r|^2 -- float compute, widened once for the reduction.
            const float2 t1 = cmulf(pa, sA);
            const float2 t2 = cmulf(pb, sB);
            const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
            const double re = (double)(dd.x * r.x + dd.y * r.y); // Re(d * conj(r))
            const double im = (double)(dd.y * r.x - dd.x * r.y); // Im(d * conj(r))
            const double ee = (double)(r.x * r.x + r.y * r.y);
            acc[t].x += re;
            acc[t].y += im;
            e[t] += ee;
            // P_HEAD: the prompt's own MAC, gated on the hop. The replica is already in hand, so
            // the head segment costs one predicated add -- no second gather, no second carrier.
            if (t == 1 && in_head) {
                acc[3].x += re;
                acc[3].y += im;
                e[3] += ee;
            }
        }
    }

    // Block reduction over hop lanes (each lane already summed its strided hops above), run as
    // four sequential passes over ONE set of shared arrays: four at once would be 24 KB/block.
    __shared__ double sh_re[256], sh_im[256], sh_e[256];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // m_head == 0 (no code-period boundary in this window -- e.g. the plain 3-trial contract
        // of despread_batch) leaves row 3 identically zero: write it and skip the tree. The test
        // is uniform across the block, and the row must still be WRITTEN because the assembler
        // reads all four unconditionally.
        if (j == 3 && job.m_head == 0) {
            if (m == 0) {
                corr[(size_t)(4 * b + 3) * gridDim.y + ci] = make_double2(0.0, 0.0);
                energy[(size_t)(4 * b + 3) * gridDim.y + ci] = 0.0;
            }
            break;
        }
        __syncthreads(); // the previous pass's reads must finish before we overwrite
        sh_re[m] = acc[j].x;
        sh_im[m] = acc[j].y;
        sh_e[m] = e[j];
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (m < off) {
                sh_re[m] += sh_re[m + off];
                sh_im[m] += sh_im[m + off];
                sh_e[m] += sh_e[m + off];
            }
            __syncthreads();
        }
        if (m == 0) {
            corr[(size_t)(4 * b + j) * gridDim.y + ci] = make_double2(sh_re[0], sh_im[0]);
            energy[(size_t)(4 * b + j) * gridDim.y + ci] = sh_e[0];
        }
    }
}

} // namespace

namespace gnss_cuda {

namespace {
// Block = min(pow2 >= n_hops, 256); longer records grid-stride inside the kernel (the
// 20 MSPS wide front end runs 1000-hop L1 / 4000-hop E1C records).
inline int despread_block(int n_hops) {
    int block = 1;
    while (block < n_hops && block < 256)
        block <<= 1;
    return block;
}
} // namespace

cudaError_t launch_despread(const float2* data, const int8_t* code, const DespreadJob* jobs,
                            int n_spec, int n_chan, const DespreadParams& p, double2* corr,
                            double* energy, cudaStream_t stream) {
    dim3 grid(n_spec, n_chan); // one block per (PRN quad, channel) -- 4 output rows each
    gnss_despread_kernel<float2>
        <<<grid, despread_block(p.n_hops), 0, stream>>>(data, nullptr, code, jobs, p, corr, energy);
    return cudaGetLastError();
}

cudaError_t launch_despread_q(const unsigned char* data, const float* chan_scale,
                              const int8_t* code, const DespreadJob* jobs, int n_spec, int n_chan,
                              const DespreadParams& p, double2* corr, double* energy,
                              cudaStream_t stream) {
    dim3 grid(n_spec, n_chan);
    gnss_despread_kernel<unsigned char><<<grid, despread_block(p.n_hops), 0, stream>>>(
        data, chan_scale, code, jobs, p, corr, energy);
    return cudaGetLastError();
}

__global__ void gnss_chan_ingest_q_kernel(const float2* __restrict__ frame,
                                          unsigned char* __restrict__ ring,
                                          const float* __restrict__ chan_inv_scale, int n_hops_f,
                                          int n_chan, long long ring_hops, long long write_hop,
                                          unsigned int* __restrict__ rail_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hops_f * n_chan)
        return;
    const int m = idx / n_chan; // hop within the frame
    const int c = idx % n_chan; // channel (coalesced across threads: frame is hop-major)
    int railed = 0;
    const unsigned char b = gnss_cuda::pack_44(frame[idx], chan_inv_scale[c], &railed);
    ring[(size_t)c * ring_hops + (size_t)((write_hop + m) % ring_hops)] = b;
    if (railed && rail_count)
        atomicAdd(&rail_count[c], 1u); // per-channel railfrac numerator (RFI/CW watchdog)
}

cudaError_t launch_chan_ingest_q(const float2* frame, unsigned char* ring,
                                 const float* chan_inv_scale, int n_hops_f, int n_chan,
                                 long long ring_hops, long long write_hop,
                                 unsigned int* rail_count, cudaStream_t stream) {
    const int total = n_hops_f * n_chan;
    const int block = 256;
    gnss_chan_ingest_q_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        frame, ring, chan_inv_scale, n_hops_f, n_chan, ring_hops, write_hop, rail_count);
    return cudaGetLastError();
}

__global__ void gnss_ring_zero_q_kernel(unsigned char* __restrict__ ring, int n_chan,
                                        long long ring_hops, long long write_hop,
                                        long long count) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * n_chan)
        return;
    const long long m = idx / n_chan;
    const int c = (int)(idx % n_chan);
    // Offset-encoded zero, NOT 0x00: 0x00 decodes to (-8, -8), a large DC spike.
    ring[(size_t)c * ring_hops + (size_t)((write_hop + m) % ring_hops)] = gnss44::ZERO;
}

cudaError_t launch_ring_zero_q(unsigned char* ring, int n_chan, long long ring_hops,
                               long long write_hop, long long count, cudaStream_t stream) {
    const long long total = count * n_chan;
    const int block = 256;
    gnss_ring_zero_q_kernel<<<(unsigned)((total + block - 1) / block), block, 0, stream>>>(
        ring, n_chan, ring_hops, write_hop, count);
    return cudaGetLastError();
}

__global__ void gnss_chan_power_kernel(const float2* __restrict__ frame, int n_hops_f, int n_chan,
                                       double* __restrict__ sumsq) {
    // One block per channel; accumulate sum|v|^2 over the frame's hops for the bandpass measure.
    const int c = blockIdx.x;
    double s = 0.0;
    for (int m = threadIdx.x; m < n_hops_f; m += blockDim.x) {
        const float2 v = frame[(size_t)m * n_chan + c];
        s += (double)v.x * v.x + (double)v.y * v.y;
    }
    __shared__ double sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off)
            sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        sumsq[c] += sh[0]; // accumulate across frames; host divides by the sample count
}

cudaError_t launch_chan_power(const float2* frame, int n_hops_f, int n_chan, double* sumsq,
                              cudaStream_t stream) {
    gnss_chan_power_kernel<<<n_chan, 256, 0, stream>>>(frame, n_hops_f, n_chan, sumsq);
    return cudaGetLastError();
}

__global__ void gnss_chan_ingest_kernel(const float2* __restrict__ frame,
                                        float2* __restrict__ ring, int n_hops_f, int n_chan,
                                        long long ring_hops, long long write_hop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hops_f * n_chan)
        return;
    const int m = idx / n_chan; // hop within the frame
    const int c = idx % n_chan; // channel (coalesced across threads: frame is hop-major)
    ring[(size_t)c * ring_hops + (size_t)((write_hop + m) % ring_hops)] = frame[idx];
}

cudaError_t launch_chan_ingest(const float2* frame, float2* ring, int n_hops_f, int n_chan,
                               long long ring_hops, long long write_hop, cudaStream_t stream) {
    const int total = n_hops_f * n_chan;
    const int block = 256;
    gnss_chan_ingest_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        frame, ring, n_hops_f, n_chan, ring_hops, write_hop);
    return cudaGetLastError();
}

__global__ void gnss_ring_zero_kernel(float2* __restrict__ ring, int n_chan,
                                      long long ring_hops, long long write_hop,
                                      long long count) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * n_chan)
        return;
    const long long m = idx / n_chan;
    const int c = (int)(idx % n_chan);
    ring[(size_t)c * ring_hops + (size_t)((write_hop + m) % ring_hops)] = make_float2(0.f, 0.f);
}

cudaError_t launch_ring_zero(float2* ring, int n_chan, long long ring_hops, long long write_hop,
                             long long count, cudaStream_t stream) {
    const long long total = count * n_chan;
    const int block = 256;
    gnss_ring_zero_kernel<<<(unsigned)((total + block - 1) / block), block, 0, stream>>>(
        ring, n_chan, ring_hops, write_hop, count);
    return cudaGetLastError();
}

} // namespace gnss_cuda
