#include "cudaGnssDespreadKernel.hpp"

#include <cuda_runtime.h>
#include <math.h>

// GPU port of gnss::ChannelizedReplicaBank::hoprate_stream + gnss::channelized_despread,
// FUSED: each thread generates one (channel, hop) replica sample from the precomputed
// cumulative filter tables (PhiA/PhiB, Doppler-bucketed, built on CPU) and immediately
// multiplies it against the data -- the correlate-at-data despread as one batched kernel.
//
// Geometry (one launch = one record window of one signal band):
//   grid.x  = batch index b (one per PRN x {E,P,L} correlator trial)
//   grid.y  = covering-channel index ci
//   block.x = hops (m), one thread per hop (125 @ L1/5MSPS; <=1024 always for our bands)
// Each block computes partial sums over its hops via shared-memory reduction ->
// corr[b][ci], energy[b][ci]; the tiny cross-channel sum happens on the host (or a
// follow-up 1-block kernel when batches grow).
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

__global__ void gnss_despread_kernel(const float2* __restrict__ data, // [nchan][n_hops]
                                     const int8_t* __restrict__ code, // [batch-shared code table]
                                     const gnss_cuda::DespreadJob* __restrict__ jobs, // [nbatch]
                                     gnss_cuda::DespreadParams p,
                                     double2* __restrict__ corr,    // [nbatch][nchan]
                                     double* __restrict__ energy) { // [nbatch][nchan]
    const int b = blockIdx.x;  // batch (PRN x correlator)
    const int ci = blockIdx.y; // covering channel
    const int m = threadIdx.x; // hop lane: handles hops m, m+blockDim, ... (grid-stride over
                               // the record, so records LONGER than the block work -- 1000-hop
                               // L1 / 4000-hop E1C records at the 20 MSPS wide front end)
    const gnss_cuda::DespreadJob job = jobs[b];

    double2 acc = make_double2(0.0, 0.0);
    double e = 0.0;
    const bool covered = (job.chan_mask >> ci) & 1ULL; // channel in this PRN's covering set?
    // Hop range: full-record trials run [0, n_hops); a P_HEAD segment trial stops at the
    // code-period boundary hop (host-computed) so the two sides of the secondary-overlay
    // sign flip are never summed blind (gnssRecord.hpp slots 16-18).
    const int hop_end = (job.hop_hi < p.n_hops) ? job.hop_hi : p.n_hops;
    for (int mh = job.hop_lo + m; covered && mh < hop_end; mh += blockDim.x) {
        // Per-hop code phase at the hop's reference sample (absolute anchoring): DOUBLE.
        const long long n_m = p.n0 + (long long)mh * p.fft_len;
        const double C = job.cp0 + (double)n_m * job.cps;
        const long long chip0 = (long long)floor(C);
        const double phi = C - (double)chip0;

        // Carrier phasor pa = e^{i wc n_m}: range-reduce in DOUBLE, trig in float.
        const double ang = fmod(job.wc * (double)n_m, 2.0 * M_PI);
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        const float2 pa = make_float2(cn, sn);
        const float2 pb = make_float2(cn, -sn);

        // Chip gather over the filter span: sA/sB = sum_d code[chip0-d] * (Phi[khi+1]-Phi[klo]).
        //
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
        const float2* phiA = job.phiA + (size_t)ci * (p.Lf + 1);
        const float2* phiB = job.phiB + (size_t)ci * (p.Lf + 1);
        float2 sA = make_float2(0.f, 0.f), sB = make_float2(0.f, 0.f);
        {
            const double base = phi * job.inv_cps;
            const int ks = (int)job.inv_cps;
            const float kf = (float)(job.inv_cps - ks);
            long long i0 = chip0 % (long long)job.code_len;
            if (i0 < 0)
                i0 += job.code_len;
            int idx = (int)i0;                                   // (b) walks with d
            int prev_khi = -ks + (int)floorf((float)base - kf);  // (a) seed: khi(-1)
            float f = (float)base;                               // (c) base + d*kf
            int khi_base = 0;                                    // (c) d*ks
            for (int d = 0; d < job.n_chips; ++d) {
                const int khi = khi_base + (int)floorf(f);
                const int klo = prev_khi + 1;
                prev_khi = khi;
                f += kf;
                khi_base += ks;
                int kh = khi, kl = klo;
                if (kl < 0)
                    kl = 0;
                if (kh > p.Lf - 1)
                    kh = p.Lf - 1;
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
        // Replica channel sample r = 0.5 (pa sA + conj(pa) sB); then the despread MAC:
        // acc += data * conj(r), e += |r|^2 -- float compute, widened once for the reduction.
        const float2 t1 = cmulf(pa, sA);
        const float2 t2 = cmulf(pb, sB);
        const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
        const float2 dd = data[(size_t)ci * p.data_stride + mh];
        acc.x += (double)(dd.x * r.x + dd.y * r.y); // Re(d * conj(r))
        acc.y += (double)(dd.y * r.x - dd.x * r.y); // Im(d * conj(r))
        e += (double)(r.x * r.x + r.y * r.y);
    }

    // Block reduction over hop lanes (each lane already summed its strided hops above).
    __shared__ double sh_re[256], sh_im[256], sh_e[256];
    sh_re[m] = acc.x;
    sh_im[m] = acc.y;
    sh_e[m] = e;
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
        corr[(size_t)b * gridDim.y + ci] = make_double2(sh_re[0], sh_im[0]);
        energy[(size_t)b * gridDim.y + ci] = sh_e[0];
    }
}

} // namespace

namespace gnss_cuda {

cudaError_t launch_despread(const float2* data, const int8_t* code, const DespreadJob* jobs,
                            int n_batch, int n_chan, const DespreadParams& p, double2* corr,
                            double* energy, cudaStream_t stream) {
    // Block = min(pow2 >= n_hops, 256); longer records grid-stride inside the kernel (the
    // 20 MSPS wide front end runs 1000-hop L1 / 4000-hop E1C records).
    int block = 1;
    while (block < p.n_hops && block < 256)
        block <<= 1;
    dim3 grid(n_batch, n_chan);
    gnss_despread_kernel<<<grid, block, 0, stream>>>(data, code, jobs, p, corr, energy);
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
