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
    const int m = threadIdx.x; // hop
    const gnss_cuda::DespreadJob job = jobs[b];

    double2 acc = make_double2(0.0, 0.0);
    double e = 0.0;
    const bool covered = (job.chan_mask >> ci) & 1ULL; // channel in this PRN's covering set?
    if (covered && m < p.n_hops) {
        // Per-hop code phase at the hop's reference sample (absolute anchoring): DOUBLE.
        const long long n_m = p.n0 + (long long)m * p.fft_len;
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
        // Boundary indices in double (multiply by inv_cps -- no FP64 divide); gather in float.
        const float2* phiA = job.phiA + (size_t)ci * (p.Lf + 1);
        const float2* phiB = job.phiB + (size_t)ci * (p.Lf + 1);
        float2 sA = make_float2(0.f, 0.f), sB = make_float2(0.f, 0.f);
        for (int d = 0; d < job.n_chips; ++d) {
            int klo = (int)floor((phi + d - 1.0) * job.inv_cps) + 1;
            int khi = (int)floor((phi + d) * job.inv_cps);
            if (klo < 0)
                klo = 0;
            if (khi > p.Lf - 1)
                khi = p.Lf - 1;
            if (khi < klo)
                continue;
            long long cidx = chip0 - d;
            long long idx = cidx % (long long)job.code_len;
            if (idx < 0)
                idx += job.code_len;
            const float cv = (float)code[job.code_offset + idx];
            sA.x += cv * (phiA[khi + 1].x - phiA[klo].x);
            sA.y += cv * (phiA[khi + 1].y - phiA[klo].y);
            sB.x += cv * (phiB[khi + 1].x - phiB[klo].x);
            sB.y += cv * (phiB[khi + 1].y - phiB[klo].y);
        }
        // Replica channel sample r = 0.5 (pa sA + conj(pa) sB); then the despread MAC:
        // acc += data * conj(r), e += |r|^2 -- float compute, widened once for the reduction.
        const float2 t1 = cmulf(pa, sA);
        const float2 t2 = cmulf(pb, sB);
        const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
        const float2 dd = data[(size_t)ci * p.n_hops + m];
        acc.x = (double)(dd.x * r.x + dd.y * r.y); // Re(d * conj(r))
        acc.y = (double)(dd.y * r.x - dd.x * r.y); // Im(d * conj(r))
        e = (double)(r.x * r.x + r.y * r.y);
    }

    // Block reduction over hops (blockDim.x <= 1024; power-of-two padded loop).
    __shared__ double sh_re[256], sh_im[256], sh_e[256];
    // hops <= 256 for all current bands (125 @ L1, 250 @ L5-half...); assert via launcher.
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
    if (p.n_hops > 256)
        return cudaErrorInvalidValue; // reduction buffer sized for <=256 hops/record
    // round block up to a power of two >= n_hops for the reduction
    int block = 1;
    while (block < p.n_hops)
        block <<= 1;
    dim3 grid(n_batch, n_chan);
    gnss_despread_kernel<<<grid, block, 0, stream>>>(data, code, jobs, p, corr, energy);
    return cudaGetLastError();
}

} // namespace gnss_cuda
