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
// Numerics: code phase C = cp0 + n*cps uses ABSOLUTE sample indices (~1e10) -> double is
// mandatory (float has 2^24 integer ceiling); carrier phase wc*n is range-reduced with fmod
// before sincos. This mirrors the CPU code exactly (renormalised phasor there, fmod here).

namespace {

__device__ inline double2 cmul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void gnss_despread_kernel(const float2* __restrict__ data, // [nchan][n_hops]
                                     const int8_t* __restrict__ code, // [batch-shared code table]
                                     const double2* __restrict__ PhiA, // [nchan][Lf+1]
                                     const double2* __restrict__ PhiB, // [nchan][Lf+1]
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
    if (m < p.n_hops) {
        // Per-hop code phase at the hop's reference sample (absolute anchoring).
        const long long n_m = p.n0 + (long long)m * p.fft_len;
        const double C = job.cp0 + (double)n_m * job.cps;
        const long long chip0 = (long long)floor(C);
        const double phi = C - (double)chip0;

        // Carrier phasor pa = e^{i wc n_m}, range-reduced.
        const double ang = fmod(job.wc * (double)n_m, 2.0 * M_PI);
        double s, c;
        sincos(ang, &s, &c);
        const double2 pa = make_double2(c, s);
        const double2 pb = make_double2(c, -s);

        // Chip gather over the filter span: sA/sB = sum_d code[chip0-d] * (Phi[khi+1]-Phi[klo]).
        const double2* phiA = PhiA + (size_t)ci * (p.Lf + 1);
        const double2* phiB = PhiB + (size_t)ci * (p.Lf + 1);
        double2 sA = make_double2(0.0, 0.0), sB = make_double2(0.0, 0.0);
        for (int d = 0; d < p.n_chips; ++d) {
            int klo = (int)floor((phi + d - 1.0) / job.cps) + 1;
            int khi = (int)floor((phi + d) / job.cps);
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
            const double cv = (double)code[job.code_offset + idx];
            const double2 dA =
                make_double2(phiA[khi + 1].x - phiA[klo].x, phiA[khi + 1].y - phiA[klo].y);
            const double2 dB =
                make_double2(phiB[khi + 1].x - phiB[klo].x, phiB[khi + 1].y - phiB[klo].y);
            sA.x += cv * dA.x;
            sA.y += cv * dA.y;
            sB.x += cv * dB.x;
            sB.y += cv * dB.y;
        }
        // Replica channel sample r = 0.5 (pa sA + conj(pa) sB); then the despread MAC:
        // acc += data * conj(r), e += |r|^2.
        const double2 t1 = cmul(pa, sA);
        const double2 t2 = cmul(pb, sB);
        const double2 r = make_double2(0.5 * (t1.x + t2.x), 0.5 * (t1.y + t2.y));
        const float2 dd = data[(size_t)ci * p.n_hops + m];
        acc.x = (double)dd.x * r.x + (double)dd.y * r.y;  // Re(d * conj(r))
        acc.y = (double)dd.y * r.x - (double)dd.x * r.y;  // Im(d * conj(r))
        e = r.x * r.x + r.y * r.y;
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

cudaError_t launch_despread(const float2* data, const int8_t* code, const double2* PhiA,
                            const double2* PhiB, const DespreadJob* jobs, int n_batch, int n_chan,
                            const DespreadParams& p, double2* corr, double* energy,
                            cudaStream_t stream) {
    if (p.n_hops > 256)
        return cudaErrorInvalidValue; // reduction buffer sized for <=256 hops/record
    // round block up to a power of two >= n_hops for the reduction
    int block = 1;
    while (block < p.n_hops)
        block <<= 1;
    dim3 grid(n_batch, n_chan);
    gnss_despread_kernel<<<grid, block, 0, stream>>>(data, code, PhiA, PhiB, jobs, p, corr,
                                                     energy);
    return cudaGetLastError();
}

} // namespace gnss_cuda
