#include "cudaGnssDespreadKernel.hpp"

#include "cudaGnssReplicaDevice.cuh"

#include <cuda_runtime.h>
#include <math.h>

// GPU port of gnss::ChannelizedReplicaBank::hoprate_stream and gnss::channelized_despread,
// fused: each thread generates (channel, hop) replica samples from the precomputed cumulative
// filter tables (PhiA/PhiB, per Doppler bucket, built on the host) and multiply-accumulates
// them against the data.
//
// Geometry (one launch = one record window of one signal band):
//   grid.x  = spec index b, one PRN's correlator triple (E, P, L)
//   grid.y  = covering-channel index ci
//   block.x = hop lanes, min(pow2 >= n_hops, 256); lane m handles hops m, m + blockDim.x, ...
// Each block reduces its lanes in shared memory into rows 4*b + 0..2 of corr/energy; the caller
// sums across channels. Row 4*b + 3 (P_HEAD in the four-row layout) is written as zeros: this
// kernel does not form the head-gated prompt (the waveform path in cudaGnssChordDespread.cu
// does), and its only production caller, GnssCudaDespread::despread_batch, wants E/P/L alone.
//
// E, P and L are one block rather than three jobs because they differ only in code phase: they
// share a Doppler, so one carrier phasor, and they read the same data sample. Fused, the carrier
// (a double fmod and a sincosf) and the load happen once per (PRN, channel, hop), and only the
// chip gather runs three times.
//
// Fusion costs registers, and so occupancy. It pays when each thread walks several hops: the
// kernel is then throughput-bound and the shared carrier and load dominate. With ~1 hop per
// thread it is latency-bound (one dependent fmod -> sincosf -> gather chain per thread),
// occupancy is what hides that latency, and fusion loses. The throughput bench in
// cudaGnssDespreadTest.cpp runs at several hops per thread for this reason, not at the ~1 of
// its correctness geometry.
//
// Numerics (mixed precision: FP64 throughput is a small fraction of FP32 on the target GPUs):
//   double: the code phase C = cp_ref + dn*cps and its floor/frac split, base = frac*inv_cps,
//     and the carrier reduction fmod(wc*dn, 2*pi). The tap-boundary index
//     floor(base + d*inv_cps) splits into an exact integer d*ks and a float remainder
//     base + d*kf (see chip_gather3).
//   float: everything per chip after the reduction: Phi tables and the telescoped gather (values
//     O(1); ~2e-6 relative cancellation error per chip difference), the carrier phasor (sincosf
//     on the reduced angle), the replica sample, the data MAC and the energy.
//   double: each lane's sums over its hops and the block reduction across lanes.
// Agrees with the all-double CPU reference to ~1e-6 relative; the test gates at 1e-5.

namespace {

// The replica primitives are shared (cudaGnssReplicaDevice.cuh) so the despread and the CHORD
// waveform kernels use the same gather.
using gnss_cuda::cmulf;

__global__ void gnss_despread_kernel(const float2* __restrict__ data, // [nchan][data_stride]
                                     const int8_t* __restrict__ code, // [batch-shared code table]
                                     const gnss_cuda::DespreadJob* __restrict__ jobs, // [n_spec]
                                     gnss_cuda::DespreadParams p,
                                     double2* __restrict__ corr,    // [4*n_spec][nchan]
                                     double* __restrict__ energy) { // [4*n_spec][nchan]
    const int b = blockIdx.x;  // spec: one PRN's E/P/L triple
    const int ci = blockIdx.y; // covering channel
    const int m = threadIdx.x; // hop lane: hops m, m + blockDim.x, ... (records may be longer
                               // than the block)
    const gnss_cuda::DespreadJob job = jobs[b];

    // acc/e rows: 0 = Early, 1 = Prompt, 2 = Late.
    double2 acc[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    double e[3] = {0.0, 0.0, 0.0};
    const bool covered = (job.chan_mask >> ci) & 1ULL; // channel in this PRN's covering set?

    // Invariant across hops and trials.
    const int ks = (int)job.inv_cps;
    const float kf = (float)(job.inv_cps - ks);
    const float2* phiA = job.phiA + (size_t)ci * (p.Lf + 1);
    const float2* phiB = job.phiB + (size_t)ci * (p.Lf + 1);

    for (int mh = m; covered && mh < p.n_hops; mh += blockDim.x) {
        // Intra-record sample offset of this hop, exact in double.
        const double dn = (double)((long long)mh * (long long)p.fft_len);
        // Prompt code phase from the window's reference (DespreadJob::cp_ref), so cps multiplies
        // only dn. Early is C_P - ds, Late C_P + ds.
        const double C_P = job.cp_ref + dn * job.cps;

        // Carrier phasor pa = e^{i wc n}, one per hop for all three trials. The phase comes from
        // the window's reference (DespreadJob::ang0), so wc multiplies only dn; reduce in double
        // before the float sincosf.
        const double ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        const float2 pa = make_float2(cn, sn);
        const float2 pb = make_float2(cn, -sn);

        // One load for the three trials.
        const float2 dd = data[(size_t)ci * p.data_stride + mh];

        // Trials run in the order {P, E, L}, the waveform kernel's order
        // (cudaGnssChordDespread.cu), so the two paths do the same arithmetic. The accumulator
        // index t keeps its meaning (0 = E, 1 = P, 2 = L).
        //
        // One walk of the chip loop serves all three trials (chip_gather3): their taps at a
        // given chip are neighbours in the Phi slice, so they share L1 lines.
        float2 g3A[3], g3B[3];
        const double Cs[3] = {C_P, C_P - job.ds, C_P + job.ds};
        // Shared tables need the SHARED instantiation with Psi and ddw passed explicitly:
        // chip_gather3's defaults (SHARED = false, null Psi) read the Doppler-free tables with no
        // correction.
        if (p.shared)
            gnss_cuda::chip_gather3<float2, true>(
                job.inv_cps, job.code_offset, job.code_len, job.n_chips, job.d_first, p.Lf, code,
                phiA, phiB, ks, kf, Cs, g3A, g3B, (const float2*)job.psiA + (size_t)ci * (p.Lf + 1),
                (const float2*)job.psiB + (size_t)ci * (p.Lf + 1), job.ddw);
        else
            gnss_cuda::chip_gather3(job.inv_cps, job.code_offset, job.code_len, job.n_chips,
                                    job.d_first, p.Lf, code, phiA, phiB, ks, kf, Cs, g3A, g3B);
#pragma unroll
        for (int tt = 0; tt < 3; ++tt) {
            const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2; // -> P / E (C_P-ds) / L (C_P+ds)
            // Replica channel sample r = 0.5 (pa sA + conj(pa) sB); then the despread MAC:
            // acc += data * conj(r), e += |r|^2 -- float compute, widened once for the reduction.
            const float2 t1 = cmulf(pa, g3A[tt]);
            const float2 t2 = cmulf(pb, g3B[tt]);
            const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
            const double re = (double)(dd.x * r.x + dd.y * r.y); // Re(d * conj(r))
            const double im = (double)(dd.y * r.x - dd.x * r.y); // Im(d * conj(r))
            const double ee = (double)(r.x * r.x + r.y * r.y);
            acc[t].x += re;
            acc[t].y += im;
            e[t] += ee;
        }
    }

    // Block reduction over hop lanes (each lane already summed its strided hops), run as three
    // sequential passes over one set of shared arrays: three at once would be 18 KB per block.
    // They hold 256 lanes, the most despread_block launches; the tree needs a power-of-2 block.
    __shared__ double sh_re[256], sh_im[256], sh_e[256];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
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
    // Row 3: zeros, so that no reader of the four-row layout sees stale data.
    if (m == 0) {
        corr[(size_t)(4 * b + 3) * gridDim.y + ci] = make_double2(0.0, 0.0);
        energy[(size_t)(4 * b + 3) * gridDim.y + ci] = 0.0;
    }
}

} // namespace

namespace gnss_cuda {

namespace {
// Block = min(pow2 >= n_hops, 256); longer records grid-stride inside the kernel.
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
    dim3 grid(n_spec, n_chan); // one block per (PRN, channel), 4 output rows each
    gnss_despread_kernel<<<grid, despread_block(p.n_hops), 0, stream>>>(data, code, jobs, p, corr,
                                                                        energy);
    return cudaGetLastError();
}

} // namespace gnss_cuda
