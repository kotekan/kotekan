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
//   grid.x  = spec index b, one PRN's correlator quad (E, P, L, P_HEAD)
//   grid.y  = covering-channel index ci
//   block.x = hop lanes, min(pow2 >= n_hops, 256); lane m handles hops m, m + blockDim.x, ...
// Each block reduces its lanes in shared memory into rows out_rows_spec*b + 0..3 of corr/energy;
// the caller sums across channels.
//
// The quad is one block rather than four jobs because E, P and L differ only in code phase: they
// share a Doppler, so one carrier phasor, and they read the same data sample. Fused, the carrier
// (a double fmod and a sincosf) and the load happen once per (PRN, channel, hop), and only the
// chip gather runs three times. P_HEAD is the prompt's own MAC, gated on `mh < m_head`.
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
//     base + d*kf (see chip_gather).
//   float: everything per chip after the reduction: Phi tables and the telescoped gather (values
//     O(1); ~2e-6 relative cancellation error per chip difference), the carrier phasor (sincosf
//     on the reduced angle), the replica sample, the data MAC and the energy.
//   double: each lane's sums over its hops and the block reduction across lanes.
// Agrees with the all-double CPU reference to ~1e-6 relative; the test gates at 1e-5.

namespace {

// The replica primitives are shared (cudaGnssReplicaDevice.cuh) so the despread, the peel and
// the CHORD waveform kernels all use the same gather.
using gnss_cuda::chip_gather;
using gnss_cuda::cmulf;
using gnss_cuda::load_v;

/// Kernel-split ablation, a benchmarking template parameter that separates the cost of replica
/// synthesis (which CHORD's N^2 correlator cannot absorb) from the correlation MAC (which it
/// could). NO_MAC keeps one add so the gather is not eliminated; its difference from ABL_NONE
/// is the MAC's cost. NO_SYN replaces the gather with a constant, keeping the carrier and the
/// MAC; its difference is synthesis. launch_despread and launch_despread_q instantiate
/// ABL_NONE only.
enum AblMode { ABL_NONE = 0, ABL_NO_MAC = 1, ABL_NO_SYN = 2 };

template <typename T, bool XC, int ABL = ABL_NONE>
__global__ void gnss_despread_kernel(const T* __restrict__ data,      // [nchan][data_stride]
                                     const float* __restrict__ chan_scale, // [nchan], 4+4b only
                                     const int8_t* __restrict__ code, // [batch-shared code table]
                                     const gnss_cuda::DespreadJob* __restrict__ jobs, // [n_spec]
                                     gnss_cuda::DespreadParams p,
                                     double2* __restrict__ corr,  // [out_rows_spec*n_spec][nchan]
                                     double* __restrict__ energy, // [out_rows_spec*n_spec][nchan]
                                     double2* __restrict__ xcorr) { // [4*n_spec][nchan], XC only
    const int b = blockIdx.x;  // spec: one PRN's E/P/L/P_HEAD quad
    const int ci = blockIdx.y; // covering channel
    const int m = threadIdx.x; // hop lane: hops m, m + blockDim.x, ... (records may be longer
                               // than the block)
    const gnss_cuda::DespreadJob job = jobs[b];

    // acc/e rows: 0 = Early, 1 = Prompt, 2 = Late, 3 = P_HEAD (the prompt over [0, m_head)).
    double2 acc[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    double e[4] = {0.0, 0.0, 0.0, 0.0};
    // Reference cross-correlations for the analytic peel add-back (XC only; with XC false these
    // and their reduction compile away).
    //   0 = <R_P, R_E>          1 = <R_P, R_L>            (whole window)
    //   2 = <R_P, R_E>|head     3 = <R_P, R_L>|head       (hops [0, m_head) only)
    // The peel subtracts a different gain on each side of the code-period boundary, so the
    // add-back is
    //   V_full[k] = V'[k] + a_head*<R_P 1_head, R_k> + a_tail*<R_P 1_tail, R_k>,
    // with the tail term (whole - head). Like P_HEAD, these use the replicas already in
    // registers.
    double2 xc[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
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

        // Carrier phasor pa = e^{i wc n}, one per hop for the whole quad. The phase comes from
        // the window's reference (DespreadJob::ang0), so wc multiplies only dn; reduce in double
        // before the float sincosf.
        const double ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        const float2 pa = make_float2(cn, sn);
        const float2 pb = make_float2(cn, -sn);

        // One load for the quad (fp32 complex, or a 4+4b byte unpacked with this channel's scale)
        const float2 dd = load_v(data, (size_t)ci * p.data_stride + mh, chan_scale, ci);
        const bool in_head = (mh < job.m_head);

        // Trials run in the order {P, E, L} so the prompt replica is in hand when Early and Late
        // are formed, and <R_P, R_E>, <R_P, R_L> need no second gather. The accumulator index t
        // keeps its meaning (0 = E, 1 = P, 2 = L).
        //
        // XC adds no arithmetic to acc/e, but it raises register pressure, and nvcc may then
        // contract or schedule the identical float MACs differently: corr/energy can differ by
        // ~1 ULP between the XC instantiations. The test gates that at 1e-6, not bit-identity.
        float2 r_P = make_float2(0.f, 0.f);
        // One walk of the chip loop serves all three trials (chip_gather3): their taps at a
        // given chip are neighbours in the Phi slice, so they share L1 lines.
        float2 g3A[3], g3B[3];
        if (ABL != ABL_NO_SYN) {
            const double Cs[3] = {C_P, C_P - job.ds, C_P + job.ds};
            // Shared tables need the SHARED instantiation with Psi and ddw passed explicitly:
            // chip_gather3's defaults (SHARED = false, null Psi) read the Doppler-free tables
            // with no correction.
            if (p.shared)
                gnss_cuda::chip_gather3<float2, false, false, true>(
                    job.inv_cps, job.code_offset, job.code_len, job.n_chips, job.d_first, p.Lf,
                    code, phiA, phiB, ks, kf, Cs, g3A, g3B,
                    (const float2*)job.psiA + (size_t)ci * (p.Lf + 1),
                    (const float2*)job.psiB + (size_t)ci * (p.Lf + 1), job.ddw);
            else
                gnss_cuda::chip_gather3(job.inv_cps, job.code_offset, job.code_len, job.n_chips,
                                        job.d_first, p.Lf, code, phiA, phiB, ks, kf, Cs, g3A,
                                        g3B);
        }
#pragma unroll
        for (int tt = 0; tt < 3; ++tt) {
            const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2; // -> P / E (C_P-ds) / L (C_P+ds)
            float2 sA, sB;
            if (ABL == ABL_NO_SYN) {
                // Constant replica: the gather disappears, the carrier apply and MAC remain.
                sA = make_float2(1.0f, 0.0f);
                sB = make_float2(1.0f, 0.0f);
            } else {
                sA = g3A[tt];
                sB = g3B[tt];
            }
            // Replica channel sample r = 0.5 (pa sA + conj(pa) sB); then the despread MAC:
            // acc += data * conj(r), e += |r|^2 -- float compute, widened once for the reduction.
            const float2 t1 = cmulf(pa, sA);
            const float2 t2 = cmulf(pb, sB);
            const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
            if (ABL == ABL_NO_MAC) {
                // One add keeps the gather live; without it nvcc eliminates the synthesis.
                acc[t].x += (double)r.x;
                continue;
            }
            const double re = (double)(dd.x * r.x + dd.y * r.y); // Re(d * conj(r))
            const double im = (double)(dd.y * r.x - dd.x * r.y); // Im(d * conj(r))
            const double ee = (double)(r.x * r.x + r.y * r.y);
            acc[t].x += re;
            acc[t].y += im;
            e[t] += ee;
            // P_HEAD: the prompt's own MAC, gated on the hop; one predicated add.
            if (t == 1 && in_head) {
                acc[3].x += re;
                acc[3].y += im;
                e[3] += ee;
            }
            if (XC) {
                // The prompt is kept for the other two; <R_P, R_P> is energy row 1 already.
                if (t == 1) {
                    r_P = r;
                } else {
                    const int x = (t == 0) ? 0 : 1; // 0 = Early, 1 = Late
                    // <R_P, R_t> = sum_hop r_P conj(r_t): the despread's <a, b> = sum a conj(b),
                    // in the order the add-back V[k] = V'[k] + a*<R_P, R_k> uses.
                    const double xre = (double)(r_P.x * r.x + r_P.y * r.y);
                    const double xim = (double)(r_P.y * r.x - r_P.x * r.y);
                    xc[x].x += xre;
                    xc[x].y += xim;
                    if (in_head) { // head-restricted rows 2/3
                        xc[x + 2].x += xre;
                        xc[x + 2].y += xim;
                    }
                }
            }
        }
    }

    // Block reduction over hop lanes (each lane already summed its strided hops), run as four
    // sequential passes over one set of shared arrays: four at once would be 24 KB per block.
    // They hold 256 lanes, the most despread_block launches; the tree needs a power-of-2 block.
    __shared__ double sh_re[256], sh_im[256], sh_e[256];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // m_head == 0 (despread_batch's plain E/P/L contract) leaves row 3 identically zero:
        // write zeros and skip the tree. The test is uniform across the block, and the row is
        // still written so no reader sees stale data.
        if (j == 3 && job.m_head == 0) {
            if (m == 0) {
                corr[(size_t)(p.out_rows_spec * b + 3) * gridDim.y + ci] = make_double2(0.0, 0.0);
                energy[(size_t)(p.out_rows_spec * b + 3) * gridDim.y + ci] = 0.0;
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
            corr[(size_t)(p.out_rows_spec * b + j) * gridDim.y + ci] = make_double2(sh_re[0], sh_im[0]);
            energy[(size_t)(p.out_rows_spec * b + j) * gridDim.y + ci] = sh_e[0];
        }
    }

    // Reference cross-correlations, same tree over the same shared arrays (sh_e unused here).
    if (XC) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __syncthreads();
            sh_re[m] = xc[j].x;
            sh_im[m] = xc[j].y;
            __syncthreads();
            for (int off = blockDim.x / 2; off > 0; off >>= 1) {
                if (m < off) {
                    sh_re[m] += sh_re[m + off];
                    sh_im[m] += sh_im[m + off];
                }
                __syncthreads();
            }
            if (m == 0)
                xcorr[(size_t)(4 * b + j) * gridDim.y + ci] = make_double2(sh_re[0], sh_im[0]);
        }
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
                            double* energy, cudaStream_t stream, double2* xcorr) {
    dim3 grid(n_spec, n_chan); // one block per (PRN quad, channel), 4 output rows each
    // XC is a template parameter, not a runtime flag: with it false the cross-term accumulators
    // and their reduction compile away, so a despread without the peel keeps its register count
    // (the fusion win is register-sensitive; see the note at the top of this file).
    if (xcorr != nullptr)
        gnss_despread_kernel<float2, true><<<grid, despread_block(p.n_hops), 0, stream>>>(
            data, nullptr, code, jobs, p, corr, energy, xcorr);
    else
        gnss_despread_kernel<float2, false><<<grid, despread_block(p.n_hops), 0, stream>>>(
            data, nullptr, code, jobs, p, corr, energy, nullptr);
    return cudaGetLastError();
}

cudaError_t launch_despread_q(const unsigned char* data, const float* chan_scale,
                              const int8_t* code, const DespreadJob* jobs, int n_spec, int n_chan,
                              const DespreadParams& p, double2* corr, double* energy,
                              cudaStream_t stream, double2* xcorr) {
    dim3 grid(n_spec, n_chan);
    if (xcorr != nullptr)
        gnss_despread_kernel<unsigned char, true><<<grid, despread_block(p.n_hops), 0, stream>>>(
            data, chan_scale, code, jobs, p, corr, energy, xcorr);
    else
        gnss_despread_kernel<unsigned char, false><<<grid, despread_block(p.n_hops), 0, stream>>>(
            data, chan_scale, code, jobs, p, corr, energy, nullptr);
    return cudaGetLastError();
}

} // namespace gnss_cuda

namespace {

/**
 * Voltage peel. One thread owns one (channel, hop) output sample: load the voltage once,
 * subtract every job's prompt replica, store the residual once.
 *
 * The grid is (hop tile, channel), unlike the despread's (PRN, channel), because there is no
 * reduction: the output is a voltage, not a correlation. Handling every job inside the thread
 * keeps the accumulation deterministic and atomic-free, which is valid because the gains are
 * feed-forward: they do not depend on the data, so the subtractions commute.
 *
 * The replica comes from the same chip gather, Phi tables and code phase as the despread's,
 * because the analytic add-back assumes the subtracted waveform and the correlated reference
 * are the same. Psi and ddw are not applied (see PeelJob::psiA).
 */
/// Hops each thread owns. One is fastest here, unlike the despread: the despread shares one
/// carrier phasor and one voltage load among three gathers per hop, but the peel has one gather
/// per hop, so extra hops per thread add only live state (v[H] and the unrolled gathers), and
/// the register pressure costs the occupancy that hides the fp64 fmod latency. A different
/// geometry or GPU could move the optimum; re-sweep rather than assume.
#define GNSS_PEEL_HOPS_PER_THREAD 1

template <typename T>
__global__ void gnss_peel_kernel(const T* __restrict__ data,           // [nchan][data_stride]
                                 const float* __restrict__ chan_scale, // [nchan], 4+4b only
                                 const int8_t* __restrict__ code,
                                 const gnss_cuda::PeelJob* __restrict__ jobs, // [n_job]
                                 int n_job, gnss_cuda::DespreadParams p,
                                 float2* __restrict__ resid) { // [nchan][n_hops], packed
    constexpr int H = GNSS_PEEL_HOPS_PER_THREAD;
    const int ci = blockIdx.y;
    // A thread's hops are strided by blockDim.x, not consecutive, so each of the H loads and
    // stores stays coalesced across the warp.
    const int base = blockIdx.x * blockDim.x * H + threadIdx.x;

    float2 v[H];
    bool act[H];
#pragma unroll
    for (int t = 0; t < H; ++t) {
        const int mh = base + t * (int)blockDim.x;
        act[t] = (mh < p.n_hops);
        if (act[t])
            v[t] = load_v(data, (size_t)ci * p.data_stride + mh, chan_scale, ci);
    }

    for (int b = 0; b < n_job; ++b) {
        const gnss_cuda::PeelJob job = jobs[b];
        if (!((job.chan_mask >> ci) & 1ULL))
            continue; // carrier not in this channel: nothing of this PRN to remove here
        // Per-segment gains: m_head is the despread's P_HEAD boundary, where the overlay or nav
        // symbol can flip sign, so each side has its own gain.
        const float2 ah = job.a_head[ci];
        const float2 at = job.a_tail[ci];
        if (ah.x == 0.f && ah.y == 0.f && at.x == 0.f && at.y == 0.f)
            continue; // zero gains leave this (PRN, channel) unpeeled
        const int ks = (int)job.inv_cps;
        const float kf = (float)(job.inv_cps - ks);
        const float2* const phiA = job.phiA + (size_t)ci * (p.Lf + 1);
        const float2* const phiB = job.phiB + (size_t)ci * (p.Lf + 1);

#pragma unroll
        for (int t = 0; t < H; ++t) {
            if (!act[t])
                continue;
            const int mh = base + t * (int)blockDim.x;
            const double dn = (double)((long long)mh * (long long)p.fft_len);
            // Code phase and carrier from the window's reference, as in the despread kernel.
            const double C_P = job.cp_ref + dn * job.cps;
            const double ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
            float sn, cn;
            sincosf((float)ang, &sn, &cn);
            const float2 pa = make_float2(cn, sn);
            const float2 pb = make_float2(cn, -sn);
            float2 sA, sB;
            chip_gather(job.inv_cps, job.code_offset, job.code_len, job.n_chips, job.d_first,
                        p.Lf, code, phiA, phiB, ks, kf, C_P, sA, sB);
            const float2 t1 = cmulf(pa, sA);
            const float2 t2 = cmulf(pb, sB);
            const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
            const float2 a = (mh < job.m_head) ? ah : at;
            v[t].x -= a.x * r.x - a.y * r.y; // v -= a * r
            v[t].y -= a.x * r.y + a.y * r.x;
        }
    }

#pragma unroll
    for (int t = 0; t < H; ++t)
        if (act[t])
            resid[(size_t)ci * p.n_hops + base + t * (int)blockDim.x] = v[t];
}

} // namespace

namespace gnss_cuda {

namespace {
inline int peel_block(int n_hops) {
    return (n_hops < 256 * GNSS_PEEL_HOPS_PER_THREAD) ? 64 : 256;
}
inline dim3 peel_grid(int n_hops, int block, int n_chan) {
    const int per_block = block * GNSS_PEEL_HOPS_PER_THREAD;
    return dim3((unsigned)((n_hops + per_block - 1) / per_block), (unsigned)n_chan);
}
} // namespace

cudaError_t launch_peel(const float2* data, const int8_t* code, const PeelJob* jobs, int n_job,
                        int n_chan, const DespreadParams& p, float2* resid, cudaStream_t stream) {
    const int block = peel_block(p.n_hops);
    gnss_peel_kernel<float2><<<peel_grid(p.n_hops, block, n_chan), block, 0, stream>>>(
        data, nullptr, code, jobs, n_job, p, resid);
    return cudaGetLastError();
}

cudaError_t launch_peel_q(const unsigned char* data, const float* chan_scale, const int8_t* code,
                          const PeelJob* jobs, int n_job, int n_chan, const DespreadParams& p,
                          float2* resid, cudaStream_t stream) {
    const int block = peel_block(p.n_hops);
    gnss_peel_kernel<unsigned char><<<peel_grid(p.n_hops, block, n_chan), block, 0, stream>>>(
        data, chan_scale, code, jobs, n_job, p, resid);
    return cudaGetLastError();
}

} // namespace gnss_cuda
