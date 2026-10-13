#include "cudaGnssChordDespread.hpp"

#include "cudaGnssReplicaDevice.cuh"

#include <cuda_runtime.h>
#include <math.h>

// The synthesis kernel is the fused kernel's replica path (cudaGnssDespreadKernel.cu) without the
// data load and MAC, and the correlation kernel is that MAC with an element axis added.
// cudaGnssDespreadTest requires the split and fused paths to agree exactly, so keep the
// arithmetic textually the same.

namespace {

using gnss_cuda::cmulf;

/// grid.x = job (PRN), grid.y = covering channel, block.x = hop lane (grid-stride), as in
/// gnss_despread_kernel, so the per-hop arithmetic is identical.
///
/// MAXT is the block width this instantiation is compiled for, via __launch_bounds__. Registers
/// cap the width (without the bound the gather's register use rules out a 1024-thread block),
/// and the width sets the DRAM traffic (see WAVE_THREADS), so the bound lets ptxas trade
/// registers for width. The 512 and 1024 instantiations both exist; launch_waveform takes the
/// wider one when the driver reports that it can launch.
/// PHI = __half2 reads fp16 tables.
template<int MAXT, class PHI = float2>
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
    const PHI* phiA = (const PHI*)job.phiA + (size_t)ci * (p.Lf + 1);
    const PHI* phiB = (const PHI*)job.phiB + (size_t)ci * (p.Lf + 1);

    for (int mh = m; mh < p.n_hops; mh += blockDim.x) {
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

        // Trial order {P, E, L}, as in the fused kernel, so the arithmetic matches: the same
        // three code phases in the same order, one walk of the chip loop.
        const double Cs[3] = {C_P, C_P - job.ds, C_P + job.ds};
        float2 sA[3], sB[3];
        gnss_cuda::chip_gather3<PHI>(job.inv_cps, job.code_offset, job.code_len, job.n_chips,
                                     job.d_first, p.Lf, code, phiA, phiB, ks, kf, Cs, sA, sB);
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

cudaError_t launch_waveform(const int8_t* code, const DespreadJob* jobs, int n_job, int n_chan,
                            const DespreadParams& p, float2* wave, double* energy,
                            cudaStream_t stream) {
    int threads = p.n_hops < WAVE_THREADS ? p.n_hops : WAVE_THREADS;
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
    // p.phi_half selects the __half2 gather (despite job.phiA/phiB's float2 type); the tables
    // the jobs point at must already be half storage, which GnssCudaDespread::ensure_phi
    // guarantees.
    if (p.phi_half) {
        if (threads > 512) {
            static const int k16 = ceiling((const void*)gnss_waveform_kernel<1024, __half2>);
            if (threads <= k16) {
                gnss_waveform_kernel<1024, __half2>
                    <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
                return cudaGetLastError();
            }
            threads = 512;
        }
        gnss_waveform_kernel<512, __half2>
            <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (threads > 512) {
        static const int kmaxF = ceiling((const void*)gnss_waveform_kernel<1024>);
        if (threads <= kmaxF) {
            gnss_waveform_kernel<1024><<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
            return cudaGetLastError();
        }
        threads = 512;
    }
    static const int k5F = ceiling((const void*)gnss_waveform_kernel<512>);
    if (threads > k5F)
        threads = k5F;
    gnss_waveform_kernel<512><<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
    return cudaGetLastError();
}

void phi_to_half(const float2* src, void* dst, size_t n) {
    // Here rather than in GnssCudaDespread.cpp so __half2 stays out of g++-compiled code.
    __half2* d = (__half2*)dst;
    for (size_t i = 0; i < n; ++i)
        d[i] = __floats2half2_rn(src[i].x, src[i].y);
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
