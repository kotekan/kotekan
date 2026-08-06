#include "cudaGnssChordDespread.hpp"

#include "cudaGnssReplicaDevice.cuh"

#include <cuda_runtime.h>
#include <math.h>

// The generation half is line-for-line the fused kernel's replica path with the data load and
// the MAC removed, and the correlation half is that MAC with an element axis added. Keeping the
// two halves recognisably the same code as cudaGnssDespreadKernel.cu is the point: the split is
// gated by a test that requires the two paths to agree exactly, and the cheapest way to keep
// that true is for the arithmetic to be visibly identical rather than merely equivalent.

namespace {

using gnss_cuda::chip_gather;
using gnss_cuda::cmulf;

/// grid.x = job (PRN), grid.y = covering channel, block.x = hop lane (grid-stride).
/// Mirrors gnss_despread_kernel's geometry so the per-hop arithmetic is identical.
///
/// MAXT is the block width this instantiation is COMPILED for, via __launch_bounds__. It is a
/// template parameter and not just a launch-time choice because the width is capped by registers
/// -- unbounded, the gather compiles to ~72 registers on sm_86 and the driver refuses anything
/// past 512 threads. Naming the bound lets ptxas trade registers for the width, and the width is
/// what sets the DRAM traffic (see WAVE_THREADS). Whether that trade wins is a measurement, so
/// both instantiations exist and scripts/gnss/wavebench.cpp compares them.
/// FUSE3 walks the chip loop ONCE for all three E/P/L trials (@ref gnss_cuda::chip_gather3)
/// instead of three times. Bit-identical either way; it trades registers for DRAM traffic, and
/// registers are what cap MAXT, so the two knobs pull against each other.
template<int MAXT, bool FUSE3, class PHI = float2>
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

    // Hoisted exactly as in the fused kernel: invariant across hops and across trials.
    const int ks = (int)job.inv_cps;
    const float kf = (float)(job.inv_cps - ks);
    const PHI* phiA = (const PHI*)job.phiA + (size_t)ci * (p.Lf + 1);
    const PHI* phiB = (const PHI*)job.phiB + (size_t)ci * (p.Lf + 1);

    for (int mh = m; mh < p.n_hops; mh += blockDim.x) {
        // An uncovered channel still has to WRITE its replica rows -- the correlator reads the
        // wave array unconditionally, and a stale/uninitialised sample would contribute garbage
        // rather than nothing. Zero is the correct replica for a channel this PRN does not
        // occupy, and its energy contribution is zero too.
        if (!covered) {
#pragma unroll
            for (int t = 0; t < 3; ++t)
                wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] =
                    make_float2(0.f, 0.f);
            continue;
        }

        const long long n_m = p.n0 + (long long)mh * p.fft_len;
        const double C_P = job.cp0 + (double)n_m * job.cps;

        // TWO-PRODUCT before the reduction. wc ~ 2.31 rad/sample and n_m ~ 2.95e15 after ten
        // days of F-engine uptime, so wc*n_m ~ 6.8e15 lands in the binade [2^52, 2^53) where a
        // double's ULP is EXACTLY ONE RADIAN -- and n_m advances per HOP, so the error re-rolls
        // 2048 times inside every record rather than once per record. Measured 0.604 rad rms,
        // costing exp(-sigma^2/2) = 0.83 of the correlation amplitude.
        //
        // fma recovers the part the product rounded away EXACTLY (n_m < 2^53 is exact as a
        // double, so wc*n_m = pr + er with no error), and reducing pr while carrying er
        // separately is congruent mod 2pi. What remains is wc's own representation error times
        // n_m -- large in absolute terms but varying by only ~8e-9 rad across a whole record,
        // i.e. a constant phase offset, which a correlation does not care about.
        //
        // long double is the fix on the host side (see gnssChannelizedReplica.cpp); CUDA maps
        // long double to double, so the same trick is not available here. This is invisible at
        // prototype scale -- airspy's n_m ~ 1e9 gives a 4.8e-7 rad ULP.
        const double pr = job.wc * (double)n_m;
        const double er = fma(job.wc, (double)n_m, -pr);
        const double ang = fmod(pr, 2.0 * M_PI) + er;
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        const float2 pa = make_float2(cn, sn);
        const float2 pb = make_float2(cn, -sn);
        const bool in_head = (mh < job.m_head);

        // Trial order {P, E, L} as in the fused kernel -- kept so the arithmetic matches even
        // though nothing here needs the prompt in hand first.
        //
        // ⚠️ THE TWO BRANCHES DUPLICATE THE CARRIER/WRITE/ENERGY BLOCK ON PURPOSE. Hoisting it
        // into one loop fed by a float2 gA[3] array -- the obvious tidy-up, and the first thing
        // written here -- moved the FUSE3=false path's codegen and broke bit-identity with
        // gnss_despread_kernel by 3.1e-07, on the UNFUSED path, where nothing should have
        // changed at all. The split-vs-fused test caught it; the bench could not, because it
        // compares this kernel against itself. Keep the else branch textually what it was.
        if (FUSE3) {
            // Same three code phases in the same order, one walk of the chip loop.
            const double Cs[3] = {C_P, C_P - job.ds, C_P + job.ds};
            float2 sA[3], sB[3];
            gnss_cuda::chip_gather3(job.inv_cps, job.code_offset, job.code_len, job.n_chips, p.Lf,
                                    code, phiA, phiB, ks, kf, Cs, sA, sB);
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
        } else {
#pragma unroll
            for (int tt = 0; tt < 3; ++tt) {
                const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2; // -> cp0 / cp0-ds / cp0+ds
                float2 sA, sB;
                chip_gather(job.inv_cps, job.code_offset, job.code_len, job.n_chips, p.Lf, code,
                            (const float2*)phiA, (const float2*)phiB, ks, kf,
                            C_P + (double)(t - 1) * job.ds, sA, sB);
                const float2 t1 = cmulf(pa, sA);
                const float2 t2 = cmulf(pb, sB);
                const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
                wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] = r;

                const double ee = (double)(r.x * r.x + r.y * r.y);
                e[t] += ee;
                if (t == 1 && in_head)
                    e[3] += ee;
            }
        }
    }

    // Block reduction over hop lanes, one row at a time (as the fused kernel does, for the same
    // shared-memory reason). Sized for THIS instantiation's width -- 8 KB at the widest, against
    // the SM's 100 KB, so shared memory is never the occupancy limiter here (registers are).
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
    // The ONLY job field this kernel needs: everything else is already baked into `wave`.
    // Uniform across the block, so it is one broadcast load per block.
    const int m_head = jobs[b].m_head;

    extern __shared__ double2 sh[]; // [blockDim.x * blockDim.y]

    // Round the element loop UP to a whole number of blockDim.x passes and predicate the memory
    // accesses instead of trimming the loop. The reduction below calls __syncthreads(), so every
    // thread in the block must execute the same number of iterations -- letting the tail elements
    // drop out early would leave the block's threads at different barriers, which is undefined
    // behaviour (in practice, a hang). n_elem is not a multiple of the block width in general:
    // the pathfinder has 32 live elements of 128 allocated, and both numbers move.
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
                // Uniform across threadIdx.x -> broadcast, not a per-element load.
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

cudaError_t launch_waveform_tuned(const int8_t* code, const DespreadJob* jobs, int n_job,
                                  int n_chan, const DespreadParams& p, float2* wave, double* energy,
                                  int threads_hint, int fuse3, int phi16,
                                  cudaStream_t stream) {
    const bool fuse = fuse3 < 0 ? WAVE_FUSE3 : (fuse3 != 0);
    int cap = threads_hint > 0 ? threads_hint : WAVE_THREADS;
    if (cap > 1024)
        cap = 1024; // sh_e is sized for this
    int threads = p.n_hops < cap ? p.n_hops : cap;
    // The shared reduction tree halves blockDim.x, so it must be a power of two.
    int pow2 = 1;
    while (pow2 * 2 <= threads)
        pow2 *= 2;
    threads = pow2;
    // ASK THE DRIVER, do not assume. Each instantiation's ceiling is whatever ptxas gave it, it
    // moves with the architecture and with any edit to the gather, and exceeding it does not
    // degrade -- the launch fails outright with "too many resources requested". Clamping here is
    // what makes WAVE_THREADS a portable REQUEST rather than a promise the hardware may not keep.
    auto ceiling = [](const void* fn) {
        cudaFuncAttributes at{};
        return cudaFuncGetAttributes(&at, fn) == cudaSuccess ? at.maxThreadsPerBlock : 256;
    };
    const dim3 grid(n_job, n_chan);
    // BENCH ONLY: fp16 Phi. job.phiA/phiB then point at __half2 tables (the struct field type is
    // a lie the caller opts into). Answers whether the kernel is byte-limited or request-limited;
    // production never takes this path.
    if (phi16) {
        auto ceil16 = [](const void* fn) {
            cudaFuncAttributes at{};
            return cudaFuncGetAttributes(&at, fn) == cudaSuccess ? at.maxThreadsPerBlock : 256;
        };
        if (threads > 512) {
            static const int k16 = ceil16((const void*)gnss_waveform_kernel<1024, true, __half2>);
            if (threads <= k16) {
                gnss_waveform_kernel<1024, true, __half2>
                    <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
                return cudaGetLastError();
            }
            threads = 512;
        }
        gnss_waveform_kernel<512, true, __half2>
            <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (threads > 512) {
        static const int kmaxF = ceiling((const void*)gnss_waveform_kernel<1024, true>);
        static const int kmaxU = ceiling((const void*)gnss_waveform_kernel<1024, false>);
        if (threads <= (fuse ? kmaxF : kmaxU)) {
            if (fuse)
                gnss_waveform_kernel<1024, true>
                    <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
            else
                gnss_waveform_kernel<1024, false>
                    <<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
            return cudaGetLastError();
        }
        threads = 512;
    }
    static const int k5F = ceiling((const void*)gnss_waveform_kernel<512, true>);
    static const int k5U = ceiling((const void*)gnss_waveform_kernel<512, false>);
    const int k5 = fuse ? k5F : k5U;
    if (threads > k5)
        threads = k5;
    if (fuse)
        gnss_waveform_kernel<512, true><<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
    else
        gnss_waveform_kernel<512, false><<<grid, threads, 0, stream>>>(code, jobs, p, wave, energy);
    return cudaGetLastError();
}

cudaError_t launch_waveform(const int8_t* code, const DespreadJob* jobs, int n_job, int n_chan,
                            const DespreadParams& p, float2* wave, double* energy,
                            cudaStream_t stream) {
    return launch_waveform_tuned(code, jobs, n_job, n_chan, p, wave, energy, 0, -1, 0, stream);
}

cudaError_t launch_correlate_nm(const unsigned char* data, const float* chan_scale,
                                const int* chan_ids, const float2* wave, const DespreadJob* jobs,
                                int n_job, int n_chan, int n_elem, int elem_stride,
                                int frame_chan_stride, const DespreadParams& p, double2* corr,
                                cudaStream_t stream) {
    // Elements across x (coalescing), hops across y (parallelism). n_elem is small on the
    // pathfinder (32 live of 128), so without the hop lanes the kernel would run at a few
    // thousand threads and leave the device idle.
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

} // namespace gnss_cuda
