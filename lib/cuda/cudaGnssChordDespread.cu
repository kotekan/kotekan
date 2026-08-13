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
template<int MAXT, bool FUSE3, class PHI = float2, bool ABL_NOLOAD = false, bool ILV = false,
         bool HOPPERM = false>
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
    // ⚠️ The channel stride is in ELEMENTS of the table's own type. Under ILV the table is
    // float4, so advancing a float2* by ci*(Lf+1) lands at half the right offset and every
    // channel above 0 reads out of bounds -- which is exactly how the first --ilv run died.
    const PHI* phiA = ILV ? (const PHI*)((const float4*)job.phiA + (size_t)ci * (p.Lf + 1))
                          : (const PHI*)job.phiA + (size_t)ci * (p.Lf + 1);
    const PHI* phiB = (const PHI*)job.phiB + (size_t)ci * (p.Lf + 1);

    // PRELOAD the permutation. Read inside the loop, hop_perm[] is a dependent global load at
    // the head of every hop: it must land before n_m -> C_P -> base -> the whole 212-step chain
    // can start, and with ~2 hops per thread there is nothing to hide it behind. Measured 0.81x
    // that way. Issuing all of a thread's lookups up front lets them overlap each other and the
    // prologue instead.
    constexpr int MAX_HOPS_PER_THREAD = 8;
    int hop_id[MAX_HOPS_PER_THREAD];
    if (HOPPERM) {
#pragma unroll
        for (int k = 0; k < MAX_HOPS_PER_THREAD; ++k) {
            const int mh_i = m + k * (int)blockDim.x;
            hop_id[k] = (mh_i < p.n_hops) ? p.hop_perm[(size_t)b * p.n_hops + mh_i] : 0;
        }
    }
    int kk = 0;
    for (int mh_i = m; mh_i < p.n_hops; mh_i += blockDim.x, ++kk) {
        const int mh = HOPPERM ? hop_id[kk < MAX_HOPS_PER_THREAD ? kk : 0] : mh_i;
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

        const long long n_m = p.n0 + (long long)mh * p.fft_len; // arm 0 only (see below)
        // Intra-record sample offset: exact as a double, and the ONLY thing the per-sample
        // rates are allowed to multiply now (tasks #52, #54).
        const double dn = (double)((long long)mh * (long long)p.fft_len);
        // CODE PHASE FROM A REFERENCE (task #54). This was cp0 + n_m*cps on the ABSOLUTE
        // sample: n_m*cps reaches 6.1e12 chips at CHORD uptime, where a double's ulp is
        // ~1e-3 chips, and it re-rolls every hop. Measured GPU-vs-CPU replica error on the
        // PROMPT row, with the carrier already referenced via ang0: 7.9e-08 at 0.007 days,
        // 8.2e-07 at 0.068, 9.6e-04 at 0.678, 3.6e-02 at 6.8. Zero at prototype scale, 3.6%
        // at the real thing -- and WITHIN a record, so no per-record correction can reach it.
        //
        // job.cp_ref is the prompt code phase at p.n0, reduced mod code_len on the host in
        // long double. cps then multiplies only the intra-record offset (<= 3.4e7 samples),
        // so its rounding contributes ~1e-11 chips instead of ~1e-3.
        const double C_P = job.cp_ref + dn * job.cps;

        // PHASE FROM A REFERENCE, NOT FROM THE ABSOLUTE SAMPLE (task #52, 2026-08-13).
        //
        // This used to evaluate wc*n_m with a two-product, to survive n_m ~ 2.95e15 landing in
        // the binade where a double's ULP is EXACTLY ONE RADIAN. That trick is exact for the
        // PRODUCT, and the note it carried said the leftover -- "wc's own representation error
        // times n_m" -- was harmless because it varies by only ~8e-9 rad across a record, i.e.
        // is a constant phase offset "which a correlation does not care about".
        //
        // ⚠️ THAT LAST STEP IS THE BUG. It is constant WITHIN a record and re-rolls BETWEEN
        // them, because the Doppler is re-propagated every record and wc changes with it. A
        // per-record constant is exactly what every CROSS-RECORD estimator lives on: the deep
        // fold, the ADR arc, and the per-channel spectrum window sum. Measured with e2e (24
        // realizations per point, records at the live 10.486 ms cadence): flat at 0.012 rad
        // below ~0.2 days of uptime, then dead linear in absolute time -- 0.023 / 0.066 / 0.217
        // rad at 0.6 / 2 / 6.8 days. On sky the per-record phase floor reads 0.745 rad, and it
        // reproduces here with NO NOISE.
        //
        // The fix is not more precision, it is a shorter lever: take the phase at the window's
        // reference sample from the host (job.ang0, long double) and let wc multiply only the
        // intra-record offset, which never exceeds 3.4e7 samples. Every rounding of the ~1.18
        // GHz carrier then contributes 1.5e-8 rad instead of tenths. Same medicine as #45 step
        // 6 gave the code phase.
        //
        // dn is exact as a double (< 2^25 hops * 2^14 samples), and wc*dn <= 7.8e7 rad has a
        // 1.5e-8 rad ULP, so the reduction below loses nothing that matters. Reduce BEFORE the
        // float cast: sincosf on 7.8e7 rad would be meaningless (float ULP there is ~8 rad).
        // A/B ARM (task #52/#55, TEMPORARY). Arm 1 is the fix: ang0 carries the phase at the
        // window reference and wc multiplies only the intra-record offset, so the ~1.18 GHz
        // carrier's rounding cannot reach the absolute sample index. Arm 0 is exactly what
        // shipped before 86349ac4d, kept textually intact so the comparison is honest.
        double ang;
        if (p.carrier_phase_from_ref) {
            ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
        } else {
            const double pr = job.wc * (double)n_m;
            const double er = fma(job.wc, (double)n_m, -pr);
            ang = fmod(pr, 2.0 * M_PI) + er;
        }
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
            gnss_cuda::chip_gather3<PHI, ABL_NOLOAD, ILV>(job.inv_cps, job.code_offset, job.code_len,
                                                    job.n_chips, p.Lf, code, phiA, phiB, ks, kf,
                                                    Cs, sA, sB);
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
    if (phi16 == 5) { // BENCH: hop-sorted lane->hop mapping (coalesces the Phi gather)
        gnss_waveform_kernel<1024, true, float2, false, false, true>
            <<<grid, threads > 1024 ? 1024 : threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (phi16 == 6) { // BENCH: hop-sorted AND fp16
        gnss_waveform_kernel<1024, true, __half2, false, false, true>
            <<<grid, threads > 1024 ? 1024 : threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (phi16 == 4) { // BENCH: interleaved float4 Phi -- one 16 B load instead of two 8 B
        gnss_waveform_kernel<1024, true, float2, false, true>
            <<<grid, threads > 1024 ? 1024 : threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
    if (phi16 == 2) { // BENCH: no-load ablation -- all the index math and flops, no Phi traffic
        gnss_waveform_kernel<1024, true, float2, true>
            <<<grid, threads > 1024 ? 1024 : threads, 0, stream>>>(code, jobs, p, wave, energy);
        return cudaGetLastError();
    }
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

// ---------------------------------------------------------------------------------------------
// Path B: quantize the materialised replicas into synthetic 4+4b N^2 stations.
// ---------------------------------------------------------------------------------------------

namespace {

/// One thread per (hop, covering channel, lane); consecutive threads take consecutive lanes so
/// the 128-byte station row of each (hop, chan) is one coalesced store. Lane identity is
/// STABLE: lane = 4*prn_slot + trial (E/P/L/P_HEAD), so the tile consumer's lane -> PRN map
/// survives satellites rising and setting; slots without a live spec this record write the
/// 0x88 background (0+0j). The P_HEAD lane is the PROMPT gated to hops [0, m_head) -- a
/// time-gated replica IS just another lane, which is how P_HEAD survives nt_outer=1.
///
/// SCALE: s = 7 / (3 * rms) per (lane, channel), rms from the energy row launch_waveform
/// already computed (row 3 = the head-gated prompt energy over m_head hops). The consumer
/// never needs s: the M^2 diagonal carries the quantized replica's own energy, so
/// conj(V_mixed)/M^2_diag is the amplitude with the scale cancelled -- the same
/// correlation/replica_energy normalization the despread uses. Values CLAMP to [-7,7]:
/// -8 silently corrupts the whole N^2 launch (n2k's negate_4bit).
__global__ void gnss_pack44_kernel(const float2* __restrict__ wave,
                                   const double* __restrict__ energy,
                                   const gnss_cuda::DespreadJob* __restrict__ jobs,
                                   const int* __restrict__ slot2spec, int n_slot, int n_chan,
                                   int n_hops, const int* __restrict__ chan_map,
                                   int frame_chan_stride, int num_synth, int conj_replica,
                                   unsigned char* __restrict__ synth) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = idx % num_synth;
    const int c = (idx / num_synth) % n_chan;
    const int m = idx / (num_synth * n_chan);
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
                // CONJUGATE THE REPLICA, not the data (DespreadParams::conj_data's job on the
                // fused path). The N^2 kernel is production's and has no conj flag, and the
                // antenna input is shared -- so the F-engine's conjugation must be absorbed
                // here. It CANNOT be undone downstream: the correlator forms
                // sum_t R conj(D), while the tracker forms sum_t conj(D) conj(R), and those
                // are not conjugates of each other once summed. Injecting conj(R) makes the
                // mixed tile equal the tracker's answer directly.
                int qi = __float2int_rn(conj_replica ? -s * w.y : s * w.y);
                qr = qr < -7 ? -7 : (qr > 7 ? 7 : qr);
                qi = qi < -7 ? -7 : (qi > 7 ? 7 : qi);
                out = (unsigned char)(((qr + 8) << 4) | ((qi + 8) & 0xf));
            }
        }
    }
    synth[((size_t)m * frame_chan_stride + chan_map[c]) * num_synth + lane] = out;
}

} // namespace

cudaError_t launch_pack44(const float2* wave, const double* energy, const DespreadJob* jobs,
                          const int* slot2spec, int n_slot, int n_chan, int n_hops,
                          const int* chan_map, int frame_chan_stride, int num_synth,
                          bool conj_replica, unsigned char* synth, cudaStream_t stream) {
    const long total = (long)n_hops * n_chan * num_synth;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    gnss_pack44_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        wave, energy, jobs, slot2spec, n_slot, n_chan, n_hops, chan_map, frame_chan_stride,
        num_synth, conj_replica ? 1 : 0, synth);
    return cudaGetLastError();
}

} // namespace gnss_cuda
