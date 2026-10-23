#include "cudaGnssPsiSynth.hpp"
#include "cudaGnssReplicaDevice.cuh" // for cmulf

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include <string>

// The prototype of docs/gnss_gpu_search.md's code-phase-table synthesis. Nothing here is shared
// with the production kernels: cudaGnssChordDespread.cu stays as it is, and this kernel is
// judged against it at the correlation level by cudaGnssPsiBench.cpp.

namespace {

using gnss_cuda::cmulf;

/// Table element load: float2, or __half2 widened to float2 (DespreadParams::phi_half).
__device__ inline float2 tab_ld(const float2* t, int i) {
    return t[i];
}
__device__ inline float2 tab_ld(const __half2* t, int i) {
    return __half22float2(t[i]);
}

/// Table value at absolute grid index @c a in [0, G*L): the plain table, or for a tiled code
/// the period-sign split sec[q] Psi_plain[m'] + (sec[q-1] - sec[q]) Psi_prev[m'] (see
/// PsiJob). @c Ns, @c GP, @c n_prev are block-uniform, so the branches do not diverge.
template<class TAB>
__device__ inline float2 psi_at(const TAB* __restrict__ tab, const TAB* __restrict__ prev,
                                const int8_t* __restrict__ sec, int Ns, int GP, int n_prev, int a) {
    if (Ns == 1)
        return tab_ld(tab, a);
    const int q = a / GP;
    const int mp = a - q * GP;
    const float sq = (float)sec[q];
    float2 v = tab_ld(tab, mp);
    v.x *= sq;
    v.y *= sq;
    if (mp < n_prev) {
        const float sp = (float)sec[q == 0 ? Ns - 1 : q - 1] - sq;
        const float2 pv = tab_ld(prev, mp);
        v.x = fmaf(sp, pv.x, v.x);
        v.y = fmaf(sp, pv.y, v.y);
    }
    return v;
}

__device__ inline double warp_sum(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

/// Cubic Lagrange weights at fraction f in [0, 1) for the stencil j = -1, 0, 1, 2.
__device__ inline void lagrange4(float f, float w[4]) {
    const float fm1 = f - 1.f, fm2 = f - 2.f, fp1 = f + 1.f;
    w[0] = -f * fm1 * fm2 * (1.f / 6.f);
    w[1] = fp1 * fm1 * fm2 * 0.5f;
    w[2] = -fp1 * f * fm2 * 0.5f;
    w[3] = fp1 * f * fm1 * (1.f / 6.f);
}

/// Per-block constants of the Psi evaluation, hoisted once per (job, channel). The waveform
/// kernel and the fused pack (gnss_psi_pack_kernel) share this and the per-hop functions below,
/// so the two synthesise bit-identical samples and the pack's bytes equal launch_psi_waveform's
/// rows put through launch_pack44.
template<class TAB>
struct PsiBlk {
    const TAB* tab;
    const TAB* prev;
    const int8_t* sec;
    int Ns, GP, GL, n_prev;
    float th;
    float2 Ej[4];    ///< e^{+i th j}, j = -1, 0, 1, 2
    double dsG, GLd; ///< fp64 path constants (grid units)
    long long GLf;   ///< fixed-point path: GL << shift
    int fsh;         ///< shift - 24: the fraction is read from the top 24 bits below the point
};

template<class TAB>
__device__ __forceinline__ void psi_blk_setup(const gnss_cuda::DespreadJob& job,
                                              const gnss_cuda::PsiJob& pj, float th, int ci,
                                              PsiBlk<TAB>& B) {
    const int G = pj.G;
    B.GP = G * pj.P;
    B.GL = G * job.code_len;
    B.tab = (const TAB*)pj.psi + (size_t)ci * B.GP;
    B.prev = pj.psi_prev ? (const TAB*)pj.psi_prev + (size_t)ci * pj.n_prev : nullptr;
    B.sec = pj.sec;
    B.Ns = pj.Ns;
    B.n_prev = pj.n_prev;
    B.th = th;
    float s1, c1;
    sincosf(th, &s1, &c1);
    B.Ej[1] = make_float2(1.f, 0.f);
    B.Ej[2] = make_float2(c1, s1);
    B.Ej[0] = make_float2(c1, -s1);
    B.Ej[3] = cmulf(B.Ej[2], B.Ej[2]);
    B.dsG = job.ds * (double)G;
    B.GLd = (double)B.GL;
    // 32-bit conversions only below: a 64-bit int to float conversion runs on the fp64 pipe.
    B.GLf = (long long)B.GL << pj.shift;
    B.fsh = pj.shift - 24;
}

/// The hop's prompt code phase in grid units reduced to [0, GL) (CGr on the fp64 path, Cfr on
/// the fixed-point path) and the conjugate carrier phasor pb.
///
/// FIXED: the per-hop code phase and carrier are 64-bit fixed-point accumulations (PsiJob's
/// cg_ref/dcg/dsg in 2^-shift grid units, ang0_t/dturn_t in 2^-64 turns) instead of the fp64
/// DFMA + fmod of the production kernel. On GA102 fp64 runs at 1/32 of fp32, and the per-hop
/// fp64 fmod alone is a sizeable share of a 10-20 us launch.
template<bool FIXED, class TAB>
__device__ __forceinline__ void
psi_hop(const gnss_cuda::DespreadJob& job, const gnss_cuda::PsiJob& pj, const PsiBlk<TAB>& B,
        int fft_len, int mh, double& CGr, long long& Cfr, float2& pb) {
    float2 pa;
    if (FIXED) {
        // The wrap count (<= ~11 per record at CHORD) from a float estimate, then one
        // conditional fix-up each way: a 64-bit modulo is emulated through fp64.
        const int q = (int)(fmaf((float)mh, pj.dcg_r, pj.cg_r));
        Cfr = pj.cg_ref + (long long)mh * pj.dcg - (long long)q * B.GLf;
        Cfr += (Cfr < 0) ? B.GLf : 0;
        Cfr -= (Cfr >= B.GLf) ? B.GLf : 0;
        CGr = 0.0;
        const unsigned long long at = pj.ang0_t + (unsigned long long)mh * pj.dturn_t;
        // top 24 bits of the turn fraction -> [0, 2) half-turns for sincospif
        const float x = (float)(unsigned)(at >> 40) * (2.f / 16777216.f);
        float sn, cn;
        sincospif(x, &sn, &cn);
        pa = make_float2(cn, sn);
    } else {
        // As the production kernel: references at p.n0 reduced on the host, the rates
        // multiply only the intra-record offset dn (cudaGnssChordDespread.cu:69-85).
        const double dn = (double)((long long)mh * (long long)fft_len);
        const double C_P = job.cp_ref + dn * job.cps;
        const double CG = C_P * (double)pj.G;
        CGr = CG - floor(CG / B.GLd) * B.GLd;
        Cfr = 0;
        const double ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        pa = make_float2(cn, sn);
    }
    pb = make_float2(pa.x, -pa.y);
}

/// One trial of the hop: tt = 0 prompt (C), 1 early (C - ds), 2 late (C + ds). Per trial:
///     S = e^{-i th f} sum_{j=-1..2} w_j(f) e^{+i th j} Psi[(q + j) mod GL],
///     r = 0.5 conj(pa) S,
/// q = floor(C), f = C - q; one sincosf for the fraction ramp, four table loads, ~40 flops.
template<bool FIXED, class TAB>
__device__ __forceinline__ float2 psi_trial(const gnss_cuda::PsiJob& pj, const PsiBlk<TAB>& B,
                                            double CGr, long long Cfr, int tt, float2 pb) {
    int qi;
    float f;
    if (FIXED) {
        long long Cf = Cfr + (tt == 0 ? 0 : (tt == 1 ? -pj.dsg : pj.dsg));
        Cf += (Cf < 0) ? B.GLf : 0;
        Cf -= (Cf >= B.GLf) ? B.GLf : 0;
        qi = (int)(Cf >> pj.shift);
        f = (float)((unsigned)(Cf >> B.fsh) & 0xffffffu) * (1.f / 16777216.f);
    } else {
        const double Cg = CGr + (tt == 0 ? 0.0 : (tt == 1 ? -B.dsG : B.dsG));
        const double qd = floor(Cg);
        f = (float)(Cg - qd);
        qi = (int)qd;
        qi += (qi < 0) ? B.GL : 0;
        qi -= (qi >= B.GL) ? B.GL : 0;
    }
    float w[4];
    lagrange4(f, w);
    float2 S = make_float2(0.f, 0.f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int a = qi + (j - 1);
        a += (a < 0) ? B.GL : 0;
        a -= (a >= B.GL) ? B.GL : 0;
        const float2 v = psi_at<TAB>(B.tab, B.prev, B.sec, B.Ns, B.GP, B.n_prev, a);
        const float2 wE = make_float2(w[j] * B.Ej[j].x, w[j] * B.Ej[j].y);
        const float2 c = cmulf(wE, v);
        S.x += c.x;
        S.y += c.y;
    }
    // The fraction ramp e^{-i th f}.
    float sf, cf;
    sincosf(-B.th * f, &sf, &cf);
    S = cmulf(S, make_float2(cf, sf));
    const float2 t2 = cmulf(pb, S);
    return make_float2(0.5f * t2.x, 0.5f * t2.y);
}

__device__ __forceinline__ float psi_e2(float2 r) {
    return r.x * r.x + r.y * r.y;
}

/// Block reduction of the four energy accumulators over the hop lanes: warp shuffles, one
/// barrier, warp 0 finishes and lane 0 writes energy_row[j * stride]. The production kernel's
/// shared tree takes 44 barriers for the four rows, which at two hops per thread was the
/// largest stall in the profile. blockDim.x must be a multiple of 32.
template<int MAXT>
__device__ __forceinline__ void psi_reduce(double e[4], int m, double (*sh_w)[4],
                                           double* energy_row, size_t stride) {
    const int lane = m & 31, warp = m >> 5, n_warp = blockDim.x >> 5;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        e[j] = warp_sum(e[j]);
        if (lane == 0)
            sh_w[warp][j] = e[j];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const double v = warp_sum(lane < n_warp ? sh_w[lane][j] : 0.0);
            if (lane == 0)
                energy_row[(size_t)j * stride] = v;
        }
    }
}

/// grid.x = job (PRN), grid.y = covering channel, block.x = hop lane (grid-stride): the shape
/// of gnss_waveform_kernel, so the output rows, energy rows and hop order are identical.
template<int MAXT, bool ROT, class TAB, bool FIXED>
__global__ __launch_bounds__(MAXT) void gnss_psi_waveform_kernel(
    const gnss_cuda::DespreadJob* __restrict__ jobs, const gnss_cuda::PsiJob* __restrict__ pjobs,
    const float* __restrict__ theta, // [n_job][nchan]
    gnss_cuda::DespreadParams p,
    const float* __restrict__ rot, // ROT only: [n_job][nchan]
    float2* __restrict__ wave,     // [3*n_job][nchan][n_hops]
    double* __restrict__ energy) { // [4*n_job][nchan]
    const int b = blockIdx.x;
    const int ci = blockIdx.y;
    const int m = threadIdx.x;
    const int n_chan = gridDim.y;
    const gnss_cuda::DespreadJob job = jobs[b];
    const gnss_cuda::PsiJob pj = pjobs[b];

    // E, P, L, P_HEAD. The fixed-point path sums its few hops per thread in float (2 at CHORD
    // geometry, 6e-8 relative) and widens once: a double add per hop runs on GA102's 1/32-rate
    // fp64 pipe, which the profile showed busy 40% of the launch.
    double e[4] = {0.0, 0.0, 0.0, 0.0};
    float ef[4] = {0.f, 0.f, 0.f, 0.f};
    const bool covered = (job.chan_mask >> ci) & 1ULL;

    PsiBlk<TAB> B;
    psi_blk_setup<TAB>(job, pj, theta[(size_t)b * n_chan + ci], ci, B);
    float2 rotE = make_float2(1.f, 0.f), rotL = rotE;
    if (ROT) {
        float sr, cr;
        sincosf(rot[(size_t)b * n_chan + ci], &sr, &cr);
        rotE = make_float2(cr, sr);
        rotL = make_float2(cr, -sr);
    }
    const size_t row_stride = (size_t)n_chan * p.n_hops; // wave rows E, P, L

    for (int mh = m; mh < p.n_hops; mh += blockDim.x) {
        // An uncovered channel still writes zero replica rows (launch_correlate_nm and the
        // pack read every channel's rows) and must not touch a table not built for it.
        if (!covered) {
#pragma unroll
            for (int t = 0; t < 3; ++t)
                wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] = make_float2(0.f, 0.f);
            continue;
        }
        double CGr;
        long long Cfr;
        float2 pb;
        psi_hop<FIXED, TAB>(job, pj, B, p.fft_len, mh, CGr, Cfr, pb);
        const bool in_head = (mh < job.m_head);

        // Trial order {P, E, L} as the production kernel: rows 1, 0, 2 = C, C-ds, C+ds.
        constexpr int NT = ROT ? 1 : 3;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
            const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2;
            const float2 r = psi_trial<FIXED, TAB>(pj, B, CGr, Cfr, tt, pb);
            const size_t o = ((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh;
            const float eef = psi_e2(r);
            const double ee = (double)eef;
            if (ROT) {
                wave[o - row_stride] = cmulf(r, rotE);
                wave[o] = r;
                wave[o + row_stride] = cmulf(r, rotL);
                if (FIXED) {
                    ef[1] += eef;
                    if (in_head)
                        ef[3] += eef;
                } else {
                    e[1] += ee;
                    if (in_head)
                        e[3] += ee;
                }
            } else {
                wave[o] = r;
                if (FIXED) {
                    ef[t] += eef;
                    if (t == 1 && in_head)
                        ef[3] += eef;
                } else {
                    e[t] += ee;
                    if (t == 1 && in_head)
                        e[3] += ee;
                }
            }
        }
    }

    if (FIXED) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            e[j] = (double)ef[j];
    }
    if (ROT) {
        e[0] = e[1];
        e[2] = e[1];
    }
    __shared__ double sh_w[MAXT / 32][4];
    psi_reduce<MAXT>(e, m, sh_w, energy + (size_t)(4 * b) * n_chan + ci, (size_t)n_chan);
}

// ---------------------------------------------------------------------------------------------
// Fused synthesis + 4+4b pack (launch_psi_pack).
// ---------------------------------------------------------------------------------------------

/// gnss_pack44_kernel's byte for sample @c w at scale @c s (0 when the energy row is 0: 0x88).
/// The imaginary part is negated before quantising (the pack conjugates the replica, see
/// launch_pack44); values clamp to [-7, 7] since -8 corrupts the N^2 (n2k's negate_4bit).
__device__ __forceinline__ unsigned char pack44_byte(float2 w, float s) {
    if (!(s > 0.0f))
        return 0x88;
    int qr = __float2int_rn(s * w.x);
    int qi = __float2int_rn(-s * w.y);
    qr = qr < -7 ? -7 : (qr > 7 ? 7 : qr);
    qi = qi < -7 ? -7 : (qi > 7 ? 7 : qi);
    return (unsigned char)(((qr + 8) << 4) | ((qi + 8) & 0xf));
}

/// The slot's LPS adjacent lane bytes of one hop (b[0..LPS)): one aligned 16- or 32-bit store
/// when the launcher verified the alignment and all LPS lanes are inside [0, n_lanes), else
/// the n_valid in-range bytes one by one.
template<int LPS>
__device__ __forceinline__ void store_lanes(unsigned char* dst, const unsigned char* b, bool vec,
                                            int n_valid) {
    if (vec) {
        if (LPS == 2)
            *(unsigned short*)dst = (unsigned short)((unsigned)b[0] | ((unsigned)b[1] << 8));
        else
            *(unsigned*)dst = (unsigned)b[0] | ((unsigned)b[1] << 8) | ((unsigned)b[2] << 16)
                              | ((unsigned)b[3] << 24);
    } else {
#pragma unroll
        for (int tl = 0; tl < LPS; ++tl)
            if (tl < n_valid)
                dst[tl] = b[tl];
    }
}

/// Sweeper unit: lanes [lo, hi) of one hop row set to 0x88, one aligned 16-byte store when the
/// unit is a whole aligned chunk, byte stores otherwise.
__device__ __forceinline__ void fill88(unsigned char* row, int lo, int hi, bool align16) {
    if (align16 && hi - lo == 16 && (lo & 15) == 0) {
        *(uint4*)(row + lo) = make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
        return;
    }
    for (int i = lo; i < hi; ++i)
        row[i] = 0x88;
}

/// grid.x = PRN slot, then the sweeper blocks for the lanes beyond the last slot; grid.y =
/// covering channel; block.x = hop lane. The hop -> thread map and the energy reduction are
/// the waveform kernel's at 1024 threads, so the energy rows are bit-identical to its; a
/// 512-thread block (MAXT 512, two blocks per SM) emulates two of the 1024-thread kernel's
/// threads per thread (virtual thread m and m + 512) and reduces in the same order, so its
/// rows are bit-identical too.
///
/// Record 0 (energy_scale == nullptr): each thread synthesises its HPT hops into registers
/// (the prompt, plus the exact E/L under LPS 4 without rot), the block reduces the energy,
/// lane 0 of warp 0 broadcasts the totals through shared memory, and every thread quantises
/// what it holds with the pack's scale. Records 1..: the scale is known from energy_scale up
/// front, so each hop is quantised and stored inside the synthesis loop (the stores drain
/// under the next hop's work) and the energy rows are still reduced and written.
/// Blocks of inactive slots and of uncovered channels only write 0x88 (an uncovered channel's
/// energy rows are 0, as the waveform kernel's). The sweeper blocks write the lanes beyond
/// the last slot, one (hop, 16-lane chunk) unit per thread, spread over as many blocks as the
/// units need (one block per channel serialised 2048 x 40 B on one SM and cost 17 us).
/// @c flags bit 0: synth, lane_base and lane_pitch are multiples of LPS (vector lane stores);
/// bit 1: multiples of 16 (the sweeper's uint4 stores).
template<int MAXT, int HPT, int LPS, bool ROT, class TAB, bool FIXED>
__global__ __launch_bounds__(MAXT, MAXT == 512 ? 2 : 1) void gnss_psi_pack_kernel(
    const gnss_cuda::DespreadJob* __restrict__ jobs, const gnss_cuda::PsiJob* __restrict__ pjobs,
    const float* __restrict__ theta, gnss_cuda::DespreadParams p, const float* __restrict__ rot,
    const int* __restrict__ slot2spec, int n_slot, const int* __restrict__ chan_map,
    int frame_chan_stride, int n_lanes, int lane_base, int lane_pitch, int flags,
    const double* __restrict__ energy_scale, double* __restrict__ energy,
    unsigned char* __restrict__ synth) {
    constexpr int VT = (MAXT == 512) ? 2 : 1; // virtual 1024-thread-kernel threads per thread
    const int slot = blockIdx.x;
    const int ci = blockIdx.y;
    const int m = threadIdx.x;
    const int n_chan = gridDim.y;
    const int vec_ok = flags & 1;
    unsigned char* row0 = synth + (size_t)chan_map[ci] * lane_pitch + lane_base;
    const size_t hop_stride = (size_t)frame_chan_stride * lane_pitch;
    if (slot >= n_slot) {
        // Sweeper unit u = (hop, chunk): chunks of 16 lanes from the 16-lane boundary at or
        // below the first beyond-slot lane, clipped to [lo, n_lanes).
        const int lo = LPS * n_slot;
        const int c0 = lo & ~15;
        const int n_chunk = (n_lanes - c0 + 15) >> 4;
        const int u = (slot - n_slot) * blockDim.x + m;
        const int mh = u / n_chunk, ch = u - mh * n_chunk;
        if (lo >= n_lanes || mh >= p.n_hops)
            return;
        const int a = c0 + 16 * ch, b = a + 16;
        fill88(row0 + (size_t)mh * hop_stride, a < lo ? lo : a, b > n_lanes ? n_lanes : b,
               (flags & 2) != 0);
        return;
    }
    const int lane0 = LPS * slot;
    const int n_valid = (n_lanes - lane0 < LPS) ? n_lanes - lane0 : LPS; // lanes inside n_lanes
    const bool vec = vec_ok && n_valid == LPS;
    const int spec = slot2spec[slot];
    unsigned char* dst0 = row0 + lane0;

    const unsigned char zb[4] = {0x88, 0x88, 0x88, 0x88};
    if (spec < 0) {
        if (n_valid > 0)
            for (int mh = m; mh < p.n_hops; mh += blockDim.x)
                store_lanes<LPS>(dst0 + (size_t)mh * hop_stride, zb, vec, n_valid);
        return;
    }
    const gnss_cuda::DespreadJob job = jobs[spec];
    double* erow = energy + (size_t)(4 * spec) * n_chan + ci;
    if (!((job.chan_mask >> ci) & 1ULL)) {
        if (n_valid > 0)
            for (int mh = m; mh < p.n_hops; mh += blockDim.x)
                store_lanes<LPS>(dst0 + (size_t)mh * hop_stride, zb, vec, n_valid);
        if (m == 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                erow[(size_t)j * n_chan] = 0.0;
        }
        return;
    }
    const gnss_cuda::PsiJob pj = pjobs[spec];
    PsiBlk<TAB> B;
    psi_blk_setup<TAB>(job, pj, theta[(size_t)spec * n_chan + ci], ci, B);
    __shared__ double sh_w[32][4];
    __shared__ double sh_tot[4];
    float2 rotE = make_float2(1.f, 0.f), rotL = rotE;
    if (ROT) {
        float sr, cr;
        sincosf(rot[(size_t)spec * n_chan + ci], &sr, &cr);
        rotE = make_float2(cr, sr);
        rotL = make_float2(cr, -sr);
    }
    // Exact early/late lookups only when they are packed (LPS 4 without rot): under LPS 2
    // nobody reads them, and their energy rows are written equal to P's (see the header).
    constexpr bool EXACT = (!ROT) && (LPS == 4);
    const bool prescale = (energy_scale != nullptr);

    // The pack's scale per lane, exactly as gnss_pack44_kernel forms it: rms over n_hops, the
    // head lane over this job's m_head. From energy_scale now, or from this block's totals
    // after the reduction.
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    if (prescale) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const double E = energy_scale[(size_t)(4 * spec + t) * n_chan + ci];
            const int nh = (t == 3) ? (job.m_head > 0 ? job.m_head : 1) : p.n_hops;
            const float rms = sqrtf((float)(E / nh));
            s[t] = (rms > 0.0f) ? 7.0f / (3.0f * rms) : 0.0f;
        }
    }

    double e[VT][4];
    float ef[VT][4];
#pragma unroll
    for (int v = 0; v < VT; ++v)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            e[v][j] = 0.0;
            ef[v][j] = 0.f;
        }
    float2 vP[HPT], vE[HPT], vL[HPT];
#pragma unroll
    for (int k = 0; k < HPT; ++k) {
        vP[k] = make_float2(0.f, 0.f);
        vE[k] = vP[k];
        vL[k] = vP[k];
        const int mh = m + k * MAXT;
        // Virtual thread of the 1024-thread kernel this hop belongs to (MAXT 512: k odd ->
        // m + 512), so each accumulator sums the same hops in the same order.
        const int v = (VT == 2) ? (k & 1) : 0;
        if (mh < p.n_hops) {
            double CGr;
            long long Cfr;
            float2 pb;
            psi_hop<FIXED, TAB>(job, pj, B, p.fft_len, mh, CGr, Cfr, pb);
            const bool in_head = (mh < job.m_head);
            // The waveform kernel's accumulation order per row: P first, hops ascending.
            const float2 r = psi_trial<FIXED, TAB>(pj, B, CGr, Cfr, 0, pb);
            const float eP = psi_e2(r);
            if (FIXED) {
                ef[v][1] += eP;
                if (in_head)
                    ef[v][3] += eP;
            } else {
                e[v][1] += (double)eP;
                if (in_head)
                    e[v][3] += (double)eP;
            }
            float2 rE = make_float2(0.f, 0.f), rL = rE;
            if (EXACT) {
                rE = psi_trial<FIXED, TAB>(pj, B, CGr, Cfr, 1, pb);
                const float eE = psi_e2(rE);
                if (FIXED)
                    ef[v][0] += eE;
                else
                    e[v][0] += (double)eE;
                rL = psi_trial<FIXED, TAB>(pj, B, CGr, Cfr, 2, pb);
                const float eL = psi_e2(rL);
                if (FIXED)
                    ef[v][2] += eL;
                else
                    e[v][2] += (double)eL;
            }
            if (prescale) {
                if (n_valid > 0) {
                    unsigned char b[4];
                    const unsigned char ph = in_head ? pack44_byte(r, s[3]) : (unsigned char)0x88;
                    if (LPS == 4) {
                        b[0] = pack44_byte(ROT ? cmulf(r, rotE) : rE, s[0]);
                        b[1] = pack44_byte(r, s[1]);
                        b[2] = pack44_byte(ROT ? cmulf(r, rotL) : rL, s[2]);
                        b[3] = ph;
                    } else {
                        b[0] = pack44_byte(r, s[1]);
                        b[1] = ph;
                        b[2] = 0x88;
                        b[3] = 0x88;
                    }
                    store_lanes<LPS>(dst0 + (size_t)mh * hop_stride, b, vec, n_valid);
                }
            } else {
                vP[k] = r;
                vE[k] = rE;
                vL[k] = rL;
            }
        }
    }
    // Reduction in the 1024-thread kernel's order: per virtual thread the float sum widened
    // once, warp sums into sh_w[virtual warp], warp 0 sums the 32 virtual warps.
#pragma unroll
    for (int v = 0; v < VT; ++v) {
        if (FIXED) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                e[v][j] = (double)ef[v][j];
        }
        if (!EXACT) {
            e[v][0] = e[v][1];
            e[v][2] = e[v][1];
        }
    }
    {
        const int lane = m & 31, warp = m >> 5;
        const int n_vwarp = (blockDim.x >> 5) * VT;
#pragma unroll
        for (int v = 0; v < VT; ++v)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const double ws = warp_sum(e[v][j]);
                if (lane == 0)
                    sh_w[warp + v * (MAXT / 32)][j] = ws;
            }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const double tv = warp_sum(lane < n_vwarp ? sh_w[lane][j] : 0.0);
                if (lane == 0) {
                    erow[(size_t)j * n_chan] = tv;
                    sh_tot[j] = tv;
                }
            }
        }
    }
    if (prescale || n_valid <= 0)
        return;
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int nh = (t == 3) ? (job.m_head > 0 ? job.m_head : 1) : p.n_hops;
        const float rms = sqrtf((float)(sh_tot[t] / nh));
        s[t] = (rms > 0.0f) ? 7.0f / (3.0f * rms) : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < HPT; ++k) {
        const int mh = m + k * MAXT;
        if (mh >= p.n_hops)
            break;
        unsigned char b[4];
        const unsigned char ph = (mh < job.m_head) ? pack44_byte(vP[k], s[3]) : (unsigned char)0x88;
        if (LPS == 4) {
            b[0] = pack44_byte(ROT ? cmulf(vP[k], rotE) : vE[k], s[0]);
            b[1] = pack44_byte(vP[k], s[1]);
            b[2] = pack44_byte(ROT ? cmulf(vP[k], rotL) : vL[k], s[2]);
            b[3] = ph;
        } else {
            b[0] = pack44_byte(vP[k], s[1]);
            b[1] = ph;
            b[2] = 0x88;
            b[3] = 0x88;
        }
        store_lanes<LPS>(dst0 + (size_t)mh * hop_stride, b, vec, n_valid);
    }
}

/// launch_psi_pack's arguments, carried through the template dispatch.
struct PackArgs {
    const gnss_cuda::DespreadJob* jobs;
    const gnss_cuda::PsiJob* pjobs;
    const float* theta;
    gnss_cuda::DespreadParams p;
    const float* rot;
    const int* slot2spec;
    int n_slot;
    const int* chan_map;
    int frame_chan_stride;
    int n_lanes;
    int lane_base;
    int lane_pitch;
    int flags;
    const double* energy_scale;
    double* energy;
    unsigned char* synth;
    dim3 grid;
    int threads;
    cudaStream_t stream;
};

template<int MAXT, int HPT, int LPS, bool ROT, class TAB, bool FIXED>
cudaError_t pack_launch(const PackArgs& a) {
    gnss_psi_pack_kernel<MAXT, HPT, LPS, ROT, TAB, FIXED><<<a.grid, a.threads, 0, a.stream>>>(
        a.jobs, a.pjobs, a.theta, a.p, a.rot, a.slot2spec, a.n_slot, a.chan_map,
        a.frame_chan_stride, a.n_lanes, a.lane_base, a.lane_pitch, a.flags, a.energy_scale,
        a.energy, a.synth);
    return cudaGetLastError();
}

// 1024-thread kernels for every (lanes, rot, table, phase) combination; the 512-thread variant
// (two blocks per SM) for fixed-point phases only.
template<int LPS, bool ROT, class TAB>
cudaError_t pack_fixed(const PackArgs& a, int fixed, int hpt) {
    if (a.threads == 512)
        return pack_launch<512, 4, LPS, ROT, TAB, true>(a);
    if (hpt <= 2)
        return fixed ? pack_launch<1024, 2, LPS, ROT, TAB, true>(a)
                     : pack_launch<1024, 2, LPS, ROT, TAB, false>(a);
    return fixed ? pack_launch<1024, 4, LPS, ROT, TAB, true>(a)
                 : pack_launch<1024, 4, LPS, ROT, TAB, false>(a);
}

template<int LPS, bool ROT>
cudaError_t pack_tab(const PackArgs& a, int fixed, int hpt) {
    return a.p.phi_half ? pack_fixed<LPS, ROT, __half2>(a, fixed, hpt)
                        : pack_fixed<LPS, ROT, float2>(a, fixed, hpt);
}

template<int LPS>
cudaError_t pack_rot(const PackArgs& a, int fixed, int hpt) {
    return a.rot ? pack_tab<LPS, true>(a, fixed, hpt) : pack_tab<LPS, false>(a, fixed, hpt);
}

int block_ceiling_psi(const void* fn) {
    cudaFuncAttributes at{};
    return cudaFuncGetAttributes(&at, fn) == cudaSuccess ? at.maxThreadsPerBlock : 256;
}

template<bool ROT, class TAB, bool FIXED>
cudaError_t launch_psi_t(const gnss_cuda::DespreadJob* jobs, const gnss_cuda::PsiJob* pjobs,
                         const float* theta, dim3 grid, int threads,
                         const gnss_cuda::DespreadParams& p, const float* rot, float2* wave,
                         double* energy, cudaStream_t stream) {
    if (threads > 512) {
        static const int k10 =
            block_ceiling_psi((const void*)gnss_psi_waveform_kernel<1024, ROT, TAB, FIXED>);
        if (threads <= k10) {
            gnss_psi_waveform_kernel<1024, ROT, TAB, FIXED>
                <<<grid, threads, 0, stream>>>(jobs, pjobs, theta, p, rot, wave, energy);
            return cudaGetLastError();
        }
        threads = 512;
    }
    static const int k5 =
        block_ceiling_psi((const void*)gnss_psi_waveform_kernel<512, ROT, TAB, FIXED>);
    if (threads > k5)
        threads = k5;
    gnss_psi_waveform_kernel<512, ROT, TAB, FIXED>
        <<<grid, threads, 0, stream>>>(jobs, pjobs, theta, p, rot, wave, energy);
    return cudaGetLastError();
}

template<class TAB, bool FIXED>
cudaError_t launch_psi_rot(const gnss_cuda::DespreadJob* jobs, const gnss_cuda::PsiJob* pjobs,
                           const float* theta, dim3 grid, int threads,
                           const gnss_cuda::DespreadParams& p, const float* rot, float2* wave,
                           double* energy, cudaStream_t stream) {
    return rot ? launch_psi_t<true, TAB, FIXED>(jobs, pjobs, theta, grid, threads, p, rot, wave,
                                                energy, stream)
               : launch_psi_t<false, TAB, FIXED>(jobs, pjobs, theta, grid, threads, p, rot, wave,
                                                 energy, stream);
}


double reduce_pi(double x) {
    double t = std::fmod(x + M_PI, 2.0 * M_PI);
    if (t < 0.0)
        t += 2.0 * M_PI;
    return t - M_PI; // [-pi, pi)
}

} // namespace

namespace gnss_cuda {

cudaError_t launch_psi_waveform(const DespreadJob* jobs, const PsiJob* pjobs, const float* theta,
                                int n_job, int n_chan, const DespreadParams& p, const float* rot,
                                int fixed, float2* wave, double* energy, cudaStream_t stream) {
    int threads = p.n_hops < 1024 ? p.n_hops : 1024;
    int pow2 = 1;
    while (pow2 * 2 <= threads)
        pow2 *= 2;
    threads = pow2;
    if (threads < 32)
        return cudaErrorInvalidValue; // the warp-shuffle reduction needs whole warps
    const dim3 grid(n_job, n_chan);
    if (p.phi_half)
        return fixed ? launch_psi_rot<__half2, true>(jobs, pjobs, theta, grid, threads, p, rot,
                                                     wave, energy, stream)
                     : launch_psi_rot<__half2, false>(jobs, pjobs, theta, grid, threads, p, rot,
                                                      wave, energy, stream);
    return fixed ? launch_psi_rot<float2, true>(jobs, pjobs, theta, grid, threads, p, rot, wave,
                                                energy, stream)
                 : launch_psi_rot<float2, false>(jobs, pjobs, theta, grid, threads, p, rot, wave,
                                                 energy, stream);
}

cudaError_t launch_psi_pack_opts(const DespreadJob* jobs, const PsiJob* pjobs, const float* theta,
                                 const float* rot, int fixed, const int* slot2spec, int n_slot,
                                 int lanes_per_slot, int n_chan, const DespreadParams& p,
                                 const int* chan_map, int frame_chan_stride, int n_lanes,
                                 int lane_base, int lane_pitch, const double* energy_scale,
                                 double* energy, unsigned char* synth, int opts,
                                 cudaStream_t stream) {
    if (lanes_per_slot != 4 && lanes_per_slot != 2)
        return cudaErrorInvalidValue;
    if (n_lanes <= 0 || n_chan <= 0 || n_slot < 0 || p.n_hops <= 0)
        return cudaErrorInvalidValue;
    // The waveform kernel's block: the energy rows reduce in its order and are bit-identical.
    int threads = p.n_hops < 1024 ? p.n_hops : 1024;
    int pow2 = 1;
    while (pow2 * 2 <= threads)
        pow2 *= 2;
    threads = pow2;
    if (threads < 32)
        return cudaErrorInvalidValue; // the warp-shuffle reduction needs whole warps
    // Hops per thread, held in registers between the energy reduction and the quantisation.
    const int hpt = (p.n_hops + threads - 1) / threads;
    if (hpt > 4)
        return cudaErrorInvalidValue;
    // 512-thread blocks emulate the 1024-thread reduction only where that would be the
    // reference block (n_hops >= 1024) and four hops per thread cover the record.
    if ((opts & PSI_PACK_T512) && fixed && threads == 1024 && p.n_hops <= 2048)
        threads = 512;
    PackArgs a;
    a.jobs = jobs;
    a.pjobs = pjobs;
    a.theta = theta;
    a.p = p;
    a.rot = rot;
    a.slot2spec = slot2spec;
    a.n_slot = n_slot;
    a.chan_map = chan_map;
    a.frame_chan_stride = frame_chan_stride;
    a.n_lanes = n_lanes;
    a.lane_base = lane_base;
    a.lane_pitch = lane_pitch;
    const uintptr_t al = (uintptr_t)synth | (uintptr_t)lane_base | (uintptr_t)lane_pitch;
    a.flags = ((al % (uintptr_t)lanes_per_slot) == 0 ? 1 : 0) | ((al % 16) == 0 ? 2 : 0);
    a.energy_scale = energy_scale;
    a.energy = energy;
    a.synth = synth;
    // One block per slot, then the sweeper blocks: one unit per thread over (hop, 16-lane
    // chunk) of the lanes beyond the last slot.
    int n_sweep = 0;
    const int lo = lanes_per_slot * n_slot;
    if (lo < n_lanes) {
        const long units = (long)p.n_hops * ((n_lanes - (lo & ~15) + 15) >> 4);
        n_sweep = (int)((units + threads - 1) / threads);
    }
    a.grid = dim3(n_slot + n_sweep, n_chan);
    a.threads = threads;
    a.stream = stream;
    return lanes_per_slot == 4 ? pack_rot<4>(a, fixed, hpt) : pack_rot<2>(a, fixed, hpt);
}

cudaError_t launch_psi_pack(const DespreadJob* jobs, const PsiJob* pjobs, const float* theta,
                            const float* rot, int fixed, const int* slot2spec, int n_slot,
                            int lanes_per_slot, int n_chan, const DespreadParams& p,
                            const int* chan_map, int frame_chan_stride, int n_lanes, int lane_base,
                            int lane_pitch, const double* energy_scale, double* energy,
                            unsigned char* synth, cudaStream_t stream) {
    // The 512-thread kernel wins at every slot count under lanes 2 (28.7 vs 33.8 us at 12
    // slots, 68.6 vs 82.8 at 36, cx52) and only from ~24 slots under lanes 4 (43.0 vs 34.8 us
    // at 12 slots, 48.1 vs 61.4 at 24): fixtures/gpu_budget/psi_fused_cx52_20261021.txt.
    const int opts = (lanes_per_slot == 2) ? PSI_PACK_T512 : 0;
    return launch_psi_pack_opts(jobs, pjobs, theta, rot, fixed, slot2spec, n_slot, lanes_per_slot,
                                n_chan, p, chan_map, frame_chan_stride, n_lanes, lane_base,
                                lane_pitch, energy_scale, energy, synth, opts, stream);
}

int psi_primary_period(const int8_t* code, long L, int n_chips) {
    if (!code || L <= 0 || n_chips < 0)
        return 0;
    for (long P = (long)n_chips + 1; P <= L; ++P) {
        if (L % P != 0)
            continue;
        // The sign of each period against the first comes from the first chip that is non-
        // zero in the first period (a TDM stream has zeros); every chip must then agree.
        long i0 = 0;
        while (i0 < P && code[i0] == 0)
            ++i0;
        if (i0 == P)
            continue;
        bool ok = true;
        for (long q = 1; q < L / P && ok; ++q) {
            const int sg = (int)code[q * P + i0] * (int)code[i0];
            if (sg == 0) {
                ok = false;
                break;
            }
            for (long i = 0; i < P; ++i)
                if ((int)code[q * P + i] != sg * (int)code[i]) {
                    ok = false;
                    break;
                }
        }
        if (ok)
            return (int)P;
    }
    return 0;
}

int psi_grid_for(double inv_cps) {
    const long g = std::lround(inv_cps / 312.8);
    return g < 1 ? 1 : (int)g;
}

void psi_to_half(const float2* src, void* dst, size_t n) {
    __half2* d = (__half2*)dst;
    for (size_t i = 0; i < n; ++i)
        d[i] = __floats2half2_rn(src[i].x, src[i].y);
}

double psi_theta(int chan, int fft_len, double wc, double* theta_far) {
    const double off = 2.0 * M_PI * (double)chan / (double)fft_len;
    const double thB = reduce_pi(off - wc);
    const double thA = reduce_pi(off + wc);
    if (theta_far)
        *theta_far = std::fabs(thA);
    return thB;
}

void psi_fixed_setup(const DespreadJob& j, PsiJob& pj, int fft_len, int n_hops) {
    // Grid-unit fixed point with the largest shift that keeps the record's whole code-phase
    // range below 2^62: the rounding of the per-hop advance dcg is then <= 2^-(shift+1) grid
    // units per hop, accumulated over n_hops.
    const long double G = (long double)pj.G;
    const long double dC = (long double)fft_len * (long double)j.cps * G;
    const long double maxC =
        ((long double)j.code_len + (long double)j.ds + 2.0L) * G + (long double)n_hops * dC;
    int shift = (int)std::floor(62.0L - std::log2(maxC));
    if (shift > 60)
        shift = 60;
    if (shift < 0)
        throw std::runtime_error("psi_fixed_setup: code phase range does not fit int64");
    const long double scale = std::ldexp(1.0L, shift);
    pj.shift = shift;
    pj.cg_ref = (long long)std::llround((long double)j.cp_ref * G * scale);
    pj.dcg = (long long)std::llround(dC * scale);
    pj.dsg = (long long)std::llround((long double)j.ds * G * scale);
    const long double GLf = (long double)j.code_len * G * scale;
    pj.dcg_r = (float)((long double)pj.dcg / GLf);
    pj.cg_r = (float)((long double)pj.cg_ref / GLf);
    constexpr long double TWO_PI_L = 6.283185307179586476925286766559005768L;
    const long double two64 = 18446744073709551616.0L;
    long double fr0 = std::fmod((long double)j.ang0 / TWO_PI_L, 1.0L);
    if (fr0 < 0.0L)
        fr0 += 1.0L;
    long double frd = std::fmod((long double)j.wc * (long double)fft_len / TWO_PI_L, 1.0L);
    if (frd < 0.0L)
        frd += 1.0L;
    long double a0 = fr0 * two64, ad = frd * two64;
    if (a0 >= two64)
        a0 = two64 - 1.0L;
    if (ad >= two64)
        ad = two64 - 1.0L;
    pj.ang0_t = (unsigned long long)a0;
    pj.dturn_t = (unsigned long long)ad;
}

} // namespace gnss_cuda
