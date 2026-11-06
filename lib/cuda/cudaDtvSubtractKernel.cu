#include "cudaDtvSubtractKernel.hpp"

#include <cuda_pipeline.h>
#include <mma.h>

namespace gnss_cuda {

namespace {

constexpr int NI = DTV_INPUTS;
constexpr int HIST = DTV_HIST;
constexpr int ROW_THREADS = 256; // 8 warps: one warp per sample, lane = 4 inputs
constexpr int ROW_SAMPLES = 64;  // samples per block in the row kernels
constexpr int TT = 256;          // samples per tile in the R_ss kernel
constexpr int COV_THREADS = 256;
constexpr int COV_TT = 32;
constexpr int COV_PAIRS = NI * NI / COV_THREADS;

// The fused pass: 8 warps; tiles of F_TT samples of x (both planes, rows padded to F_LDX floats)
// double-buffered with asynchronous copies, the timestream segment the tile's lags reach, and the
// lagged timestream as three TF32 planes (re, im, -im; rows padded to F_LDS).
constexpr int F_THREADS = 256;
constexpr int F_TT = 32;
constexpr int F_LDX = NI + 4;
constexpr int F_LDS = DTV_MAX_LAGS + 4;
constexpr int F_SEG = F_TT + DTV_MAX_LAGS;
constexpr size_t F_SMEM =
    (size_t)(2 * 2 * F_TT * F_LDX + 3 * F_TT * F_LDS + 2 * 2 * F_SEG) * sizeof(float);
static_assert(F_SMEM <= 99 * 1024, "the fused pass's shared memory exceeds sm_86's per block");
static_assert(DTV_MAX_LAGS == 64, "the fused pass's warps cover four 16-lag blocks");

__device__ __forceinline__ float2 decode44(unsigned char b) {
    return make_float2((float)((b >> 4) & 0x0f) - 8.0f, (float)(b & 0x0f) - 8.0f);
}

__device__ __forceinline__ unsigned char encode44(float re_f, float im_f) {
    int re = __float2int_rn(re_f), im = __float2int_rn(im_f);
    re = re < -7 ? -7 : (re > 7 ? 7 : re);
    im = im < -7 ? -7 : (im > 7 ? 7 : im);
    return (unsigned char)(((re + 8) << 4) | (im + 8));
}

__device__ __forceinline__ bool any_active(const int* act, int c, int n_src, int level) {
    for (int k = 0; k < n_src; k++)
        if (act[c * n_src + k] >= level)
            return true;
    return false;
}

// Per-input sums over the block's samples: each lane holds 4 inputs; the block's 8 warps add
// their lanes' partial sums in shared memory, then one atomic per input per block.
__device__ __forceinline__ void add_input_sums(const float* v4, float* sh, float* out) {
    const int lane = threadIdx.x & 31;
    for (int k = 0; k < 4; k++)
        atomicAdd(&sh[4 * lane + k], v4[k]);
    __syncthreads();
    for (int i = threadIdx.x; i < NI; i += blockDim.x)
        atomicAdd(&out[i], sh[i]);
}

// One sample of a beam: the lane holds inputs 4 lane .. 4 lane + 3; the warp's sum.
__device__ __forceinline__ float2 beam_sample(const float2* ws, const float* re, const float* im) {
    const int lane = threadIdx.x & 31;
    float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 wv = ws[4 * lane + j];
        acc.x += wv.x * re[j] - wv.y * im[j];
        acc.y += wv.x * im[j] + wv.y * re[j];
    }
    for (int o = 16; o > 0; o >>= 1) {
        acc.x += __shfl_xor_sync(0xffffffffu, acc.x, o);
        acc.y += __shfl_xor_sync(0xffffffffu, acc.y, o);
    }
    return acc;
}

__global__ void dtv_decode_kernel(const unsigned char* __restrict__ in, const int* __restrict__ sel,
                                  const float2* __restrict__ w, const int* __restrict__ act,
                                  int n_src, int n_freq, int n_times, float* __restrict__ x,
                                  float* __restrict__ pw, float2* __restrict__ s,
                                  float* __restrict__ ps) {
    __shared__ float sh[NI];
    __shared__ float2 ws[NI];
    __shared__ float psh;
    const int c = blockIdx.y;
    const int f = sel[c];
    const bool beam = act[c * n_src] >= 1;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < NI; i += blockDim.x) {
        sh[i] = 0.0f;
        if (beam)
            ws[i] = w[(size_t)c * n_src * NI + i];
    }
    if (threadIdx.x == 0)
        psh = 0.0f;
    __syncthreads();
    float* xr = x + (size_t)c * 2 * n_times * NI;
    float* xi = xr + (size_t)n_times * NI;
    float2* s0 = s + (size_t)c * n_src * (HIST + n_times) + HIST;
    float p4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float pacc = 0.0f;
    for (int it = 0; it < ROW_SAMPLES / (ROW_THREADS / 32); it++) {
        const int t = blockIdx.x * ROW_SAMPLES + it * (ROW_THREADS / 32) + warp;
        if (t >= n_times)
            break;
        const uchar4 b =
            *reinterpret_cast<const uchar4*>(in + ((size_t)t * n_freq + f) * NI + 4 * lane);
        const unsigned char bb[4] = {b.x, b.y, b.z, b.w};
        float re[4], im[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float2 v = decode44(bb[k]);
            re[k] = v.x;
            im[k] = v.y;
            p4[k] += v.x * v.x + v.y * v.y;
        }
        *reinterpret_cast<float4*>(xr + (size_t)t * NI + 4 * lane) =
            make_float4(re[0], re[1], re[2], re[3]);
        *reinterpret_cast<float4*>(xi + (size_t)t * NI + 4 * lane) =
            make_float4(im[0], im[1], im[2], im[3]);
        if (beam) {
            const float2 acc = beam_sample(ws, re, im);
            if (lane == 0) {
                s0[t] = acc;
                pacc += acc.x * acc.x + acc.y * acc.y;
            }
        }
    }
    add_input_sums(p4, sh, pw + c * NI);
    if (beam) {
        if (lane == 0)
            atomicAdd(&psh, pacc);
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(&ps[c * n_src], psh);
    }
}

__global__ void dtv_shift_kernel(float2* __restrict__ s, int n_times) {
    float2* sk = s + (size_t)blockIdx.x * (HIST + n_times);
    const float2 v = sk[n_times + threadIdx.x];
    __syncthreads();
    sk[threadIdx.x] = v;
}

__global__ void dtv_beam_kernel(const float* __restrict__ x, const float2* __restrict__ w,
                                const int* __restrict__ act, int k, int n_src, int n_times,
                                float2* __restrict__ s, float* __restrict__ ps) {
    __shared__ float2 ws[NI];
    __shared__ float psh;
    const int c = blockIdx.y;
    if (act[c * n_src + k] < 1)
        return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < NI; i += blockDim.x)
        ws[i] = w[(c * n_src + k) * NI + i];
    if (threadIdx.x == 0)
        psh = 0.0f;
    __syncthreads();
    const float* xr = x + (size_t)c * 2 * n_times * NI;
    const float* xi = xr + (size_t)n_times * NI;
    float2* sk = s + (size_t)(c * n_src + k) * (HIST + n_times) + HIST;
    float pacc = 0.0f;
    for (int it = 0; it < ROW_SAMPLES / (ROW_THREADS / 32); it++) {
        const int t = blockIdx.x * ROW_SAMPLES + it * (ROW_THREADS / 32) + warp;
        if (t >= n_times)
            break;
        const float4 r4 = *reinterpret_cast<const float4*>(xr + (size_t)t * NI + 4 * lane);
        const float4 i4 = *reinterpret_cast<const float4*>(xi + (size_t)t * NI + 4 * lane);
        const float re[4] = {r4.x, r4.y, r4.z, r4.w}, im[4] = {i4.x, i4.y, i4.z, i4.w};
        const float2 acc = beam_sample(ws, re, im);
        if (lane == 0) {
            sk[t] = acc;
            pacc += acc.x * acc.x + acc.y * acc.y;
        }
    }
    if (lane == 0)
        atomicAdd(&psh, pacc);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(&ps[c * n_src + k], psh);
}

// rss[d] = sum_t s(t) conj(s(t - d)) for d = 0..n_lag_c-1: every lag difference the normal
// equations need (negative ones by conjugate symmetry). d < DTV_HIST, so s(t - d) is always held.
// One block per (channel, tile of TT samples); the tiles' partial sums are added with atomics
// into memory the caller zeroed.
__global__ void dtv_rss_kernel(const float2* __restrict__ s, const int* __restrict__ act,
                               const int* __restrict__ lags, int k, int n_src, int n_times,
                               int n_lag, float2* __restrict__ rss) {
    const int c = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= lags[2 * c + 1] || act[c * n_src + k] < 1)
        return;
    const float2* sk = s + (size_t)(c * n_src + k) * (HIST + n_times) + HIST;
    const int t0 = blockIdx.y * TT;
    const int t1 = t0 + TT < n_times ? t0 + TT : n_times;
    float2 acc = make_float2(0.0f, 0.0f);
    for (int t = t0; t < t1; t++) {
        const float2 a = sk[t], b = sk[t - d];
        acc.x += a.x * b.x + a.y * b.y;
        acc.y += a.y * b.x - a.x * b.y;
    }
    float2* dst = rss + (size_t)(c * n_src + k) * n_lag + d;
    atomicAdd(&dst->x, acc.x);
    atomicAdd(&dst->y, acc.y);
}

namespace wm = nvcuda::wmma;
using FragArow = wm::fragment<wm::matrix_a, 16, 16, 8, wm::precision::tf32, wm::row_major>;
using FragAcol = wm::fragment<wm::matrix_a, 16, 16, 8, wm::precision::tf32, wm::col_major>;
using FragB = wm::fragment<wm::matrix_b, 16, 16, 8, wm::precision::tf32, wm::row_major>;
using FragC = wm::fragment<wm::accumulator, 16, 16, 8, float>;

template<typename F>
__device__ __forceinline__ void to_tf32(F& f) {
    for (int e = 0; e < f.num_elements; e++)
        f.x[e] = wm::__float_to_tf32(f.x[e]);
}

template<typename F>
__device__ __forceinline__ void negated(F& o, const F& i) {
    for (int e = 0; e < i.num_elements; e++)
        o.x[e] = -i.x[e];
}

// Start the copies of tile t0: its x rows into Br/Bi (zero past the frame's end), and the
// timestream samples its lags reach, seg[m] = s[HIST + t0 - lag_min - (l16 - 1) + m] (zero
// outside s: only lags beyond the channel's window reach there).
__device__ __forceinline__ void fused_issue(const float* xr, const float* xi, const float2* sk,
                                            int t0, int n_times, int lag_min, int l16, float* Br,
                                            float* Bi, float2* seg) {
    for (int q = threadIdx.x; q < F_TT * 2 * (NI / 4); q += blockDim.x) {
        const int t = q / (2 * (NI / 4)), r = q % (2 * (NI / 4));
        const int part = r / (NI / 4), ch = r % (NI / 4);
        float* dst = (part ? Bi : Br) + t * F_LDX + 4 * ch;
        if (t0 + t < n_times)
            __pipeline_memcpy_async(dst, (part ? xi : xr) + (size_t)(t0 + t) * NI + 4 * ch, 16);
        else
            *reinterpret_cast<float4*>(dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const int base = HIST + t0 - lag_min - (l16 - 1);
    for (int m = threadIdx.x; m < F_TT + l16 - 1; m += blockDim.x) {
        const int idx = base + m;
        if (idx >= 0 && idx < HIST + n_times)
            __pipeline_memcpy_async(&seg[m], &sk[idx], 8);
        else
            seg[m] = make_float2(0.0f, 0.0f);
    }
}

// The fused pass of source k over tiles_per_block tiles of one channel (blockIdx.y). Per tile:
// the lagged timestream S (F_TT x l16) into shared memory; R += S^H X, warp w holding lag block
// w % 4 and input blocks 4 (w / 4) .. +3 across the tiles; X -= S G, warp w on input block w with
// its g fragments in registers for the whole block; the next source's beam from the residual
// tile; the tile back to x. NK = the g stride / 8; the channel's own window takes nk <= NK.
template<int NK>
__global__ void __launch_bounds__(F_THREADS, 1)
    dtv_fused_kernel(float* __restrict__ x, float2* __restrict__ s, const float* __restrict__ gs,
                     const float2* __restrict__ w, const int* __restrict__ act,
                     const int* __restrict__ lags, int k, int n_src, int n_times, int n_lag,
                     int tiles_per_block, float2* __restrict__ rxs, float* __restrict__ ps) {
    constexpr int L16 = 8 * NK;
    extern __shared__ __align__(32) float sm[];
    float* B[2][2] = {{sm, sm + F_TT * F_LDX}, {sm + 2 * F_TT * F_LDX, sm + 3 * F_TT * F_LDX}};
    float* Sr = sm + 4 * F_TT * F_LDX;
    float* Si = Sr + F_TT * F_LDS;
    float* NSi = Si + F_TT * F_LDS;
    float2* seg[2] = {reinterpret_cast<float2*>(NSi + F_TT * F_LDS),
                      reinterpret_cast<float2*>(NSi + F_TT * F_LDS) + F_SEG};
    const int c = blockIdx.y;
    const int a = act[c * n_src + k];
    const bool corr = a >= 1, sub = a >= 2;
    const bool nxt = k + 1 < n_src && act[c * n_src + k + 1] >= 1;
    const int t_begin = blockIdx.x * tiles_per_block * F_TT;
    const int n_tiles = min(tiles_per_block, (n_times - t_begin + F_TT - 1) / F_TT);
    if ((!corr && !nxt) || n_tiles <= 0)
        return;
    const int lag_min = lags[2 * c], n_lag_c = lags[2 * c + 1];
    const int l16 = (n_lag_c + 15) & ~15, nk = l16 / 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float2* sk = s + (size_t)(c * n_src + k) * (HIST + n_times);
    float2* sn = s + (size_t)(c * n_src + k + 1) * (HIST + n_times);
    float* xr = x + (size_t)c * 2 * n_times * NI;
    float* xi = xr + (size_t)n_times * NI;
    float2 wv[4];
    if (nxt)
        for (int m = 0; m < 4; m++)
            wv[m] = w[(size_t)(c * n_src + k + 1) * NI + lane + 32 * m];
    FragB gr[NK], gi[NK];
    if (sub) {
        const float* Gr = gs + (size_t)(c * n_src + k) * 2 * L16 * NI;
        const float* Gi = Gr + (size_t)L16 * NI;
#pragma unroll
        for (int kk = 0; kk < NK; kk++)
            if (kk < nk) {
                wm::load_matrix_sync(gr[kk], Gr + kk * 8 * NI + warp * 16, NI);
                wm::load_matrix_sync(gi[kk], Gi + kk * 8 * NI + warp * 16, NI);
                to_tf32(gr[kk]);
                to_tf32(gi[kk]);
            }
    }
    const int jb = warp & 3, ib0 = (warp >> 2) * 4;
    const bool rwarp = corr && jb * 16 < l16;
    FragC Rre[4], Rim[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
        wm::fill_fragment(Rre[q], 0.0f);
        wm::fill_fragment(Rim[q], 0.0f);
    }
    float psum = 0.0f;
    fused_issue(xr, xi, sk, t_begin, n_times, lag_min, l16, B[0][0], B[0][1], seg[0]);
    __pipeline_commit();
    for (int tile = 0; tile < n_tiles; tile++) {
        const int t0 = t_begin + tile * F_TT, cur = tile & 1;
        if (tile + 1 < n_tiles)
            fused_issue(xr, xi, sk, t0 + F_TT, n_times, lag_min, l16, B[cur ^ 1][0], B[cur ^ 1][1],
                        seg[cur ^ 1]);
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        float* Xr = B[cur][0];
        float* Xi = B[cur][1];
        const float2* sg = seg[cur];
        if (corr)
            for (int q = threadIdx.x; q < F_TT * l16; q += blockDim.x) {
                const int t = q / l16, j = q % l16;
                const float2 v = j < n_lag_c ? sg[t + (l16 - 1) - j] : make_float2(0.0f, 0.0f);
                const float re = wm::__float_to_tf32(v.x), im = wm::__float_to_tf32(v.y);
                Sr[t * F_LDS + j] = re;
                Si[t * F_LDS + j] = im;
                NSi[t * F_LDS + j] = -im;
            }
        __syncthreads();
        if (rwarp) // R[j][i] += sum_t conj(S[t][j]) X[t][i]: re = Sr Xr + Si Xi, im = Sr Xi - Si Xr
#pragma unroll
            for (int kk = 0; kk < F_TT / 8; kk++) {
                FragAcol ar, ai, nai;
                wm::load_matrix_sync(ar, Sr + kk * 8 * F_LDS + jb * 16, F_LDS);
                wm::load_matrix_sync(ai, Si + kk * 8 * F_LDS + jb * 16, F_LDS);
                wm::load_matrix_sync(nai, NSi + kk * 8 * F_LDS + jb * 16, F_LDS);
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    FragB br, bi;
                    wm::load_matrix_sync(br, Xr + kk * 8 * F_LDX + (ib0 + q) * 16, F_LDX);
                    wm::load_matrix_sync(bi, Xi + kk * 8 * F_LDX + (ib0 + q) * 16, F_LDX);
                    to_tf32(br);
                    to_tf32(bi);
                    wm::mma_sync(Rre[q], ar, br, Rre[q]);
                    wm::mma_sync(Rre[q], ai, bi, Rre[q]);
                    wm::mma_sync(Rim[q], ar, bi, Rim[q]);
                    wm::mma_sync(Rim[q], nai, br, Rim[q]);
                }
            }
        __syncthreads();
        if (sub) { // X -= S G: re += -Sr Gr + Si Gi, im += -Sr Gi - Si Gr
            FragC Ore[2], Oim[2];
#pragma unroll
            for (int q = 0; q < 2; q++) {
                wm::load_matrix_sync(Ore[q], Xr + q * 16 * F_LDX + warp * 16, F_LDX,
                                     wm::mem_row_major);
                wm::load_matrix_sync(Oim[q], Xi + q * 16 * F_LDX + warp * 16, F_LDX,
                                     wm::mem_row_major);
            }
#pragma unroll
            for (int kk = 0; kk < NK; kk++)
                if (kk < nk)
#pragma unroll
                    for (int q = 0; q < 2; q++) {
                        FragArow ar, ai, nai, nar;
                        wm::load_matrix_sync(ar, Sr + q * 16 * F_LDS + kk * 8, F_LDS);
                        wm::load_matrix_sync(ai, Si + q * 16 * F_LDS + kk * 8, F_LDS);
                        wm::load_matrix_sync(nai, NSi + q * 16 * F_LDS + kk * 8, F_LDS);
                        negated(nar, ar);
                        wm::mma_sync(Ore[q], nar, gr[kk], Ore[q]);
                        wm::mma_sync(Ore[q], ai, gi[kk], Ore[q]);
                        wm::mma_sync(Oim[q], nar, gi[kk], Oim[q]);
                        wm::mma_sync(Oim[q], nai, gr[kk], Oim[q]);
                    }
#pragma unroll
            for (int q = 0; q < 2; q++) {
                wm::store_matrix_sync(Xr + q * 16 * F_LDX + warp * 16, Ore[q], F_LDX,
                                      wm::mem_row_major);
                wm::store_matrix_sync(Xi + q * 16 * F_LDX + warp * 16, Oim[q], F_LDX,
                                      wm::mem_row_major);
            }
        }
        __syncthreads();
        const int t_end = min(F_TT, n_times - t0);
        if (nxt) // the next source's beam: a warp per sample, lanes on inputs lane + 32 m
            for (int t = warp; t < t_end; t += F_THREADS / 32) {
                float re = 0.0f, im = 0.0f;
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int i = lane + 32 * m;
                    const float xa = Xr[t * F_LDX + i], xb = Xi[t * F_LDX + i];
                    re += wv[m].x * xa - wv[m].y * xb;
                    im += wv[m].x * xb + wv[m].y * xa;
                }
                for (int o = 16; o > 0; o >>= 1) {
                    re += __shfl_xor_sync(0xffffffffu, re, o);
                    im += __shfl_xor_sync(0xffffffffu, im, o);
                }
                if (lane == 0) {
                    sn[HIST + t0 + t] = make_float2(re, im);
                    psum += re * re + im * im;
                }
            }
        if (sub)
            for (int q = threadIdx.x; q < t_end * 2 * (NI / 4); q += blockDim.x) {
                const int t = q / (2 * (NI / 4)), r = q % (2 * (NI / 4));
                const int part = r / (NI / 4), ch = r % (NI / 4);
                *reinterpret_cast<float4*>((part ? xi : xr) + (size_t)(t0 + t) * NI + 4 * ch) =
                    *reinterpret_cast<const float4*>((part ? Xi : Xr) + t * F_LDX + 4 * ch);
            }
        __syncthreads();
    }
    if (rwarp) { // the accumulators into rxs through shared memory (the tiles are done)
        float* scr = sm + warp * 512;
        for (int q = 0; q < 4; q++) {
            wm::store_matrix_sync(scr, Rre[q], 16, wm::mem_row_major);
            wm::store_matrix_sync(scr + 256, Rim[q], 16, wm::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int j = jb * 16 + e / 16, i = (ib0 + q) * 16 + e % 16;
                if (j < n_lag_c) {
                    float2* d = rxs + ((size_t)(c * n_src + k) * n_lag + j) * NI + i;
                    atomicAdd(&d->x, scr[e]);
                    atomicAdd(&d->y, scr[256 + e]);
                }
            }
            __syncwarp();
        }
    }
    if (nxt && lane == 0)
        atomicAdd(&ps[c * n_src + k + 1], psum);
}

// One block per (channel, source); thread i rotates input i's weight and its g at every lag.
__global__ void dtv_steer_kernel(const float2* __restrict__ w_anchor,
                                 const float* __restrict__ gs_anchor, const float* __restrict__ psi,
                                 int l16, float2* __restrict__ w, float* __restrict__ gs) {
    const size_t ck = blockIdx.x;
    const int i = threadIdx.x;
    float sn, cs;
    sincosf(psi[ck * NI + i], &sn, &cs);
    const float2 a = w_anchor[ck * NI + i];
    w[ck * NI + i] = make_float2(a.x * cs + a.y * sn, a.y * cs - a.x * sn); // a e^{-i psi}
    const float* gr = gs_anchor + ck * 2 * l16 * NI;
    const float* gi = gr + (size_t)l16 * NI;
    float* hr = gs + ck * 2 * l16 * NI;
    float* hi = hr + (size_t)l16 * NI;
    for (int j = 0; j < l16; j++) {
        const float re = gr[j * NI + i], im = gi[j * NI + i];
        hr[j * NI + i] = re * cs - im * sn; // g e^{+i psi}
        hi[j * NI + i] = im * cs + re * sn;
    }
}

__global__ void dtv_encode_kernel(const float* __restrict__ x, const int* __restrict__ sel,
                                  const int* __restrict__ act, int n_src, int n_freq, int n_times,
                                  unsigned char* __restrict__ out, float* __restrict__ pw) {
    __shared__ float sh[NI];
    const int c = blockIdx.y;
    if (!any_active(act, c, n_src, 2))
        return;
    const int f = sel[c];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < NI; i += blockDim.x)
        sh[i] = 0.0f;
    __syncthreads();
    const float* xr = x + (size_t)c * 2 * n_times * NI;
    const float* xi = xr + (size_t)n_times * NI;
    float p4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int it = 0; it < ROW_SAMPLES / (ROW_THREADS / 32); it++) {
        const int t = blockIdx.x * ROW_SAMPLES + it * (ROW_THREADS / 32) + warp;
        if (t >= n_times)
            break;
        const float4 r4 = *reinterpret_cast<const float4*>(xr + (size_t)t * NI + 4 * lane);
        const float4 i4 = *reinterpret_cast<const float4*>(xi + (size_t)t * NI + 4 * lane);
        const float re[4] = {r4.x, r4.y, r4.z, r4.w}, im[4] = {i4.x, i4.y, i4.z, i4.w};
        unsigned char bb[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            p4[k] += re[k] * re[k] + im[k] * im[k];
            bb[k] = encode44(re[k], im[k]);
        }
        *reinterpret_cast<uchar4*>(out + ((size_t)t * n_freq + f) * NI + 4 * lane) =
            make_uchar4(bb[0], bb[1], bb[2], bb[3]);
    }
    add_input_sums(p4, sh, pw + c * NI);
}

__global__ void dtv_roll_accum_kernel(const float2* __restrict__ rxs,
                                      const float2* __restrict__ rss,
                                      const float* __restrict__ pw_raw,
                                      const float* __restrict__ pw_res, int n_chan, int n_src,
                                      int n_lag, DtvDecay lam, float2* __restrict__ acc_rxs,
                                      float2* __restrict__ acc_rss, float* __restrict__ acc_pw) {
    const size_t n_rxs = (size_t)n_chan * n_src * n_lag * NI,
                 n_rss = (size_t)n_chan * n_src * n_lag;
    const size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n_rxs) {
        const float l = lam.lam[(q / ((size_t)n_lag * NI)) % n_src];
        const float2 a = acc_rxs[q], r = rxs[q];
        acc_rxs[q] = make_float2(l * a.x + r.x, l * a.y + r.y);
    } else if (q < n_rxs + n_rss) {
        const size_t p = q - n_rxs;
        const float l = lam.lam[(p / n_lag) % n_src];
        const float2 a = acc_rss[p], r = rss[p];
        acc_rss[p] = make_float2(l * a.x + r.x, l * a.y + r.y);
    } else if (q < n_rxs + n_rss + n_chan) {
        const int c = (int)(q - n_rxs - n_rss);
        float raw = 0.0f, res = 0.0f;
        for (int i = 0; i < NI; i++) {
            raw += pw_raw[c * NI + i];
            res += pw_res[c * NI + i];
        }
        acc_pw[c] = lam.lam[0] * acc_pw[c] + raw;
        acc_pw[n_chan + c] += raw;
        acc_pw[2 * n_chan + c] += res;
    }
}

__global__ void dtv_cov_kernel(const float* __restrict__ x, int n_times, float2* __restrict__ cov) {
    __shared__ float2 tile[COV_TT * NI];
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int t_per = n_times / gridDim.y;
    const int t_begin = blockIdx.y * t_per, t_end = t_begin + t_per;
    const float* xr = x + (size_t)c * 2 * n_times * NI;
    const float* xi = xr + (size_t)n_times * NI;
    float2 acc[COV_PAIRS];
    for (int k = 0; k < COV_PAIRS; k++)
        acc[k] = make_float2(0.0f, 0.0f);
    for (int t0 = t_begin; t0 < t_end; t0 += COV_TT) {
        __syncthreads();
        for (int m = tid; m < COV_TT * NI; m += COV_THREADS)
            tile[m] = make_float2(xr[(size_t)t0 * NI + m], xi[(size_t)t0 * NI + m]);
        __syncthreads();
        for (int k = 0; k < COV_PAIRS; k++) {
            const int p = tid + k * COV_THREADS;
            const int i = p / NI, j = p % NI;
            float re = 0.0f, im = 0.0f;
            for (int tt = 0; tt < COV_TT; tt++) {
                const float2 a = tile[tt * NI + i];
                const float2 b = tile[tt * NI + j];
                re += a.x * b.x + a.y * b.y;
                im += a.y * b.x - a.x * b.y;
            }
            acc[k].x += re;
            acc[k].y += im;
        }
    }
    for (int k = 0; k < COV_PAIRS; k++) {
        float2* dst = cov + (size_t)c * NI * NI + tid + k * COV_THREADS;
        atomicAdd(&dst->x, acc[k].x);
        atomicAdd(&dst->y, acc[k].y);
    }
}

__global__ void dtv_pick_ref_kernel(const float2* __restrict__ cov, int* __restrict__ ref) {
    __shared__ float pw[NI];
    __shared__ int idx[NI];
    const int c = blockIdx.x, i = threadIdx.x;
    pw[i] = cov[(size_t)c * NI * NI + i * NI + i].x;
    idx[i] = i;
    __syncthreads();
    for (int o = NI / 2; o > 0; o >>= 1) {
        if (i < o && pw[i + o] > pw[i]) {
            pw[i] = pw[i + o];
            idx[i] = idx[i + o];
        }
        __syncthreads();
    }
    if (i == 0)
        ref[c] = idx[0];
}

template<int NK>
void fused_launch(dim3 grid, cudaStream_t stream, float* x, float2* s, const float* gs,
                  const float2* w, const int* act, const int* lags, int k, int n_src, int n_times,
                  int n_lag, int tiles_per_block, float2* rxs, float* ps) {
    cudaFuncSetAttribute(dtv_fused_kernel<NK>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)F_SMEM);
    dtv_fused_kernel<NK><<<grid, F_THREADS, F_SMEM, stream>>>(
        x, s, gs, w, act, lags, k, n_src, n_times, n_lag, tiles_per_block, rxs, ps);
}

} // namespace

void launch_dtv_shift(float2* s, int n_chan, int n_src, int n_times, cudaStream_t stream) {
    if (n_chan * n_src <= 0)
        return;
    dtv_shift_kernel<<<n_chan * n_src, HIST, 0, stream>>>(s, n_times);
}

void launch_dtv_decode(const unsigned char* in, const int* sel, const float2* w, const int* act,
                       int n_chan, int n_src, int n_freq, int n_times, float* x, float* pw_raw,
                       float2* s, float* ps, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    cudaMemsetAsync(pw_raw, 0, sizeof(float) * NI * n_chan, stream);
    const dim3 grid((n_times + ROW_SAMPLES - 1) / ROW_SAMPLES, n_chan);
    dtv_decode_kernel<<<grid, ROW_THREADS, 0, stream>>>(in, sel, w, act, n_src, n_freq, n_times, x,
                                                        pw_raw, s, ps);
}

void launch_dtv_beam(const float* x, const float2* w, const int* act, int k, int n_chan, int n_src,
                     int n_times, float2* s, float* ps, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    const dim3 grid((n_times + ROW_SAMPLES - 1) / ROW_SAMPLES, n_chan);
    dtv_beam_kernel<<<grid, ROW_THREADS, 0, stream>>>(x, w, act, k, n_src, n_times, s, ps);
}

void launch_dtv_rss(const float2* s, const int* act, const int* lags, int k, int n_chan, int n_src,
                    int n_times, int n_lag, float2* rss, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    const int n_tiles = (n_times + TT - 1) / TT;
    dtv_rss_kernel<<<dim3(n_chan, n_tiles), DTV_MAX_LAGS, 0, stream>>>(s, act, lags, k, n_src,
                                                                       n_times, n_lag, rss);
}

void launch_dtv_fused(float* x, float2* s, const float* gs, const float2* w, const int* act,
                      const int* lags, int k, int n_chan, int n_src, int n_times, int n_lag,
                      float2* rxs, float* ps, cudaStream_t stream) {
    if (n_chan <= 0 || n_times <= 0)
        return;
    // about two blocks per SM in all (one is resident per SM), shared by the channels
    int dev = 0, n_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
    const int tiles = (n_times + F_TT - 1) / F_TT;
    int per_chan = (2 * n_sm + n_chan - 1) / n_chan;
    per_chan = per_chan < 1 ? 1 : (per_chan > tiles ? tiles : per_chan);
    const int tpb = (tiles + per_chan - 1) / per_chan;
    const dim3 grid((tiles + tpb - 1) / tpb, n_chan);
    switch (dtv_l16(n_lag) / 8) {
        case 2:
            fused_launch<2>(grid, stream, x, s, gs, w, act, lags, k, n_src, n_times, n_lag, tpb,
                            rxs, ps);
            break;
        case 4:
            fused_launch<4>(grid, stream, x, s, gs, w, act, lags, k, n_src, n_times, n_lag, tpb,
                            rxs, ps);
            break;
        case 6:
            fused_launch<6>(grid, stream, x, s, gs, w, act, lags, k, n_src, n_times, n_lag, tpb,
                            rxs, ps);
            break;
        default:
            fused_launch<8>(grid, stream, x, s, gs, w, act, lags, k, n_src, n_times, n_lag, tpb,
                            rxs, ps);
            break;
    }
}

void launch_dtv_steer(const float2* w_anchor, const float* gs_anchor, const float* psi, int n_chan,
                      int n_src, int n_lag, float2* w, float* gs, cudaStream_t stream) {
    if (n_chan * n_src <= 0)
        return;
    dtv_steer_kernel<<<n_chan * n_src, NI, 0, stream>>>(w_anchor, gs_anchor, psi, dtv_l16(n_lag), w,
                                                        gs);
}

void launch_dtv_encode(const float* x, const int* sel, const int* act, int n_chan, int n_src,
                       int n_freq, int n_times, unsigned char* out, float* pw_res,
                       cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    cudaMemsetAsync(pw_res, 0, sizeof(float) * NI * n_chan, stream);
    const dim3 grid((n_times + ROW_SAMPLES - 1) / ROW_SAMPLES, n_chan);
    dtv_encode_kernel<<<grid, ROW_THREADS, 0, stream>>>(x, sel, act, n_src, n_freq, n_times, out,
                                                        pw_res);
}

void launch_dtv_roll_accum(const float2* rxs, const float2* rss, const float* pw_raw,
                           const float* pw_res, int n_chan, int n_src, int n_lag, DtvDecay lam,
                           float2* acc_rxs, float2* acc_rss, float* acc_pw, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    const size_t total = (size_t)n_chan * n_src * n_lag * (NI + 1) + n_chan;
    const int threads = 256;
    dtv_roll_accum_kernel<<<(unsigned)((total + threads - 1) / threads), threads, 0, stream>>>(
        rxs, rss, pw_raw, pw_res, n_chan, n_src, n_lag, lam, acc_rxs, acc_rss, acc_pw);
}

void launch_dtv_cov(const float* x, int n_chan, int n_times, float2* cov, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    const int n_split = n_times / (8 * COV_TT) > 0 ? n_times / (8 * COV_TT) : 1;
    dtv_cov_kernel<<<dim3(n_chan, n_split), COV_THREADS, 0, stream>>>(x, n_times, cov);
}

void launch_dtv_pick_ref(const float2* cov, int n_chan, int* ref, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    dtv_pick_ref_kernel<<<n_chan, NI, 0, stream>>>(cov, ref);
}

} // namespace gnss_cuda
