#include "cudaDtvSubtractKernel.hpp"

namespace gnss_cuda {

namespace {

constexpr int NI = DTV_INPUTS;
constexpr int HIST = DTV_HIST;
constexpr int ROW_THREADS = 256;     // 8 warps: one warp per sample, lane = 4 inputs
constexpr int ROW_SAMPLES = 64;      // samples per block in the row kernels
constexpr int TT = 256;              // samples per tile in the lag kernels
constexpr int TILE = TT + HIST + 16; // s entries a tile needs: lags -16 .. HIST-1
constexpr int LAG_GROUP = 16;        // lags per block in the correlation kernel
constexpr int COV_THREADS = 256;
constexpr int COV_TT = 32;
constexpr int COV_PAIRS = NI * NI / COV_THREADS;

__device__ __forceinline__ float2 decode44(unsigned char b) {
    return make_float2((float)((b >> 4) & 0x0f) - 8.0f, (float)(b & 0x0f) - 8.0f);
}

__device__ __forceinline__ unsigned char encode44(float2 v) {
    int re = __float2int_rn(v.x), im = __float2int_rn(v.y);
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

__global__ void dtv_decode_kernel(const unsigned char* __restrict__ in, const int* __restrict__ sel,
                                  int n_freq, int n_times, float2* __restrict__ x,
                                  float* __restrict__ pw) {
    __shared__ float sh[NI];
    const int c = blockIdx.y;
    const int f = sel[c];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < NI; i += blockDim.x)
        sh[i] = 0.0f;
    __syncthreads();
    float p4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int it = 0; it < ROW_SAMPLES / (ROW_THREADS / 32); it++) {
        const int t = blockIdx.x * ROW_SAMPLES + it * (ROW_THREADS / 32) + warp;
        if (t >= n_times)
            break;
        const uchar4 b =
            *reinterpret_cast<const uchar4*>(in + ((size_t)t * n_freq + f) * NI + 4 * lane);
        const unsigned char bb[4] = {b.x, b.y, b.z, b.w};
        float2* row = x + ((size_t)c * n_times + t) * NI + 4 * lane;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float2 v = decode44(bb[k]);
            row[k] = v;
            p4[k] += v.x * v.x + v.y * v.y;
        }
    }
    add_input_sums(p4, sh, pw + c * NI);
}

__global__ void dtv_shift_kernel(float2* __restrict__ s, int n_times) {
    float2* sk = s + (size_t)blockIdx.x * (HIST + n_times);
    const float2 v = sk[n_times + threadIdx.x];
    __syncthreads();
    sk[threadIdx.x] = v;
}

__global__ void dtv_beam_kernel(const float2* __restrict__ x, const float2* __restrict__ w,
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
    float2* sk = s + (size_t)(c * n_src + k) * (HIST + n_times) + HIST;
    float pacc = 0.0f;
    for (int it = 0; it < ROW_SAMPLES / (ROW_THREADS / 32); it++) {
        const int t = blockIdx.x * ROW_SAMPLES + it * (ROW_THREADS / 32) + warp;
        if (t >= n_times)
            break;
        const float2* row = x + ((size_t)c * n_times + t) * NI + 4 * lane;
        float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 wv = ws[4 * lane + j];
            const float2 xv = row[j];
            acc.x += wv.x * xv.x - wv.y * xv.y;
            acc.y += wv.x * xv.y + wv.y * xv.x;
        }
        for (int o = 16; o > 0; o >>= 1) {
            acc.x += __shfl_xor_sync(0xffffffffu, acc.x, o);
            acc.y += __shfl_xor_sync(0xffffffffu, acc.y, o);
        }
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

// The s entries a tile of TT samples needs, for lags lag_min..lag_min+n_lag-1 (>= -16, < HIST):
// tile[m] = s[HIST + t0 - HIST + m], i.e. s index HIST + t - lag = tile index t - t0 - lag + HIST.
__device__ __forceinline__ void load_s_tile(const float2* __restrict__ sk, int t0, int n_times,
                                            float2* tile) {
    for (int m = threadIdx.x; m < TILE; m += blockDim.x) {
        const int idx = t0 + m; // s index HIST + t0 - HIST + m
        tile[m] = idx < HIST + n_times ? sk[idx] : make_float2(0.0f, 0.0f);
    }
}

// One block per (lag group, channel, tile of TT samples); the tiles' partial sums are added with
// atomics into memory the caller zeroed.
__global__ void dtv_lagcorr_kernel(const float2* __restrict__ x, const float2* __restrict__ s,
                                   const int* __restrict__ act, int k, int n_src, int n_times,
                                   int n_lag, int lag_min, float2* __restrict__ rxs) {
    __shared__ float2 tile[TILE];
    const int c = blockIdx.y;
    if (act[c * n_src + k] < 1)
        return;
    const int i = threadIdx.x;
    const int j0 = blockIdx.x * LAG_GROUP;
    const int nj = n_lag - j0 < LAG_GROUP ? n_lag - j0 : LAG_GROUP;
    const int t0 = blockIdx.z * TT;
    const float2* sk = s + (size_t)(c * n_src + k) * (HIST + n_times);
    const float2* xc = x + (size_t)c * n_times * NI;
    float2 acc[LAG_GROUP];
#pragma unroll
    for (int j = 0; j < LAG_GROUP; j++)
        acc[j] = make_float2(0.0f, 0.0f);
    load_s_tile(sk, t0, n_times, tile);
    __syncthreads();
    const int tn = n_times - t0 < TT ? n_times - t0 : TT;
    for (int tt = 0; tt < tn; tt++) {
        const float2 xv = xc[(size_t)(t0 + tt) * NI + i];
        const float2* sp = tile + tt + HIST - lag_min - j0; // lag j: tile[tt + HIST - lag_j]
#pragma unroll
        for (int j = 0; j < LAG_GROUP; j++) {
            if (j < nj) {
                const float2 sv = sp[-j];
                acc[j].x += xv.x * sv.x + xv.y * sv.y;
                acc[j].y += xv.y * sv.x - xv.x * sv.y;
            }
        }
    }
    for (int j = 0; j < nj; j++) {
        float2* dst = rxs + ((size_t)(c * n_src + k) * n_lag + j0 + j) * NI + i;
        atomicAdd(&dst->x, acc[j].x);
        atomicAdd(&dst->y, acc[j].y);
    }
}

// rss[d] = sum_t s(t) conj(s(t - d)) for d = 0..n_lag-1: every lag difference the normal
// equations need (negative ones by conjugate symmetry). d < DTV_HIST, so s(t - d) is always held.
// One block per (channel, tile of TT samples), added with atomics like dtv_lagcorr_kernel.
__global__ void dtv_rss_kernel(const float2* __restrict__ s, const int* __restrict__ act, int k,
                               int n_src, int n_times, int n_lag, float2* __restrict__ rss) {
    const int c = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= n_lag || act[c * n_src + k] < 1)
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

__global__ void dtv_subtract_kernel(float2* __restrict__ x, const float2* __restrict__ s,
                                    const float2* __restrict__ g, const int* __restrict__ act,
                                    int k, int n_src, int n_times, int n_lag, int lag_min) {
    __shared__ float2 tile[TILE];
    const int c = blockIdx.y;
    if (act[c * n_src + k] < 2)
        return;
    const int i = threadIdx.x;
    const int t0 = blockIdx.x * TT;
    const float2* sk = s + (size_t)(c * n_src + k) * (HIST + n_times);
    const float2* gk = g + (size_t)(c * n_src + k) * n_lag * NI + i;
    load_s_tile(sk, t0, n_times, tile);
    __syncthreads();
    const int tn = n_times - t0 < TT ? n_times - t0 : TT;
    for (int tt = 0; tt < tn; tt++) {
        float2 acc = make_float2(0.0f, 0.0f);
        const float2* sp = tile + tt + HIST - lag_min; // lag j: sp[-j]
        for (int j = 0; j < n_lag; j++) {
            const float2 gv = gk[(size_t)j * NI];
            const float2 sv = sp[-j];
            acc.x += gv.x * sv.x - gv.y * sv.y;
            acc.y += gv.x * sv.y + gv.y * sv.x;
        }
        float2* xv = x + ((size_t)c * n_times + t0 + tt) * NI + i;
        xv->x -= acc.x;
        xv->y -= acc.y;
    }
}

__global__ void dtv_encode_kernel(const float2* __restrict__ x, const int* __restrict__ sel,
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
    float p4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int it = 0; it < ROW_SAMPLES / (ROW_THREADS / 32); it++) {
        const int t = blockIdx.x * ROW_SAMPLES + it * (ROW_THREADS / 32) + warp;
        if (t >= n_times)
            break;
        const float2* row = x + ((size_t)c * n_times + t) * NI + 4 * lane;
        unsigned char bb[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float2 v = row[k];
            p4[k] += v.x * v.x + v.y * v.y;
            bb[k] = encode44(v);
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

__global__ void dtv_cov_kernel(const float2* __restrict__ x, int n_times,
                               float2* __restrict__ cov) {
    __shared__ float2 tile[COV_TT * NI];
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int t_per = n_times / gridDim.y;
    const int t_begin = blockIdx.y * t_per, t_end = t_begin + t_per;
    const float2* xc = x + (size_t)c * n_times * NI;
    float2 acc[COV_PAIRS];
    for (int k = 0; k < COV_PAIRS; k++)
        acc[k] = make_float2(0.0f, 0.0f);
    for (int t0 = t_begin; t0 < t_end; t0 += COV_TT) {
        __syncthreads();
        for (int m = tid; m < COV_TT * NI; m += COV_THREADS)
            tile[m] = xc[(size_t)t0 * NI + m];
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

} // namespace

void launch_dtv_decode(const unsigned char* in, const int* sel, int n_chan, int n_freq, int n_times,
                       float2* x, float* pw_raw, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    cudaMemsetAsync(pw_raw, 0, sizeof(float) * NI * n_chan, stream);
    const dim3 grid((n_times + ROW_SAMPLES - 1) / ROW_SAMPLES, n_chan);
    dtv_decode_kernel<<<grid, ROW_THREADS, 0, stream>>>(in, sel, n_freq, n_times, x, pw_raw);
}

void launch_dtv_shift(float2* s, int n_chan, int n_src, int n_times, cudaStream_t stream) {
    if (n_chan * n_src <= 0)
        return;
    dtv_shift_kernel<<<n_chan * n_src, HIST, 0, stream>>>(s, n_times);
}

void launch_dtv_beam(const float2* x, const float2* w, const int* act, int k, int n_chan, int n_src,
                     int n_times, float2* s, float* ps, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    const dim3 grid((n_times + ROW_SAMPLES - 1) / ROW_SAMPLES, n_chan);
    dtv_beam_kernel<<<grid, ROW_THREADS, 0, stream>>>(x, w, act, k, n_src, n_times, s, ps);
}

void launch_dtv_lagcorr(const float2* x, const float2* s, const int* act, int k, int n_chan,
                        int n_src, int n_times, int n_lag, int lag_min, float2* rxs, float2* rss,
                        cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    const int n_tiles = (n_times + TT - 1) / TT;
    const dim3 grid((n_lag + LAG_GROUP - 1) / LAG_GROUP, n_chan, n_tiles);
    dtv_lagcorr_kernel<<<grid, NI, 0, stream>>>(x, s, act, k, n_src, n_times, n_lag, lag_min, rxs);
    dtv_rss_kernel<<<dim3(n_chan, n_tiles), DTV_MAX_LAGS, 0, stream>>>(s, act, k, n_src, n_times,
                                                                       n_lag, rss);
}

void launch_dtv_subtract(float2* x, const float2* s, const float2* g, const int* act, int k,
                         int n_chan, int n_src, int n_times, int n_lag, int lag_min,
                         cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    const dim3 grid((n_times + TT - 1) / TT, n_chan);
    dtv_subtract_kernel<<<grid, NI, 0, stream>>>(x, s, g, act, k, n_src, n_times, n_lag, lag_min);
}

void launch_dtv_encode(const float2* x, const int* sel, const int* act, int n_chan, int n_src,
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

void launch_dtv_cov(const float2* x, int n_chan, int n_times, float2* cov, cudaStream_t stream) {
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
