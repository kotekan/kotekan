#include "cudaGnssBeamDumpKernel.hpp"

#include <cuda_fp16.h>

namespace gnss_cuda {

namespace {

constexpr int NI = BEAM_INPUTS;
constexpr int COV_THREADS = 256;
constexpr int COV_TT = 64; // samples per shared-memory tile
constexpr int PAIRS_PER_THREAD = NI * NI / COV_THREADS;
constexpr float PW_EPS = 1e-3f; // a dead input's power floor (its samples are 0x88 = 0)

__device__ __forceinline__ float2 decode44(unsigned char b) {
    return make_float2((float)((b >> 4) & 0x0f) - 8.0f, (float)(b & 0x0f) - 8.0f);
}

// cov[c][i][j] = sum_t x_i conj(x_j). Blocks (channel, time split); each thread owns a fixed
// set of ordered pairs and walks its share of the frame in tiles of COV_TT samples staged in
// shared memory, so the 128 contiguous bytes of each sample are read once per block. The
// splits add their partial sums with float atomics, exact for these integer-valued sums.
__global__ void beam_cov_kernel(const unsigned char* __restrict__ in, const int* __restrict__ sel,
                                int n_freq, int n_times, float2* __restrict__ cov) {
    __shared__ __align__(16) unsigned char tile[COV_TT * NI];
    const int c = blockIdx.x;
    const int f = sel[c];
    const int tid = threadIdx.x;
    const int t_per = n_times / gridDim.y;
    const int t_begin = blockIdx.y * t_per, t_end = t_begin + t_per;
    float2 acc[PAIRS_PER_THREAD];
    for (int k = 0; k < PAIRS_PER_THREAD; k++)
        acc[k] = make_float2(0.0f, 0.0f);
    for (int t0 = t_begin; t0 < t_end; t0 += COV_TT) {
        __syncthreads();
        // 16 bytes per copy; a sample's 128 inputs are 128 contiguous, 16-aligned bytes.
        for (int b = tid; b < COV_TT * NI / 16; b += COV_THREADS) {
            const int tt = b / (NI / 16), part = b % (NI / 16);
            const int4 v = *reinterpret_cast<const int4*>(in + ((size_t)(t0 + tt) * n_freq + f) * NI
                                                          + part * 16);
            *reinterpret_cast<int4*>(tile + tt * NI + part * 16) = v;
        }
        __syncthreads();
        for (int k = 0; k < PAIRS_PER_THREAD; k++) {
            const int p = tid + k * COV_THREADS;
            const int i = p / NI, j = p % NI;
            float re = 0.0f, im = 0.0f;
            for (int tt = 0; tt < COV_TT; tt++) {
                const float2 a = decode44(tile[tt * NI + i]);
                const float2 b = decode44(tile[tt * NI + j]);
                re += a.x * b.x + a.y * b.y; // a conj(b)
                im += a.y * b.x - a.x * b.y;
            }
            acc[k].x += re;
            acc[k].y += im;
        }
    }
    for (int k = 0; k < PAIRS_PER_THREAD; k++) {
        float2* dst = cov + (size_t)c * NI * NI + tid + k * COV_THREADS;
        atomicAdd(&dst->x, acc[k].x);
        atomicAdd(&dst->y, acc[k].y);
    }
}

__device__ __forceinline__ float block_sum_128(float v, float* scratch) {
    // 128 threads = 4 warps
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0)
        scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    const float s = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float2 block_csum_128(float2 v, float* scratch) {
    const float re = block_sum_128(v.x, scratch);
    const float im = block_sum_128(v.y, scratch);
    return make_float2(re, im);
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ float2 cmulc(float2 a, float2 b) { // a conj(b)
    return make_float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

// One block of NI threads per channel, thread i = input i. W = D^-1/2 cov D^-1/2 / n_times is
// applied on the fly from global memory (L2-resident, 128 KB per channel).
__global__ void beam_eig_kernel(const float2* __restrict__ cov, int n_times, int ref, int n_iter,
                                float2* __restrict__ q1s, float2* __restrict__ q2s,
                                float2* __restrict__ w, float* __restrict__ pw_out,
                                float* __restrict__ lam_out, const int* __restrict__ ref_dev) {
    __shared__ float2 q[NI];
    __shared__ float2 r[NI];
    __shared__ float wt[NI];
    __shared__ float scratch[4];
    const int c = blockIdx.x, i = threadIdx.x;
    if (ref_dev)
        ref = ref_dev[c];
    const float2* V = cov + (size_t)c * NI * NI;
    const float inv_t = 1.0f / (float)n_times;
    const float pw = V[i * NI + i].x * inv_t;
    wt[i] = rsqrtf(fmaxf(pw, PW_EPS));
    pw_out[c * NI + i] = pw;
    // warm start; zeros mean a cold start
    float2 qi = q1s[c * NI + i];
    float2 ri = q2s[c * NI + i];
    q[i] = qi;
    r[i] = ri;
    __syncthreads();
    float n1 = block_sum_128(qi.x * qi.x + qi.y * qi.y, scratch);
    if (n1 < 1e-12f)
        q[i] = make_float2(rsqrtf((float)NI), 0.0f);
    float n2 = block_sum_128(ri.x * ri.x + ri.y * ri.y, scratch);
    if (n2 < 1e-12f)
        r[i] = make_float2((i & 1) ? -rsqrtf((float)NI) : rsqrtf((float)NI), 0.0f);
    __syncthreads();
    float lam1 = 0.0f, lam2 = 0.0f;
    for (int it = 0; it < n_iter; it++) {
        // y = W q
        float2 y = make_float2(0.0f, 0.0f);
        const float wi = wt[i] * inv_t;
        for (int j = 0; j < NI; j++) {
            const float2 Wij = V[i * NI + j];
            const float s = wi * wt[j];
            y.x += s * (Wij.x * q[j].x - Wij.y * q[j].y);
            y.y += s * (Wij.x * q[j].y + Wij.y * q[j].x);
        }
        const float ny = sqrtf(block_sum_128(y.x * y.x + y.y * y.y, scratch));
        lam1 = ny;
        const float inv = ny > 0.0f ? 1.0f / ny : 0.0f;
        __syncthreads();
        q[i] = make_float2(y.x * inv, y.y * inv);
        __syncthreads();
    }
    // the second component: power iteration on W deflated by lambda1 q q^H, re-orthogonalised
    for (int it = 0; it < n_iter; it++) {
        float2 y = make_float2(0.0f, 0.0f);
        const float wi = wt[i] * inv_t;
        for (int j = 0; j < NI; j++) {
            const float2 Wij = V[i * NI + j];
            const float s = wi * wt[j];
            y.x += s * (Wij.x * r[j].x - Wij.y * r[j].y);
            y.y += s * (Wij.x * r[j].y + Wij.y * r[j].x);
        }
        // y -= q (q^H y)
        const float2 qy = block_csum_128(cmulc(y, q[i]), scratch); // sum conj(q_i) y_i
        y.x -= q[i].x * qy.x - q[i].y * qy.y;
        y.y -= q[i].x * qy.y + q[i].y * qy.x;
        const float ny = sqrtf(block_sum_128(y.x * y.x + y.y * y.y, scratch));
        lam2 = ny;
        const float inv = ny > 0.0f ? 1.0f / ny : 0.0f;
        __syncthreads();
        r[i] = make_float2(y.x * inv, y.y * inv);
        __syncthreads();
    }
    q1s[c * NI + i] = q[i];
    q2s[c * NI + i] = r[i];
    // phase reference: q[ref] real and positive
    const float2 qr = q[ref];
    const float ar = sqrtf(qr.x * qr.x + qr.y * qr.y);
    float2 ph = make_float2(1.0f, 0.0f);
    if (ar > 1e-6f)
        ph = make_float2(qr.x / ar, -qr.y / ar);
    const float2 qi_ref = cmul(q[i], ph);
    // beam = sum_i conj(q_i) wt_i x_i * |q_ref| / wt_ref  (the reference input's units)
    const float scale = wt[i] * ar / wt[ref];
    w[c * NI + i] = make_float2(qi_ref.x * scale, -qi_ref.y * scale);
    if (i == 0) {
        lam_out[c * 4 + 0] = lam1;
        lam_out[c * 4 + 1] = lam2;
        lam_out[c * 4 + 2] = 0.0f;
        lam_out[c * 4 + 3] = 0.0f;
    }
}

// One warp per (channel, sample): lane l reads inputs 4l..4l+3 (a 128-byte coalesced row).
__global__ void beam_form_kernel(const unsigned char* __restrict__ in, const int* __restrict__ sel,
                                 int n_freq, int n_times, const float2* __restrict__ w,
                                 __half2* __restrict__ out) {
    __shared__ float2 ws[NI];
    const int c = blockIdx.y;
    const int f = sel[c];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = (blockIdx.x * (blockDim.x >> 5)) + warp;
    for (int k = threadIdx.x; k < NI; k += blockDim.x)
        ws[k] = w[c * NI + k];
    __syncthreads();
    if (t >= n_times)
        return;
    const unsigned char* row = in + ((size_t)t * n_freq + f) * NI;
    const uchar4 b = *reinterpret_cast<const uchar4*>(row + 4 * lane);
    float2 acc = make_float2(0.0f, 0.0f);
    const unsigned char bb[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const float2 x = decode44(bb[k]);
        const float2 wk = ws[4 * lane + k];
        acc.x += wk.x * x.x - wk.y * x.y;
        acc.y += wk.x * x.y + wk.y * x.x;
    }
    for (int o = 16; o > 0; o >>= 1) {
        acc.x += __shfl_xor_sync(0xffffffffu, acc.x, o);
        acc.y += __shfl_xor_sync(0xffffffffu, acc.y, o);
    }
    if (lane == 0)
        out[(size_t)c * n_times + t] = __floats2half2_rn(acc.x, acc.y);
}

} // namespace

void launch_beam_cov(const unsigned char* in, const int* sel, int n_chan, int n_freq, int n_times,
                     float2* cov, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    // 256 samples per block: 32 blocks per channel at CHORD's 8192-sample frame
    const int n_split = n_times / (4 * COV_TT) > 0 ? n_times / (4 * COV_TT) : 1;
    cudaMemsetAsync(cov, 0, (size_t)n_chan * NI * NI * sizeof(float2), stream);
    beam_cov_kernel<<<dim3(n_chan, n_split), COV_THREADS, 0, stream>>>(in, sel, n_freq, n_times,
                                                                       cov);
}

void launch_beam_eig(const float2* cov, int n_chan, int n_times, int ref, int n_iter, float2* q1,
                     float2* q2, float2* w, float* pw, float* lam, cudaStream_t stream,
                     const int* ref_dev) {
    if (n_chan <= 0)
        return;
    beam_eig_kernel<<<n_chan, NI, 0, stream>>>(cov, n_times, ref, n_iter, q1, q2, w, pw, lam,
                                               ref_dev);
}

void launch_beam_form(const unsigned char* in, const int* sel, int n_chan, int n_freq, int n_times,
                      const float2* w, void* out, cudaStream_t stream) {
    if (n_chan <= 0)
        return;
    constexpr int threads = 256, warps = threads / 32;
    const dim3 grid((n_times + warps - 1) / warps, n_chan);
    beam_form_kernel<<<grid, threads, 0, stream>>>(in, sel, n_freq, n_times, w,
                                                   static_cast<__half2*>(out));
}

} // namespace gnss_cuda
