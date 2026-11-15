#include "tensorFFT.hpp"

#include <cassert>
#include <cstdint>

// Each CTA stages ELEMS contiguous complex fp16 elements (one 8192-point transform, or
// ELEMS / n shorter ones) through shared memory and runs every FFT stage as tensor-core
// mma. A length-16 DFT in the real form G (32x32, 2x2 blocks [[Re,-Im],[Im,Re]] of the
// complex matrix) maps interleaved re/im input to interleaved output, so one
// m16n8k16.f32.f16.f16.f32 mma with the data tile as the A operand and G^T as the B
// operand transforms 16 butterflies at once and leaves (re, im) pairs together in the
// fp32 accumulators for the inter-stage twiddles.
static constexpr int ELEMS = 8192;
static constexpr int THREADS = 256;
static constexpr int THREADS_F32 = 256;

// Swizzled shared index: folds upper address bits into the bank bits so that every
// butterfly access family (stride 1 << lc, lc = 0..9, crossed with the 16-consecutive
// butterfly rows) touches 32 distinct banks. No padding, so three CTAs fit per SM.
__device__ static inline int pad(int i) {
    return i ^ ((i >> 4) & 4) ^ (((i >> 2) ^ (i >> 4) ^ (i >> 6)) & 24) ^ ((i >> 9) & 4)
           ^ ((i >> 11) & 2);
}

// D += A * B: A 16x16 fp16 row-major, B 16x8 fp16 column-major, D 16x8 fp32.
__device__ static inline void mma_f32_m16_n8_k16(float d[4], const __half2 a[4],
                                                 const __half2 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(*(const unsigned*)&a[0]), "r"(*(const unsigned*)&a[1]), "r"(*(const unsigned*)&a[2]),
          "r"(*(const unsigned*)&a[3]), "r"(*(const unsigned*)&b[0]), "r"(*(const unsigned*)&b[1]));
}

// Columns (2j, 2j+1) of row 2k + im_row of the real form of I_{16/r} (x) F_r, the
// block-diagonal radix-r DFT matrix (r = 16 is the plain DFT-16): (cos t, sin t) on the
// even row and (-sin t, cos t) on the odd, t = 2 pi ((k j) mod r) / r within a block.
__device__ static inline __half2 dft_pair(int k, int j, int r, bool im_row) {
    if ((k / r) != (j / r))
        return __floats2half2_rn(0.0f, 0.0f);
    float s, c;
    sincospif(2.0f * ((k % r) * (j % r) % r) / r, &s, &c);
    return im_row ? __floats2half2_rn(-s, c) : __floats2half2_rn(c, s);
}

// This thread's B fragments of G^T for the radix-r stage matrix: bw[ct][kc] covers
// output columns 8 ct .. 8 ct + 7 and k-rows 16 kc .. 16 kc + 15 of the 32x32 product.
__device__ static void load_dft_fragments(__half2 bw[4][2][2], int r, int grp, int tg) {
#pragma unroll
    for (int ct = 0; ct < 4; ct++)
#pragma unroll
        for (int kc = 0; kc < 2; kc++) {
            const int k = (8 * ct + grp) >> 1;
            const int j = 8 * kc + tg;
            bw[ct][kc][0] = dft_pair(k, j, r, grp & 1);
            bw[ct][kc][1] = dft_pair(k, j + 4, r, grp & 1);
        }
}

// Position inside a transform that holds output bin k after all stages ran in place:
// the per-stage digit of k, little-endian base 16 with the leftover radix r = n / 16^s
// on top, stored big-endian.
__device__ static inline int bin_pos(int k, int s, int lr) {
    int q = 0, low = k & ((1 << (4 * s)) - 1);
#pragma unroll
    for (int i = 0; i < 3; i++)
        if (i < s) {
            q = (q << 4) | (low & 15);
            low >>= 4;
        }
    return (q << lr) | (k >> (4 * s));
}

__global__ static void __launch_bounds__(THREADS, 3)
    tensor_fft(const __half2* __restrict__ in, __half2* __restrict__ out, int log2n, size_t total) {
    __shared__ __half2 sm[ELEMS];
    const size_t base = (size_t)blockIdx.x * ELEMS;

    // Vector loads on full, aligned CTA chunks; the tail CTA and odd pointers go scalar.
    const bool vec = base + ELEMS <= total && ((uintptr_t)in & 15) == 0;
    if (vec) {
        for (int i = threadIdx.x; i < ELEMS / 4; i += THREADS) {
            const float4 v = ((const float4*)in)[base / 4 + i];
            sm[pad(4 * i)] = *(const __half2*)&v.x;
            sm[pad(4 * i + 1)] = *(const __half2*)&v.y;
            sm[pad(4 * i + 2)] = *(const __half2*)&v.z;
            sm[pad(4 * i + 3)] = *(const __half2*)&v.w;
        }
    } else {
        for (int i = threadIdx.x; i < ELEMS; i += THREADS) {
            const size_t g = base + i;
            sm[pad(i)] = g < total ? in[g] : __floats2half2_rn(0.0f, 0.0f);
        }
    }

    const int s = log2n >> 2; // radix-16 stages
    const int lr = log2n & 3; // leftover radix r = 1 << lr
    const int grp = (threadIdx.x & 31) >> 2;
    const int tg = threadIdx.x & 3;

    __half2 bw[4][2][2];
    load_dft_fragments(bw, 16, grp, tg);

    // Decimation in frequency: blocks shrink 16x per stage, butterflies at stride
    // c = b / 16 inside each block, twiddled by W_b^(j2 k1); the leftover radix runs as
    // one block-diagonal stage over contiguous 16-element chunks.
    const int nstages = s + (lr > 0);
    for (int t = 0; t < nstages; t++) {
        const int lb = t == s ? 4 : log2n - 4 * t;
        const int lc = lb - 4;
        if (t == s)
            load_dft_fragments(bw, 1 << lr, grp, tg);
        __syncthreads();
        for (int tile = threadIdx.x >> 5; tile < ELEMS / 256; tile += THREADS / 32) {
            // 16 butterflies 16 tile + row; this thread covers rows grp and grp + 8.
            const int beta0 = 16 * tile + grp;
            const int blk0 = (beta0 >> lc) << lb, j2_0 = beta0 & ((1 << lc) - 1);
            const int blk1 = ((beta0 + 8) >> lc) << lb, j2_1 = (beta0 + 8) & ((1 << lc) - 1);
            float d[4][4] = {}; // d[ct] = (re, im) of element 4 ct + tg, both rows
#pragma unroll
            for (int kc = 0; kc < 2; kc++) {
                const int j = 8 * kc + tg;
                __half2 a[4];
                a[0] = sm[pad(blk0 + j * (1 << lc) + j2_0)];
                a[1] = sm[pad(blk1 + j * (1 << lc) + j2_1)];
                a[2] = sm[pad(blk0 + (j + 4) * (1 << lc) + j2_0)];
                a[3] = sm[pad(blk1 + (j + 4) * (1 << lc) + j2_1)];
#pragma unroll
                for (int ct = 0; ct < 4; ct++)
                    mma_f32_m16_n8_k16(d[ct], a, bw[ct][kc]);
            }
            if (lb > 4) {
                // Twiddle W_b^(j2 k1), k1 = tg, tg + 4, tg + 8, tg + 12: one base angle
                // and a constant W_b^(4 j2) step per butterfly, carried in fp32.
                const float a1 = -6.2831853e0f / (1 << lb);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int j2 = h ? j2_1 : j2_0;
                    float wr, wi, pr, pi;
                    __sincosf(a1 * (j2 * tg), &wi, &wr);
                    __sincosf(a1 * (j2 * 4), &pi, &pr);
#pragma unroll
                    for (int ct = 0; ct < 4; ct++) {
                        const float re = d[ct][2 * h], im = d[ct][2 * h + 1];
                        d[ct][2 * h] = re * wr - im * wi;
                        d[ct][2 * h + 1] = re * wi + im * wr;
                        const float nr = wr * pr - wi * pi;
                        wi = wr * pi + wi * pr;
                        wr = nr;
                    }
                }
            }
#pragma unroll
            for (int ct = 0; ct < 4; ct++) {
                const int k1 = 4 * ct + tg;
                sm[pad(blk0 + k1 * (1 << lc) + j2_0)] = __floats2half2_rn(d[ct][0], d[ct][1]);
                sm[pad(blk1 + k1 * (1 << lc) + j2_1)] = __floats2half2_rn(d[ct][2], d[ct][3]);
            }
        }
    }

    __syncthreads();
    const int m = (1 << log2n) - 1;
    if (vec && ((uintptr_t)out & 15) == 0) {
        for (int i = threadIdx.x; i < ELEMS / 4; i += THREADS) {
            float4 v;
            *(__half2*)&v.x = sm[pad((4 * i & ~m) | bin_pos(4 * i & m, s, lr))];
            *(__half2*)&v.y = sm[pad(((4 * i + 1) & ~m) | bin_pos((4 * i + 1) & m, s, lr))];
            *(__half2*)&v.z = sm[pad(((4 * i + 2) & ~m) | bin_pos((4 * i + 2) & m, s, lr))];
            *(__half2*)&v.w = sm[pad(((4 * i + 3) & ~m) | bin_pos((4 * i + 3) & m, s, lr))];
            ((float4*)out)[base / 4 + i] = v;
        }
    } else {
        for (int i = threadIdx.x; i < ELEMS; i += THREADS) {
            const size_t g = base + i;
            if (g < total)
                out[g] = sm[pad((i & ~m) | bin_pos(i & m, s, lr))];
        }
    }
}

// ---- TF32 variant: the same decimation-in-frequency structure on fp32 data. Operands
// round to TF32 per mma (m16n8k8), accumulators, twiddles and inter-stage storage stay
// fp32. Shared memory is dynamic float2 with a +1-per-32 skew instead of the fp16 swizzle;
// 66 KiB per CTA leaves one CTA per SM — this path is for accuracy, not speed.

__device__ static inline int padf(int i) {
    return i + (i >> 5);
}

__device__ static inline unsigned to_tf32(float x) {
    unsigned u;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
    return u;
}

// D += A * B: A 16x8 tf32 row-major, B 8x8 tf32 column-major, D 16x8 fp32.
__device__ static inline void mma_f32_m16_n8_k8(float d[4], const unsigned a[4],
                                                const unsigned b[2]) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Element (p, q) of the 32x32 real form of I_{16/r} (x) F_r: output real row p = 2k + im,
// input real column q = 2j + comp.
__device__ static inline float dft_scalar(int p, int q, int r) {
    const int k = p >> 1, j = q >> 1;
    if ((k / r) != (j / r))
        return 0.0f;
    float s, c;
    sincospif(2.0f * ((k % r) * (j % r) % r) / r, &s, &c);
    if (p & 1)
        return (q & 1) ? c : -s;
    return (q & 1) ? s : c;
}

// This thread's tf32 B fragments of G^T: bw[ct][kc] covers output columns 8 ct .. 8 ct + 7
// and k-rows 8 kc .. 8 kc + 7.
__device__ static void load_dft_fragments_f32(unsigned bw[4][4][2], int r, int grp, int tg) {
#pragma unroll
    for (int ct = 0; ct < 4; ct++)
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            const int p = 8 * ct + grp;
            bw[ct][kc][0] = to_tf32(dft_scalar(p, 8 * kc + tg, r));
            bw[ct][kc][1] = to_tf32(dft_scalar(p, 8 * kc + tg + 4, r));
        }
}

__global__ static void __launch_bounds__(THREADS_F32, 1)
    tensor_fft_f32(const float2* __restrict__ in, float2* __restrict__ out, int log2n,
                   size_t total) {
    extern __shared__ float2 smf[];
    const size_t base = (size_t)blockIdx.x * ELEMS;

    for (int i = threadIdx.x; i < ELEMS; i += THREADS_F32) {
        const size_t g = base + i;
        smf[padf(i)] = g < total ? in[g] : make_float2(0.0f, 0.0f);
    }

    const int s = log2n >> 2;
    const int lr = log2n & 3;
    const int grp = (threadIdx.x & 31) >> 2;
    const int tg = threadIdx.x & 3;

    unsigned bw[4][4][2];
    load_dft_fragments_f32(bw, 16, grp, tg);

    const int nstages = s + (lr > 0);
    for (int t = 0; t < nstages; t++) {
        const int lb = t == s ? 4 : log2n - 4 * t;
        const int lc = lb - 4;
        if (t == s)
            load_dft_fragments_f32(bw, 1 << lr, grp, tg);
        __syncthreads();
        for (int tile = threadIdx.x >> 5; tile < ELEMS / 256; tile += THREADS_F32 / 32) {
            const int beta0 = 16 * tile + grp;
            const int blk0 = (beta0 >> lc) << lb, j2_0 = beta0 & ((1 << lc) - 1);
            const int blk1 = ((beta0 + 8) >> lc) << lb, j2_1 = (beta0 + 8) & ((1 << lc) - 1);
            float d[4][4] = {};
#pragma unroll
            for (int kc = 0; kc < 4; kc++) {
                // A columns 8 kc + tg and + 4: real component (q & 1) of complex element
                // q >> 1 of each butterfly row.
                const int q0 = 8 * kc + tg, q1 = q0 + 4;
                const float2 e00 = smf[padf(blk0 + (q0 >> 1) * (1 << lc) + j2_0)];
                const float2 e01 = smf[padf(blk1 + (q0 >> 1) * (1 << lc) + j2_1)];
                const float2 e10 = smf[padf(blk0 + (q1 >> 1) * (1 << lc) + j2_0)];
                const float2 e11 = smf[padf(blk1 + (q1 >> 1) * (1 << lc) + j2_1)];
                unsigned a[4];
                a[0] = to_tf32((q0 & 1) ? e00.y : e00.x);
                a[1] = to_tf32((q0 & 1) ? e01.y : e01.x);
                a[2] = to_tf32((q1 & 1) ? e10.y : e10.x);
                a[3] = to_tf32((q1 & 1) ? e11.y : e11.x);
#pragma unroll
                for (int ct = 0; ct < 4; ct++)
                    mma_f32_m16_n8_k8(d[ct], a, bw[ct][kc]);
            }
            if (lb > 4) {
                const float a1 = -6.2831853e0f / (1 << lb);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int j2 = h ? j2_1 : j2_0;
                    float wr, wi, pr, pi;
                    __sincosf(a1 * (j2 * tg), &wi, &wr);
                    __sincosf(a1 * (j2 * 4), &pi, &pr);
#pragma unroll
                    for (int ct = 0; ct < 4; ct++) {
                        const float re = d[ct][2 * h], im = d[ct][2 * h + 1];
                        d[ct][2 * h] = re * wr - im * wi;
                        d[ct][2 * h + 1] = re * wi + im * wr;
                        const float nr = wr * pr - wi * pi;
                        wi = wr * pi + wi * pr;
                        wr = nr;
                    }
                }
            }
#pragma unroll
            for (int ct = 0; ct < 4; ct++) {
                const int k1 = 4 * ct + tg;
                smf[padf(blk0 + k1 * (1 << lc) + j2_0)] = make_float2(d[ct][0], d[ct][1]);
                smf[padf(blk1 + k1 * (1 << lc) + j2_1)] = make_float2(d[ct][2], d[ct][3]);
            }
        }
    }

    __syncthreads();
    const int m = (1 << log2n) - 1;
    for (int i = threadIdx.x; i < ELEMS; i += THREADS_F32) {
        const size_t g = base + i;
        if (g < total)
            out[g] = smf[padf((i & ~m) | bin_pos(i & m, s, lr))];
    }
}

void gpu_tensor_fft_f32(const float2* in, float2* out, size_t n, size_t batch,
                        cudaStream_t stream) {
    assert(n >= 32 && n <= 8192 && (n & (n - 1)) == 0);
    if (batch == 0)
        return;
    int log2n = 0;
    while ((size_t(1) << log2n) < n)
        log2n++;
    const size_t total = n * batch;
    const int blocks = (int)((total + ELEMS - 1) / ELEMS);
    const int smem = (ELEMS + ELEMS / 32) * sizeof(float2);
    cudaFuncSetAttribute(tensor_fft_f32, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    tensor_fft_f32<<<blocks, THREADS_F32, smem, stream>>>(in, out, log2n, total);
}

void gpu_tensor_fft(const __half2* in, __half2* out, size_t n, size_t batch, cudaStream_t stream) {
    assert(n >= 32 && n <= 8192 && (n & (n - 1)) == 0);
    if (batch == 0)
        return;
    int log2n = 0;
    while ((size_t(1) << log2n) < n)
        log2n++;
    const size_t total = n * batch;
    const int blocks = (int)((total + ELEMS - 1) / ELEMS);
    tensor_fft<<<blocks, THREADS, 0, stream>>>(in, out, log2n, total);
}

// ---- Fused last pass: FFT2 + untangle + gain + quantize in one kernel. Each CTA owns a
// conjugate PAIR of FFT2 transforms (rows k1 and m1 - k1 of the twiddled matrix; rows 0 and
// m1/2 pair with themselves), so after the in-smem FFT both Z[k] and Z[m - k] are resident
// and the spectrum never returns to global memory: the quantized bytes leave in the
// transform-local [k1][k2] order and a cheap byte transpose puts them in bin order. This
// removes FFT2's store and the quantizer's two spectrum reads, about a fifth of the
// pipeline's traffic. Gains and stats use the same transform-local layout
// (gpu_reorder_f32 converts); requires m2 >= 256 and power-of-two sizes.

__device__ static inline uint8_t encode_fused(float vr, float vi, float g,
                                              const float* __restrict__ gains,
                                              float* __restrict__ stats, size_t k) {
    if (stats)
        stats[k] += vr * vr + vi * vi;
    if (gains)
        g *= gains[k];
    const int re = max(-7, min(7, __float2int_rn(g * vr)));
    const int im = max(-7, min(7, __float2int_rn(g * vi)));
    return (uint8_t)(((re + 8) << 4) | (im + 8));
}

__device__ static inline float2 twiddle_hilo(const float2* __restrict__ w_hi,
                                             const float2* __restrict__ w_lo, uint64_t e) {
    const float2 wh = w_hi[e >> 12];
    const float2 wl = w_lo[e & 4095];
    return make_float2(wh.x * wl.x - wh.y * wl.y, wh.x * wl.y + wh.y * wl.x);
}

// Which half of the pair-CTA's shared array a packed index lives in, and its offset there.
__device__ static inline int half_of(int i, int n) {
    return i >= n ? n : 0;
}
__device__ static inline int in_half(int i, int n) {
    return i >= n ? i - n : i;
}

static constexpr int THREADS_FQ = 512;

__global__ static void __launch_bounds__(THREADS_FQ, 1)
    tensor_fft2_quantize(const __half2* __restrict__ in, size_t m1, size_t m2,
                         const float2* __restrict__ w_hi, const float2* __restrict__ w_lo,
                         float scale, float gain, const float* __restrict__ gains_t,
                         float* __restrict__ stats_t, uint8_t* __restrict__ out) {
    extern __shared__ __half2 smp[];
    const size_t c = blockIdx.x;
    const bool self = c == 0 || c == m1 / 2;
    const size_t rowB = self ? c : m1 - c;
    const int n = (int)m2;
    int log2n = 0;
    while ((1 << log2n) < n)
        log2n++;
    const int elems = self ? n : 2 * n;

    // Rows are contiguous in global memory; halves are pad()-swizzled like the base kernel
    // (pad only permutes within 32-aligned blocks, so it stays inside each half).
    for (int i = threadIdx.x; i < n; i += THREADS_FQ) {
        smp[pad(i)] = in[c * m2 + i];
        if (!self)
            smp[n + pad(i)] = in[rowB * m2 + i];
    }

    const int s = log2n >> 2;
    const int lr = log2n & 3;
    const int grp = (threadIdx.x & 31) >> 2;
    const int tg = threadIdx.x & 3;
    __half2 bw[4][2][2];
    load_dft_fragments(bw, 16, grp, tg);

    const int nstages = s + (lr > 0);
    for (int t = 0; t < nstages; t++) {
        const int lb = t == s ? 4 : log2n - 4 * t;
        const int lc = lb - 4;
        if (t == s)
            load_dft_fragments(bw, 1 << lr, grp, tg);
        __syncthreads();
        for (int tile = threadIdx.x >> 5; tile < elems / 256; tile += THREADS_FQ / 32) {
            const int beta0 = 16 * tile + grp;
            const int blk0 = (beta0 >> lc) << lb, j2_0 = beta0 & ((1 << lc) - 1);
            const int blk1 = ((beta0 + 8) >> lc) << lb, j2_1 = (beta0 + 8) & ((1 << lc) - 1);
            float d[4][4] = {};
#pragma unroll
            for (int kc = 0; kc < 2; kc++) {
                const int j = 8 * kc + tg;
                __half2 a[4];
                a[0] = smp[half_of(blk0, n) + pad(in_half(blk0, n) + j * (1 << lc) + j2_0)];
                a[1] = smp[half_of(blk1, n) + pad(in_half(blk1, n) + j * (1 << lc) + j2_1)];
                a[2] = smp[half_of(blk0, n) + pad(in_half(blk0, n) + (j + 4) * (1 << lc) + j2_0)];
                a[3] = smp[half_of(blk1, n) + pad(in_half(blk1, n) + (j + 4) * (1 << lc) + j2_1)];
#pragma unroll
                for (int ct = 0; ct < 4; ct++)
                    mma_f32_m16_n8_k16(d[ct], a, bw[ct][kc]);
            }
            if (lb > 4) {
                const float a1 = -6.2831853e0f / (1 << lb);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int j2 = h ? j2_1 : j2_0;
                    float wr, wi, pr, pi;
                    __sincosf(a1 * (j2 * tg), &wi, &wr);
                    __sincosf(a1 * (j2 * 4), &pi, &pr);
#pragma unroll
                    for (int ct = 0; ct < 4; ct++) {
                        const float re = d[ct][2 * h], im = d[ct][2 * h + 1];
                        d[ct][2 * h] = re * wr - im * wi;
                        d[ct][2 * h + 1] = re * wi + im * wr;
                        const float nr = wr * pr - wi * pi;
                        wi = wr * pi + wi * pr;
                        wr = nr;
                    }
                }
            }
#pragma unroll
            for (int ct = 0; ct < 4; ct++) {
                const int k1 = 4 * ct + tg;
                const int p0 = blk0 + k1 * (1 << lc) + j2_0;
                const int p1 = blk1 + k1 * (1 << lc) + j2_1;
                smp[half_of(p0, n) + pad(in_half(p0, n))] = __floats2half2_rn(d[ct][0], d[ct][1]);
                smp[half_of(p1, n) + pad(in_half(p1, n))] = __floats2half2_rn(d[ct][2], d[ct][3]);
            }
        }
    }

    __syncthreads();
    for (int k2 = threadIdx.x; k2 < n; k2 += THREADS_FQ) {
        const int q2 =
            self ? (c == 0 ? (int)((m2 - k2) % m2) : (int)(m2 - 1 - k2)) : (int)(m2 - 1 - k2);
        const float2 za = __half22float2(smp[pad(bin_pos(k2, s, lr))]);
        const float2 zb =
            __half22float2(self ? smp[pad(bin_pos(q2, s, lr))] : smp[n + pad(bin_pos(q2, s, lr))]);
        // Bin (c, k2): Z[k] = za, Z[m - k] = zb.
        {
            const size_t k = (size_t)k2 * m1 + c;
            const float er = 0.5f * (za.x + zb.x), ei = 0.5f * (za.y - zb.y);
            const float our = 0.5f * (za.y + zb.y), oui = 0.5f * (zb.x - za.x);
            const float2 w = twiddle_hilo(w_hi, w_lo, k);
            const float xr = scale * (er + w.x * our - w.y * oui);
            const float xi = scale * (ei + w.x * oui + w.y * our);
            const size_t kt = c * m2 + k2;
            out[kt] = encode_fused(xr, xi, gain, gains_t, stats_t, kt);
        }
        // Bin (m1 - c, q2): Z[k] = zb, Z[m - k] = za (the same pair, swapped).
        if (!self) {
            const size_t k = (size_t)q2 * m1 + rowB;
            const float er = 0.5f * (zb.x + za.x), ei = 0.5f * (zb.y - za.y);
            const float our = 0.5f * (zb.y + za.y), oui = 0.5f * (za.x - zb.x);
            const float2 w = twiddle_hilo(w_hi, w_lo, k);
            const float xr = scale * (er + w.x * our - w.y * oui);
            const float xi = scale * (ei + w.x * oui + w.y * our);
            const size_t kt = rowB * m2 + q2;
            out[kt] = encode_fused(xr, xi, gain, gains_t, stats_t, kt);
        }
    }
}

void gpu_tensor_fft2_quantize(const __half2* in, size_t m1, size_t m2, const float2* w_hi,
                              const float2* w_lo, float scale, float gain, const float* gains_t,
                              float* stats_t, uint8_t* out, cudaStream_t stream) {
    assert(m2 >= 256 && m2 <= 8192 && (m2 & (m2 - 1)) == 0 && m1 >= 2);
    const int smem = 2 * (int)m2 * sizeof(__half2);
    cudaFuncSetAttribute(tensor_fft2_quantize, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    tensor_fft2_quantize<<<(unsigned)(m1 / 2 + 1), THREADS_FQ, smem, stream>>>(
        in, m1, m2, w_hi, w_lo, scale, gain, gains_t, stats_t, out);
}

// Bytes [k1][k2] -> bin order [k2 * m1 + k1], 32x32 tiles.
__global__ static void bytes_transpose(const uint8_t* __restrict__ in, size_t m1, size_t m2,
                                       uint8_t* __restrict__ out) {
    __shared__ uint8_t tile[32][33];
    const size_t k10 = blockIdx.x * 32;
    const size_t k20 = blockIdx.y * 32;
    for (int l = threadIdx.y; l < 32; l += 8)
        tile[l][threadIdx.x] = in[(k10 + l) * m2 + k20 + threadIdx.x];
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8)
        out[(k20 + r) * m1 + k10 + threadIdx.x] = tile[threadIdx.x][r];
}

void gpu_bytes_transpose(const uint8_t* in, size_t m1, size_t m2, uint8_t* out,
                         cudaStream_t stream) {
    dim3 grid(m1 / 32, m2 / 32), block(32, 8);
    bytes_transpose<<<grid, block, 0, stream>>>(in, m1, m2, out);
}

// Floats in bin order [k2 * m1 + k1] -> transform-local [k1 * m2 + k2] (or back, by
// swapping the roles of m1 and m2).
__global__ static void reorder_f32(const float* __restrict__ in, size_t m1, size_t m2,
                                   float* __restrict__ out) {
    __shared__ float tile[32][33];
    const size_t k10 = blockIdx.x * 32;
    const size_t k20 = blockIdx.y * 32;
    for (int l = threadIdx.y; l < 32; l += 8)
        tile[l][threadIdx.x] = in[(k20 + l) * m1 + k10 + threadIdx.x];
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8)
        out[(k10 + threadIdx.x) * m2 + k20 + r] = tile[threadIdx.x][r];
}

void gpu_reorder_f32(const float* in, size_t m1, size_t m2, float* out, cudaStream_t stream) {
    dim3 grid(m1 / 32, m2 / 32), block(32, 8);
    reorder_f32<<<grid, block, 0, stream>>>(in, m1, m2, out);
}
