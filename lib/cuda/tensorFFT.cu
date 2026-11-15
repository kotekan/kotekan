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
