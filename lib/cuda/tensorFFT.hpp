// Tensor-core batched FFT for the cudaSparkFEngine fp16 four-step path.

#ifndef TENSOR_FFT_HPP
#define TENSOR_FFT_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/// The longest transform gpu_tensor_fft runs in one CTA's shared memory.
constexpr size_t TENSOR_FFT_ELEMS = 8192;

/// How gpu_tensor_fft splits a length-n transform: log2(n / 8192) above 8192, else 0.
inline int tensor_fft_split(size_t n) {
    int split = 0;
    while ((TENSOR_FFT_ELEMS << split) < n)
        split++;
    return split;
}

/// The bin at position p of a gpu_tensor_fft output with the given split (identity for 0).
__host__ __device__ inline size_t tensor_fft_bin(size_t p, int split) {
    return split ? ((p & (TENSOR_FFT_ELEMS - 1)) << split) | (p / TENSOR_FFT_ELEMS) : p;
}

/// The position of bin k in a gpu_tensor_fft output with the given split.
__host__ __device__ inline size_t tensor_fft_pos(size_t k, int split) {
    return split ? (k & ((size_t(1) << split) - 1)) * TENSOR_FFT_ELEMS + (k >> split) : k;
}

/**
 * Forward C2C DFT of length n on each of batch contiguous transforms:
 * out[b * n + k] = sum_j in[b * n + j] * exp(-2 pi i j k / n), unscaled, natural order in
 * and out — the same semantics as a batched forward cuFFT CUDA_C_16F plan. n must be a
 * power of two in 32..32768; up to 8192, in and out must be identical or non-overlapping.
 *
 * Radix-16 decimation-in-frequency stages run as m16n8k16 mma against the 16-point DFT
 * matrix in its real [[Re,-Im],[Im,Re]] form, with fp32 accumulators and fp32 twiddles
 * between stages; a leftover factor of 2, 4 or 8 is one more mma stage against a
 * block-diagonal DFT matrix. Each CTA stages 8192 contiguous elements (one transform, or
 * 8192 / n of them) through shared memory.
 *
 * Longer transforms, n = 8192 << split, run on 2^split CTAs each, still in one pass: CTA c
 * computes output c of the first, radix-2^split, decimation-in-frequency stage and the
 * 8192-point FFT of that, which holds bins c, c + 2^split, c + 2 * 2^split, ... The bins
 * then land in split order, bin k at position tensor_fft_pos(k, split), and in and out
 * must not overlap.
 */
void gpu_tensor_fft(const __half2* in, __half2* out, size_t n, size_t batch, cudaStream_t stream);

/**
 * The same transform on fp32 data with TF32 tensor cores ("fp32 mode": m16n8k8 mma on
 * tf32-rounded operands, fp32 accumulators, fp32 storage between stages). Each mma input
 * rounds to TF32's 10-bit mantissa, but twiddles and inter-stage values stay fp32, so the
 * composite error sits well below the fp16 path's. Needs 66 KiB of dynamic shared memory
 * per CTA (one per SM); built for accuracy comparison, not tuned for speed.
 */
void gpu_tensor_fft_f32(const float2* in, float2* out, size_t n, size_t batch, cudaStream_t stream);

/**
 * The fused last pass of the four-step: runs the second batched FFT (length m2, batch m1)
 * AND the untangle + gain + quantize in one kernel by giving each CTA a conjugate pair of
 * transforms (rows k1 and m1 - k1), so the spectrum never returns to global memory. Bytes
 * land in transform-local [k1][k2] order in out; gpu_bytes_transpose puts them in bin
 * order. gains_t and stats_t, when given, use the same transform-local layout
 * (gpu_reorder_f32 converts a bin-ordered table, and converts back with m1 and m2
 * swapped). scale and the encoding match gpu_quantize_fourstep. Needs m2 in 256..8192
 * (power of two) and m1 >= 2.
 */
void gpu_tensor_fft2_quantize(const __half2* in, size_t m1, size_t m2, const float2* w_hi,
                              const float2* w_lo, float scale, float gain, const float* gains_t,
                              float* stats_t, uint8_t* out, cudaStream_t stream);

/// Bytes [k1][k2] (m1 rows of m2) to bin order [k2 * m1 + k1]. m1 and m2 must be multiples
/// of 128, and out 4-byte aligned.
void gpu_bytes_transpose(const uint8_t* in, size_t m1, size_t m2, uint8_t* out,
                         cudaStream_t stream);

/// Floats from bin order [k2 * m1 + k1] to transform-local [k1 * m2 + k2]; swap the m1 and
/// m2 arguments to convert back.
void gpu_reorder_f32(const float* in, size_t m1, size_t m2, float* out, cudaStream_t stream);

#endif // TENSOR_FFT_HPP
