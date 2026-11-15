// Tensor-core batched FFT for the cudaSparkFEngine fp16 four-step path.

#ifndef TENSOR_FFT_HPP
#define TENSOR_FFT_HPP

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * Forward C2C DFT of length n on each of batch contiguous transforms:
 * out[b * n + k] = sum_j in[b * n + j] * exp(-2 pi i j k / n), unscaled, natural order in
 * and out — the same semantics as a batched forward cuFFT CUDA_C_16F plan. n must be a
 * power of two in 32..8192; in and out must be identical or non-overlapping.
 *
 * Radix-16 decimation-in-frequency stages run as m16n8k16 mma against the 16-point DFT
 * matrix in its real [[Re,-Im],[Im,Re]] form, with fp32 accumulators and fp32 twiddles
 * between stages; a leftover factor of 2, 4 or 8 is one more mma stage against a
 * block-diagonal DFT matrix. Each CTA stages 8192 contiguous elements (one transform, or
 * 8192 / n of them) through shared memory.
 */
void gpu_tensor_fft(const __half2* in, __half2* out, size_t n, size_t batch, cudaStream_t stream);

#endif // TENSOR_FFT_HPP
