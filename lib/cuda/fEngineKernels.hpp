// F-engine kernels: 12-bit packet gather/unpack and the real-FFT untangle + quantize.

#ifndef FENGINE_KERNELS_HPP
#define FENGINE_KERNELS_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * Gathers the window's packets through the lookup table and unpacks the 12-bit samples to
 * float, (raw - 2048) / 2048. lut holds num_packets device-readable payload-slot addresses
 * (see fftWindowJob.hpp); out gets num_packets * samples_per_packet floats in time order.
 */
void gpu_unpack12(const uint64_t* lut, int num_packets, int samples_per_packet, float* out,
                  cudaStream_t stream);

/**
 * Turns the M-point complex FFT of the real-as-complex packed window into the real spectrum
 * and quantizes it: X[k] = Ze + W^k * Zo per the standard untangle (fft_length = 2 * m,
 * W = exp(-2 pi i / fft_length)), bins 0..m-1 (DC kept, Nyquist dropped), byte k =
 * offset-encoded 4+4 of scale * X[k] (round to nearest even, clamp +-7, real in the high
 * nibble). The twiddle W^k is w_hi[k >> 12] * w_lo[k & 4095]: w_lo[j] = W^j (4096 entries),
 * w_hi[j] = W^(4096 j) ((m + 4095) / 4096 entries), both computed on the host in double.
 * out may be host memory (coherent on the DGX Spark); scale folds the 2 / fft_length
 * spectrum convention and the gain together.
 */
void gpu_untangle_quantize(const float2* spec, size_t m, const float2* w_hi, const float2* w_lo,
                           float scale, uint8_t* out, cudaStream_t stream);

// The fp16 four-step path: m = m1 * m2 (both multiples of 32), z viewed as A[a][b] with
// time index a * m2 + b. The twiddle tables of all three kernels index roots of unity of
// order fft_length = 2 * m: w_lo[j] = W^j with W = exp(-2 pi i / fft_length), 4096 entries,
// and w_hi[j] = W^(4096 j), (fft_length + 4095) / 4096 entries.

/**
 * Gather + unpack as gpu_unpack12, but to fp16 complex pairs z[t] = x[2t] + i x[2t+1]
 * stored TRANSPOSED: out[b * m1 + a] = z[a * m2 + b], so the first batched FFT (length m1,
 * batch m2) runs contiguous. samples_per_packet must be a multiple of 64.
 */
void gpu_unpack12_fp16_t(const uint64_t* lut, int samples_per_packet, size_t m1, size_t m2,
                         __half2* out, cudaStream_t stream);

/**
 * The four-step middle: in[b][k1] (the first FFT's output, m2 rows of m1) becomes
 * out[k1][b] = in[b][k1] * W^(2 * ((b * k1) mod m)) / m1 — the W_m^(b k1) twiddle and the
 * 1/m1 scaling, transposed so the second batched FFT (length m2, batch m1) runs contiguous.
 */
void gpu_twiddle_transpose(const __half2* in, size_t m1, size_t m2, const float2* w_hi,
                           const float2* w_lo, float inv_m1, __half2* out, cudaStream_t stream);

/**
 * Untangle + quantize for the four-step output: spec[k1][k2] (m1 rows of m2) holds
 * Z[k2 * m1 + k1]; bin k's conjugate partner Z[m - k] sits at (m1 - k1, m2 - 1 - k2), or in
 * row 0 at (0, (m2 - k2) mod m2). The untangle, scaling and 4+4 encoding match
 * gpu_untangle_quantize; scale must also fold in the m1 the twiddle pass divided out.
 */
void gpu_quantize_fourstep(const __half2* spec, size_t m1, size_t m2, const float2* w_hi,
                           const float2* w_lo, float scale, uint8_t* out, cudaStream_t stream);

#endif // FENGINE_KERNELS_HPP
