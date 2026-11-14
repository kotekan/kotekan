// F-engine kernels: 12-bit packet gather/unpack and the real-FFT untangle + quantize.

#ifndef FENGINE_KERNELS_HPP
#define FENGINE_KERNELS_HPP

#include <cstddef>
#include <cstdint>
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

#endif // FENGINE_KERNELS_HPP
