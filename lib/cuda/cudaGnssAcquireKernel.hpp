#ifndef CUDA_GNSS_ACQUIRE_KERNEL_HPP
#define CUDA_GNSS_ACQUIRE_KERNEL_HPP

#include <cuda_runtime.h>

/**
 * @file
 * @brief GPU acquisition kernels: correlate front end, surface aggregate and peak reduction.
 *
 * The aggregate is the GPU version of gnss::aggregate_accumulate (gnssChannelizedAcquire.cpp),
 * which dominates a blind search on the CPU:
 *
 *     D(q,i) = sum_c P_c(q) e^{+i 2 pi f_c (i * s_step) / sph},   surf(q,i) += |D(q,i)|^2
 *
 * The literal sum costs nc * s_cols complex MACs per row (nc=79, s_cols=128 at CHORD's blind
 * dimensions, with n_dop=321, Mp=3125). When `s_cols * s_step == s_stored`, the ramp depends on
 * the channel only through `(f_c - f_0)/(sph/s_stored) mod s_cols`, up to a channel-independent
 * phase factor that |D|^2 drops. The sum is then an s_cols-point inverse DFT of the channels
 * folded into s_cols bins: an identity about the ramp, exact for any signal. The tables shrink
 * to nc bin indices and an s_cols/2 twiddle table (staged in shared memory), and the flop count
 * drops ~18x. GnssCudaAcquire runs this folded form over the channel-major layout
 * (@ref launch_aggregate_cdq).
 *
 * Precision: the CPU reference accumulates the surface in double, these kernels in float.
 */
namespace gnss_cuda {

/// Folded aggregate over the channel-major layout `P[nc][n_dop][Mp]`, cuFFT's natural inverse
/// output, accumulated into `surf` [n_dop][Mp][s_cols] float (zero it before window 0).
/// @c bin is the device [nc] int fold bins and @c twiddle the device [s_cols/2] complex
/// e^{+2 pi i j / s_cols}. cuFFT can write the channel-fastest layout only with
/// `ostride = nc`, i.e. 8-byte writes at a 632-byte stride, each in its own memory sector. Here
/// the transpose happens through a shared-memory tile on the read side instead, where it costs
/// no extra traffic.
/// Each block runs one thread per fine-lag column with (1.5*s_cols + 32*nc) float2 of dynamic
/// shared memory and no opt-in, so s_cols must be at most 1024 and that total at most 48 KB;
/// outside those limits the launch fails and the launcher does not report it.
void launch_aggregate_cdq(const float2* P, float* surf, const int* bin, const float2* twiddle,
                          int nc, int n_dop, int Mp, int s_cols, cudaStream_t stream);

/// Per-Doppler max, sum and argmax, written to host arrays of length `n_dop`; synchronizes
/// `stream`. The search needs each Doppler bin's own max over lag (the pair-sum plane selection,
/// and gnss::peak_from_reduction's sub-grid Doppler refine and peak_interp), and the global peak
/// and mean follow from the same reduction. `h_idx[d]` is the argmax within slice d, in
/// [0, cells_per_dop). `scratch` must hold @ref peak_dop_scratch_bytes bytes.
void launch_peak_dop(const float* surf, int n_dop, long cells_per_dop, float* h_max, double* h_sum,
                     long long* h_idx, void* scratch, cudaStream_t stream);

/// Bytes of scratch @ref launch_peak_dop needs.
size_t peak_dop_scratch_bytes(long cells_per_dop, int n_dop);

// ---------------------------------------------------------------------------------------------
// Correlate front end, so the whole chain stays on the device.
// ---------------------------------------------------------------------------------------------

/// Gather the covering channels out of the stage's [hop][n_chan] snapshot into per-channel
/// contiguous rows, zero-padded from the data window M out to the replica period Mp.
/// @param snap  device [M][n_chan_total] complex, exactly as GnssChannelizedSearch holds it
/// @param cov   device [nc] local channel indices (columns of @c snap)
/// @param out   device [nc][Mp] complex, zero beyond M
void launch_gather_pad(const float2* snap, const int* cov, float2* out, int n_chan_total, int M,
                       int Mp, int nc, cudaStream_t stream);

/// Materialise one secondary-code alignment of a PRN's replica:
///   R[c][m] = head[c][m] * sgn0[m] + tail[c][m] * sgn1[m]
/// The sign pair depends on the hop and the alignment but not the channel, so the caller passes
/// it as two [Mp] tables.
void launch_materialise(const float2* head, const float2* tail, const float* sgn0,
                        const float* sgn1, float2* R, int nc, int Mp, cudaStream_t stream);

/// Doppler wipe and replica conjugate multiply, in the frequency domain:
///   X[(c*n_dop + d)*Mp + k] = A[c][(k + bshift[d]) mod Mp] * conj(B[c][k]) / Mp
/// with every bshift[d] in [0, Mp).
///
/// The cyclic shift is exact: wiping the data by f_d multiplies hop m by
/// e^{-i 2 pi f_d m sph/Fs}, a pure cyclic shift of its spectrum whenever f_d is a whole number
/// of this transform's bins (Fs/(Mp*sph) = 62.5 Hz at CHORD). One forward transform of the data
/// then serves every Doppler trial. GnssCudaAcquire::set_doppler_grid refuses any other grid;
/// sub-grid Doppler comes from the refine in gnss::peak_from_reduction.
///
/// The 1/Mp is cuFFT's missing inverse normalization, applied here because |D|^2 is quadratic
/// in P and scaling later would need a second pass over the surface.
///
/// The layout is channel-major, Doppler-minor, so the inverse transform runs as one contiguous
/// batch over Doppler per channel and leaves P[nc][n_dop][Mp] for @ref launch_aggregate_cdq.
void launch_dopmul(const float2* A, const float2* B, const int* bshift, float2* X, int nc,
                   int n_dop, int Mp, cudaStream_t stream);

} // namespace gnss_cuda

#endif // CUDA_GNSS_ACQUIRE_KERNEL_HPP
