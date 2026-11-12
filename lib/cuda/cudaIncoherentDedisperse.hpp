// Incoherent-beam dedispersion kernels.
//
// Search the feed-summed power (S012tilde, from the first-stage RFI statistics) for dispersed
// pulses. Three kernels run on each chunk of `num_times` new samples:
//
//  1. normalize: turn the power of each frequency into unit-variance deviations from a running
//     mean, and drop samples without data or beyond `clip_sigmas`.
//  2. zero_dm: subtract, from each time sample, its mean over the frequencies, and store the result
//     in the history. This removes undispersed broadband signals.
//  3. dedisperse: for each dispersion measure (DM), sum the history along the dispersion delays,
//     and find the most significant sum over a set of pulse widths and sub-bands.
//
// See cudaIncoherentDedisperse.cu for the kernel implementations.

#ifndef CUDA_INCOHERENT_DEDISPERSE_HPP
#define CUDA_INCOHERENT_DEDISPERSE_HPP

#include <cstddef>        // for ptrdiff_t
#include <cstdint>        // for int32_t, int64_t, uint32_t, uint64_t
#include <driver_types.h> // for cudaStream_t

// These are free functions (not in a namespace), with a CPU reference for each launcher; this
// mirrors the launch_/cpu_ pairing in cudaPLMaskFRB1.hpp.

// The dedispersed sums are kept in this many sub-bands, each an equal share of the frequency axis.
constexpr int incoherent_dedisperse_num_subbands = 4;
// Pulse widths searched are 1, 2, 4, ... samples, up to this one.
constexpr int incoherent_dedisperse_max_width = 16;
// Sub-band sets searched: all four sub-bands, each half, and each single sub-band.
constexpr int incoherent_dedisperse_num_templates = 7;
// Largest `num_times` the dedisperse kernel accepts.
constexpr int incoherent_dedisperse_max_times = 256;

// Normalize a chunk of feed-summed power.
//
// Array layouts (C index ordering):
//     uint64 S012tilde[num_times][num_frequencies][3]   (input; S0 = sample count, S1 = power sum)
//     float  z        [num_times][num_frequencies]      (output)
//     float  mean, variance[num_frequencies]            (running statistics, updated)
//     uint32 count    [num_frequencies]                 (samples seen, saturating; updated)
//
// The power of a sample is `x = S1 / S0`. The running statistics follow `x` with weight
// `max(alpha, 1 / count)` per sample, with the deviation limited to `clip_sigmas` standard
// deviations so that outliers have a bounded effect. The output is `(x - mean) / sqrt(variance)`,
// or NaN for a sample with `S0 == 0`, beyond `clip_sigmas`, or among the first `1 / alpha` samples
// of its frequency. `mean`, `variance` and `count` must start at zero.
void launch_incoherent_normalize(float* z, float* mean, float* variance, std::uint32_t* count,
                                 const std::uint64_t* S012tilde, std::ptrdiff_t num_frequencies,
                                 std::ptrdiff_t num_times, float alpha, float clip_sigmas,
                                 cudaStream_t stream);
void cpu_incoherent_normalize(float* z, float* mean, float* variance, std::uint32_t* count,
                              const std::uint64_t* S012tilde, std::ptrdiff_t num_frequencies,
                              std::ptrdiff_t num_times, float alpha, float clip_sigmas);

// Subtract the mean over frequencies from each time sample, and store the chunk in the history.
//
// Array layouts:
//     float z      [num_times][num_frequencies]      (input)
//     float history[num_frequencies][history_size]   (output; a ring in time)
//
// Sample `time_begin + t` is stored at history position `(time_begin + t) % history_size`. The
// mean is taken over the samples that are not NaN, and NaN samples are stored as zero.
void launch_incoherent_zero_dm(float* history, const float* z, std::ptrdiff_t num_frequencies,
                               std::ptrdiff_t num_times, std::ptrdiff_t history_size,
                               std::int64_t time_begin, cudaStream_t stream);
void cpu_incoherent_zero_dm(float* history, const float* z, std::ptrdiff_t num_frequencies,
                            std::ptrdiff_t num_times, std::ptrdiff_t history_size,
                            std::int64_t time_begin);

// Dedisperse the newly complete samples of each DM, and find the most significant one.
//
// Array layouts:
//     float history  [num_frequencies][history_size]   (input; see above)
//     int32 delay    [num_dms][num_frequencies]        (input; dispersion delay in samples, >= 0)
//     int32 max_delay[num_dms]                         (input; largest delay of each DM)
//     float weight   [num_frequencies]                 (input; 0 excludes a frequency, else 1)
//     float sums     [num_dms][num_subbands][sums_size] (dedispersed sums, a ring in time; updated)
//     float best_snr [num_dms]                         (output)
//     int32 best_time, best_width, best_template[num_dms]   (output)
//
// After the chunk `[time_begin, time_begin + num_times)` is in the history, the samples of DM `d`
// that arrived at the reference frequency (the one with delay 0) at times
// `time_begin - max_delay[d] + [0, num_times)` are complete. For each such time `t0` and each
// sub-band, the kernel stores the sum over the sub-band's frequencies `f` of
// `weight[f] * history[f][t0 + delay[d][f]]` in `sums`, at position `t0 % sums_size`.
//
// For each of these `t0`, each width `w` and each template, the significance is the sum over the
// template's sub-bands and the `w` samples ending at `t0`, divided by the square root of the number
// of samples summed. The kernel returns the largest significance for each DM in `best_snr`, with
// `t0 - (time_begin - max_delay[d])` in `best_time`, the width in `best_width`, and the template in
// `best_template` (0: all sub-bands; 1, 2: lower and upper half of the frequency axis; 3 to 6: a
// single sub-band).
//
// Requires `max_delay[d] + num_times <= history_size`,
// `num_times + incoherent_dedisperse_max_width <= sums_size + 1`, and
// `num_times <= incoherent_dedisperse_max_times`. `history` and `sums` must start at zero.
void launch_incoherent_dedisperse(float* best_snr, std::int32_t* best_time,
                                  std::int32_t* best_width, std::int32_t* best_template,
                                  float* sums, const float* history, const std::int32_t* delay,
                                  const std::int32_t* max_delay, const float* weight,
                                  std::ptrdiff_t num_dms, std::ptrdiff_t num_frequencies,
                                  std::ptrdiff_t num_times, std::ptrdiff_t history_size,
                                  std::ptrdiff_t sums_size, std::int64_t time_begin,
                                  cudaStream_t stream);
void cpu_incoherent_dedisperse(float* best_snr, std::int32_t* best_time, std::int32_t* best_width,
                               std::int32_t* best_template, float* sums, const float* history,
                               const std::int32_t* delay, const std::int32_t* max_delay,
                               const float* weight, std::ptrdiff_t num_dms,
                               std::ptrdiff_t num_frequencies, std::ptrdiff_t num_times,
                               std::ptrdiff_t history_size, std::ptrdiff_t sums_size,
                               std::int64_t time_begin);

#endif // CUDA_INCOHERENT_DEDISPERSE_HPP
