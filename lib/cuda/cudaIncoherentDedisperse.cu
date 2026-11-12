// Incoherent-beam dedispersion kernels.
//
// See cudaIncoherentDedisperse.hpp for the public launcher API. The GPU kernels and the CPU
// references call the same per-element functions, so that they differ only in rounding.

#include "cudaIncoherentDedisperse.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace {

    constexpr int num_subbands = incoherent_dedisperse_num_subbands;
    constexpr int max_width = incoherent_dedisperse_max_width;
    constexpr int num_templates = incoherent_dedisperse_num_templates;
    constexpr int max_times = incoherent_dedisperse_max_times;

    // First sub-band and number of sub-bands of each template
    __host__ __device__ inline int template_first(const int tpl) {
        return tpl == 0 ? 0 : tpl <= 2 ? 2 * (tpl - 1) : tpl - 3;
    }
    __host__ __device__ inline int template_count(const int tpl) {
        return tpl == 0 ? 4 : tpl <= 2 ? 2 : 1;
    }

    // Position of time `time` in a ring of `size` elements; `time` may be negative
    __host__ __device__ inline std::ptrdiff_t ring_position(const std::int64_t time,
                                                            const std::ptrdiff_t size) {
        const std::int64_t r = time % size;
        return r < 0 ? r + size : r;
    }

    // Normalize all time samples of frequency `f`
    __host__ __device__ inline void
    normalize_frequency(float* __restrict__ const z, float* __restrict__ const mean,
                        float* __restrict__ const variance, std::uint32_t* __restrict__ const count,
                        const std::uint64_t* __restrict__ const S012tilde,
                        const std::ptrdiff_t num_frequencies, const std::ptrdiff_t num_times,
                        const float alpha, const float clip_sigmas, const std::ptrdiff_t f) {
        const std::uint32_t warmup = std::uint32_t(ceilf(1 / alpha));
        float m = mean[f];
        float v = variance[f];
        std::uint32_t c = count[f];
        for (std::ptrdiff_t t = 0; t < num_times; ++t) {
            const std::ptrdiff_t tf = f + num_frequencies * t;
            const std::uint64_t s0 = S012tilde[3 * tf + 0];
            const std::uint64_t s1 = S012tilde[3 * tf + 1];
            if (s0 == 0) {
                z[tf] = NAN;
                continue;
            }
            const float x = float(double(s1) / double(s0));
            const bool warm = c >= warmup;
            if (!warm)
                ++c;
            const float a = fmaxf(alpha, 1.0f / float(c));
            const float sigma = sqrtf(v);
            float diff = x - m;
            bool keep = false;
            if (warm) {
                const float limit = clip_sigmas * sigma;
                keep = sigma > 0 && fabsf(diff) <= limit;
                z[tf] = keep ? diff / sigma : NAN;
                diff = fminf(fmaxf(diff, -limit), limit);
            } else {
                z[tf] = NAN;
            }
            // Exponentially weighted mean and variance
            const float incr = a * diff;
            m += incr;
            v = (1 - a) * (v + diff * incr);
        }
        mean[f] = m;
        variance[f] = v;
        count[f] = c;
    }

    // Subtract the mean over frequencies from time sample `t`, and store it in the history
    __host__ __device__ inline void
    zero_dm_time(float* __restrict__ const history, const float* __restrict__ const z,
                 const std::ptrdiff_t num_frequencies, const std::ptrdiff_t history_size,
                 const std::int64_t time_begin, const std::ptrdiff_t t) {
        float sum = 0;
        int num = 0;
        for (std::ptrdiff_t f = 0; f < num_frequencies; ++f) {
            const float zf = z[f + num_frequencies * t];
            if (zf == zf) { // not NaN
                sum += zf;
                ++num;
            }
        }
        const float zmean = num > 0 ? sum / float(num) : 0;
        const std::ptrdiff_t pos = ring_position(time_begin + t, history_size);
        for (std::ptrdiff_t f = 0; f < num_frequencies; ++f) {
            const float zf = z[f + num_frequencies * t];
            history[pos + history_size * f] = zf == zf ? zf - zmean : 0;
        }
    }

    // Reference-frequency time of output sample `j` of DM `d`
    __host__ __device__ inline std::int64_t output_time(const std::int32_t* const max_delay,
                                                        const std::int64_t time_begin,
                                                        const std::ptrdiff_t d,
                                                        const std::ptrdiff_t j) {
        return time_begin - max_delay[d] + j;
    }

    // Dedisperse output sample `j` of DM `d` into `sums`. Returns the number of frequencies
    // summed in each sub-band in `subband_num`.
    __host__ __device__ inline void
    dedisperse_sample(float* __restrict__ const sums, float* const subband_num,
                      const float* __restrict__ const history,
                      const std::int32_t* __restrict__ const delay,
                      const std::int32_t* __restrict__ const max_delay,
                      const float* __restrict__ const weight, const std::ptrdiff_t num_frequencies,
                      const std::ptrdiff_t history_size, const std::ptrdiff_t sums_size,
                      const std::int64_t time_begin, const std::ptrdiff_t d,
                      const std::ptrdiff_t j) {
        const std::int64_t t0 = output_time(max_delay, time_begin, d, j);
        const std::ptrdiff_t sums_pos = ring_position(t0, sums_size);
        // All positions are non-negative after this offset, since delay <= max_delay
        const std::ptrdiff_t pos0 = ring_position(t0, history_size);
        for (int b = 0; b < num_subbands; ++b) {
            const std::ptrdiff_t fbegin = b * num_frequencies / num_subbands;
            const std::ptrdiff_t fend = (b + 1) * num_frequencies / num_subbands;
            float sum = 0;
            float num = 0;
            for (std::ptrdiff_t f = fbegin; f < fend; ++f) {
                std::ptrdiff_t pos = pos0 + delay[f + num_frequencies * d];
                pos = pos >= history_size ? pos - history_size : pos;
                sum += weight[f] * history[pos + history_size * f];
                num += weight[f];
            }
            sums[sums_pos + sums_size * (b + num_subbands * d)] = sum;
            subband_num[b] = num;
        }
    }

    // Find the most significant width and template of output sample `j` of DM `d`
    __host__ __device__ inline void
    detect_sample(float& snr, int& width, int& tpl, const float* __restrict__ const sums,
                  const float* const subband_num, const std::int32_t* __restrict__ const max_delay,
                  const std::ptrdiff_t sums_size, const std::int64_t time_begin,
                  const std::ptrdiff_t d, const std::ptrdiff_t j) {
        const std::int64_t t0 = output_time(max_delay, time_begin, d, j);
        snr = 0;
        width = 0;
        tpl = 0;
        for (int p = 0; p < num_templates; ++p) {
            const int bfirst = template_first(p);
            const int bcount = template_count(p);
            float num = 0;
            for (int b = bfirst; b < bfirst + bcount; ++b)
                num += subband_num[b];
            if (num == 0)
                continue;
            float sum = 0;
            int k = 0;
            for (int w = 1; w <= max_width; w *= 2) {
                for (; k < w; ++k) {
                    const std::ptrdiff_t pos = ring_position(t0 - k, sums_size);
                    for (int b = bfirst; b < bfirst + bcount; ++b)
                        sum += sums[pos + sums_size * (b + num_subbands * d)];
                }
                const float s = sum / sqrtf(float(w) * num);
                if (s > snr) {
                    snr = s;
                    width = w;
                    tpl = p;
                }
            }
        }
    }

    __global__ void incoherent_normalize(float* __restrict__ const z, float* __restrict__ const mean,
                                         float* __restrict__ const variance,
                                         std::uint32_t* __restrict__ const count,
                                         const std::uint64_t* __restrict__ const S012tilde,
                                         const std::ptrdiff_t num_frequencies,
                                         const std::ptrdiff_t num_times, const float alpha,
                                         const float clip_sigmas) {
        const std::ptrdiff_t f = threadIdx.x + std::ptrdiff_t(blockDim.x) * blockIdx.x;
        if (f < num_frequencies)
            normalize_frequency(z, mean, variance, count, S012tilde, num_frequencies, num_times,
                                alpha, clip_sigmas, f);
    }

    __global__ void incoherent_zero_dm(float* __restrict__ const history,
                                       const float* __restrict__ const z,
                                       const std::ptrdiff_t num_frequencies,
                                       const std::ptrdiff_t num_times,
                                       const std::ptrdiff_t history_size,
                                       const std::int64_t time_begin) {
        const std::ptrdiff_t t = threadIdx.x + std::ptrdiff_t(blockDim.x) * blockIdx.x;
        if (t < num_times)
            zero_dm_time(history, z, num_frequencies, history_size, time_begin, t);
    }

    // One block per DM, one thread per output sample
    __global__ void incoherent_dedisperse(
        float* __restrict__ const best_snr, std::int32_t* __restrict__ const best_time,
        std::int32_t* __restrict__ const best_width, std::int32_t* __restrict__ const best_template,
        float* __restrict__ const sums, const float* __restrict__ const history,
        const std::int32_t* __restrict__ const delay,
        const std::int32_t* __restrict__ const max_delay, const float* __restrict__ const weight,
        const std::ptrdiff_t num_frequencies, const std::ptrdiff_t num_times,
        const std::ptrdiff_t history_size, const std::ptrdiff_t sums_size,
        const std::int64_t time_begin) {
        __shared__ float shared_snr[max_times];
        __shared__ int shared_width[max_times];
        __shared__ int shared_template[max_times];

        const std::ptrdiff_t d = blockIdx.x;
        const std::ptrdiff_t j = threadIdx.x;

        float subband_num[num_subbands];
        dedisperse_sample(sums, subband_num, history, delay, max_delay, weight, num_frequencies,
                          history_size, sums_size, time_begin, d, j);
        // The widths read the sums of the other threads
        __syncthreads();

        detect_sample(shared_snr[j], shared_width[j], shared_template[j], sums, subband_num,
                      max_delay, sums_size, time_begin, d, j);
        __syncthreads();

        if (j == 0) {
            int best = 0;
            for (int i = 1; i < num_times; ++i)
                if (shared_snr[i] > shared_snr[best])
                    best = i;
            best_snr[d] = shared_snr[best];
            best_time[d] = best;
            best_width[d] = shared_width[best];
            best_template[d] = shared_template[best];
        }
    }

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Launchers (externally visible)

void launch_incoherent_normalize(float* const z, float* const mean, float* const variance,
                                 std::uint32_t* const count, const std::uint64_t* const S012tilde,
                                 const std::ptrdiff_t num_frequencies,
                                 const std::ptrdiff_t num_times, const float alpha,
                                 const float clip_sigmas, const cudaStream_t stream) {
    assert(z && mean && variance && count && S012tilde);
    assert(num_frequencies > 0 && num_times > 0);
    assert(alpha > 0 && alpha <= 1 && clip_sigmas > 0);

    constexpr int threads = 128;
    const dim3 nblocks((num_frequencies + threads - 1) / threads, 1, 1);
    incoherent_normalize<<<nblocks, dim3(threads, 1, 1), 0, stream>>>(
        z, mean, variance, count, S012tilde, num_frequencies, num_times, alpha, clip_sigmas);
}

void launch_incoherent_zero_dm(float* const history, const float* const z,
                               const std::ptrdiff_t num_frequencies, const std::ptrdiff_t num_times,
                               const std::ptrdiff_t history_size, const std::int64_t time_begin,
                               const cudaStream_t stream) {
    assert(history && z);
    assert(num_frequencies > 0 && num_times > 0 && num_times <= history_size);

    constexpr int threads = 32;
    const dim3 nblocks((num_times + threads - 1) / threads, 1, 1);
    incoherent_zero_dm<<<nblocks, dim3(threads, 1, 1), 0, stream>>>(
        history, z, num_frequencies, num_times, history_size, time_begin);
}

void launch_incoherent_dedisperse(float* const best_snr, std::int32_t* const best_time,
                                  std::int32_t* const best_width, std::int32_t* const best_template,
                                  float* const sums, const float* const history,
                                  const std::int32_t* const delay,
                                  const std::int32_t* const max_delay, const float* const weight,
                                  const std::ptrdiff_t num_dms, const std::ptrdiff_t num_frequencies,
                                  const std::ptrdiff_t num_times, const std::ptrdiff_t history_size,
                                  const std::ptrdiff_t sums_size, const std::int64_t time_begin,
                                  const cudaStream_t stream) {
    assert(best_snr && best_time && best_width && best_template);
    assert(sums && history && delay && max_delay && weight);
    assert(num_dms > 0 && num_frequencies >= num_subbands);
    assert(num_times > 0 && num_times <= max_times);
    assert(num_times <= history_size);
    assert(num_times + max_width <= sums_size + 1);

    incoherent_dedisperse<<<dim3(num_dms, 1, 1), dim3(num_times, 1, 1), 0, stream>>>(
        best_snr, best_time, best_width, best_template, sums, history, delay, max_delay, weight,
        num_frequencies, num_times, history_size, sums_size, time_begin);
}

////////////////////////////////////////////////////////////////////////////////
// CPU references

void cpu_incoherent_normalize(float* const z, float* const mean, float* const variance,
                              std::uint32_t* const count, const std::uint64_t* const S012tilde,
                              const std::ptrdiff_t num_frequencies, const std::ptrdiff_t num_times,
                              const float alpha, const float clip_sigmas) {
    for (std::ptrdiff_t f = 0; f < num_frequencies; ++f)
        normalize_frequency(z, mean, variance, count, S012tilde, num_frequencies, num_times, alpha,
                            clip_sigmas, f);
}

void cpu_incoherent_zero_dm(float* const history, const float* const z,
                            const std::ptrdiff_t num_frequencies, const std::ptrdiff_t num_times,
                            const std::ptrdiff_t history_size, const std::int64_t time_begin) {
    for (std::ptrdiff_t t = 0; t < num_times; ++t)
        zero_dm_time(history, z, num_frequencies, history_size, time_begin, t);
}

void cpu_incoherent_dedisperse(float* const best_snr, std::int32_t* const best_time,
                               std::int32_t* const best_width, std::int32_t* const best_template,
                               float* const sums, const float* const history,
                               const std::int32_t* const delay, const std::int32_t* const max_delay,
                               const float* const weight, const std::ptrdiff_t num_dms,
                               const std::ptrdiff_t num_frequencies, const std::ptrdiff_t num_times,
                               const std::ptrdiff_t history_size, const std::ptrdiff_t sums_size,
                               const std::int64_t time_begin) {
    std::vector<float> subband_num(num_subbands);
    for (std::ptrdiff_t d = 0; d < num_dms; ++d) {
        for (std::ptrdiff_t j = 0; j < num_times; ++j)
            dedisperse_sample(sums, subband_num.data(), history, delay, max_delay, weight,
                              num_frequencies, history_size, sums_size, time_begin, d, j);
        best_snr[d] = 0;
        best_time[d] = 0;
        best_width[d] = 0;
        best_template[d] = 0;
        for (std::ptrdiff_t j = 0; j < num_times; ++j) {
            float snr;
            int width, tpl;
            detect_sample(snr, width, tpl, sums, subband_num.data(), max_delay, sums_size,
                          time_begin, d, j);
            if (j == 0 || snr > best_snr[d]) {
                best_snr[d] = snr;
                best_time[d] = j;
                best_width[d] = width;
                best_template[d] = tpl;
            }
        }
    }
}
