#include "DataType.hpp"
#include "cudaQuantizeKernel8Chime.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////

// Global constants

// Input array layout
static constexpr int in_ntimes = 256;
static constexpr int in_nfreqs = 256; // 16 coarse channels, upchannelized by 16
static constexpr int in_nbeams = 1024;

// Output array layout
// The output is quantized in "chunks". Each chunk has a different offset and scale.
static constexpr int out_ntimes_chunk = 16;
static constexpr int out_nfreqs_chunk = 16;
// Chunks are combined into packets.
static constexpr int out_nfreqs_packet = 4;
static constexpr int out_nbeams_packet = 8;
// There are several packets.
static constexpr int out_ntimes_outer = 16;
static constexpr int out_nfreqs_outer = 4;
static constexpr int out_nbeams_outer = 128;

static_assert(out_ntimes_chunk * out_ntimes_outer == in_ntimes);
static_assert(out_nfreqs_chunk * out_nfreqs_packet * out_nfreqs_outer == in_nfreqs);
static_assert(out_nbeams_packet * out_nbeams_outer == in_nbeams);

// For efficiency we proceess several chunks at once on the GPU.
static constexpr int nchunks = 32;

static_assert(out_nbeams_outer % nchunks == 0);

////////////////////////////////////////////////////////////////////////////////

// Helper types and functions

// Load 4 `float` values from memory simultaneously. This is more efficient than loading each
// `float` separately.
__device__ static inline float4 load_float4(const float* __restrict__ const ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// Calculate the maximum but do not ignore not-a-numbers
__device__ __host__ static inline float sane_fmax(const float x, const float y) {
    using std::fmax, std::isnan;
    if (isnan(x) || isnan(y))
        return 0.0f / 0.0f;
    return fmax(x, y);
}

// Calculate the minimum but do not ignore not-a-numbers
__device__ __host__ static inline float sane_fmin(const float x, const float y) {
    using std::fmin, std::isnan;
    if (isnan(x) || isnan(y))
        return 0.0f / 0.0f;
    return fmin(x, y);
}

// Round to nearest integer, avoiding all the implementation-defined cases
__device__ __host__ static inline int sane_lrint(const float x) {
    using std::isnan, std::lrint;
    if (x < INT_MIN)
        return INT_MIN;
    if (x > INT_MAX)
        return INT_MAX;
    if (isnan(x))
        return 0; // we could mask to 0 instead
    return int(lrint(x));
}

////////////////////////////////////////////////////////////////////////////////

// Kernels (called by the drivers below)

// Quantize a "chunk". This is a CPU implementation that is
// straightforward and thus "obviously correct". If the GPU version
// produces a different result, then the GPU version is wrong.
void cpu_quantize8chime_chunk(const float* __restrict__ const input,
                              __half* __restrict__ const outputf,
                              std::uint8_t* __restrict__ const outputi) {
    using std::isfinite, std::max, std::min;

    // Find the minimum and maximum of all input values.
    float minval = +1.0f / 0.0f, maxval = -1.0f / 0.0f;
    for (int freq = 0; freq < out_nfreqs_chunk; ++freq) {
        for (int time = 0; time < out_ntimes_chunk; ++time) {
            const int in_ind = time + in_ntimes * freq;
            const float x = input[in_ind];
            minval = sane_fmin(minval, x);
            maxval = sane_fmax(maxval, x);
        }
    }
    // Calculate offset and scale. Ensure the scale is nonzero.
    // There are 254 possible uint8 values since we don't use 0 or 255.
    // Values are reconstructed as `offset + scale * n`, with `uint8 n`.
    constexpr int outi_min = 1;
    constexpr int outi_max = 254;
    constexpr float outf_min = outi_min - 0.5f;
    constexpr float outf_max = outi_max + 0.5f;
    // We want this mapping:
    //     minval => outf_min
    //     maxval => outf_max
    const float outf_range = outf_max - outf_min;
    const float in_range = maxval - minval;

    float scale = in_range / outf_range;
    float offset = minval - outf_min * scale;
    __half offseth = __float2half(offset);
    __half scaleh = __float2half(scale);

    // Avoid non-finite numbers
    if (!isfinite(__half2float(offseth)) || !isfinite(__half2float(scaleh))) {
        offset = 0.0f;
        scale = 0.0f;
        offseth = __float2half(offset);
        scaleh = __float2half(scale);
    }

    // Store offset and scale in the output array
    outputf[0] = __float2half(offset);
    outputf[1] = __float2half(scale);

    // We encode values as unsigned 8-bit integers.
    for (int freq = 0; freq < out_nfreqs_chunk; ++freq) {
        for (int time = 0; time < out_ntimes_chunk; ++time) {
            const int in_ind = time + in_ntimes * freq;
            const int out_ind = time + out_ntimes_chunk * freq;
            // Get value
            const float x = input[in_ind];
            // Scale
            const float y = (x - offset) / scale;
            // Round
            const int j = sane_lrint(y);
            // Clamp; use 0 for nan
            const int k = isnan(y) ? 0 : max(outi_min, min(outi_max, j));
            // Store
            outputi[out_ind] = k;
        }
    }
}

// Quantize a set of "chunk"s. This is an efficient GPU implementation.
__global__ void gpu_quantize8chime_chunks(const float* __restrict__ const input,
                                          __half* __restrict__ const outputf,
                                          std::uint8_t* __restrict__ const outputi) {
    using std::isfinite, std::max, std::min;

    // Find our position in the arrays
    const int thread = threadIdx.x;
    const int time_outer = blockIdx.x;
    const int freq_packet = blockIdx.y % out_nfreqs_packet;
    const int freq_outer = blockIdx.y / out_nfreqs_packet;
    const int beam_packet = blockIdx.z % out_nbeams_packet;
    const int beam_outer0 = nchunks * (blockIdx.z / out_nbeams_packet); // not adding threadIdx.x

    // Calculate input indices
    const auto time = [&](int time_chunk, int time_outer) {
        return time_chunk + out_ntimes_chunk * time_outer;
    };
    const auto freq = [&](int freq_chunk, int freq_packet, int freq_outer) {
        return freq_chunk + out_nfreqs_chunk * (freq_packet + out_nfreqs_packet * freq_outer);
    };
    const auto beam = [&](int beam_packet, int beam_outer) {
        return beam_packet + out_nbeams_packet * beam_outer;
    };

    const auto input_offset = [&](int time, int freq, int beam) {
        return time + in_ntimes * (freq + in_nfreqs * beam);
    };

    // Calculate output indices
    const auto outputf_offset = [&](int freq_packet, int beam_packet, int time_outer,
                                    int freq_outer, int beam_outer) {
        return freq_packet
               + out_nfreqs_packet
                     * (beam_packet
                        + out_nbeams_packet
                              * (time_outer
                                 + out_ntimes_outer
                                       * (freq_outer + out_nfreqs_outer * beam_outer)));
    };
    const auto outputi_offset = [&](int time_chunk, int freq_chunk, int freq_packet,
                                    int beam_packet, int time_outer, int freq_outer,
                                    int beam_outer) {
        return time_chunk
               + out_ntimes_chunk
                     * (freq_chunk
                        + out_nfreqs_chunk
                              * outputf_offset(freq_packet, beam_packet, time_outer, freq_outer,
                                               beam_outer));
    };

    // We store offset and scale only after finishing the loop over all chunks. Each thread stores
    // one offset/scale pair.
    __half2 outf;
    // Loop over all beams we're handling
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const int beam_outer = beam_outer0 + chunk;

        // Load all values in this chunk. We load all values at once in the beginning.
        // There are 256 values and 32 threads, so each thread loads 8 values.
        // We load these 8 values in 2 batches of 4 times each.
        // We space these batches 4 * 32 = 128 elements apart to be cache-efficient.
        const int in_time_chunk = 4 * threadIdx.x % 4; // 0...15 in steps of 4
        const int in_freq_chunk = threadIdx.x / 4;     // 0...7
        const int input_offset_0123 = input_offset(time(in_time_chunk, time_outer),
                                                   freq(in_freq_chunk + 0, freq_packet, freq_outer),
                                                   beam(beam_packet, beam_outer));
        const int input_offset_4567 = input_offset(time(in_time_chunk, time_outer),
                                                   freq(in_freq_chunk + 8, freq_packet, freq_outer),
                                                   beam(beam_packet, beam_outer));

        const float4 xs0123 = load_float4(input + input_offset_0123);
        const float4 xs4567 = load_float4(input + input_offset_4567);
        const float x0 = xs0123.x;
        const float x1 = xs0123.y;
        const float x2 = xs0123.z;
        const float x3 = xs0123.w;
        const float x4 = xs4567.x;
        const float x5 = xs4567.y;
        const float x6 = xs4567.z;
        const float x7 = xs4567.w;

        float minval = sane_fmin(sane_fmin(sane_fmin(x0, x1), sane_fmin(x2, x3)),
                                 sane_fmin(sane_fmin(x4, x5), sane_fmin(x6, x7)));
        float maxval = sane_fmax(sane_fmax(sane_fmax(x0, x1), sane_fmax(x2, x3)),
                                 sane_fmax(sane_fmax(x4, x5), sane_fmax(x6, x7)));

        minval = sane_fmin(minval, __shfl_sync(0xffffffff, minval, thread ^ 0x01));
        maxval = sane_fmax(maxval, __shfl_sync(0xffffffff, maxval, thread ^ 0x01));
        minval = sane_fmin(minval, __shfl_sync(0xffffffff, minval, thread ^ 0x02));
        maxval = sane_fmax(maxval, __shfl_sync(0xffffffff, maxval, thread ^ 0x02));
        minval = sane_fmin(minval, __shfl_sync(0xffffffff, minval, thread ^ 0x04));
        maxval = sane_fmax(maxval, __shfl_sync(0xffffffff, maxval, thread ^ 0x04));
        minval = sane_fmin(minval, __shfl_sync(0xffffffff, minval, thread ^ 0x08));
        maxval = sane_fmax(maxval, __shfl_sync(0xffffffff, maxval, thread ^ 0x08));
        minval = sane_fmin(minval, __shfl_sync(0xffffffff, minval, thread ^ 0x10));
        maxval = sane_fmax(maxval, __shfl_sync(0xffffffff, maxval, thread ^ 0x10));

        constexpr int outi_min = 1;
        constexpr int outi_max = 254;
        constexpr float outf_min = outi_min - 0.5f;
        constexpr float outf_max = outi_max + 0.5f;
        const float outf_range = outf_max - outf_min;
        const float in_range = maxval - minval;
        const float scale = in_range / outf_range;
        const float offset = minval - outf_min * scale;

        const float inv_offset = -offset / scale;
        const float inv_scale = 1.0f / scale;
        __half offseth = __float2half_rn(offset);
        __half scaleh = __float2half_rn(scale);
        __half inv_offseth = __float2half_rn(inv_offset);
        __half inv_scaleh = __float2half_rn(inv_scale);

        // Avoid non-finite numbers
        if (!isfinite(__half2float(offseth)) || !isfinite(__half2float(scaleh))) {
            // Offset or scale are non-finite
            offseth = __float2half_rn(0.0f);
            scaleh = __float2half_rn(0.0f);
            inv_offseth = __float2half_rn(0.0f);
            inv_scaleh = __float2half_rn(0.0f);
        } else if (!isfinite(__half2float(inv_offseth)) || !isfinite(__half2float(inv_scaleh))) {
            // Offset and scale are fine, but the inverse scale is non-finite. This means the scale
            // is too close to zero.
            scaleh = __float2half_rn(0.0f);
            inv_offseth = -offseth;
            inv_scaleh = __float2half_rn(0.0f);
        }

        if (chunk == thread)
            outf = {offset, scale};

        const __half2 minh = __half2half2(__int2half_rn(outi_min));
        const __half2 maxh = __half2half2(__int2half_rn(outi_max));
        const __half2 inv_offseth2 = __half2half2(inv_offseth);
        const __half2 inv_scaleh2 = __half2half2(inv_scaleh);

        // Note:
        //     float2int_rn(+/-0) returns 0.
        //     float2int_rn(+Inf) returns INT_MAX = 0x7FFFFFFF.
        //     float2int_rn(-Inf) returns INT_MIN = 0x80000000.
        //     float2int_rn(NaN)  returns 0.
        // This is consistent with what we need. There is also no performance penalty.

        const auto half2int = [](const __half x) {
            // This special case is not necessary since float2int_rn maps nan to 0 anyway
            // if (__hisnan(x))
            //     return 0;
            return __half2int_rn(x);
        };

        const __half2 x01 = __floats2half2_rn(x0, x1);
        const __half2 y01 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x01, inv_scaleh2, inv_offseth2)));
        const int i0 = half2int(__low2half(y01));
        const int i1 = half2int(__high2half(y01));

        const __half2 x23 = __floats2half2_rn(x2, x3);
        const __half2 y23 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x23, inv_scaleh2, inv_offseth2)));
        const int i2 = half2int(__low2half(y23));
        const int i3 = half2int(__high2half(y23));

        const __half2 x45 = __floats2half2_rn(x4, x5);
        const __half2 y45 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x45, inv_scaleh2, inv_offseth2)));
        const int i4 = half2int(__low2half(y45));
        const int i5 = half2int(__high2half(y45));

        const __half2 x67 = __floats2half2_rn(x6, x7);
        const __half2 y67 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x67, inv_scaleh2, inv_offseth2)));
        const int i6 = half2int(__low2half(y67));
        const int i7 = half2int(__high2half(y67));

        const std::uint64_t outi = (std::uint64_t(i0) << 0x00) | (std::uint64_t(i1) << 0x08)
                                   | (std::uint64_t(i2) << 0x10) | (std::uint64_t(i3) << 0x18)
                                   | (std::uint64_t(i4) << 0x20) | (std::uint64_t(i5) << 0x28)
                                   | (std::uint64_t(i6) << 0x30) | (std::uint64_t(i7) << 0x38);

        const int out_time_chunk = 8 * (chunk % 2); // 0...15 in steps of 8
        const int out_freq_chunk = chunk / 2;       // 0...7
        *(std::uint64_t*)(outputi
                          + outputi_offset(out_time_chunk, out_freq_chunk, freq_packet, beam_packet,
                                           time_outer, freq_outer, beam_outer)) = outi;
    }
    const int out_beam_outer = beam_outer0 + thread;
    *((__half2*)outputf
      + outputf_offset(freq_packet, beam_packet, time_outer, freq_outer, out_beam_outer)) = outf;
}

////////////////////////////////////////////////////////////////////////////////

// Drivers (externally visible)

void cpu_quantize8chime(const float* __restrict__ const input, __half* __restrict__ const outputf,
                        std::uint8_t* __restrict__ const outputi) {
    assert(input);
    assert(outputf);
    assert(outputi);

    // Loop over all packets
    for (int beam_outer = 0; beam_outer < out_nbeams_outer; ++beam_outer) {
        for (int freq_outer = 0; freq_outer < out_nfreqs_outer; ++freq_outer) {
            for (int time_outer = 0; time_outer < out_ntimes_outer; ++time_outer) {

                // Loop over the chunks in each packets
                for (int beam_packet = 0; beam_packet < out_nbeams_packet; ++beam_packet) {
                    for (int freq_packet = 0; freq_packet < out_nfreqs_packet; ++freq_packet) {

                        // Process one chunk
                        const int time = out_ntimes_chunk * time_outer;
                        const int freq =
                            out_nfreqs_chunk * (freq_packet + out_nfreqs_packet * freq_outer);
                        const int beam = beam_packet + out_nbeams_packet * beam_outer;
                        const int input_offset = time + in_ntimes * (freq + in_nfreqs * beam);

                        const int outputf_offset =
                            freq_packet
                            + out_nfreqs_packet
                                  * (beam_packet
                                     + out_nbeams_packet
                                           * (time_outer
                                              + out_ntimes_outer
                                                    * (freq_outer
                                                       + out_nfreqs_outer * beam_outer)));
                        const int outputi_offset =
                            out_ntimes_chunk * out_nfreqs_chunk * outputf_offset;

                        cpu_quantize8chime_chunk(input + input_offset, outputf + outputf_offset,
                                                 outputi + outputi_offset);
                    }
                }
            }
        }
    }
}

void gpu_quantize8chime(const float* __restrict__ const input, __half* __restrict__ const outputf,
                        std::uint8_t* __restrict__ const outputi, cudaStream_t stream) {
    assert(input);
    assert(outputf);
    assert(outputi);

    dim3 nthreads;
    nthreads.x = nchunks;
    nthreads.y = 1;
    nthreads.z = 1;
    dim3 nblocks;
    nblocks.x = out_ntimes_outer;
    nblocks.y = out_nfreqs_packet * out_nfreqs_outer;
    nblocks.z = out_nbeams_packet * out_nbeams_outer / nchunks;
    const int shmem_nbytes = 0;

    gpu_quantize8chime_chunks<<<nblocks, nthreads, shmem_nbytes, stream>>>(input, outputf, outputi);
}
