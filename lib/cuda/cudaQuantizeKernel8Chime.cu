// 8-bit FRB beam quantizer for CHIME, using the chromatic CHIME float32 FRB beamformer and CHIME's
// sending-to-Bonsai network format

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
static int sane_lrint(const float x) {
    using std::isnan, std::lrint;
    // INT_MIN = -2^31 is exact in float; INT_MAX = 2^31 - 1 rounds up to 2^31
    // when promoted, so the upper check uses `>=`.
    if (isnan(x))
        return 0; // we could mask to -8 instead
    if (x < INT_MIN)
        return INT_MIN;
    if (x >= INT_MAX)
        return INT_MAX;
    return int(lrint(x));
}

////////////////////////////////////////////////////////////////////////////////

// Kernels (called by the drivers below)

// Quantize a "chunk". This is a CPU implementation that is
// straightforward and thus "obviously correct". If the GPU version
// produces a different result, then the GPU version is wrong.
void cpu_quantize8chime_chunk(const float* __restrict__ const input,
                              float* __restrict__ const outputf,
                              std::uint8_t* __restrict__ const outputi) {
    using std::isfinite, std::isnan, std::max, std::min;

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
    // Calculate offset and scale.
    // There are 254 possible uint8 values since we don't use 0 or 255.
    // Values are reconstructed as `offset + scale * n`, with `uint8 n`.
    // NaN in the input is represented as `n = 0`.
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

    // Avoid non-finite numbers
    if (!isfinite(offset) || !isfinite(scale)) {
        offset = 0.0f;
        scale = 0.0f;
    }

    // Store offset and scale in the output array
    outputf[0] = offset;
    outputf[1] = scale;

    // We encode values as unsigned 8-bit integers.
    for (int freq = 0; freq < out_nfreqs_chunk; ++freq) {
        for (int time = 0; time < out_ntimes_chunk; ++time) {
            const int in_ind = time + in_ntimes * freq;
            const int out_ind = time + out_ntimes_chunk * freq;
            // Get value
            const float x = input[in_ind];
            // Scale
            const float y = scale == 0 ? float(outi_min + outi_max) / 2.0f : (x - offset) / scale;
            // Round
            const int j = sane_lrint(y);
            // Clamp; use 0 for nan
            const int k = isnan(x) ? 0 : max(outi_min, min(outi_max, j));
            // Store
            outputi[out_ind] = k;
        }
    }
}

// Quantize a set of "chunk"s. This is an efficient GPU implementation.
__global__ void gpu_quantize8chime_chunks(const float* __restrict__ const input,
                                          float* __restrict__ const outputf,
                                          std::uint8_t* __restrict__ const outputi) {
    using std::isfinite;

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
        return 2
               * (freq_packet
                  + out_nfreqs_packet
                        * (beam_packet
                           + out_nbeams_packet
                                 * (time_outer
                                    + out_ntimes_outer
                                          * (freq_outer + out_nfreqs_outer * beam_outer))));
    };
    const auto outputi_offset = [&](int time_chunk, int freq_chunk, int freq_packet,
                                    int beam_packet, int time_outer, int freq_outer,
                                    int beam_outer) {
        return time_chunk
               + out_ntimes_chunk
                     * (freq_chunk
                        + out_nfreqs_chunk
                              * (freq_packet
                                 + out_nfreqs_packet
                                       * (beam_packet
                                          + out_nbeams_packet
                                                * (time_outer
                                                   + out_ntimes_outer
                                                         * (freq_outer
                                                            + out_nfreqs_outer * beam_outer)))));
    };

    // We store offset and scale only after finishing the loop over all chunks. Each thread stores
    // one offset/scale pair.
    float2 outf;
    // Loop over all beams we're handling
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const int beam_outer = beam_outer0 + chunk;

        // Load all values in this chunk. We load all values at once in the beginning.
        // There are 256 values and 32 threads, so each thread loads 8 values.
        // We load these 8 values in 2 batches of 4 times each.
        // We space these batches 4 * 32 = 128 elements apart.
        // (This is not cache-efficient because the frequencies are not adjacent.)
        const int in_time_chunk = 4 * (threadIdx.x % 4); // 0...15 in steps of 4
        const int in_freq_chunk = threadIdx.x / 4;       // 0...7
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
        float scale = in_range / outf_range;
        float offset = minval - outf_min * scale;

        float inv_offset = -offset / scale;
        float inv_scale = 1.0f / scale;

        // Avoid non-finite numbers
        if (!isfinite(offset) || !isfinite(scale)) {
            // Offset or scale are non-finite
            offset = 0.0f;
            scale = 0.0f;
            inv_offset = 0.0f;
            inv_scale = 0.0f;
        } else if (!isfinite(inv_offset) || !isfinite(inv_scale)) {
            // Offset and scale are fine, but the inverse scale is non-finite. This means the scale
            // is too close to zero.
            scale = 0.0f;
            inv_offset = 0.0f;
            inv_scale = 0.0f;
        }

        if (chunk == thread)
            outf = {offset, scale};

        const float minf = outi_min;
        const float maxf = outi_max;

        // Clamp, then convert to int. Map nan to 0.
        const auto clamp_and_convert = [](const float x, const float xlo, const float xhi) {
            if (isnan(x) || isnan(xlo) || isnan(xhi))
                return 0;
            return __float2int_rn(fmaxf(xlo, fminf(xhi, x)));
        };

        const int i0 = clamp_and_convert(fmaf(x0, inv_scale, inv_offset), minf, maxf);
        const int i1 = clamp_and_convert(fmaf(x1, inv_scale, inv_offset), minf, maxf);
        const int i2 = clamp_and_convert(fmaf(x2, inv_scale, inv_offset), minf, maxf);
        const int i3 = clamp_and_convert(fmaf(x3, inv_scale, inv_offset), minf, maxf);
        const int i4 = clamp_and_convert(fmaf(x4, inv_scale, inv_offset), minf, maxf);
        const int i5 = clamp_and_convert(fmaf(x5, inv_scale, inv_offset), minf, maxf);
        const int i6 = clamp_and_convert(fmaf(x6, inv_scale, inv_offset), minf, maxf);
        const int i7 = clamp_and_convert(fmaf(x7, inv_scale, inv_offset), minf, maxf);

        // The thread loaded 4 contiguous times at freq+0 (x0..x3) and 4 contiguous times at
        // freq+8 (x4..x7), so the encoded bytes span two freqs and need two separate 4-byte
        // writes — one per freq, at the same (in_time_chunk, in_freq_chunk) tile it loaded.
        const std::uint32_t outi_lo = std::uint32_t(i0) << 0x00 | std::uint32_t(i1) << 0x08
                                      | std::uint32_t(i2) << 0x10 | std::uint32_t(i3) << 0x18;
        const std::uint32_t outi_hi = std::uint32_t(i4) << 0x00 | std::uint32_t(i5) << 0x08
                                      | std::uint32_t(i6) << 0x10 | std::uint32_t(i7) << 0x18;
        *(std::uint32_t*)(outputi
                          + outputi_offset(in_time_chunk, in_freq_chunk + 0, freq_packet,
                                           beam_packet, time_outer, freq_outer, beam_outer)) =
            outi_lo;
        *(std::uint32_t*)(outputi
                          + outputi_offset(in_time_chunk, in_freq_chunk + 8, freq_packet,
                                           beam_packet, time_outer, freq_outer, beam_outer)) =
            outi_hi;
    }
    const int out_beam_outer = beam_outer0 + thread;
    *(float2*)(outputf
               + outputf_offset(freq_packet, beam_packet, time_outer, freq_outer, out_beam_outer)) =
        outf;
}

////////////////////////////////////////////////////////////////////////////////

// Drivers (externally visible)

void cpu_quantize8chime(const float* __restrict__ const input, float* __restrict__ const outputf,
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
                            2
                            * (freq_packet
                               + out_nfreqs_packet
                                     * (beam_packet
                                        + out_nbeams_packet
                                              * (time_outer
                                                 + out_ntimes_outer
                                                       * (freq_outer
                                                          + out_nfreqs_outer * beam_outer))));
                        const int outputi_offset =
                            out_ntimes_chunk * out_nfreqs_chunk
                            * (freq_packet
                               + out_nfreqs_packet
                                     * (beam_packet
                                        + out_nbeams_packet
                                              * (time_outer
                                                 + out_ntimes_outer
                                                       * (freq_outer
                                                          + out_nfreqs_outer * beam_outer))));

                        cpu_quantize8chime_chunk(input + input_offset, outputf + outputf_offset,
                                                 outputi + outputi_offset);
                    }
                }
            }
        }
    }
}

void gpu_quantize8chime(const float* __restrict__ const input, float* __restrict__ const outputf,
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
