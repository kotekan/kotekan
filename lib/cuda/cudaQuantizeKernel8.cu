// 8-bit FRB beam quantizer for CHIME, using the float16 FRB beamformer and CHIME's
// sending-to-Bonsai network format

#include "DataType.hpp"
#include "cudaQuantizeKernel8.hpp"

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
static constexpr int out_nbeams_packet = 4;
// There are several packets.
static constexpr int out_ntimes_outer = 16;
static constexpr int out_nfreqs_outer = 4;
static constexpr int out_nbeams_outer = 256;

static_assert(out_ntimes_chunk * out_ntimes_outer == in_ntimes);
static_assert(out_nfreqs_chunk * out_nfreqs_packet * out_nfreqs_outer == in_nfreqs);
static_assert(out_nbeams_packet * out_nbeams_outer == in_nbeams);

// For efficiency we proceess several chunks at once on the GPU.
static constexpr int nchunks = 32;

static_assert(out_nbeams_outer % nchunks == 0);

////////////////////////////////////////////////////////////////////////////////

// Helper types and functions

// A struct holding 4 consecutive `half2` values
struct half2x4 {
    __half2 x, y, z, w;
};

// Load 4 `half2` values from memory simultaneously. This is more efficient than loading each
// `half2` separately.
__device__ static inline half2x4 load_half2x4(const __half* __restrict__ const ptr) {
    const int4 data = *reinterpret_cast<const int4*>(ptr);
    return reinterpret_cast<const half2x4&>(data);
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
void cpu_quantize8_chunk(const __half* __restrict__ const input,
                         const std::uint8_t* __restrict__ const pl_mask, const int pl_mask_stride,
                         __half* __restrict__ const outputf,
                         std::uint8_t* __restrict__ const outputi) {
    using std::isfinite, std::isnan, std::max, std::min;

    // Find the minimum and maximum of all valid input values, skipping samples that the packet
    // loss mask marks as invalid. Calculate everything as `float` for convenience.
    float minval = +1.0f / 0.0f, maxval = -1.0f / 0.0f;
    for (int freq = 0; freq < out_nfreqs_chunk; ++freq) {
        for (int time = 0; time < out_ntimes_chunk; ++time) {
            if (!pl_mask[time * pl_mask_stride + freq])
                continue;
            const int in_ind = time + in_ntimes * freq;
            const float x = __half2float(input[in_ind]);
            minval = sane_fmin(minval, x);
            maxval = sane_fmax(maxval, x);
        }
    }
    // Calculate offset and scale.
    // There are 254 possible uint8 values since we don't use 0 or 255 for regular values.
    // Values are reconstructed as `offset + scale * n`, with `uint8 n`.
    // NaN in the input, and samples the packet loss mask marks as invalid, are represented as
    // `n = 0`.
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
    outputf[0] = offseth;
    outputf[1] = scaleh;

    // Recalculate offset and scale from the stored half precision values
    offset = __half2float(offseth);
    scale = __half2float(scaleh);

    // We encode values as unsigned 8-bit integers.
    for (int freq = 0; freq < out_nfreqs_chunk; ++freq) {
        for (int time = 0; time < out_ntimes_chunk; ++time) {
            const int in_ind = time + in_ntimes * freq;
            const int out_ind = time + out_ntimes_chunk * (out_nfreqs_chunk - 1 - freq);
            // Get value
            const float x = __half2float(input[in_ind]);
            // Scale. (A zero scale maps all values to `outi_min`, as on the GPU.)
            const float y = scale == 0 ? 0.0f : (x - offset) / scale;
            // Round
            const int j = sane_lrint(y);
            // Clamp; use 0 for nan and for invalid samples
            const bool valid = pl_mask[time * pl_mask_stride + freq];
            const int k = !valid || isnan(x) ? 0 : max(outi_min, min(outi_max, j));
            // Store
            outputi[out_ind] = k;
        }
    }
}

// Quantize a set of "chunk"s. This is an efficient GPU implementation.
__global__ void gpu_quantize8_chunks(const __half* __restrict__ const input,
                                     const std::uint8_t* __restrict__ const pl_mask,
                                     const int pl_mask_stride, __half* __restrict__ const outputf,
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
                     * ((out_nfreqs_chunk - 1 - freq_chunk)
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

    // Per-thread tile within each chunk: 8 contiguous times × 1 freq. 32 threads
    // cover the 16 × 16 chunk as 2 time-groups × 16 freqs = 32 tiles.
    const int in_time_chunk = 8 * (threadIdx.x % 2); // 0 or 8
    const int in_freq_chunk = threadIdx.x / 2;       // 0..15

    // The packet loss mask is the same for all beams. Load this thread's 8 values once; masked
    // samples are skipped when finding the minimum and maximum, and are encoded as 0.
    bool valid[8];
    for (int i = 0; i < 8; ++i)
        valid[i] = pl_mask[time(in_time_chunk + i, time_outer) * pl_mask_stride
                           + freq(in_freq_chunk, freq_packet, freq_outer)];
    constexpr float inf = 1.0f / 0.0f;

    // We store offset and scale only after finishing the loop over all chunks. Each thread stores
    // one offset/scale pair.
    __half2 outf;
    // Loop over all beams we're handling
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const int beam_outer = beam_outer0 + chunk;

        // Load all values in this chunk. We load all values at once in the beginning.
        // There are 256 values and 32 threads, so each thread loads 8 values.
        // We load these 8 values as a batch of 8 times.
        // (This is not cache-efficient because the frequencies are not adjacent.)
        const int input_offset_01234567 = input_offset(time(in_time_chunk, time_outer),
                                                       freq(in_freq_chunk, freq_packet, freq_outer),
                                                       beam(beam_packet, beam_outer));
        const half2x4 xs01234567 = load_half2x4(input + input_offset_01234567);
        const __half2 x01 = xs01234567.x;
        const __half2 x23 = xs01234567.y;
        const __half2 x45 = xs01234567.z;
        const __half2 x67 = xs01234567.w;

        const float x0 = __half2float(__low2half(x01));
        const float x1 = __half2float(__high2half(x01));
        const float x2 = __half2float(__low2half(x23));
        const float x3 = __half2float(__high2half(x23));
        const float x4 = __half2float(__low2half(x45));
        const float x5 = __half2float(__high2half(x45));
        const float x6 = __half2float(__low2half(x67));
        const float x7 = __half2float(__high2half(x67));

        const float m0 = valid[0] ? x0 : +inf, M0 = valid[0] ? x0 : -inf;
        const float m1 = valid[1] ? x1 : +inf, M1 = valid[1] ? x1 : -inf;
        const float m2 = valid[2] ? x2 : +inf, M2 = valid[2] ? x2 : -inf;
        const float m3 = valid[3] ? x3 : +inf, M3 = valid[3] ? x3 : -inf;
        const float m4 = valid[4] ? x4 : +inf, M4 = valid[4] ? x4 : -inf;
        const float m5 = valid[5] ? x5 : +inf, M5 = valid[5] ? x5 : -inf;
        const float m6 = valid[6] ? x6 : +inf, M6 = valid[6] ? x6 : -inf;
        const float m7 = valid[7] ? x7 : +inf, M7 = valid[7] ? x7 : -inf;

        float minval = sane_fmin(sane_fmin(sane_fmin(m0, m1), sane_fmin(m2, m3)),
                                 sane_fmin(sane_fmin(m4, m5), sane_fmin(m6, m7)));
        float maxval = sane_fmax(sane_fmax(sane_fmax(M0, M1), sane_fmax(M2, M3)),
                                 sane_fmax(sane_fmax(M4, M5), sane_fmax(M6, M7)));

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
            inv_offseth = __float2half_rn(0.0f);
            inv_scaleh = __float2half_rn(0.0f);
        }

        if (chunk == thread)
            outf = __halves2half2(offseth, scaleh);

        const __half2 minh = __half2half2(__int2half_rn(outi_min));
        const __half2 maxh = __half2half2(__int2half_rn(outi_max));
        const __half2 inv_offseth2 = __half2half2(inv_offseth);
        const __half2 inv_scaleh2 = __half2half2(inv_scaleh);

        // Note:
        //     __half2int_rn(+/-0) returns 0.
        //     __half2int_rn(+Inf) returns INT_MAX = 0x7FFFFFFF.
        //     __half2int_rn(-Inf) returns INT_MIN = 0x80000000.
        //     __half2int_rn(NaN)  returns 0.
        // This is consistent with what we need. There is also no performance penalty.

        const auto half2int = [](const __half x) {
            // This special case is not necessary since __half2int_rn maps nan to 0 anyway
            // if (__hisnan(x))
            //     return 0;
            return __half2int_rn(x);
        };

        const __half2 y01 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x01, inv_scaleh2, inv_offseth2)));
        const int i0 = valid[0] ? half2int(__low2half(y01)) : 0;
        const int i1 = valid[1] ? half2int(__high2half(y01)) : 0;

        const __half2 y23 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x23, inv_scaleh2, inv_offseth2)));
        const int i2 = valid[2] ? half2int(__low2half(y23)) : 0;
        const int i3 = valid[3] ? half2int(__high2half(y23)) : 0;

        const __half2 y45 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x45, inv_scaleh2, inv_offseth2)));
        const int i4 = valid[4] ? half2int(__low2half(y45)) : 0;
        const int i5 = valid[5] ? half2int(__high2half(y45)) : 0;

        const __half2 y67 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x67, inv_scaleh2, inv_offseth2)));
        const int i6 = valid[6] ? half2int(__low2half(y67)) : 0;
        const int i7 = valid[7] ? half2int(__high2half(y67)) : 0;

        const std::uint64_t outi = (std::uint64_t(i0) << 0x00) | (std::uint64_t(i1) << 0x08)
                                   | (std::uint64_t(i2) << 0x10) | (std::uint64_t(i3) << 0x18)
                                   | (std::uint64_t(i4) << 0x20) | (std::uint64_t(i5) << 0x28)
                                   | (std::uint64_t(i6) << 0x30) | (std::uint64_t(i7) << 0x38);

        // Each thread writes its own tile, the same (in_time_chunk, in_freq_chunk) it loaded from.
        *(std::uint64_t*)(outputi
                          + outputi_offset(in_time_chunk, in_freq_chunk, freq_packet, beam_packet,
                                           time_outer, freq_outer, beam_outer)) = outi;
    }
    const int out_beam_outer = beam_outer0 + thread;
    *(__half2*)(outputf
                + outputf_offset(freq_packet, beam_packet, time_outer, freq_outer,
                                 out_beam_outer)) = outf;
}

////////////////////////////////////////////////////////////////////////////////

// Drivers (externally visible)

void cpu_quantize8(const __half* __restrict__ const input,
                   const std::uint8_t* __restrict__ const pl_mask, const int pl_mask_stride,
                   __half* __restrict__ const outputf, std::uint8_t* __restrict__ const outputi) {
    assert(input);
    assert(pl_mask);
    assert(pl_mask_stride >= in_nfreqs);
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

                        const int pl_mask_offset = time * pl_mask_stride + freq;

                        cpu_quantize8_chunk(input + input_offset, pl_mask + pl_mask_offset,
                                            pl_mask_stride, outputf + outputf_offset,
                                            outputi + outputi_offset);
                    }
                }
            }
        }
    }
}

void gpu_quantize8(const __half* __restrict__ const input,
                   const std::uint8_t* __restrict__ const pl_mask, const int pl_mask_stride,
                   __half* __restrict__ const outputf, std::uint8_t* __restrict__ const outputi,
                   cudaStream_t stream) {
    assert(input);
    assert(pl_mask);
    assert(pl_mask_stride >= in_nfreqs);
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

    gpu_quantize8_chunks<<<nblocks, nthreads, shmem_nbytes, stream>>>(
        input, pl_mask, pl_mask_stride, outputf, outputi);
}
