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

// The output is quantized in "chunks". Each chunk has a different offset and scale.
static constexpr int chunk_size1 = 16; // goes with a stride of 1
static constexpr int chunk_size2 = 16; // goes with chunk_stride2

// For efficiency we proceess several chunks at once on the GPU.
static constexpr int nchunks = 32;

// Number of threads per block
static constexpr int nthreads = nchunks;

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
void cpu_quantize8chime_chunk(const float* __restrict__ const input, const int chunk_stride2,
                              float16_t* __restrict__ const outputf,
                              std::uint8_t* __restrict__ const outputi) {
    static_assert(chunk_size1 >= 0);
    static_assert(chunk_size2 >= 0);
    assert(chunk_stride2 > 0);

    using std::isfinite, std::max, std::min;

    // Find the minimum and maximum of all input values. Calculate
    // everything as `float` for convenience.
    float minval = +1.0f / 0.0f, maxval = -1.0f / 0.0f;
    for (int ind2 = 0; ind2 < chunk_size2; ++ind2) {
        for (int ind1 = 0; ind1 < chunk_size1; ++ind1) {
            const int i = ind1 + chunk_stride2 * ind2;
            const float x = input[i];
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
    for (int ind2 = 0; ind2 < chunk_size2; ++ind2) {
        for (int ind1 = 0; ind1 < chunk_size1; ++ind1) {
            const int indin = ind1 + chunk_stride2 * ind2; // input
            const int indout = ind1 + chunk_size1 * ind2;  // output
            // Get value
            const float x = input[indin];
            // Scale
            const float y = (x - offset) / scale;
            // Round
            const int j = sane_lrint(y);
            // Clamp; use 0 for nan
            const int k = isnan(y) ? 0 : max(outi_min, min(outi_max, j));
            // Store
            outputi[indout] = k;
        }
    }
}

// Quantize a set of "chunk"s. This is an efficient GPU implementation.
__global__ void gpu_quantize8chime_chunks(
    const float* __restrict__ const input0, __half* __restrict__ const outputf0,
    std::uint8_t* __restrict__ const outputi0, const int input_size1, const int input_size2,
    const int input_size3, const int input_stride2, const int input_stride3,
    const int outputf_size1, const int outputf_size2, const int outputf_size3,
    const int outputf_stride2, const int outputf_stride3, const int outputi_size1,
    const int outputi_size2, const int outputi_size3, const int outputi_stride2,
    const int outputi_stride3) {
    static_assert(chunk_size1 >= 0);
    static_assert(chunk_size2 >= 0);
    static_assert(chunk_size1 * chunk_size2 % (8 * nthreads) == 0);

    // Calculate array indices and convert to SIMD reference
    const auto input = [&](const int dim1, const int dim2, const int dim3) -> const float& {
        assert(dim1 >= 0 && dim1 < input_size1);
        assert(dim2 >= 0 && dim2 < input_size2);
        assert(dim3 >= 0 && dim3 < input_size3);
        return *(const float*)&(input0[dim1 + input_stride2 * dim2 + input_stride3 * dim3]);
    };
    const auto outputi = [&](const int dim1, const int dim2, const int dim3) -> std::uint64_t& {
        assert(dim1 >= 0 && dim1 < outputi_size1);
        assert(dim2 >= 0 && dim2 < outputi_size2);
        assert(dim3 >= 0 && dim3 < outputi_size3);
        return *(std::uint64_t*)&(outputi0[dim1 + outputi_stride2 * dim2 + outputi_stride3 * dim3]);
    };
    const auto outputf = [&](const int dim1, const int dim2, const int dim3) -> __half2& {
        assert(dim1 >= 0 && dim1 < outputf_size1);
        assert(dim2 >= 0 && dim2 < outputf_size2);
        assert(dim3 >= 0 && dim3 < outputf_size3);
        return *(__half2*)&(outputf0[dim1 + outputf_stride2 * dim2 + outputf_stride3 * dim3]);
    };

    // // Find our position in the arrays
    const int thread = threadIdx.x;
    const int dim1 = blockIdx.x * chunk_size1;
    const int dim2 = blockIdx.y * chunk_size2;
    const int dim3 = blockIdx.z * nchunks;
    const int outf_dim1 = 2 * blockIdx.x;
    const int outf_dim2 = blockIdx.y;
    const int outf_dim3 = blockIdx.z * nchunks;
    const int outi_dim1 = (chunk_size1 * chunk_size2) * blockIdx.x;
    const int outi_dim2 = blockIdx.y;
    const int outi_dim3 = blockIdx.z * nchunks;

    static_assert(nchunks == nthreads);
    // We store offset and scale only after finishing the loop over all chunks. Each thread stores
    // one offset/scale pair.
    __half2 outf;
    // Loop over all chunks in the frame
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        static_assert(chunk_size1 * chunk_size2 == 8 * nthreads);

        // Load all values in this chunk. We load all values at once in the beginning.
        // Note: This way of loading values is not cache-efficient; we are loading only half of each
        // cache line. We need to fix this if it's too slow.
        const int ind = 8 * thread;
        const int ind1 = ind % chunk_size1;
        const int ind2 = ind / chunk_size1;
        const float4 xs0123 = load_float4(&input(dim1 + ind1, dim2 + ind2, dim3 + chunk));
        const float4 xs4567 = load_float4(&input(dim1 + ind1, dim2 + ind2, dim3 + chunk));
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
        using std::isfinite;
        if (!isfinite(__half2float(offseth)) || !isfinite(__half2float(scaleh))) {
            // Offset or scale are non-finite
            offseth = __float2half_rn(0.0f);
            scaleh = __float2half_rn(0.0f);
            inv_offseth = __float2half_rn(inv_offset);
            inv_scaleh = __float2half_rn(inv_scale);
        } else if (!isfinite(__half2float(inv_offseth)) || !isfinite(__half2float(inv_scaleh))) {
            // Offset and scale are fine, but the inverse scale is non-finite. This means the scale
            // is too close to zero.
            scaleh = __float2half_rn(0.0f);
            inv_offseth = -offseth;
            inv_scaleh = __float2half_rn(0.0f);
        }

        if (chunk == thread)
            outf = __halves2half2(offseth, scaleh);

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

        outputi(outi_dim1 + ind, outi_dim2, outi_dim3 + chunk) = outi;
    }
    outputf(outf_dim1, outf_dim2, outf_dim3 + thread) = outf;
}

////////////////////////////////////////////////////////////////////////////////

// Drivers (externally visible)

void cpu_quantize8chime(const float* __restrict__ const input,
                        float16_t* __restrict__ const outputf,
                        std::uint8_t* __restrict__ const outputi, const int input_size1,
                        const int input_size2, const int input_size3, const int input_stride2,
                        const int input_stride3, const int outputf_size1, const int outputf_size2,
                        const int outputf_size3, const int outputf_stride2,
                        const int outputf_stride3, const int outputi_size1, const int outputi_size2,
                        const int outputi_size3, const int outputi_stride2,
                        const int outputi_stride3) {
    assert(input);
    assert(outputf);
    assert(outputi);

    assert(input_size1 >= 0);
    assert(input_size2 >= 0);
    assert(input_size3 >= 0);
    assert(input_stride2 >= 0);
    assert(input_stride3 >= 0);
    assert(outputf_size1 >= 0);
    assert(outputf_size2 >= 0);
    assert(outputf_size3 >= 0);
    assert(outputf_stride2 >= 0);
    assert(outputf_stride3 >= 0);
    assert(outputi_size1 >= 0);
    assert(outputi_size2 >= 0);
    assert(outputi_size3 >= 0);
    assert(outputi_stride2 >= 0);
    assert(outputi_stride3 >= 0);
    static_assert(chunk_size1 > 0);
    static_assert(chunk_size2 > 0);

    assert(input_size1 % chunk_size1 == 0);
    assert(input_size2 % chunk_size2 == 0);

    assert(outputf_size1 == 2 * (input_size1 / chunk_size1));
    assert(outputf_size2 == input_size2 / chunk_size2);
    assert(outputf_size3 == input_size3);
    assert(outputi_size1 == (chunk_size1 * chunk_size2) * (input_size1 / chunk_size1));
    assert(outputi_size2 == input_size2 / chunk_size2);
    assert(outputi_size3 == input_size3);

    for (int dim3 = 0; dim3 < input_size3; ++dim3) {
        for (int dim2 = 0; dim2 < input_size2; dim2 += chunk_size2) {
            for (int dim1 = 0; dim1 < input_size1; dim1 += chunk_size1) {
                const int input_dim1 = dim1;
                const int input_dim2 = dim2;
                const int outputf_dim1 = 2 * (dim1 / chunk_size1);
                const int outputf_dim2 = dim2 / chunk_size2;
                const int outputi_dim1 = (chunk_size1 * chunk_size2) * (dim1 / chunk_size1);
                const int outputi_dim2 = dim2 / chunk_size2;
                const int input_offset =
                    input_dim1 + input_stride2 * input_dim2 + input_stride3 * dim3;
                const int outputf_offset =
                    outputf_dim1 + outputf_stride2 * outputf_dim2 + outputf_stride3 * dim3;
                const int outputi_offset =
                    outputi_dim1 + outputi_stride2 * outputi_dim2 + outputi_stride3 * dim3;
                cpu_quantize8chime_chunk(input + input_offset, input_stride2,
                                         outputf + outputf_offset, outputi + outputi_offset);
            }
        }
    }
}

void gpu_quantize8chime(const float* __restrict__ const input,
                        float16_t* __restrict__ const outputf,
                        std::uint8_t* __restrict__ const outputi, const int input_size1,
                        const int input_size2, const int input_size3, const int input_stride2,
                        const int input_stride3, const int outputf_size1, const int outputf_size2,
                        const int outputf_size3, const int outputf_stride2,
                        const int outputf_stride3, const int outputi_size1, const int outputi_size2,
                        const int outputi_size3, const int outputi_stride2,
                        const int outputi_stride3, cudaStream_t stream) {
    assert(input);
    assert(outputf);
    assert(outputi);

    assert(input_size1 >= 0);
    assert(input_size2 >= 0);
    assert(input_size3 >= 0);
    assert(input_stride2 >= 0);
    assert(input_stride3 >= 0);
    assert(outputf_size1 >= 0);
    assert(outputf_size2 >= 0);
    assert(outputf_size3 >= 0);
    assert(outputf_stride2 >= 0);
    assert(outputf_stride3 >= 0);
    assert(outputi_size1 >= 0);
    assert(outputi_size2 >= 0);
    assert(outputi_size3 >= 0);
    assert(outputi_stride2 >= 0);
    assert(outputi_stride3 >= 0);
    static_assert(chunk_size1 > 0);
    static_assert(chunk_size2 > 0);
    static_assert(nchunks > 0);

    assert(input_size1 % chunk_size1 == 0);
    assert(input_size2 % chunk_size2 == 0);
    assert(input_size3 % nchunks == 0);

    assert(outputf_size1 == 2 * (input_size1 / chunk_size1));
    assert(outputf_size2 == input_size2 / chunk_size2);
    assert(outputf_size3 == input_size3);
    assert(outputi_size1 == (chunk_size1 * chunk_size2) * (input_size1 / chunk_size1));
    assert(outputi_size2 == input_size2 / chunk_size2);
    assert(outputi_size3 == input_size3);

    dim3 nblocks;
    nblocks.x = input_size1 / chunk_size1;
    nblocks.y = input_size2 / chunk_size2;
    nblocks.z = input_size3 / nchunks;
    const int shmem_nbytes = 0;
    gpu_quantize8chime_chunks<<<nblocks, nthreads, shmem_nbytes, stream>>>(
        input, outputf, outputi, input_size1, input_size2, input_size3, input_stride2,
        input_stride3, outputf_size1, outputf_size2, outputf_size3, outputf_stride2,
        outputf_stride3, outputi_size1, outputi_size2, outputi_size3, outputi_stride2,
        outputi_stride3);
}
