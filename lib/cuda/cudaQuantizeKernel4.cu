#include "DataType.hpp"
#include "cudaQuantizeKernel4.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////

// Global constants

// The output is quantized in "chunks". Each chunk has a different offset and scale.
static constexpr int chunk_size = 256;

// For efficiency we proceess several chunks at once on the GPU.
static constexpr int nchunks = 32;

// Number of threads per block
static constexpr int nthreads = nchunks;

////////////////////////////////////////////////////////////////////////////////

// Helper types and functions

// A struct holding 4 consecutive `half2` values
struct half2x4 {
    __half2 x, y, z, w;
};

// Load 4 `half2` values from memory simultaneously. This is more efficient than loading each
// `half2` separately.
__device__ static inline half2x4 load_half2x4(const __half2* __restrict__ const ptr) {
    const int4 data = *reinterpret_cast<const int4*>(ptr);
    return reinterpret_cast<const half2x4&>(data);
}

// Round to nearest integer, avoiding all the implementation-defined cases
static int sane_lrint(const float x) {
    using std::isnan, std::lrint;
    if (x < INT_MIN)
        return INT_MIN;
    if (x > INT_MAX)
        return INT_MAX;
    if (isnan(x))
        return 0; // we could mask to -8 instead
    return int(lrint(x));
}

////////////////////////////////////////////////////////////////////////////////

// Kernels (called by the drivers below)

// Quantize a "chunk". This is a CPU implementation that is
// straightforward and thus "obviously correct". If the GPU version
// produces a different result, then the GPU version is wrong.
void cpu_quantize4_chunk(const __half* __restrict__ const input, __half* __restrict__ const outputf,
                         kotekan::int4x2_t* __restrict__ const outputi, const float stddev_cutoff) {
    static_assert(chunk_size >= 0);
    // The chunk size must be even because we process 2 `half` values simultaneously to create 1
    // output byte.
    static_assert(chunk_size % 2 == 0);

    using std::fmax, std::isfinite, std::isnan, std::max, std::min, std::sqrt;

    // Find the sum and sum-squared of all input values. Calculate everything as `float` to
    // avoid overflow.
    float sum = 0, sum2 = 0;
    for (int i = 0; i < chunk_size; ++i) {
        const float x = __half2float(input[i]);
        sum += x;
        sum2 += x * x;
    }
    // Calculate the mean and corrected standard deviation
    using std::fmax, std::sqrt;
    const float mean = sum / chunk_size;
    const float stddev = sqrt(fmax(0.0f, sum2 / chunk_size - mean * mean));
    // Calculate offset and scale. Ensure the scale is nonzero.
    // There are 15 possible int4 values since we don't use 0.
    // Values are reconstructed as `offset + scale * n`, with `int4 n`.
    constexpr int outi_min = -7;
    constexpr int outi_max = +7;
    constexpr float outf_min = outi_min - 0.5f;
    constexpr float outf_max = outi_max + 0.5f;
    // We want this mapping:
    //     mean                          => 0
    //     mean - stddev * stddev_cutoff => outf_min
    //     mean + stddev * stddev_cutoff => outf_max
    const float outf_range = outf_max - outf_min;
    const float in_range = 2 * stddev * stddev_cutoff;

    float offset = mean;
    float scale = in_range / outf_range;
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

    // We encode values as signed 4-bit integers. We combine two 4-bit integers into one byte. The
    // lower 4 bits hold the first value, the upper 4 bits hold the second value.
    for (int i = 0; i < chunk_size; i += 2) {
        // Get values
        const float x0 = __half2float(input[i + 0]);
        const float x1 = __half2float(input[i + 1]);
        // Scale
        const float y0 = (x0 - offset) / scale;
        const float y1 = (x1 - offset) / scale;
        // Round
        const int j0 = sane_lrint(y0);
        const int j1 = sane_lrint(y1);
        // Clamp; use -8 for nan
        const int k0 = isnan(y0) ? -8 : max(outi_min, min(outi_max, j0));
        const int k1 = isnan(y1) ? -8 : max(outi_min, min(outi_max, j1));
        // Combine
        const kotekan::int4x2_t k01(k0, k1);
        // Store
        outputi[i / 2] = k01;
    }
}

// Quantize a set of "chunk"s. This is an efficient GPU implementation.
__global__ void
gpu_quantize4_chunks(const __half* __restrict__ const input0, __half* __restrict__ const outputf0,
                     kotekan::int4x2_t* __restrict__ const outputi0, const int input_size1,
                     const int input_size2, const int input_size3, const int input_stride2,
                     const int input_stride3, const int outputf_size1, const int outputf_size2,
                     const int outputf_size3, const int outputf_stride2, const int outputf_stride3,
                     const int outputi_size1, const int outputi_size2, const int outputi_size3,
                     const int outputi_stride2, const int outputi_stride3,
                     const float stddev_cutoff) {
    static_assert(chunk_size >= 0);
    static_assert(chunk_size % (8 * nthreads) == 0);

    using std::fmax, std::isfinite, std::sqrt;

    // Calculate array indices and convert to SIMD reference
    const auto input = [&](const int dim1, const int dim2, const int dim3) -> const __half2& {
        assert(dim1 >= 0 && dim1 < input_size1);
        assert(dim2 >= 0 && dim2 < input_size2);
        assert(dim3 >= 0 && dim3 < input_size3);
        return *(const __half2*)&(input0[dim1 + input_stride2 * dim2 + input_stride3 * dim3]);
    };
    const auto outputi = [&](const int dim1, const int dim2, const int dim3) -> std::uint32_t& {
        assert(dim1 >= 0 && dim1 < outputi_size1);
        assert(dim2 >= 0 && dim2 < outputi_size2);
        assert(dim3 >= 0 && dim3 < outputi_size3);
        return *(std::uint32_t*)&(outputi0[dim1 + outputi_stride2 * dim2 + outputi_stride3 * dim3]);
    };
    const auto outputf = [&](const int dim1, const int dim2, const int dim3) -> __half2& {
        assert(dim1 >= 0 && dim1 < outputf_size1);
        assert(dim2 >= 0 && dim2 < outputf_size2);
        assert(dim3 >= 0 && dim3 < outputf_size3);
        return *(__half2*)&(outputf0[dim1 + outputf_stride2 * dim2 + outputf_stride3 * dim3]);
    };

    // // Find our position in the arrays
    const int thread = threadIdx.x;
    const int dim1 = blockIdx.x * chunk_size;
    const int dim2 = blockIdx.y;
    const int dim3 = blockIdx.z * nchunks;

    static_assert(nchunks == nthreads);
    // We store offset and scale only after finishing the loop over all chunks. Each thread stores
    // one offset/scale pair.
    __half2 outf;
    // Loop over all chunks in the frame
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        static_assert(chunk_size == 8 * nthreads);

        // Load all values in this chunk. We load all values at once in the beginning.
        const half2x4 xs = load_half2x4(&input(dim1 + 8 * thread, dim2, dim3 + chunk));
        const __half2 x01 = xs.x;
        const __half2 x23 = xs.y;
        const __half2 x45 = xs.z;
        const __half2 x67 = xs.w;

        const float x0 = __half2float(__low2half(x01));
        const float x1 = __half2float(__high2half(x01));
        const float x2 = __half2float(__low2half(x23));
        const float x3 = __half2float(__high2half(x23));
        const float x4 = __half2float(__low2half(x45));
        const float x5 = __half2float(__high2half(x45));
        const float x6 = __half2float(__low2half(x67));
        const float x7 = __half2float(__high2half(x67));

        float sum = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
        float sum2 = ((x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3))
                     + ((x4 * x4 + x5 * x5) + (x6 * x6 + x7 * x7));

        sum += __shfl_sync(0xffffffff, sum, thread ^ 0x01);
        sum2 += __shfl_sync(0xffffffff, sum2, thread ^ 0x01);
        sum += __shfl_sync(0xffffffff, sum, thread ^ 0x02);
        sum2 += __shfl_sync(0xffffffff, sum2, thread ^ 0x02);
        sum += __shfl_sync(0xffffffff, sum, thread ^ 0x04);
        sum2 += __shfl_sync(0xffffffff, sum2, thread ^ 0x04);
        sum += __shfl_sync(0xffffffff, sum, thread ^ 0x08);
        sum2 += __shfl_sync(0xffffffff, sum2, thread ^ 0x08);
        sum += __shfl_sync(0xffffffff, sum, thread ^ 0x10);
        sum2 += __shfl_sync(0xffffffff, sum2, thread ^ 0x10);

        const float mean = sum / chunk_size;
        const float stddev = sqrt(fmax(0.0f, sum2 / chunk_size - mean * mean));
        constexpr int outi_min = -7;
        constexpr int outi_max = +7;
        constexpr float outf_min = outi_min - 0.5f;
        constexpr float outf_max = outi_max + 0.5f;
        constexpr float outf_range = outf_max - outf_min;
        const float in_range = 2 * stddev * stddev_cutoff;
        const float offset = mean;
        const float scale = fmax(0.0f, in_range / outf_range);

        // Idea: Add 8 before converting to int, then the result will be positive and we don't need
        // to mask when combining. Mapping nans will also work naturally. Use a single xor to
        // subtract 8 again in the end.

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
            if (__hisnan(x))
                return -8;
            return __half2int_rn(x);
        };

        const __half2 y01 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x01, inv_scaleh2, inv_offseth2)));
        const int i0 = half2int(__low2half(y01));
        const int i1 = half2int(__high2half(y01));

        const __half2 y23 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x23, inv_scaleh2, inv_offseth2)));
        const int i2 = half2int(__low2half(y23));
        const int i3 = half2int(__high2half(y23));

        const __half2 y45 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x45, inv_scaleh2, inv_offseth2)));
        const int i4 = half2int(__low2half(y45));
        const int i5 = half2int(__high2half(y45));

        const __half2 y67 =
            __hmax2_nan(minh, __hmin2_nan(maxh, __hfma2(x67, inv_scaleh2, inv_offseth2)));
        const int i6 = half2int(__low2half(y67));
        const int i7 = half2int(__high2half(y67));

        const std::uint32_t outi =
            (std::uint32_t(i0 & 0x0f) << 0x00) | (std::uint32_t(i1 & 0x0f) << 0x04)
            | (std::uint32_t(i2 & 0x0f) << 0x08) | (std::uint32_t(i3 & 0x0f) << 0x0c)
            | (std::uint32_t(i4 & 0x0f) << 0x10) | (std::uint32_t(i5 & 0x0f) << 0x14)
            | (std::uint32_t(i6 & 0x0f) << 0x18) | (std::uint32_t(i7 & 0x0f) << 0x1c);

        outputi((dim1 + 8 * thread) / 2, dim2, dim3 + chunk) = outi;
    }
    outputf(2 * dim1 / chunk_size, dim2, dim3 + thread) = outf;
}

////////////////////////////////////////////////////////////////////////////////

// Drivers (externally visible)

void cpu_quantize4(const __half* __restrict__ const input, __half* __restrict__ const outputf,
                   kotekan::int4x2_t* __restrict__ const outputi, const int input_size1,
                   const int input_size2, const int input_size3, const int input_stride2,
                   const int input_stride3, const int outputf_size1, const int outputf_size2,
                   const int outputf_size3, const int outputf_stride2, const int outputf_stride3,
                   const int outputi_size1, const int outputi_size2, const int outputi_size3,
                   const int outputi_stride2, const int outputi_stride3,
                   const float stddev_cutoff) {
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
    static_assert(chunk_size > 0);

    static_assert(chunk_size % 2 == 0);
    assert(input_size1 % 2 == 0);
    assert(input_size1 % chunk_size == 0);

    assert(outputf_size1 == 2 * input_size1 / chunk_size);
    assert(outputf_size2 == input_size2);
    assert(outputf_size3 == input_size3);
    assert(outputi_size1 == input_size1 / 2);
    assert(outputi_size2 == input_size2);
    assert(outputi_size3 == input_size3);

    for (int dim3 = 0; dim3 < input_size3; ++dim3) {
        for (int dim2 = 0; dim2 < input_size2; ++dim2) {
            for (int dim1 = 0; dim1 < input_size1; dim1 += chunk_size) {
                const int input_dim1 = dim1;
                const int outputf_dim1 = 2 * dim1 / chunk_size;
                const int outputi_dim1 = dim1 / 2;
                const int input_offset = input_dim1 + input_stride2 * dim2 + input_stride3 * dim3;
                const int outputf_offset =
                    outputf_dim1 + outputf_stride2 * dim2 + outputf_stride3 * dim3;
                const int outputi_offset =
                    outputi_dim1 + outputi_stride2 * dim2 + outputi_stride3 * dim3;
                cpu_quantize4_chunk(input + input_offset, outputf + outputf_offset,
                                    outputi + outputi_offset, stddev_cutoff);
            }
        }
    }
}

void gpu_quantize4(const __half* __restrict__ const input, __half* __restrict__ const outputf,
                   kotekan::int4x2_t* __restrict__ const outputi, const int input_size1,
                   const int input_size2, const int input_size3, const int input_stride2,
                   const int input_stride3, const int outputf_size1, const int outputf_size2,
                   const int outputf_size3, const int outputf_stride2, const int outputf_stride3,
                   const int outputi_size1, const int outputi_size2, const int outputi_size3,
                   const int outputi_stride2, const int outputi_stride3, const float stddev_cutoff,
                   cudaStream_t stream) {
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
    static_assert(chunk_size > 0);
    static_assert(nchunks > 0);

    static_assert(chunk_size % 2 == 0);
    assert(input_size1 % 2 == 0);
    assert(input_size1 % chunk_size == 0);
    assert(input_size3 % nchunks == 0);

    assert(outputf_size1 == 2 * input_size1 / chunk_size);
    assert(outputf_size2 == input_size2);
    assert(outputf_size3 == input_size3);
    assert(outputi_size1 == input_size1 / 2);
    assert(outputi_size2 == input_size2);
    assert(outputi_size3 == input_size3);

    dim3 nblocks;
    nblocks.x = input_size1 / chunk_size;
    nblocks.y = input_size2;
    nblocks.z = input_size3 / nchunks;
    const int shmem_nbytes = 0;
    gpu_quantize4_chunks<<<nblocks, nthreads, shmem_nbytes, stream>>>(
        input, outputf, outputi, input_size1, input_size2, input_size3, input_stride2,
        input_stride3, outputf_size1, outputf_size2, outputf_size3, outputf_stride2,
        outputf_stride3, outputi_size1, outputi_size2, outputi_size3, outputi_stride2,
        outputi_stride3, stddev_cutoff);
}
