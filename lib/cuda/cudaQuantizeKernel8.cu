#include "cudaQuantizeKernel8.hpp"

#include <DataType.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////

// Global constants

// The output is quantized in "chunks". Each chunk has a different offset and scale.
static constexpr int chunk_size = 256;

// For efficiency we proceess several chunks at once as a "frame".
static constexpr int chunks_per_frame = 32;

// Number of threads per block
static constexpr int nthreads = 32;

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

////////////////////////////////////////////////////////////////////////////////

// Kernels (called by the drivers below)

// Quantize a "frame", i.e. a set of "chunks". This is a CPU implementation that is straightforward
// and thus "obviously correct". If the GPU version produces a different result, then the GPU
// version is wrong.
void cpu_quantize8_chunks(const __half* __restrict__ const input,
                          __half* __restrict__ const outputf,
                          std::uint8_t* __restrict__ const outputi, const int input_stride,
                          const int outputf_stride, const int outputi_stride) {
    static_assert(chunk_size >= 0);

    // Loop over all chunks in the frame. The chunks are all independent.
    for (int chunk = 0; chunk < chunks_per_frame; ++chunk) {
        const int input_offset = chunk * input_stride;
        const int outputf_offset = chunk * outputf_stride;
        const int outputi_offset = chunk * outputi_stride;

        // Find the minimum and maximum of all input values. Calculate
        // everything as `float` for convenience.
        float minval = +1.0f / 0.0f, maxval = -1.0f / 0.0f;
        for (int i = 0; i < chunk_size; ++i) {
            const float x = __half2float(input[i + input_offset]);
            minval = fmin(minval, x);
            maxval = fmax(maxval, x);
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
        const float scale = fmax(1.0e-4f, in_range / outf_range);
        const float offset = minval - outf_min * scale;

        // Store offset and scale for this chunk in the output array
        outputf[outputf_offset + 0] = __float2half(offset);
        outputf[outputf_offset + 1] = __float2half(scale);

        // We encode values as offset-encoded unsigned 8-bit integers.
        for (int i = 0; i < chunk_size; ++i) {
            // Get values
            const float x = __half2float(input[i + input_offset]);
            // Scale
            const float y = (x - offset) / scale;
            // Round
            const int j = lrint(y);
            // Clamp
            const int k = max(outi_min, min(outi_max, j));
            // Store
            outputi[i + outputi_offset] = k;
        }
    }
}

// Quantize a "frame", i.e. a set of "chunks". This is an efficient GPU implementation.
__global__ void
gpu_quantize8_chunks(const __half* __restrict__ const input0, __half* __restrict__ const outputf0,
                     std::uint8_t* __restrict__ const outputi0, const int input_size1,
                     const int input_size2, const int input_size3, const int input_stride2,
                     const int input_stride3, const int outputf_size1, const int outputf_size2,
                     const int outputf_size3, const int outputf_stride2, const int outputf_stride3,
                     const int outputi_size1, const int outputi_size2, const int outputi_size3,
                     const int outputi_stride2, const int outputi_stride3) {
    static_assert(chunk_size >= 0);
    static_assert(chunk_size % (8 * nthreads) == 0);

    const int thread = threadIdx.x;

    const int dim1 = blockIdx.x * chunk_size;
    const int dim2 = blockIdx.y * chunks_per_frame;
    const int dim3 = blockIdx.z;

    const __half2* __restrict__ const input1 = (const __half2*)input0;
    __half2* __restrict__ const outputf1 = (__half2*)outputf0;
    uint32_t* __restrict__ const outputi1 = (uint32_t*)outputi0;

    const int input_dim1 = dim1;
    const int outputf_dim1 = dim1 / chunk_size;
    const int outputi_dim1 = dim1 / 2;

    const __half2* __restrict__ const input2 =
        input1 + (input_dim1 + input_stride2 * dim2 + input_stride3 * dim3) / 2;
    __half2* __restrict__ const outputf =
        outputf1 + (outputf_dim1 + outputf_stride2 * dim2 + outputf_stride3 * dim3) / 2;
    uint32_t* __restrict__ const outputi2 =
        outputi1 + (outputi_dim1 + outputi_stride2 * dim2 + outputi_stride3 * dim3) / 4;

    static_assert(chunks_per_frame == nthreads);
    // We store offset and scale only after finishing the loop over all chunks. Each thread stores
    // one offset/scale pair.
    __half2 outf;
    // Loop over all chunks in the frame
    for (int chunk = 0; chunk < chunks_per_frame; ++chunk) {
        const __half2* __restrict__ const input = input2 + (chunk * input_stride3) / 2;
        uint32_t* __restrict__ const outputi = outputi2 + (chunk * outputi_stride3) / 4;

        float sum = 0, sum2 = 0;
        static_assert(chunk_size == 8 * nthreads);
        const int i = 8 * thread;

        // Load all values in this chunk. We load all values at once in the beginning.
        const half2x4 xs = load_half2x4(input + i);
        const __half2 x01 = xs.x;
        const __half2 x23 = xs.y;
        const __half2 x45 = xs.z;
        const __half2 x67 = xs.w;

        const float x0 = __half2float(__low2half(x01));
        const float x1 = __half2float(__high2half(x01));
        sum += x0 + x1;
        sum2 += x0 * x0 + x1 * x1;
        const float x2 = __half2float(__low2half(x23));
        const float x3 = __half2float(__high2half(x23));
        sum += x2 + x3;
        sum2 += x2 * x2 + x3 * x3;
        const float x4 = __half2float(__low2half(x45));
        const float x5 = __half2float(__high2half(x45));
        sum += x4 + x5;
        sum2 += x4 * x4 + x5 * x5;
        const float x6 = __half2float(__low2half(x67));
        const float x7 = __half2float(__high2half(x67));
        sum += x6 + x7;
        sum2 += x6 * x6 + x7 * x7;

        sum += __shfl_sync(0xffffffff, sum, threadIdx.x ^ 1);
        sum += __shfl_sync(0xffffffff, sum, threadIdx.x ^ 2);
        sum += __shfl_sync(0xffffffff, sum, threadIdx.x ^ 4);
        sum += __shfl_sync(0xffffffff, sum, threadIdx.x ^ 8);
        sum += __shfl_sync(0xffffffff, sum, threadIdx.x ^ 16);
        sum2 += __shfl_sync(0xffffffff, sum2, threadIdx.x ^ 1);
        sum2 += __shfl_sync(0xffffffff, sum2, threadIdx.x ^ 2);
        sum2 += __shfl_sync(0xffffffff, sum2, threadIdx.x ^ 4);
        sum2 += __shfl_sync(0xffffffff, sum2, threadIdx.x ^ 8);
        sum2 += __shfl_sync(0xffffffff, sum2, threadIdx.x ^ 16);

        const float mean = sum / chunk_size;
        const float stddev = sqrt(sum2 / (chunk_size - 1));
        const float offset = mean;
        const float scale = fmax(1.0f, stddev / 2.3f);

        const __half2 offseth_scaleh = __floats2half2_rn(offset, scale);
        if (chunk == thread)
            outf = offseth_scaleh;

        const __half2 eff_scaleh = __half2half2(__float2half(1.0f / scale));
        const __half2 eff_offseth = __half2half2(__float2half(8.0f - offset / scale));

        const __half2 minh = __half2half2(__float2half(1.0f));
        const __half2 maxh = __half2half2(__float2half(15.0f));

        uint32_t outi = 0;

        const __half2 y01 = __hmax2(minh, __hmin2(maxh, __hfma2(x01, eff_scaleh, eff_offseth)));
        const uint32_t i0 = __half2int_rd(__low2half(y01));
        const uint32_t i1 = __half2int_rd(__high2half(y01));
        outi |= (i0 << 0) | (i1 << 4);

        const __half2 y23 = __hmax2(minh, __hmin2(maxh, __hfma2(x23, eff_scaleh, eff_offseth)));
        const uint32_t i2 = __half2int_rd(__low2half(y23));
        const uint32_t i3 = __half2int_rd(__high2half(y23));
        outi |= (i2 << 8) | (i3 << 12);

        const __half2 y45 = __hmax2(minh, __hmin2(maxh, __hfma2(x45, eff_scaleh, eff_offseth)));
        const uint32_t i4 = __half2int_rd(__low2half(y45));
        const uint32_t i5 = __half2int_rd(__high2half(y45));
        outi |= (i4 << 16) | (i5 << 20);

        const __half2 y67 = __hmax2(minh, __hmin2(maxh, __hfma2(x67, eff_scaleh, eff_offseth)));
        const uint32_t i6 = __half2int_rd(__low2half(y67));
        const uint32_t i7 = __half2int_rd(__high2half(y67));
        outi |= (i6 << 24) | (i7 << 28);

        outputi[i / 8] = outi;
    }
    outputf[thread * outputf_stride3] = outf;
}

////////////////////////////////////////////////////////////////////////////////

// Drivers (externally visible)

void cpu_quantize8(const __half* __restrict__ const input, __half* __restrict__ const outputf,
                   std::uint8_t* __restrict__ const outputi, const int input_size1,
                   const int input_size2, const int input_size3, const int input_stride2,
                   const int input_stride3, const int outputf_size1, const int outputf_size2,
                   const int outputf_size3, const int outputf_stride2, const int outputf_stride3,
                   const int outputi_size1, const int outputi_size2, const int outputi_size3,
                   const int outputi_stride2, const int outputi_stride3) {
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
    static_assert(chunks_per_frame > 0);

    static_assert(chunk_size % 2 == 0);
    assert(input_size1 % 2 == 0);
    assert(input_size1 % chunk_size == 0);
    assert(input_size3 % chunks_per_frame == 0);

    assert(outputf_size1 == 2 * input_size1 / chunk_size);
    assert(outputf_size2 == input_size2);
    assert(outputf_size3 == input_size3);
    assert(outputi_size1 == input_size1 / 2);
    assert(outputi_size2 == input_size2);
    assert(outputi_size3 == input_size3);

    for (int dim3 = 0; dim3 < input_size3; dim3 += chunks_per_frame) {
        for (int dim2 = 0; dim2 < input_size2; ++dim2) {
            for (int dim1 = 0; dim1 < input_size1; dim1 += chunk_size) {
                const int input_dim1 = dim1;
                const int outputf_dim1 = dim1 / chunk_size;
                const int outputi_dim1 = dim1 / 2;
                const int input_offset = input_dim1 + input_stride2 * dim2 + input_stride3 * dim3;
                const int outputf_offset =
                    outputf_dim1 + outputf_stride2 * dim2 + outputf_stride3 * dim3;
                const int outputi_offset =
                    outputi_dim1 + outputi_stride2 * dim2 + outputi_stride3 * dim3;
                cpu_quantize8_chunks(input + input_offset, outputf + outputf_offset,
                                     outputi + outputi_offset, input_stride3, outputf_stride3,
                                     outputi_stride3);
            }
        }
    }
}

void gpu_quantize8(const __half* __restrict__ const input, __half* __restrict__ const outputf,
                   std::uint8_t* __restrict__ const outputi, const int input_size1,
                   const int input_size2, const int input_size3, const int input_stride2,
                   const int input_stride3, const int outputf_size1, const int outputf_size2,
                   const int outputf_size3, const int outputf_stride2, const int outputf_stride3,
                   const int outputi_size1, const int outputi_size2, const int outputi_size3,
                   const int outputi_stride2, const int outputi_stride3, cudaStream_t stream) {
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
    static_assert(chunks_per_frame > 0);

    static_assert(chunk_size % 2 == 0);
    assert(input_size1 % 2 == 0);
    if (!(input_size1 % chunk_size == 0))
        std::cerr << "input_size1=" << input_size1 << " chunk_size=" << chunk_size << "\n";
    assert(input_size1 % chunk_size == 0);
    if (!(input_size3 % chunks_per_frame == 0))
        std::cerr << "input_size3=" << input_size3 << " chunks_per_frame=" << chunks_per_frame
                  << "\n";
    assert(input_size3 % chunks_per_frame == 0);

    assert(outputf_size1 == 2 * input_size1 / chunk_size);
    assert(outputf_size2 == input_size2);
    assert(outputf_size3 == input_size3);
    if (!(outputi_size1 == input_size1 / 2))
        std::cerr << "outputi_size1=" << outputi_size1 << " input_size1=" << input_size1 << "\n";
    assert(outputi_size1 == input_size1 / 2);
    assert(outputi_size2 == input_size2);
    assert(outputi_size3 == input_size3);

    dim3 nblocks;
    nblocks.x = input_size1 / chunk_size;
    nblocks.y = input_size2;
    nblocks.z = input_size3 / chunks_per_frame;
    const int shmem_nbytes = 0;
    gpu_quantize8_chunks<<<nblocks, nthreads, shmem_nbytes, stream>>>(
        input, outputf, outputi, input_size1, input_size2, input_size3, input_stride2,
        input_stride3, outputf_size1, outputf_size2, outputf_size3, outputf_stride2,
        outputf_stride3, outputi_size1, outputi_size2, outputi_size3, outputi_stride2,
        outputi_stride3);
}
