#include <cuda_fp16.h>
#include <cuda_runtime.h>

const int chunksPerFrame = 32;

__device__ unsigned int clamped_round_down(__half x) {
    x = __hmax(x, __float2half(1.5f));
    x = __hmin(x, __float2half(15.5f));
    return __half2int_rd(x);
}

__device__ float sum_half2(__half2 x) {
    return __low2float(x) + __high2float(x);
}

__global__ void quantize(const __half2* __restrict__ in_base, __half2* __restrict__ outf_base,
                         unsigned int* __restrict__ outi_base,
                         const int* __restrict__ index_array) {
    __half2* __restrict__ outf = outf_base + index_array[64 * blockIdx.x + 32];
    unsigned int* __restrict__ outi = outi_base + index_array[64 * blockIdx.x + 33];
    __half2 outf_temp = __float2half2_rn(0);
    for (int chunk = 0; chunk < chunksPerFrame; chunk++) {
        const __half2* chunkIn =
            in_base + index_array[64 * blockIdx.x + chunk]; // the memory for the current chunk

        float mean = 0;
        for (int j = 0; j < 4; j++) // moving along the chunk
        {
            mean += sum_half2(chunkIn[j * 32 + threadIdx.x]);
        }
        mean += __shfl_sync(0xffffffff, mean, threadIdx.x ^ 1);
        mean += __shfl_sync(0xffffffff, mean, threadIdx.x ^ 2);
        mean += __shfl_sync(0xffffffff, mean, threadIdx.x ^ 4);
        mean += __shfl_sync(0xffffffff, mean, threadIdx.x ^ 8);
        mean += __shfl_sync(0xffffffff, mean, threadIdx.x ^ 16);
        mean /= 256;

        float scale = 0;
        for (int j = 0; j < 4; j++) {
            const __half2 diff = __hsub2(chunkIn[j * 32 + threadIdx.x], __float2half2_rn(mean));
            scale += sum_half2(__hmul2(diff, diff));
        }
        scale += __shfl_sync(0xffffffff, scale, threadIdx.x ^ 1);
        scale += __shfl_sync(0xffffffff, scale, threadIdx.x ^ 2);
        scale += __shfl_sync(0xffffffff, scale, threadIdx.x ^ 4);
        scale += __shfl_sync(0xffffffff, scale, threadIdx.x ^ 8);
        scale += __shfl_sync(0xffffffff, scale, threadIdx.x ^ 16);
        // scale = (sqrtf(scale / 255)) / 2.3f;
        scale = sqrtf(scale) / (2.3f * sqrtf(255));

        outf_temp = (threadIdx.x == chunk)
                        ? __floats2half2_rn(mean, scale)
                        : outf_temp; // for each thread, saving a single outf_temp value, then I'll
                                     // merge them all later

        scale = (scale == 0) ? 1 : scale; // avoid division by 0

        // computing i values
        __half meanh = __float2half(mean);
        __half scaleh = __float2half(scale);
        unsigned int i =
            clamped_round_down((__low2half(chunkIn[threadIdx.x]) - meanh) / scaleh + __half(8.5f));
        i = i
            | (clamped_round_down((__high2half(chunkIn[threadIdx.x]) - meanh) / scaleh
                                  + __half(8.5f))
               << 4);
        i = i
            | (clamped_round_down((__low2half(chunkIn[threadIdx.x + 32]) - meanh) / scaleh
                                  + __half(8.5f))
               << 8);
        i = i
            | (clamped_round_down((__high2half(chunkIn[threadIdx.x + 32]) - meanh) / scaleh
                                  + __half(8.5f))
               << 12);
        i = i
            | (clamped_round_down((__low2half(chunkIn[threadIdx.x + 64]) - meanh) / scaleh
                                  + __half(8.5f))
               << 16);
        i = i
            | (clamped_round_down((__high2half(chunkIn[threadIdx.x + 64]) - meanh) / scaleh
                                  + __half(8.5f))
               << 20);
        i = i
            | (clamped_round_down((__low2half(chunkIn[threadIdx.x + 96]) - meanh) / scaleh
                                  + __half(8.5f))
               << 24);
        i = i
            | (clamped_round_down((__high2half(chunkIn[threadIdx.x + 96]) - meanh) / scaleh
                                  + __half(8.5f))
               << 28);

        unsigned int i_shfl = __shfl_sync(0xffffffff, i, threadIdx.x ^ 1);
        unsigned int selector = (threadIdx.x % 2) ? (3 << 12) + (7 << 8) + (1 << 4) + (5 << 0)
                                                  : (6 << 12) + (2 << 8) + (4 << 4) + (0 << 0);
        i = __byte_perm(i, i_shfl, selector);
        i_shfl = __shfl_sync(0xffffffff, i, threadIdx.x ^ 2);
        selector = (threadIdx.x % 4 / 2) ? (3 << 12) + (2 << 8) + (7 << 4) + (6 << 0)
                                         : (5 << 12) + (4 << 8) + (1 << 4) + (0 << 0);
        i = __byte_perm(i, i_shfl, selector);

        outi[chunk * 32 + threadIdx.x / 4 + (threadIdx.x % 4) * 8] = i; // writing result to memory
    }
    // writing the means and scales for this frame to global
    outf[threadIdx.x] = outf_temp;
}
