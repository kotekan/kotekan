#include <cuda_runtime.h>
#include <cuda_fp16.h>
//#include "header.cuh"
#include <stdio.h>      /*printf*/

#define chunksPerFrame 32

__device__ unsigned int clamped_round_down(__half x)
{
    x = (__hgt(x, 1.5)) ? x : __float2half(1.5);      // max(x, 1.5)
    x = (__hlt(x, 15.5)) ? x : __float2half(15.5);  // min(x, 15.5)
    
    return __half2int_rd(x);
}

__global__ void quantize (const __half2 *__restrict__ in_base, __half2 *__restrict__ outf_base, unsigned int *__restrict__ outi_base, const int *__restrict__ index_array)
{
    __half2 * __restrict__ outf = outf_base + index_array[64*blockIdx.x + 32];
    unsigned int * __restrict__ outi = outi_base + index_array[64*blockIdx.x + 33];
    __half2 outf_temp = __float2half2_rn(0);
    for (int chunk = 0; chunk < chunksPerFrame; chunk++)
    {
        const __half2 * chunkIn = in_base + index_array[64*blockIdx.x + chunk]; //the memory for the current chunk
    
        __half2 h2sum = __half2half2(0);
        for (int j = 0; j < 4; j++) //moving along the chunk
        {
	    h2sum = __hadd2(h2sum, chunkIn[j*32 + threadIdx.x]);
        }
        __half mean = __hadd(__low2half(h2sum), __high2half(h2sum));
        mean = __hadd(mean,__shfl_sync(0xffffffff, mean, threadIdx.x ^ 1));
        mean = __hadd(mean,__shfl_sync(0xffffffff, mean, threadIdx.x ^ 2));
        mean = __hadd(mean,__shfl_sync(0xffffffff, mean, threadIdx.x ^ 4));
        mean = __hadd(mean,__shfl_sync(0xffffffff, mean, threadIdx.x ^ 8));
        mean = __hadd(mean,__shfl_sync(0xffffffff, mean, threadIdx.x ^ 16));
        mean = __hdiv(mean, 256.0);

        h2sum = __half2half2(0);
        for (int j = 0; j < 4; j++)
        {
	    const __half2 diff = __hsub2(chunkIn[j*32 + threadIdx.x], __half2half2(mean));
            h2sum = __hadd2(h2sum, __hmul2(diff, diff));
        }
	__half scale = __hadd(__low2half(h2sum), __high2half(h2sum));
        scale = __hadd(scale, __shfl_sync(0xffffffff, scale, threadIdx.x ^ 1));
        scale = __hadd(scale, __shfl_sync(0xffffffff, scale, threadIdx.x ^ 2));
        scale = __hadd(scale, __shfl_sync(0xffffffff, scale, threadIdx.x ^ 4));
        scale = __hadd(scale, __shfl_sync(0xffffffff, scale, threadIdx.x ^ 8));
        scale = __hadd(scale, __shfl_sync(0xffffffff, scale, threadIdx.x ^ 16));
        scale = __hdiv(hsqrt(__hdiv(scale, 255.0)), 2.3);

	outf_temp = (threadIdx.x == chunk) ?  __halves2half2(mean, scale) : outf_temp; //for each thread, saving a single outf_temp value, then I'll merge them all later
        
        scale = (__heq(scale, 0)) ? __float2half(1.0) : scale; //avoid division by 0
        
        //computing i values
        unsigned int i = clamped_round_down(__hadd((__low2half(chunkIn[threadIdx.x]) - mean)/scale, __half(8.5)));
	i = i | (clamped_round_down(__hadd((__high2half(chunkIn[threadIdx.x])      - mean)/scale, __half(8.5))) << 4);
	i = i | (clamped_round_down(__hadd((__low2half( chunkIn[threadIdx.x + 32]) - mean)/scale, __half(8.5))) << 8);
        i = i | (clamped_round_down(__hadd((__high2half(chunkIn[threadIdx.x + 32]) - mean)/scale, __half(8.5))) << 12);
	i = i | (clamped_round_down(__hadd((__low2half( chunkIn[threadIdx.x + 64]) - mean)/scale, __half(8.5))) << 16);
        i = i | (clamped_round_down(__hadd((__high2half(chunkIn[threadIdx.x + 64]) - mean)/scale, __half(8.5))) << 20);
	i = i | (clamped_round_down(__hadd((__low2half( chunkIn[threadIdx.x + 96]) - mean)/scale, __half(8.5))) << 24);
        i = i | (clamped_round_down(__hadd((__high2half(chunkIn[threadIdx.x + 96]) - mean)/scale, __half(8.5))) << 28);

        unsigned int i_shfl = __shfl_sync(0xffffffff, i, threadIdx.x ^ 1);
        unsigned int selector = (threadIdx.x % 2) ? (3<<12) + (7<<8) + (1<<4) + (5<<0) : (6<<12) + (2<<8) + (4<<4) + (0<<0);
        i = __byte_perm(i, i_shfl, selector);
        i_shfl = __shfl_sync(0xffffffff, i, threadIdx.x ^ 2);
        selector = (threadIdx.x%4/2) ? (3<<12) + (2<<8) + (7<<4) + (6<<0) : (5<<12) + (4<<8) + (1<<4) + (0<<0);
        i = __byte_perm(i, i_shfl, selector);

        outi[chunk*32 + threadIdx.x/4 + (threadIdx.x%4)*8] = i; //writing result to memory
    }
    //writing the means and scales for this frame to global
    outf[threadIdx.x] = outf_temp;
}
