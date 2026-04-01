#ifndef _KSGPU_DEVICE_TRANSPOSES_HPP
#define _KSGPU_DEVICE_TRANSPOSES_HPP

#include <cuda_fp16.h>              // __half2
#include "constexpr_functions.hpp"  // constexpr_is_divisible
#include "device_fp16.hpp"          // f16_perm()

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Warp transpose


// Usage: to transpose thread bit t_3 with a register bit, do
//   warp_transpose(x,y,8);  // not warp_transpose(x,y,3)!
//
// Warning: the 'thread_stride' arg must be a power of 2, but this
// isn't checked!

template<typename T>
__device__ inline void warp_transpose(T &x, T &y, uint thread_stride)
{
    bool upper = (threadIdx.x & thread_stride);
    T src = upper ? x : y;
    T z = __shfl_sync(~0u, src, threadIdx.x ^ thread_stride);
    x = upper ? z : x;
    y = upper ? y : z;
}


// Usage: suppose we have 16 registers (4 register bits), and we want
// to transpose thread bit t_3 with register bit r_2. Then:
//
//   float x[16];
//   multi_warp_transpose<4,16> (x,8);   // not (x,3)!
//
// Warning: the 'thread_stride' arg must be a power of 2, but this
// isn't checked!

template<int RegisterStride, int NumRegisters, typename T>
__device__ inline void multi_warp_transpose(T x[NumRegisters], uint thread_stride)
{
    static_assert(NumRegisters > 0);
    static_assert(RegisterStride > 0);
    static_assert(constexpr_is_divisible(NumRegisters, 2*RegisterStride));

    constexpr int R = RegisterStride;
    constexpr int S = NumRegisters / (2*R);

    #pragma unroll
    for (int s = 0; s < S; s++) {
        #pragma unroll
        for (int r = 0; r < R; r++)
            warp_transpose(x[(2*s)*R + r], x[(2*s+1)*R + r], thread_stride);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Half2 "local" transpose


// half2_transpose() replaces (a,b) by ([a0,b0],[a1,b1]).
__device__ inline void half2_transpose(__half2 &x, __half2 &y)
{
    __half2 xnew = __lows2half2(x,y);
    __half2 ynew = __highs2half2(x,y);
    
    x = xnew;
    y = ynew;
}


template<int RegisterStride, int NumRegisters>
__device__ inline void multi_half2_transpose(__half2 x[NumRegisters])
{
    static_assert(NumRegisters > 0);
    static_assert(RegisterStride > 0);
    static_assert(constexpr_is_divisible(NumRegisters, 2*RegisterStride));

    constexpr int R = RegisterStride;
    constexpr int S = NumRegisters / (2*R);

    #pragma unroll
    for (int s = 0; s < S; s++) {
        #pragma unroll
        for (int r = 0; r < R; r++)
            half2_transpose(x[(2*s)*R + r], x[(2*s+1)*R + r]);
    }    
}


// -------------------------------------------------------------------------------------------------
//
// "Warp and half2" transpose



__device__ inline void warp_and_half2_transpose(__half2 &x, uint thread_stride)
{
    bool upper = (threadIdx.x & thread_stride);
    uint s = upper ? 0x3276 : 0x5410;
    __half2 y = __shfl_sync(~0u, x, threadIdx.x ^ thread_stride);
    x = f16_perm(x,y,s);   // upper ? [y1,x1] : [x0,y0]
}


// Equivalent to:
//
//   _warp_and_half2_transpose(x, thread_stride);
//   _warp_and_half2_transpose(y, thread_stride);
//
// but uses one call to __shfl_sync() instead of two.

__device__ inline void warp_and_half2_transpose(__half2 &x, __half2 &y, uint thread_stride)
{
    bool upper = (threadIdx.x & thread_stride);
    uint s1 = upper ? 0x5410 : 0x7632;
    uint s2 = upper ? 0x3254 : 0x5410;
    uint s3 = upper ? 0x3276 : 0x7610;
    
    __half2 z = f16_perm(x,y,s1);  // upper ? [x0,y0] : [x1,y1]
    z = __shfl_sync(~0u, z, threadIdx.x ^ thread_stride);
    x = f16_perm(x,z,s2);          // upper ? [z0,x1] : [x0,z0]
    y = f16_perm(y,z,s3);          // upper ? [z1,y1] : [y0,z1]
}


} // namespace ksgpu

#endif // _KSGPU_DEVICE_TRANSPOSES_HPP
