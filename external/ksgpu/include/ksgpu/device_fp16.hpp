#ifndef _KSGPU_DEVICE_FP16_HPP
#define _KSGPU_DEVICE_FP16_HPP

#include <cuda_fp16.h>

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// half4_load, half4_store: 64-bit (4 × float16) load/store using uint2
//
// These functions provide a clean interface for loading/storing 4 half values
// using 64-bit transactions. The uint2 type is reinterpreted as 2 × __half2.


__device__ __forceinline__ void half4_load(const uint2 *p, __half2 &x, __half2 &y)
{
    uint2 u = *p;
    x = *reinterpret_cast<__half2*>(&u.x);
    y = *reinterpret_cast<__half2*>(&u.y);
}


__device__ __forceinline__ void half4_store(uint2 *p, __half2 x, __half2 y)
{
    uint2 u;
    u.x = *reinterpret_cast<uint*>(&x);
    u.y = *reinterpret_cast<uint*>(&y);
    *p = u;
}


// -------------------------------------------------------------------------------------------------
//
// half8_load, half8_store: 128-bit (8 × float16) load/store using uint4
//
// These functions provide a clean interface for loading/storing 8 half values
// using 128-bit transactions. The uint4 type is reinterpreted as 4 × __half2.


__device__ __forceinline__ void half8_load(const uint4 *p, __half2 &x, __half2 &y, __half2 &z, __half2 &w)
{
    uint4 u = *p;
    x = *reinterpret_cast<__half2*>(&u.x);
    y = *reinterpret_cast<__half2*>(&u.y);
    z = *reinterpret_cast<__half2*>(&u.z);
    w = *reinterpret_cast<__half2*>(&u.w);
}


__device__ __forceinline__ void half8_store(uint4 *p, __half2 x, __half2 y, __half2 z, __half2 w)
{
    uint4 u;
    u.x = *reinterpret_cast<uint*>(&x);
    u.y = *reinterpret_cast<uint*>(&y);
    u.z = *reinterpret_cast<uint*>(&z);
    u.w = *reinterpret_cast<uint*>(&w);
    *p = u;
}


// ----------------------------------------------------------------------------------------------
//
// Given __half2 variables a = [a0,a1] and b = [b0,b1]:
//
//    __lows2half2() returns [a0,b0]
//    __highs2half2() returns [a1,b1]
//    f16_align() returns [a1,b0]
//    f16_blend() returns [a0,b1]
//    f16_perm() returns any pair, determined by a 'selector' word.


__device__ __forceinline__
__half2 f16_align(__half2 a, __half2 b)
{
    __half2 d;
    
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
    // Note: I chose to use prmt.b32.f4e(d,a,b,2) but I think prmt.b32(d,a,b,0x5432) is equivalent.
    
    asm("prmt.b32.f4e %0, %1, %2, %3;" :
        "=r" (*(unsigned int *) &d) :
        "r" (*(const unsigned int *) &a),
        "r" (*(const unsigned int *) &b),
        "n"(2)
    );

    return d;
}

__device__ __forceinline__
__half2 f16_blend(__half2 a, __half2 b)
{
    __half2 d;
        
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
    asm("prmt.b32 %0, %1, %2, %3;" :
        "=r" (*(unsigned int *) &d) :
        "r" (*(const unsigned int *) &a),
        "r" (*(const unsigned int *) &b),
        "n"(0x7610)
    );

    return d;
}


__device__ __forceinline__
__half2 f16_perm(__half2 a, __half2 b, unsigned int s)
{
    // This is just __byte_perm(), but hacked up with casts so that the inputs/outputs have dtype __half2.
    //
    // The "selector" s should have the form 0xMMNN, where:
    //   - NN corresponds to the low 16 bits of the output __half2.
    //   - MM corresponds to the high 16 bits of the output __half2.
    //   - Each of MM, NN is {10, 32, 54, 76} for {a0, a1, b0, b1} respectively.
    //
    // Examples:
    //   - [a0,b0] can be returned with either __lows2half2() or f16_perm(0x5410).
    //   - [a1,b1] can be returned with either __highs2half2() or f16_perm(0x7632).
    //   - [a1,b0] can be returned with either f16_align() or f16_perm(0x5432).
    //   - [a0,b1] can be returned with either f16_blend() or f16_perm(0x7610).
    
    __half2 d;

    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
    asm("prmt.b32 %0, %1, %2, %3;" :
        "=r" (*(unsigned int *) &d) :
        "r" (*(const unsigned int *) &a),
        "r" (*(const unsigned int *) &b),
        "r" (s)
    );

    return d;
}


} // namespace ksgpu

#endif // _KSGPU_DEVICE_FP16_HPP

