#include "../include/pirate/ChimeBeamformer.hpp"

#include <algorithm>   // std::max
#include <cmath>       // cos, sin, M_PI
#include <complex>
#include <cuda_fp16.h>
#include <cufftdx.hpp>

#include <iostream>

#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>   // CUDA_PEEK
#include <ksgpu/KernelTimer.hpp>
#include <ksgpu/test_utils.hpp>   // assert_arrays_equal, rand_int

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Bandpass constants from kotekan (either lib/hsa/kernels/frb_upchan_amd.cl
// or lib/testing/gpuBeamformSimulate.cpp). These are used below in the CPU
// reference implementation.
static const float BP[16] = {
    0.52225748f, 0.58330915f, 0.6868705f,  0.80121821f,
    0.89386546f, 0.95477358f, 0.98662733f, 0.99942558f,
    0.99988676f, 0.98905127f, 0.95874124f, 0.90094667f,
    0.81113021f, 0.6999944f,  0.59367968f, 0.52614263f
};

// In the GPU kernel, it's more convenient to use BP48 = 1/(48*BP).
__constant__ float BP48[16] = {
    1.0f / (48 * 0.52225748f),
    1.0f / (48 * 0.58330915f),
    1.0f / (48 * 0.68687050f),
    1.0f / (48 * 0.80121821f),
    1.0f / (48 * 0.89386546f),
    1.0f / (48 * 0.95477358f),
    1.0f / (48 * 0.98662733f),
    1.0f / (48 * 0.99942558f),
    1.0f / (48 * 0.99988676f),
    1.0f / (48 * 0.98905127f),
    1.0f / (48 * 0.95874124f),
    1.0f / (48 * 0.90094667f),
    1.0f / (48 * 0.81113021f),
    1.0f / (48 * 0.69999440f),
    1.0f / (48 * 0.59367968f),
    1.0f / (48 * 0.52614263f)
};

        
// -------------------------------------------------------------------------------------------------
//
// _load_half4(), _store_half4(): load/store (4xfp16) with a single 64-bit instruction.
// The pointer 'p' must be 64-bit aligned.


__device__ __forceinline__ __half2 __uint_as_half2(unsigned int x)
{
    // According to Claude Code, memcpy() for type punning is well-defined C++,
    // and nvcc will optimize it away to a no-op (same register, reinterpreted).
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return h;
}

__device__ __forceinline__ unsigned int __half2_as_uint(__half2 h)
{
    unsigned int x;
    memcpy(&x, &h, sizeof(x));
    return x;
}

__device__ __forceinline__ void _load_half4(const __half2 *p, __half2 &x, __half2 &y)
{
    // According to Claude Code, the reinterpret_cast through (uint2 *) guarantees a single
    // 64-bit load/store instruction (LDG.64 or LDS.64) rather than two 32-bit transactions.
    uint2 tmp = *reinterpret_cast<const uint2 *>(p);
    x = __uint_as_half2(tmp.x);
    y = __uint_as_half2(tmp.y);
}

__device__ __forceinline__ void _store_half4(__half2 *p, __half2 x, __half2 y)
{
    uint2 tmp;
    tmp.x = __half2_as_uint(x);
    tmp.y = __half2_as_uint(y);
    *reinterpret_cast<uint2 *>(p) = tmp;
}


// -------------------------------------------------------------------------------------------------


// Helper for _fft128_sq() and chime_frb_upchan().
//
// This is a thread-local operation, whose input 'x' is a (2,32) array
// with register assignment:
//   register:  i
//   thread:    j4 j3 j2 j1 j0
//
// The output is an array 'y' formed by summing 'x' over one of the index
// bits j_m, to obtain a (2,16) array with register assignment:
//   thread:    j4 ... i ... j0   (j0 replaces j_m).
//
// The 'lane' argument is (1 << m).

__device__ inline float _asymmetric_reduce(float x0, float x1, uint lane)
{
    float t = (threadIdx.x & lane) ? x0 : x1;
    t = __shfl_sync(~0U, t, threadIdx.x ^ lane);
    t += (threadIdx.x & lane) ? x1 : x0;
    return t;
}


// Helper for chime_frb_upchan().
//
// This is a warp-local operation, whose input x[t] is a length-128 float16+16 array
// with register assignment:
//   simd:      ReIm
//   register:  t6 t5
//   thread:    t4 t3 t2 t1 t0
//
// We convert float16+16 -> float32+32, and take an FFT, to obtain a length-128 array y[f]
// with index bits (f6 f5 f4 f3 f2 f1 f0)_2.
//
// Next, we take the absolute square, and sum array elements in consecutive groups of 4
// to obtain a length-32 array I[]. We'll parameterize I-indices using index bits (f6 f5 f4 f3 f2)_2,
// reflecting the structure of the sum ("consecutive groups of 4" means summing over f0,f1).
//
// We return the z array in register assignment (one float32 per thread):
//   thread: f4 f3 f2 f6 f5

__device__ inline float _fft128_sq(__half2 x0, __half2 x1, __half2 x2, __half2 x3, char *smem)
{
    // The #ifdef weirdness is needed to avoid nvcc errors during the "host" pass.
    using FFT = decltype(
          cufftdx::Size<128>()
          + cufftdx::Precision<float>()       // or __half
          + cufftdx::Type<cufftdx::fft_type::c2c>()
          + cufftdx::Direction<cufftdx::fft_direction::inverse>()
          + cufftdx::ElementsPerThread<4>()
          + cufftdx::FFTsPerBlock<16>()
#ifdef __CUDA_ARCH__
          + cufftdx::SM<__CUDA_ARCH__>()
#else
          + cufftdx::SM<890>()
#endif
          + cufftdx::Block()
      );

    using complex_t = typename FFT::value_type;
    static_assert(sizeof(complex_t) == 8);
    static_assert(FFT::storage_size == 4);
    static_assert(FFT::shared_memory_size <= 16*1024);  // see "crucial note" in chime_frb_upchan()

    complex_t data[4];
    data[0] = complex_t(__half2float(x0.x), __half2float(x0.y));
    data[1] = complex_t(__half2float(x1.x), __half2float(x1.y));
    data[2] = complex_t(__half2float(x2.x), __half2float(x2.y));
    data[3] = complex_t(__half2float(x3.x), __half2float(x3.y));
    
    // FFT and absolute square.
    
    FFT().execute(data, smem);    

    float y0 = data[0].x * data[0].x + data[0].y * data[0].y;
    float y1 = data[1].x * data[1].x + data[1].y * data[1].y;
    float y2 = data[2].x * data[2].x + data[2].y * data[2].y;
    float y3 = data[3].x * data[3].x + data[3].y * data[3].y;
    
    // After these steps, 'y' has register assignment
    //   register: f6 f5
    //   thread:   f4 f3 f2 f1 f0

    // Call _asymmetric_reduce(), to sum over f0/f1, while "shuffling" f6/f5.
    float w0 = _asymmetric_reduce(y0, y1, 0x1);   // sum over f0
    float w1 = _asymmetric_reduce(y2, y3, 0x1);   // sum over f0
    float z = _asymmetric_reduce(w0, w1, 0x2);    // sum over f1
    
    // After these steps, 'z' has register assignment:
    //   thread: f4 f3 f2 f6 f5

    return z;
}


// This is the "upchannelization" part of the CHIME FRB beamforming kernel, which
// runs after the "beamforming" part.
//
//  'data': shape=(T,F,2,B), dtype=float16+16, axes (time,freq,pol,beam)
//  'results_array': shape=(B,F,T/384,16), axes (beam,cfreq,time,ufreq)
//
// where in the results_array, the index 0 <= cfreq < F represents a "coarse"
// frequency, and the index 0 <= ufreq < 16 represents an "upchannelized" frequency
// within a coarse channel. Note that the 'data' array is indexed by a coarse
// frequency 0 <= f < F.
//
// Here is a specification of the kernel:
//
//  0. Each (cfreq, beam) is processed independently, and each length-384 block
//     of time indices is also processed independently. Therefore, in order to
//     streamline notation, we can denote the input/output arrays as having the
//     following shapes:
//
//         float16+16  data[384][2];       // (time,pol)
//         float       results_array[16];  // ufreq
//
//  1. Divide the data into three length-128 time chunks, i.e. reshape to
//
//         float16+16  data[3][128][2];    // (thi,tlo,pol)
//
//  2. Fourier transform the length-128 'tlo' index, to obtain an array:
//
//         float32+32  F[3][128][2];       // (thi,f,pol)
//         F[thi,f,pol] = sum_{tlo} exp(+2pi*i*f*tlo/128) data[thi,tlo,pol]
//
//     The index 0 <= f < 128 represents a "highly upchannelized frequency".
//     Note that the FFT is performed with positive exponent exp(+2pi*i ...)
//     and without a (1/128) prefactor.
//
//  3. Take the absolute square (complex -> real), and downsample by
//     a factor 3 in time, 8 in frequency, 2 in polarization:
//
//         float32 G[16];     // ufreq
//         G[ufreq] = sum_{thi=0}^{2} sum_{pol=0}^{1}
//                      sum_{f = 8*ufreq}^{8*ufreq+7} |F[thi,f,pol]|^2
//
//  4. Write to global memory with a factor (1/48), an index shift,
//     and a bandpass correction:
//
//         results_array[ufreq] = G[ufreq ^ 8] / (48 * BP[ufreq]);
//
// Each threadblock processes (32 beams, one coarse freq, 768 times).
// Thus, gridDim = (B/32,F,T/768). We assume that B is a multiple of 32,
// and T is a multiple of 768. The values of (B,F,T) are supplied to the
// kernel via gridDims (not kernel args).
//
// The kernel is launched with 16 warps, and blockDim = {32,16}.


__global__ void __launch_bounds__(512,2)
chime_frb_upchan(const __half2 *__restrict__ data, float *__restrict__ results_array)
{
    // Shared memory layout (33 KB/threadblock):
    //
    //   __half2  smem_e[128][34];      // 17 KB
    //   char     smem_fft[16*1024];    // 16 KB
    //
    // The 'smem_e' array is used to reorganize float16+16 input data, after reading
    // from global memory, and before the FFT kernel. The 'smem_fft' array is used as
    // scratch space by cufftdx. (Crucial note: in principle, the cufftdx scratch space
    // is hardware-dependendent. There is a static_assert() in _fft128_sq() which
    // checks that the scratch space is <= 16*1024 when compiling for a new gpu arch.)
    
    __shared__ char smem_all[33*1024];

    // Block indices. Throughout this kernel, 't768' denotes a time index which has
    // been downsampled by a factor 768 relative to the 'data' array, and analogously
    // for t384, t128, etc.
    
    uint b32 = blockIdx.x;
    uint f = blockIdx.y;
    uint t768 = blockIdx.z;

    uint B32 = gridDim.x;
    uint F = gridDim.y;
    uint T768 = gridDim.z;
    
    // Apply per-block offsets to 'data' pointer.
    //   before: shape=(T,F,2,B), axes (time,freq,pol,beam)
    //   after: shape=(768,2,32), strides (2*F*B, B, 1)

    data += (768UL*t768*F + f) * (64 * B32);
    data += (b32 * 32);

    // Apply per-block offsets to 'results_array'.
    //   before: shape=(B,F,T/384,16), axes (beam,cfreq,t384,ufreq).
    //   after:  shape=(32,2,16), strides (32*F*T768, 16, 1)

    results_array += (b32 * 32UL * F + f) * ulong(T768 * 32);
    results_array += 32 * t768;

    // 'g' will accumulate intensity values for 32 beams, 16 ufreqs, 2 times.
    // This is the quantity denoted G[] in the long comment above.
    //
    // It's convenient to use the following weird register assignment
    //   register:  t384
    //   thread:    f4 f3 b0 f6 f5
    //   warp:      b4 b3 b2 b1
    //
    // Note that we denote the length-2 time index by 't384'.
    // We parametrize the length-16 ufreq axis using index bits (f6 f5 f4 f3)_2,
    // since it will obtained by summing a length-128 axis in consecutive groups of 8
    // (i.e. summing over f0,f1,f2). See related comment above _fft128_sq().
    
    float g0 = 0.0f;
    float g1 = 0.0f;

    // Outer loop over 6 blocks of 128 time samples (i.e. 768 time samples total)
    for (uint t128 = 0; t128 < 6; t128++) {
        for (uint pol = 0; pol < 2; pol++) {
            
            // Read data from global memory (128 times, 32 beams) in register assignment
            //   register:  t6 t5 b0
            //   thread:    t0 b4 b3 b2 b1
            //   warp:      t4 t3 t2 t1
            //
            // We use 64-bit load instructions here. (This kernel is expected to be GPU memory
            // bandwidth limited, so this detail determines overall performance.)
            
            uint bhi = (threadIdx.x & 0xf);                      // index bits (b4 b3 b2 b1)_2
            uint tlo = (threadIdx.x >> 4) | (threadIdx.y << 1);  // index bits (t4 t3 t2 t1 t0)_2

            // Apply remaining 'data' offsets (per-warp, per-thread, t128, pol).
            // We don't apply the (t6, t5, b0) offsets.
            // After these offsets are applied, 'data' is a per-thread shape (4,2) array.
            //   before: shape (768, 2, 32), strides (2*F*B, B, 1)
            //   after:  shape (4,2), strides (64*F*B, 1)

            ulong i = (t128 << 7) + tlo;
            i = i * (F << 1) + pol;
            i = i * (B32 << 5) + (bhi << 1);
            
            const __half2 *dp = data + i;
            ulong dstride = (ulong(F) * ulong(B32)) << 11;  // 64*F*B
            
            __half2 e0, e1, e2, e3, e4, e5, e6, e7;
            
            _load_half4(dp, e0, e1);
            _load_half4(dp + dstride, e2, e3);
            _load_half4(dp + 2*dstride, e4, e5);
            _load_half4(dp + 3*dstride, e6, e7);
            
            // Write to shared memory, using 64-bit bank-conflict-free store instructions.
            //   register:  t6 t5 b0
            //   thread:    t0 b4 b3 b2 b1
            //   warp:      t4 t3 t2 t1
            //   __half2    smem_e[128][34];   (time, beam)

            __half2 *sp = (__half2 *)smem_all + (34*tlo) + (2*bhi);

            _store_half4(sp, e0, e1);
            _store_half4(sp + 32*34, e2, e3);
            _store_half4(sp + 64*34, e4, e5);
            _store_half4(sp + 96*34, e6, e7);
            
            __syncthreads();

            // Read from shared memory, using 64-bit bank-conflict-free loads, into the
            // following register assignment (128 times, 32 beams):
            //   register:  t6 t5 b0
            //   thread:    t4 t3 t2 t1 t0
            //   warp:      b4 b3 b2 b1
            //   __half2    smem_e[128][34];   (time, beam)

            bhi = threadIdx.y;
            tlo = threadIdx.x;
            sp = (__half2 *)smem_all + (34*tlo) + (2*bhi);
            
            _load_half4(sp, e0, e1);
            _load_half4(sp + 32*34, e2, e3);
            _load_half4(sp + 64*34, e4, e5);
            _load_half4(sp + 96*34, e6, e7);

            // Since each warp does an independent FFT, it's not safe to assume that cufftdx calls __syncthreads().
            __syncthreads();

            // FFT, square, sum over f0, f1.
            char *smem_fft = smem_all + 17*1024;
            float t0 = _fft128_sq(e0, e2, e4, e6, smem_fft);
            float t1 = _fft128_sq(e1, e3, e5, e7, smem_fft);

            // After the calls to _fft128_sq(), we have intensities which have been summed
            // over 4 frequencies (bits f0,f1).
            //   register:  b0
            //   thread:    f4 f3 f2 f6 f5
            //   warp:      b4 b3 b2 b1

            // Sum over f2. We've now summed over 8 frequencies.
            float t = _asymmetric_reduce(t0, t1, 0x4);

            // Now we have intensities which have been summed over 8 freqs (bits f0,f1,f2).
            //   thread:    f4 f3 b0 f6 f5
            //   warp:      b4 b3 b2 b1
            //
            // Accumulate into (g0, g1).
            // Reminder: g register layout is
            //   register:  t384
            //   thread:    f4 f3 b0 f6 f5
            //   warp:      b4 b3 b2 b1
            
            g0 = (t128 < 3) ? (g0+t) : (g0);
            g1 = (t128 < 3) ? (g1) : (g1+t);
        } // end of pol loop
    }     // end of t128 loop

    // When we get here, the (g0,g1) registers are fully summed over 3 t128 values, 2 pols, 8 freqs.
    // Before writing to gpu global memory, we just need to permute the register assignment and
    // apply the bandpass correction.
    //
    // Reminder: g register layout is
    //   register:  t384
    //   thread:    f4 f3 b0 f6 f5
    //   warp:      b4 b3 b2 b1

    // Warp shuffle: exchange register bit with thread bit 't2'.
    float gtmp = (threadIdx.x & 0x4) ? g0 : g1;
    gtmp = __shfl_sync(~0u, gtmp, threadIdx.x ^ 0x4);
    g0 = (threadIdx.x & 0x4) ? gtmp : g0;
    g1 = (threadIdx.x & 0x4) ? g1 : gtmp;

    // Now g register layout is:
    //   register:  b0
    //   thread:    f4 f3 t384 f6 f5
    //   warp:      b4 b3 b2 b1

    // Next step is permuting threads (f4 f3 t384 f6 f5) -> (t384 f6 f5 f4 f3).
    uint lane = (threadIdx.x & 0x1c) >> 2;  // t384 f6 f5
    lane |= (threadIdx.x & 0x3) << 3;       // f4 f3

    g0 = __shfl_sync(~0u, g0, lane);
    g1 = __shfl_sync(~0u, g1, lane);
    
    // Now g register layout is:
    //   register:  b0
    //   thread:    t384 f6 f5 f4 f3
    //   warp:      b4 b3 b2 b1

    // Apply the index shift ufreq -> (ufreq ^ 8). (See step 4 in the long comment above.)
    // This could be coalesced with the previous permutation, but we keep it separate
    // for clarity, since the kernel is memory bandwidth limited anyway.

    g0 = __shfl_sync(~0u, g0, threadIdx.x ^ 0x8);
    g1 = __shfl_sync(~0u, g1, threadIdx.x ^ 0x8);

    // Multiply by 1 / (48 * BP[ufreq]). (See step 4 in the long comment above.)
    
    float bp48 = BP48[threadIdx.x & 0xf];
    g0 *= bp48;
    g1 *= bp48;

    // Before writing to global gpu memory ('results_array'),
    // apply per-warp and per-thread offsets to results_array.
    //   before: shape (32,2,16), strides (32*F*T768, 16, 1)
    //   after:  shape (2,), stride (32*F*T768)

    ulong bstride = 32 * ulong(F) * ulong(T768);
    results_array += threadIdx.y * (2 * bstride) + (threadIdx.x);
    
    results_array[0] = g0;
    results_array[bstride] = g1;
}

// 'data': shape=(T,F,2,B,2), axes (time,freq,pol,beam,ReIm)
// 'results_array': shape=(B,F,T/384,16), axes (beam,cfreq,time,ufreq)
void launch_chime_frb_upchan(const __half *data, float *results_array, long T, long F, long B, cudaStream_t stream)
{
    xassert(T > 0);
    xassert(F > 0);
    xassert(B > 0);
    xassert_divisible(T, 768);
    xassert_divisible(B, 32);

    long T768 = T / 768;
    long B32 = B / 32;

    chime_frb_upchan<<< {(uint)B32,(uint)F,(uint)T768}, {32,16}, 0, stream >>>
        (reinterpret_cast<__const __half2 *> (data), results_array);

    CUDA_PEEK("chime_frb_upchan");
}


// 'data': shape=(T,F,2,B,2), axes (time,freq,pol,beam,ReIm)
// 'results_array': shape=(B,F,T/384,16), axes (beam,cfreq,time,ufreq)
void launch_chime_frb_upchan(const Array<__half> &data, Array<float> &results_array, cudaStream_t stream)
{
    // data: shape=(T,F,2,B,2), axes (time,freq,pol,beam,ReIm)
    xassert(data.ndim == 5);
    xassert_eq(data.shape[2], 2);
    xassert_eq(data.shape[4], 2);
    xassert(data.on_gpu());
    xassert(data.is_fully_contiguous());

    long T = data.shape[0];
    long F = data.shape[1];
    long B = data.shape[3];
    
    xassert_divisible(T, 768);
    xassert_divisible(B, 32);
    
    // results_array: shape=(B,F,T/384,16), axes (beam,cfreq,time,ufreq)
    xassert_shape_eq(results_array, ({B, F, T/384, 16}));
    xassert(results_array.on_gpu());
    xassert(results_array.is_fully_contiguous());

    launch_chime_frb_upchan(data.data, results_array.data, T, F, B, stream);
}


// CPU reference implementation of chime_frb_upchan(), for testing.
// Uses O(N^2) brute-force DFT (not FFT), and float32 data (not float16).

void cpu_chime_frb_upchan(const Array<float> &data, Array<float> &results_array)
{
    // data: shape=(T,F,2,B,2), axes (time,freq,pol,beam,ReIm)
    xassert(data.ndim == 5);
    xassert_eq(data.shape[2], 2);
    xassert_eq(data.shape[4], 2);
    xassert(data.on_host());
    xassert(data.is_fully_contiguous());

    long T = data.shape[0];
    long F = data.shape[1];
    long B = data.shape[3];
    
    // results_array: shape=(B,F,T/384,16), axes (beam,cfreq,time,ufreq)
    xassert_divisible(T, 384);
    xassert_shape_eq(results_array, ({B, F, T/384, 16}));
    xassert(results_array.on_host());
    xassert(results_array.is_fully_contiguous());

    memset(results_array.data, 0, results_array.size * sizeof(float));

    // Precompute twiddle factors: ei[n] = exp(+2*pi*i*n/128) for 0 <= n < 128.
    vector<complex<float>> ei(128);
    for (long n = 0; n < 128; n++) {
        double phase = 2.0 * M_PI * n / 128.0;
        ei[n] = { float(cos(phase)), float(sin(phase)) };
    }

    vector<complex<float>> v(128);
    long tstride = data.strides[0];
    long T128 = T / 128;

    for (long t128 = 0; t128 < T128; t128++) {
        long t384 = t128 / 3;
        for (long cfreq = 0; cfreq < F; cfreq++) {
            for (long pol = 0; pol < 2; pol++) {
                for (long beam = 0; beam < B; beam++) {
                    const float *psrc = &data.at({ 128*t128, cfreq, pol, beam, 0 });
                    float *pdst = &results_array.at({ beam, cfreq, t384, 0 });
                  
                    for (long t = 0; t < 128; t++)
                        v[t] = { psrc[t*tstride], psrc[t*tstride+1] };

                    for (int f = 0; f < 128; f++) {
                        complex<float> x = { 0.0f, 0.0f };
                        
                        // DFT with positive exponent (no 1/N prefactor):
                        //   F[f] = sum_{t} data[t] * exp(+2*pi*i*f*t/128)
                        for (int t = 0; t < 128; t++)
                            x += ei[(f*t) & 127] * v[t];

                        float intensity = x.real() * x.real() + x.imag() * x.imag();

                        // Accumulate.
                        int u = (f >> 3) ^ 8;  // output index including shift u -> (u ^ 8).
                        pdst[u] += intensity / (48.0f * BP[u]);
                    }
                }
            }
        }
    }
}


void test_chime_frb_upchan()
{
    vector<long> v = ksgpu::random_integers_with_bounded_product(3, 10);
    long T = 768 * v[0];
    long F = v[1];
    long B = 32 * v[2];

    cout << "test_chime_frb_upchan: T=" << T << ", F=" << F << ", B=" << B << endl;

    Array<float> data_cpu({T, F, 2, B, 2}, af_rhost | af_random);
    Array<__half> data_gpu = data_cpu.template convert<__half>().to_gpu();

    // GPU kernel (float16, cuFFTDx).
    Array<float> results_gpu({B, F, T/384, 16}, af_gpu | af_zero);
    launch_chime_frb_upchan(data_gpu, results_gpu);
    CUDA_PEEK("test_chime_frb_upchan");
    CUDA_CALL(cudaDeviceSynchronize());

    // CPU reference (float32, O(N^2) DFT).
    Array<float> results_cpu({B, F, T/384, 16}, af_rhost | af_zero);
    cpu_chime_frb_upchan(data_cpu, results_cpu);

    // Call assert_arrays_equal() with thresholds that are appropriate for float16.
    // (Otherwise, it will use float32 thresholds, since the 'results' arrays are float32).
    double epsrel = 10 * Dtype::from_str("float16").precision();
    double epsabs = epsrel * 48 * 128 * 0.67;  // mean intensity
    
    assert_arrays_equal(results_cpu, results_gpu, "cpu", "gpu",
                        {"beam","cfreq","time","ufreq"}, epsabs, epsrel);

    cout << "test_chime_frb_upchan: pass" << endl;
}


void time_chime_frb_upchan()
{
    long T = 49152;
    long F = 16;
    long B = 1024;
    long niterations = 1000;
    long nstreams = 1;

    Array<__half> data({T, F, 2, B, 2}, af_gpu | af_zero);
    Array<float> results_array({B, F, T/384, 16}, af_gpu | af_zero);

    double gmem_gb = 1.0e-9 * (data.nbytes() + results_array.nbytes());
    KernelTimer kt(niterations, nstreams);

    while (kt.next()) {
        launch_chime_frb_upchan(data, results_array, kt.stream);

        if (kt.warmed_up && (kt.curr_iteration % 50 == 49)) {
            double gb_per_sec = gmem_gb / kt.dt;
            cout << "chime_frb_upchan: " << gb_per_sec << " GB/s (iteration " << kt.curr_iteration << ")" << endl;
        }
    }
}

} // namespace pirate

