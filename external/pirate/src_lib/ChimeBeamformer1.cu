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
#include <ksgpu/test_utils.hpp>   // assert_arrays_equal

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// This is the "beamforming" part of the CHIME FRB beamforming kernel.
// It runs before the "upchannelization" part.
//
// The input data is the uint4+4 electric field array, sampled at 4x256 feed
// locations. The length-4 axis is "east-west" and the length-256 axis
// is "north-south". The output is a float16+16 array of beamformed electric
// fields, at 4x256 beam locations.
//
//  'inputData':  shape (T,F,2,4,256), dtype uint4+4, axes (time,freq,pol,ew,ns)
//  'map':        shape (F,256), dtype uint, axes (freq,ns)
//  'co':         shape (F,4,4,2), dtype float, axes (freq,ewout,ewin,ReIm)
//  'outputData': shape (T,F,2,4,256), dtype float16+16, axes (time,freq,pol,ew,ns)
//  'gains':      shape (F,2,4,256), dtype float32+32, axes (freq,pol,ew,ns)
//
// Here is a specification of the kernel:
//
//  0. Each (time,freq,pol) is processed independently, so we streamline notation
//     by omitting those axes, and only keeping track of ew,ns axes below.
//
//  1. Unpack each uint4+4 inputData element to a float32+32 E[ew,ns].
//       real part = float(x >> 4) - 8.0
//       imag part = float(x & 0xf) - 8.0
//
//  2. Multiply by the complex conjugate of the associated gain:
//
//       F[ew,ns] = G[ew,ns]^* E[ew,ns].
//
//  3. East-west beamforming: We matrix-multiply by the 4-by-4 complex matrix co[4,4]:
//
//       H[ew_out,ns] = (1/4) sum_j co[ew_out,ew_in] F[ew_in,ns]^*
//
//     Note that in this step, we notationally distinguish the length-4 input
//     axis (ew_in) and the length-4 output axis (ew_out). The H-array can be
//     interpreted as "partially beamformed electric field": beamforming has
//     been performed along the ew-axis, but not the ns-axis.
//
//  4. North-south beamforming: we zero-pad the ns axis to length-512 and
//     take an FFT.
//
//       J[ew,ns_out] = sum_{ns_in=0}^255 H[ew,ns_in]
//                         * exp(2pi * i * ns_in * ns_out / 512)
//
//     where 0 <= ns_in < 256, and 0 <= ns_out < 512.
//
//  5. Clamping: here is where the 'map' argument comes in. We reduce the
//     ns_axis from length-512 to length-256 by selecting indices as follows:
//
//       K[ew,ns] = J[ew,map[255-ns]]
//
//     where 0 <= ns < 256, and 0 <= map[ns] < 512.
//
//  6. All math so far has been 32-bit. We convert to float16+16 and write to
//     global GPU memory.
//
// Some of the conventions above are a little unusual (uint4+4 encoding in step 1,
// complex conjugates in step 2,3, index flip (255-ns) in step 5) but are preserved
// for consistency with chime.
//
// I claim that this is equivalent to kotekan frb_beamform_amd(), but the equivalence
// is not obvious -- here is the argument. (Note that when comparing with kotekan,
// it's more convenient to use the reference implementation in gpuBeamformSimulate.cpp
// than the amd gpu kernel.)
//
//  - Combining equations from steps 1-4, we get:
//
//      J[e,n] = (1/4) sum_{e',n'} e^{2pi*i*nn'/512} co[e,e'] gain[e',n'] E[e',n']^*
//
//   - In contrast, kotekan does the following.
//
//        1. Gains:  F[e,n] = gain[e,n]^* E[e,n]
//        2. NS:     L[e,n] = sum_{n'} e^{-2pi*i*nn'/512} F[e,n']
//        3. EW:     J[e,n] = (1/4) sum_{e'} co[e,e'] L[e',n]^*
//
//      Combining these equations, we get:
//
//       J[e,n] = (1/4) sum_{e'n'} e^{2pi*i*nn'/512} co[e,e'] gain[e',n'] E[e',n']^*
//
//     which agrees with our expression above, i.e. the two are equivalent in the
//     end, although the ordering of computations and intermediate expressions are
//     inequivalent.
//
//   - Note that this analysis does not compare the clamping logic. Here,
//     Claude Code told me that the two were equivalent, but I didn't check
//     it carefully. We should confirm with a unit test. I'm not really worried
//     about it, since an arbitrary bug in the clamping logic should be fixable
//     by tweaking the host-side precomputation of the map[] array, with no
//     changes to the gpu kernel. We just need to test carefully.
//
// Each threadblock processes (128 times, 1 freq, 1 pol).
// Thus, gridDim = { T/128, F, 2 }. We assume that T is a multiple of 128.
// The values of (T,F) are supplied to the kernel via gridDims (not kernel args).
//
// The kernel is launched with 32 warps, and blockDim = { 64, 16 }.


__global__ void __launch_bounds__(1024,1)
chime_frb_beamform(
    const uint8_t *__restrict__ inputData,  // shape (T,F,2,4,256)
    const uint *__restrict__ map,           // shape (F,256)
    const float *__restrict__ co,           // shape (F,4,4,2), indexed by (ewout,ewin,ReIm)
    __half2 *__restrict__ outputData,       // shape (T,F,2,4,256)
    const float2 *__restrict__ gains)       // shape (F,2,4,256)
{
    // Shared memory layout (96 KB total):
    //
    //   uint smem_E[32][256];            // 32 KB, axes (time,ns)
    //   union {
    //       float2 smem_H[4][4][256];   // 32 KB, axes (time,ew,ns)
    //       float2 smem_J[4][4][512];   // 64 KB, axes (time,ew,ns)
    //       char smem_fft[64*1024];     // 64 KB, cufftdx scratch space
    //   };
    //
    // Kernel outline:
    //
    //   for each "outer" block of 32 times
    //      load uint4+4 E-array from global memory (T=32)
    //      store E-array in smem_E
    //      syncthreads
    //      for each "inner" block of 4 times
    //          load E-array from smem_E
    //          compute H-array: convert to float32+32, apply gains, ew beamforming
    //          store in smem_H
    //          syncthreads
    //          load from smem_H in a different ordering
    //          syncthreads
    //          compute J-array: do NS FFT
    //          syncthreads
    //          write to smem_J
    //          syncthreads
    //          compute K-array: read from smem_J with 'map' reindexing
    //          write K-array to global memory
    //          syncthreads
    //
    // Notation: in register mappings throughout the code, we use index bits (e1 e0)_2
    // for the length-4 EW axis, and index bits (n7 n6 n5 n4 n3 n2 n1 n0)_2 for the
    // length-256 NS axis.

    // Must allocate dynamically, since > 48KB.
    extern __shared__ char smem_all[];

    // First, apply per-block pointer offsets to all pointer arguments.
    uint F = gridDim.y;

    // inputData: shape (T,F,2,4,256), dtype uint8_t
    //   before: shape (T,F,2,4,256), contiguous
    //   after: shape (128,4,256), strides (F*2048, 256, 1)
    inputData += (128UL * blockIdx.x * F * 2 + blockIdx.y * 2 + blockIdx.z) * 1024;

    // map: shape (F,256), dtype uint
    //   before: shape (F,256), contiguous
    //   after: shape (256,), contiguous
    map += blockIdx.y * 256;

    // co: shape (F,4,4,2), dtype float
    //   before: shape (F,4,4,2), contiguous
    //   after: shape (4,4,2), contiguous
    co += blockIdx.y * 32;

    // outputData: shape (T,F,2,4,256), dtype __half2
    //   before: shape (T,F,2,4,256), contiguous
    //   after: shape (128,4,256), strides (F*2048, 256, 1)
    outputData += (128UL * blockIdx.x * F * 2 + blockIdx.y * 2 + blockIdx.z) * 1024;

    // gains: shape (F,2,4,256), dtype float2
    //   before: shape (F,2,4,256), contiguous
    //   after: shape (4,256), contiguous
    gains += (blockIdx.y * 2UL + blockIdx.z) * 1024;

    // Initialize "mapval" (1 register/thread).
    // This is the value of map[ns], for a specific value of ns that will be
    // useful later in the kernel.
    uint map_ns = (threadIdx.x | (threadIdx.y << 6)) & 255;
    uint mapval = map[255 - map_ns];

    // Initialize the "A-coefficients" (8 registers/thread).
    //
    // This step needs some explanation! Later in the kernel (see below), we'll
    // combine the gains and the EW beamforming into a single operation:
    //
    //   H[eo,ns] = sum_ei A[eo,ns,ei] * E[ei,ns]^*
    //
    // where:
    //
    //    0 <= eo < 4 is an output ew beam location
    //    0 <= ei < 4 is an input ew feed location
    //    0 <= ns < 256 is a ns feed location
    //
    // and we define A[eo,ns,ei] = (1/4) co[eo,ei] gain[ei,ns]
    //
    // In this step, the 1024 threads in the kernel will be mapped to (eo,ns)
    // beam locations as follows:
    //
    //   thread:  n4 n3 n2 n1 n0
    //   warp:    eo1 eo0 n7 n6 n5
    //
    // In preparation, we precompute the A-coefficients A[eo,ns,ei] for 0 <= ei < 4
    // and the appropriate per-thread value of (eo,ns).
    //

    float A0re, A0im, A1re, A1im, A2re, A2im, A3re, A3im;

    {
        // Code to compute the A-coefficients isn't particularly optimized,
        // but I didn't bother improving it, since the kernel ended up being
        // memory bandwidth limited anyway.
        
        // warpId = threadIdx.y * 2 + (threadIdx.x >> 5)
        //   w0 = n5 (threadIdx.x bit 5)
        //   w1 = n6 (threadIdx.y bit 0)
        //   w2 = n7 (threadIdx.y bit 1)
        //   w3 = eo0 (threadIdx.y bit 2)
        //   w4 = eo1 (threadIdx.y bit 3)
        uint eo = (threadIdx.y >> 2) & 3;
        uint ns = ((threadIdx.y & 3) << 6) | threadIdx.x;

        // gains: shape (4,256), dtype float2, after per-block offset
        float2 g0 = gains[ns];
        float2 g1 = gains[256 + ns];
        float2 g2 = gains[512 + ns];
        float2 g3 = gains[768 + ns];

        // co: shape (4,4,2), dtype float, after per-block offset
        // co[eo,ei] is at offset eo*8 + ei*2 (+0 for re, +1 for im)
        float co0re = co[eo*8];     float co0im = co[eo*8 + 1];
        float co1re = co[eo*8 + 2]; float co1im = co[eo*8 + 3];
        float co2re = co[eo*8 + 4]; float co2im = co[eo*8 + 5];
        float co3re = co[eo*8 + 6]; float co3im = co[eo*8 + 7];

        // A[eo,ns,ei] = (1/4) co[eo,ei] * gain[ei,ns]   (complex multiply)
        A0re = 0.25f * (co0re * g0.x - co0im * g0.y);
        A0im = 0.25f * (co0re * g0.y + co0im * g0.x);
        A1re = 0.25f * (co1re * g1.x - co1im * g1.y);
        A1im = 0.25f * (co1re * g1.y + co1im * g1.x);
        A2re = 0.25f * (co2re * g2.x - co2im * g2.y);
        A2im = 0.25f * (co2re * g2.y + co2im * g2.x);
        A3re = 0.25f * (co3re * g3.x - co3im * g3.y);
        A3im = 0.25f * (co3re * g3.y + co3im * g3.x);
    }
        
    for (int touter = 0; touter < 128; touter += 32) {
        
        // Load E-array from inputData (32 times, 4 ew feeds, 32 ns feeds).
        // Register mapping (dtype uint4+4):
        //  simd:      n1 n0
        //  register:  e1 e0 n2
        //  thread:    t0 n6 n5 n4 n3
        //  warp:      t4 t3 t2 t1 n7

        uint E0, E1, E2, E3, E4, E5, E6, E7;

        {
            uint laneId = threadIdx.x & 31;
            uint warpId = threadIdx.y * 2 + (threadIdx.x >> 5);

            // t_local = (t4 t3 t2 t1 t0)_2, range 0..31
            uint t_local = (warpId & 0x1e) | (laneId >> 4);

            // ns_base = (n7 n6 n5 n4 n3 0 0 0)_2
            uint ns_base = ((warpId & 1) << 7) | ((laneId & 0xf) << 3);

            // Pointer to inputData[touter + t_local, ew=0, ns_base]
            //   inputData shape (128,4,256), strides (F*2048, 256, 1)
            const uint8_t *p = inputData + (touter + t_local) * (ulong(F) * 2048) + ns_base;

            // 64-bit loads, one per ew value (ew stride = 256 bytes)
            uint2 tmp;
            tmp = *reinterpret_cast<const uint2 *>(p);       E0 = tmp.x; E1 = tmp.y;
            tmp = *reinterpret_cast<const uint2 *>(p + 256); E2 = tmp.x; E3 = tmp.y;
            tmp = *reinterpret_cast<const uint2 *>(p + 512); E4 = tmp.x; E5 = tmp.y;
            tmp = *reinterpret_cast<const uint2 *>(p + 768); E6 = tmp.x; E7 = tmp.y;
        }

        // Now we do a lot of shuffling operations, to change the register assignment.

        // local transpose (simd s0) <-> (register r1)
        // Pairs differing in r1: (E0,E2), (E1,E3), (E4,E6), (E5,E7)
        {
            uint t0 = __byte_perm(E0, E2, 0x6240); uint t2 = __byte_perm(E0, E2, 0x7351);
            uint t1 = __byte_perm(E1, E3, 0x6240); uint t3 = __byte_perm(E1, E3, 0x7351);
            uint t4 = __byte_perm(E4, E6, 0x6240); uint t6 = __byte_perm(E4, E6, 0x7351);
            uint t5 = __byte_perm(E5, E7, 0x6240); uint t7 = __byte_perm(E5, E7, 0x7351);
            E0=t0; E1=t1; E2=t2; E3=t3; E4=t4; E5=t5; E6=t6; E7=t7;
        }
        // local transpose (simd s1) <-> (register r2)
        // Pairs differing in r2: (E0,E4), (E1,E5), (E2,E6), (E3,E7)
        {
            uint t0 = __byte_perm(E0, E4, 0x5410); uint t4 = __byte_perm(E0, E4, 0x7632);
            uint t1 = __byte_perm(E1, E5, 0x5410); uint t5 = __byte_perm(E1, E5, 0x7632);
            uint t2 = __byte_perm(E2, E6, 0x5410); uint t6 = __byte_perm(E2, E6, 0x7632);
            uint t3 = __byte_perm(E3, E7, 0x5410); uint t7 = __byte_perm(E3, E7, 0x7632);
            E0=t0; E1=t1; E2=t2; E3=t3; E4=t4; E5=t5; E6=t6; E7=t7;
        }

        // At this point, the register assignment is (dtype uint4+4):
        //  simd:      e1 e0
        //  register:  n1 n0 n2
        //  thread:    t0 n6 n5 n4 n3
        //  warp:      t4 t3 t2 t1 n7

        // warp transpose (register r0) <-> (thread t4)
        // Pairs differing in r0: (E0,E1), (E2,E3), (E4,E5), (E6,E7)
        // t4 = laneId bit 4, mask = 0x10
        {
            uint tmp;
            tmp = (threadIdx.x & 0x10) ? E0 : E1;
            tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x10);
            E0 = (threadIdx.x & 0x10) ? tmp : E0;
            E1 = (threadIdx.x & 0x10) ? E1 : tmp;

            tmp = (threadIdx.x & 0x10) ? E2 : E3;
            tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x10);
            E2 = (threadIdx.x & 0x10) ? tmp : E2;
            E3 = (threadIdx.x & 0x10) ? E3 : tmp;

            tmp = (threadIdx.x & 0x10) ? E4 : E5;
            tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x10);
            E4 = (threadIdx.x & 0x10) ? tmp : E4;
            E5 = (threadIdx.x & 0x10) ? E5 : tmp;

            tmp = (threadIdx.x & 0x10) ? E6 : E7;
            tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x10);
            E6 = (threadIdx.x & 0x10) ? tmp : E6;
            E7 = (threadIdx.x & 0x10) ? E7 : tmp;
        }
        // warp transpose (register r2) <-> (thread t3)
        // Pairs differing in r2: (E0,E4), (E1,E5), (E2,E6), (E3,E7)
        // t3 = laneId bit 3, mask = 0x08
        {
            uint tmp;
            tmp = (threadIdx.x & 0x08) ? E0 : E4;
            tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x08);
            E0 = (threadIdx.x & 0x08) ? tmp : E0;
            E4 = (threadIdx.x & 0x08) ? E4 : tmp;

            tmp = (threadIdx.x & 0x08) ? E1 : E5;
            tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x08);
            E1 = (threadIdx.x & 0x08) ? tmp : E1;
            E5 = (threadIdx.x & 0x08) ? E5 : tmp;

            tmp = (threadIdx.x & 0x08) ? E2 : E6;
            tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x08);
            E2 = (threadIdx.x & 0x08) ? tmp : E2;
            E6 = (threadIdx.x & 0x08) ? E6 : tmp;

            tmp = (threadIdx.x & 0x08) ? E3 : E7;
            tmp = __shfl_sync(~0u, tmp, threadIdx.x ^ 0x08);
            E3 = (threadIdx.x & 0x08) ? tmp : E3;
            E7 = (threadIdx.x & 0x08) ? E7 : tmp;
        }

        // At this point, the register assignment is (dtype uint4+4):
        //  simd:      e1 e0
        //  register:  n6 n0 t0
        //  thread:    n2 n1 n5 n4 n3
        //  warp:      t4 t3 t2 t1 n7

        // We now call __shfl_sync() on each register, to permute threads
        // (n2 n1 n5 n4 n3) -> (n5 n4 n3 n2 n1).
        
        // Permute threads: (n2 n1 n5 n4 n3) -> (n5 n4 n3 n2 n1)
        // Thread at lane L (new encoding: n5 n4 n3 n2 n1) reads from
        // old lane (n2 n1 n5 n4 n3) = ((L & 0x3) << 3) | ((L >> 2) & 0x7)
        {
            uint laneId = threadIdx.x & 31;
            uint src = ((laneId & 0x3) << 3) | ((laneId >> 2) & 0x7);
            E0 = __shfl_sync(~0u, E0, src);
            E1 = __shfl_sync(~0u, E1, src);
            E2 = __shfl_sync(~0u, E2, src);
            E3 = __shfl_sync(~0u, E3, src);
            E4 = __shfl_sync(~0u, E4, src);
            E5 = __shfl_sync(~0u, E5, src);
            E6 = __shfl_sync(~0u, E6, src);
            E7 = __shfl_sync(~0u, E7, src);
        }

        // At this point, the register assignment is (dtype uint4+4)
        //  simd:      e1 e0
        //  register:  n6 n0 t0
        //  thread:    n5 n4 n3 n2 n1
        //  warp:      t4 t3 t2 t1 n7

        // Now we write to shared memory:
        //   uint smem_E[32][256];   // (time,ns)
        //
        // where there is no length-4 ew axis since we have packed four uint4+4s
        // into a uint, with simd (s1 s0) <-> (e1 e0). We can use a 64-bit,
        // bank-conflict-free store instruction here.

        // Store to smem_E using 64-bit, bank-conflict-free stores.
        // Register assignment:
        //   simd:      e1 e0       (packed in uint)
        //   register:  r2 r1 r0  <->  n6 n0 t0
        //   thread:    n5 n4 n3 n2 n1
        //   warp:      t4 t3 t2 t1 n7
        //
        // 64-bit store pairs registers differing in r1 (= n0):
        //   (E0,E2), (E1,E3), (E4,E6), (E5,E7)
        {
            uint *smem_E = reinterpret_cast<uint *>(smem_all);  // [32][256]
            uint laneId = threadIdx.x & 31;
            uint warpId = threadIdx.y * 2 + (threadIdx.x >> 5);

            uint time0 = warpId & 0x1e;       // t0=0
            uint time1 = time0 | 1;            // t0=1
            uint ns_lo = ((warpId & 1) << 7) | (laneId << 1);             // n6=0
            uint ns_hi = ns_lo | (1 << 6);                                 // n6=1

            *reinterpret_cast<uint2 *>(smem_E + time0 * 256 + ns_lo) = make_uint2(E0, E2);
            *reinterpret_cast<uint2 *>(smem_E + time0 * 256 + ns_hi) = make_uint2(E4, E6);
            *reinterpret_cast<uint2 *>(smem_E + time1 * 256 + ns_lo) = make_uint2(E1, E3);
            *reinterpret_cast<uint2 *>(smem_E + time1 * 256 + ns_hi) = make_uint2(E5, E7);
        }

        __syncthreads();

        for (int tmid = 0; tmid < 32; tmid += 4) {

            for (int tinner = 0; tinner < 4; tinner++) {
                // In this step, we load uint4+4 data from shared memory, apply gains,
                // and do EW beamforming. As explained above, these steps are coalesced
                // into the "A-matrix" multiplication.
                //
                //   E1[eo,ns] = sum_ei A[eo,ns,ei] * E0[ei,ns]^*
                //
                // We load data from smem_E:
                //   uint smem_E[32][256];   // (time,ns)
                //
                // into register assignment
                //   simd:    ei1 ei0
                //   thread:  n4 n3 n2 n1 n0
                //   warp:    eo1 eo0 n7 n6 n5

                // 32-bit load from smem_E[time][ns], bank-conflict-free.
                // ns = (n7 n6 n5 from warpId bits 2..0) | (n4..n0 from laneId)
                // eo bits in warpId don't affect the address (ew packed in simd).
                uint Eval;
                {
                    const uint *smem_E = reinterpret_cast<const uint *>(smem_all);
                    uint laneId = threadIdx.x & 31;
                    uint warpId = threadIdx.y * 2 + (threadIdx.x >> 5);
                    uint ns = ((warpId & 7) << 5) | laneId;
                    Eval = smem_E[(tmid + tinner) * 256 + ns];
                }

                // Convert uint4+4 -> float32+32, multiply by A-matrix.

                // Unpack uint4+4 -> float32+32 for all 4 EW feeds (ei=0,1,2,3).
                // Each byte: real = (byte >> 4) - 8, imag = (byte & 0xf) - 8.
                float e0re = float((Eval >> 4) & 0xf) - 8.0f;
                float e0im = float(Eval & 0xf) - 8.0f;
                float e1re = float((Eval >> 12) & 0xf) - 8.0f;
                float e1im = float((Eval >> 8) & 0xf) - 8.0f;
                float e2re = float((Eval >> 20) & 0xf) - 8.0f;
                float e2im = float((Eval >> 16) & 0xf) - 8.0f;
                float e3re = float((Eval >> 28) & 0xf) - 8.0f;
                float e3im = float((Eval >> 24) & 0xf) - 8.0f;

                // H = sum_ei A[ei] * E[ei]^*
                // A * E^* = (Are+i*Aim)(re-i*im) = (Are*re+Aim*im) + i*(Aim*re-Are*im)
                float Hre = A0re*e0re + A0im*e0im + A1re*e1re + A1im*e1im
                          + A2re*e2re + A2im*e2im + A3re*e3re + A3im*e3im;
                float Him = A0im*e0re - A0re*e0im + A1im*e1re - A1re*e1im
                          + A2im*e2re - A2re*e2im + A3im*e3re - A3re*e3im;
                float2 Hval = make_float2(Hre, Him);

                // Now we have a float2 on each thread, corresponding to the H-array in
                // register assigment:
                //   register:  ReIm
                //   thread:    n4 n3 n2 n1 n0
                //   warp:      eo1 eo0 n7 n6 n5
                //
                // Write to smem_H
                //   float2 smem_H[4][4][256];   // (tinner,ew,ns)

                // 64-bit store to smem_H[tinner][eo][ns], bank-conflict-free.
                {
                    float2 *smem_H = reinterpret_cast<float2 *>(smem_all + 32*1024);
                    uint laneId = threadIdx.x & 31;
                    uint warpId = threadIdx.y * 2 + (threadIdx.x >> 5);
                    uint eo = (warpId >> 3) & 3;
                    uint ns = ((warpId & 7) << 5) | laneId;
                    smem_H[tinner * 1024 + eo * 256 + ns] = Hval;
                }
            }

            // Load partially beamformed data from shared memory:
            //   float2 smem_H[4][4][256];   // (tinner,ew,ns)
            //
            // in the following register assignment:
            //   register:  n7 n6 ReIm
            //   thread:    n4 n3 n2 n1 n0
            //   warp:      t1 t0 e1 e0 n5

            __syncthreads();

            // 64-bit loads from smem_H[tinner_idx][ew][ns], bank-conflict-free.
            // Warp mapping: w4=t1, w3=t0, w2=e1, w1=e0, w0=n5
            float2 H0, H1, H2, H3;
            {
                const float2 *smem_H = reinterpret_cast<const float2 *>(smem_all + 32*1024);
                uint laneId = threadIdx.x & 31;
                uint warpId = threadIdx.y * 2 + (threadIdx.x >> 5);
                uint tinner_idx = (threadIdx.y >> 2) & 3;
                uint ew = threadIdx.y & 3;
                uint ns_base = ((warpId & 1) << 5) | laneId;  // (n5, n4..n0)

                uint base = tinner_idx * 1024 + ew * 256 + ns_base;
                H0 = smem_H[base];         // n7=0, n6=0
                H1 = smem_H[base + 64];    // n7=0, n6=1
                H2 = smem_H[base + 128];   // n7=1, n6=0
                H3 = smem_H[base + 192];   // n7=1, n6=1
            }
            
            __syncthreads();  // smem_H loads done; reusing union memory for FFT scratch

            // Now do the NS FFT using cufftdx.
            // The FFT is zero-padded, and takes (length 256) -> (length 512).
            // This adds an 'n8' bit to the register assignment.

            // Zero-padded 512-point FFT using cufftdx.
            // Direction: inverse (positive exponent, exp(+2pi*i*j*k/512)).
            // 16 independent FFTs per block (one per threadIdx.y).
            // Each FFT uses 64 threads (threadIdx.x = 0..63).
            // Input registers 0..3 = H0..H3 (256 ns values), registers 4..7 = zero padding.
            float2 J0, J1, J2, J3, J4, J5, J6, J7;
            {
                using FFT = decltype(
                    cufftdx::Size<512>()
                    + cufftdx::Precision<float>()
                    + cufftdx::Type<cufftdx::fft_type::c2c>()
                    + cufftdx::Direction<cufftdx::fft_direction::inverse>()
                    + cufftdx::ElementsPerThread<8>()
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
                static_assert(FFT::storage_size == 8);
                static_assert(FFT::shared_memory_size <= 64*1024);

                complex_t fft_data[8];
                fft_data[0] = complex_t(H0.x, H0.y);
                fft_data[1] = complex_t(H1.x, H1.y);
                fft_data[2] = complex_t(H2.x, H2.y);
                fft_data[3] = complex_t(H3.x, H3.y);
                fft_data[4] = complex_t(0.0f, 0.0f);
                fft_data[5] = complex_t(0.0f, 0.0f);
                fft_data[6] = complex_t(0.0f, 0.0f);
                fft_data[7] = complex_t(0.0f, 0.0f);

                char *smem_fft = smem_all + 32*1024;
                FFT().execute(fft_data, smem_fft);

                J0 = make_float2(fft_data[0].x, fft_data[0].y);
                J1 = make_float2(fft_data[1].x, fft_data[1].y);
                J2 = make_float2(fft_data[2].x, fft_data[2].y);
                J3 = make_float2(fft_data[3].x, fft_data[3].y);
                J4 = make_float2(fft_data[4].x, fft_data[4].y);
                J5 = make_float2(fft_data[5].x, fft_data[5].y);
                J6 = make_float2(fft_data[6].x, fft_data[6].y);
                J7 = make_float2(fft_data[7].x, fft_data[7].y);
            }

            __syncthreads();  // since FFT scratch memory will be reused for smem_J
            
            // After cufftdx, we get the J-array in register assignment:
            //   register:  n8 n7 n6 ReIm
            //   thread:    n4 n3 n2 n1 n0
            //   warp:      t1 t0 e1 e0 n5

            // Write to smem_J:
            //   float2 smem_J[4][4][512];  // (time,ew,ns)

            // 64-bit stores to smem_J[t][ew][ns], bank-conflict-free.
            // ns = r * 64 + threadIdx.x, where r (0..7) encodes (n8 n7 n6).
            {
                float2 *smem_J = reinterpret_cast<float2 *>(smem_all + 32*1024);
                uint t = (threadIdx.y >> 2) & 3;
                uint ew = threadIdx.y & 3;
                uint base = t * 2048 + ew * 512 + threadIdx.x;

                __syncthreads();  // FFT done; reusing union memory for smem_J
                smem_J[base]       = J0;
                smem_J[base + 64]  = J1;
                smem_J[base + 128] = J2;
                smem_J[base + 192] = J3;
                smem_J[base + 256] = J4;
                smem_J[base + 320] = J5;
                smem_J[base + 384] = J6;
                smem_J[base + 448] = J7;
            }

            __syncthreads();  // ensure all smem_J writes complete before K-array reads

            for (int tinner = 0; tinner < 4; tinner++) {
                
                // Load K-array (one float2 on each thread), in register assignment:
                //   register:  ReIm
                //   thread:    n4 n3 n2 n1 n0
                //   warp:      e1 e0 n7 n6 n5
                //
                // This can be done by loading smem_J[] at a memory location that
                // depends on 'mapval', using a 64-bit load instruction. This load
                // is generally not bank conflict free (!!). This could be improved
                // with some nontrivial effort, but I didn't bother since the kernel
                // turned out to be memory bandwidth limited with realistic 'map' values.

                // 64-bit load from smem_J[tinner][ew][mapval].
                // NOT bank-conflict-free (mapval is arbitrary).
                float2 Kval;
                {
                    const float2 *smem_J = reinterpret_cast<const float2 *>(smem_all + 32*1024);
                    uint ew = (threadIdx.y >> 2) & 3;
                    Kval = smem_J[tinner * 2048 + ew * 512 + mapval];
                }

                // Write K-array element to global memory.
                
                // Convert float32+32 -> float16+16, write to outputData.
                // outputData shape (128,4,256), strides (F*2048, 256, 1), dtype __half2.
                {
                    __half2 Kout = __floats2half2_rn(Kval.x, Kval.y);
                    uint time = touter + tmid + tinner;
                    uint ew = (threadIdx.y >> 2) & 3;
                    uint ns = ((threadIdx.y & 3) << 6) | threadIdx.x;
                    outputData[time * ulong(F * 2048) + ew * 256 + ns] = Kout;
                }
            }

            __syncthreads();  // smem_J reads done; next tmid iteration writes smem_H (same union)
        }
    }
}


void launch_chime_frb_beamform(
    const uint8_t *inputData,   // shape (T,F,2,4,256), dtype uint8_t, axes (time,freq,pol,ew,ns)
    const uint *map,            // shape (F,256), dtype uint, axes (freq,ns)
    const float *co,            // shape (F,4,4,2), dtype float, axes (freq,ewout,ewin,ReIm)  
    __half *outputData,         // shape (T,F,2,4,256), dtype float16+16, axes (time,freq,pol,ew,ns)
    const float *gains,         // shape (F,2,4,256), dtype float32+32, axes (freq,pol,ew,ns)
    long T,
    long F,
    cudaStream_t stream)
{
    xassert(T > 0);
    xassert(F > 0);
    xassert_divisible(T, 128);

    long T128 = T / 128;

    CUDA_CALL(cudaFuncSetAttribute(chime_frb_beamform, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024));
    
    chime_frb_beamform<<< {(uint)T128,(uint)F,2}, {64,16}, 96*1024, stream >>>
        (inputData, map, co,
         reinterpret_cast<__half2 *>(outputData),
         reinterpret_cast<const float2 *>(gains));

    CUDA_PEEK("chime_frb_beamform");
}


void launch_chime_frb_beamform(
    const ksgpu::Array<uint8_t> &inputData,  // shape (T,F,2,4,256), dtype uint8_t, axes (time,freq,pol,ew,ns)
    const ksgpu::Array<uint> &map,           // shape (F,256), dtype uint, axes (freq,ns)
    const ksgpu::Array<float> &co,           // shape (F,4,4,2), dtype float, axes (freq,ewout,ewin,ReIm)
    ksgpu::Array<__half> &outputData,        // shape (T,F,2,4,256), dtype float16+16, axes (time,freq,pol,ew,ns)
    const ksgpu::Array<float> &gains,        // shape (F,2,4,256), dtype float32+32, axes (freq,pol,ew,ns)
    cudaStream_t stream)
{
    // inputData: shape (T,F,2,4,256), dtype uint8_t
    xassert(inputData.ndim == 5);
    xassert_eq(inputData.shape[2], 2);
    xassert_eq(inputData.shape[3], 4);
    xassert_eq(inputData.shape[4], 256);
    xassert(inputData.on_gpu());
    xassert(inputData.is_fully_contiguous());
    
    long T = inputData.shape[0];
    long F = inputData.shape[1];

    // map: shape (F,256), dtype uint
    xassert_shape_eq(map, ({F, 256}));
    xassert(map.on_gpu());
    xassert(map.is_fully_contiguous());

    // co: shape (F,4,4,2), dtype float
    xassert_shape_eq(co, ({F, 4, 4, 2}));
    xassert(co.on_gpu());
    xassert(co.is_fully_contiguous());

    // outputData: shape (T,F,2,4,256), dtype __half (stored as float16+16, so shape has extra dim 2)
    xassert(outputData.ndim == 6);
    xassert_eq(outputData.shape[0], T);
    xassert_eq(outputData.shape[1], F);
    xassert_eq(outputData.shape[2], 2);
    xassert_eq(outputData.shape[3], 4);
    xassert_eq(outputData.shape[4], 256);
    xassert_eq(outputData.shape[5], 2);
    xassert(outputData.on_gpu());
    xassert(outputData.is_fully_contiguous());

    // gains: shape (F,2,4,256), dtype float (stored as float32+32, so shape has extra dim 2)
    xassert(gains.ndim == 5);
    xassert_eq(gains.shape[0], F);
    xassert_eq(gains.shape[1], 2);
    xassert_eq(gains.shape[2], 4);
    xassert_eq(gains.shape[3], 256);
    xassert_eq(gains.shape[4], 2);
    xassert(gains.on_gpu());
    xassert(gains.is_fully_contiguous());

    launch_chime_frb_beamform(
        inputData.data, map.data, co.data,
        outputData.data, gains.data,
        T, F, stream);
}


// CPU reference implementation of chime_frb_beamform(), for testing.
// Uses O(N^2) brute-force DFT (not FFT), and float32 output (not float16).

void cpu_chime_frb_beamform(
    const Array<uint8_t> &inputData,
    const Array<uint> &map,
    const Array<float> &co,
    Array<float> &outputData,
    const Array<float> &gains)
{
    // inputData: (T,F,2,4,256) uint8_t
    xassert(inputData.ndim == 5);
    xassert_eq(inputData.shape[2], 2);
    xassert_eq(inputData.shape[3], 4);
    xassert_eq(inputData.shape[4], 256);
    xassert(inputData.on_host());
    xassert(inputData.is_fully_contiguous());
    
    long T = inputData.shape[0];
    long F = inputData.shape[1];

    // map: (F,256) uint
    xassert_shape_eq(map, ({F, 256}));
    xassert(map.on_host());
    xassert(map.is_fully_contiguous());

    // co: (F,4,4,2) float
    xassert_shape_eq(co, ({F, 4, 4, 2}));
    xassert(co.on_host());
    xassert(co.is_fully_contiguous());

    // outputData: (T,F,2,4,256,2) float
    xassert_shape_eq(outputData, ({T, F, 2, 4, 256, 2}));
    xassert(outputData.on_host());
    xassert(outputData.is_fully_contiguous());

    // gains: (F,2,4,256,2) float
    xassert_shape_eq(gains, ({F, 2, 4, 256, 2}));
    xassert(gains.on_host());
    xassert(gains.is_fully_contiguous());

    // Precompute twiddle factors: tw[n] = exp(2*pi*i*n/512)
    vector<complex<float>> tw(512);
    for (int n = 0; n < 512; n++) {
        double phase = 2.0 * M_PI * n / 512.0;
        tw[n] = { float(cos(phase)), float(sin(phase)) };
    }

    for (long t = 0; t < T; t++) {
        for (long f = 0; f < F; f++) {
            for (int pol = 0; pol < 2; pol++) {

                // Steps 1-3: Unpack, gain multiply, EW beamform.
                //   Step 1: E = unpack(inputData)
                //   Step 2: Fval = G^* * E
                //   Step 3: H[eo,ns] = (1/4) sum_ei co[eo,ei] * Fval[ei,ns]^*

                complex<float> H[4][256];
                for (int eo = 0; eo < 4; eo++) {
                    for (int ns = 0; ns < 256; ns++) {
                        complex<float> sum(0.0f, 0.0f);
                        for (int ei = 0; ei < 4; ei++) {
                            uint8_t x = inputData.at({t, f, pol, ei, ns});
                            float re = float(x >> 4) - 8.0f;
                            float im = float(x & 0xf) - 8.0f;
                            complex<float> E(re, im);

                            float gre = gains.at({f, pol, ei, ns, 0});
                            float gim = gains.at({f, pol, ei, ns, 1});
                            complex<float> Fval = conj(complex<float>(gre, gim)) * E;

                            float core = co.at({f, eo, ei, 0});
                            float coim = co.at({f, eo, ei, 1});
                            sum += complex<float>(core, coim) * conj(Fval);
                        }
                        H[eo][ns] = sum * 0.25f;
                    }
                }

                // Step 4: NS DFT (brute-force, zero-padded 256 -> 512).
                //   J[eo,j] = sum_k exp(2*pi*i*j*k/512) H[eo,k]

                complex<float> J[4][512];
                for (int eo = 0; eo < 4; eo++) {
                    for (int j = 0; j < 512; j++) {
                        complex<float> sum(0.0f, 0.0f);
                        for (int k = 0; k < 256; k++)
                            sum += tw[(long(j)*k) % 512] * H[eo][k];
                        J[eo][j] = sum;
                    }
                }

                // Step 5+6: Clamping (map reindexing) and write output.
                for (int eo = 0; eo < 4; eo++) {
                    for (int ns = 0; ns < 256; ns++) {
                        uint m = map.at({f, 255 - ns});
                        outputData.at({t, f, pol, eo, ns, 0}) = J[eo][m].real();
                        outputData.at({t, f, pol, eo, ns, 1}) = J[eo][m].imag();
                    }
                }
            }
        }
    }
}


void test_chime_frb_beamform()
{
    vector<long> v = ksgpu::random_integers_with_bounded_product(2, 10);
    long T = 128 * v[0];
    long F = v[1];

    cout << "test_chime_frb_beamform: T=" << T << ", F=" << F << endl;

    // inputData: (T,F,2,4,256) uint8_t
    Array<uint8_t> inputData_cpu({T, F, 2, 4, 256}, af_rhost | af_random);
    Array<uint8_t> inputData_gpu = inputData_cpu.to_gpu();

    // map: (F,256) uint, random values in [0, 512)
    // (In the unit test, we use a random map instead of a realistic map.)
    Array<uint> map_cpu({F, 256}, af_rhost);
    for (long i = 0; i < F * 256; i++)
        map_cpu.data[i] = rand() % 512;
    Array<uint> map_gpu = map_cpu.to_gpu();

    // co: (F,4,4,2) float
    Array<float> co_cpu({F, 4, 4, 2}, af_rhost | af_random);
    Array<float> co_gpu = co_cpu.to_gpu();

    // gains: (F,2,4,256,2) float
    Array<float> gains_cpu({F, 2, 4, 256, 2}, af_rhost | af_random);
    Array<float> gains_gpu = gains_cpu.to_gpu();

    // GPU kernel (float16 output).
    Array<__half> outputData_gpu({T, F, 2, 4, 256, 2}, af_gpu | af_zero);
    launch_chime_frb_beamform(inputData_gpu, map_gpu, co_gpu, outputData_gpu, gains_gpu, nullptr);
    CUDA_PEEK("test_chime_frb_beamform");
    CUDA_CALL(cudaDeviceSynchronize());

    // CPU reference (float32 output).
    Array<float> outputData_cpu({T, F, 2, 4, 256, 2}, af_rhost | af_zero);
    cpu_chime_frb_beamform(inputData_cpu, map_cpu, co_cpu, outputData_cpu, gains_cpu);

    assert_arrays_equal(outputData_cpu, outputData_gpu, "cpu", "gpu",
                        {"time","freq","pol","ew","ns","ReIm"});

    cout << "test_chime_frb_beamform: pass" << endl;
}


void time_chime_frb_beamform()
{
    long T = 49152;
    long F = 16;
    long niterations = 1000;
    long nstreams = 1;

    // Realistic clamping map: F frequencies linearly spaced from 400 to 800 MHz.
    Array<double> freqs({F}, af_rhost);
    for (long f = 0; f < F; f++)
        freqs.data[f] = 400.0 + (800.0 - 400.0) * f / (F - 1);

    Array<uint8_t> inputData({T, F, 2, 4, 256}, af_gpu | af_zero);    
    Array<uint> map_arr = calculate_cl_indices(freqs, 60.0).to_gpu();
    Array<float> co({F, 4, 4, 2}, af_gpu | af_zero);
    Array<__half> outputData({T, F, 2, 4, 256, 2}, af_gpu | af_zero);
    Array<float> gains({F, 2, 4, 256, 2}, af_gpu | af_zero);

    double gmem_gb = 1.0e-9 * (inputData.nbytes() + outputData.nbytes());
    KernelTimer kt(niterations, nstreams);

    while (kt.next()) {
        launch_chime_frb_beamform(inputData, map_arr, co, outputData, gains, kt.stream);

        if (kt.warmed_up && (kt.curr_iteration % 50 == 49)) {
            double gb_per_sec = gmem_gb / kt.dt;
            cout << "chime_frb_beamform: " << gb_per_sec << " GB/s (iteration " << kt.curr_iteration << ")" << endl;
        }
    }
}

// -------------------------------------------------------------------------------------------------
//
// Standalone computation of the CHIME FRB beamformer clamping map.
//
// This logic can be copied into another project without any kotekan dependencies.
// It reimplements the logic from the following kotekan source files:
//
//   - lib/hsa/hsaBeamformKernel.hpp  (constants: LIGHT_SPEED, FEED_SEP, PI)
//   - lib/hsa/hsaBeamformKernel.cpp  (constructor: freq_ref calculation)
//   - lib/hsa/hsaBeamformKernel.cpp  (calculate_cl_index: clamping map)
//   - lib/hsa/hsaBeamformKernel.cpp  (calculate_ew_phase: EW coefficients)
//
// The clamping map selects which 256 of 512 FFT bins to use as NS beams,
// compensating for the frequency-dependent beam shift in a phased array.


// ---- Hardcoded physical constants ----
// Matches lib/hsa/hsaBeamformKernel.hpp lines 25-27.

static constexpr double LIGHT_SPEED = 299792458.;  // speed of light [m/s]
static constexpr double FEED_SEP = 0.3048;          // NS feed separation [m]
static constexpr double PI = 3.14159265;             // (kotekan uses this value, not M_PI)


// ---- Clamping map calculation ----
// Matches lib/hsa/hsaBeamformKernel.cpp lines 158-183:
//   void hsaBeamformKernel::calculate_cl_index(uint32_t* host_map, float freq_now, double freq_ref)
//
// The freq_ref calculation (kotekan constructor, line 57) is inlined at the top of this function:
//   freq_ref = (LIGHT_SPEED * (128) / (sin(_northmost_beam * PI / 180.) * FEED_SEP * 256)) / 1.e6;
//
// For each of 256 output NS beams, computes which of the 512 FFT bins
// to select at the given observing frequency.
//
// Args:
//   host_map:          output array, must have room for 256 uint32 values, each in [0, 511].
//   freq_now:          observing frequency [MHz]
//   northmost_beam:    zenith angle of the northernmost beam [degrees].
//                      Production CHIME uses 60.0.

void calculate_cl_index(uint *host_map, double freq_now, double northmost_beam) {

    // Reference frequency calculation.
    // Matches lib/hsa/hsaBeamformKernel.cpp line 57 (constructor).
    double freq_ref = (LIGHT_SPEED * 128 / (sin(northmost_beam * PI / 180.) * FEED_SEP * 256)) / 1.e6;

    // Local variables match hsaBeamformKernel::calculate_cl_index() exactly.
    double t, delta_t, beam_ref;
    int cl_index;
    double D2R = PI / 180.;
    int pad = 2;

    for (int b = 0; b < 256; ++b) {
        // Sky angle for beam b at the reference frequency.
        beam_ref =
            asin(LIGHT_SPEED * (b - 256 / 2.) / (freq_ref * 1.e6) / (256) / FEED_SEP) * 180. / PI;

        // FFT bin at the reference frequency (fractional, +0.5 for rounding).
        t = 256 * pad * (freq_ref * 1.e6) * (FEED_SEP / LIGHT_SPEED * sin(beam_ref * D2R)) + 0.5;

        // Shift due to observing at freq_now instead of freq_ref.
        delta_t = 256 * pad * (freq_now * 1e6 - freq_ref * 1e6)
                  * (FEED_SEP / LIGHT_SPEED * sin(beam_ref * D2R));

        cl_index = (int)floor(t + delta_t) + 256 * pad / 2.;

        // Wrap to [0, 512).
        if (cl_index < 0)
            cl_index = 256 * pad + cl_index;
        else if (cl_index > 256 * pad)
            cl_index = cl_index - 256 * pad;

        cl_index = cl_index - 256;
        if (cl_index < 0) {
            cl_index = 256 * pad + cl_index;
        }

        host_map[b] = cl_index;
    }
}


Array<uint> calculate_cl_indices(const Array<double> &freqs, double northmost_beam)
{
    xassert(freqs.ndim == 1);
    xassert(freqs.on_host());
    xassert(freqs.is_fully_contiguous());

    long F = freqs.shape[0];
    xassert(F > 0);

    Array<uint> ret({F, 256}, af_rhost);

    for (long f = 0; f < F; f++)
        calculate_cl_index(ret.data + f * 256, freqs.data[f], northmost_beam);

    return ret;
}


} // namespace pirate
