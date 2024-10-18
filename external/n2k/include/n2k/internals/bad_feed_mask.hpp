#ifndef _N2K_BAD_FEED_MASK_HPP
#define _N2K_BAD_FEED_MASK_HPP

#include "device_inlines.hpp"

// This source file is used internally in CUDA kernels.
// It probably won't be useful "externally" to n2k.

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// load_bad_feed_mask()
//
// Idea: we want to loop over stations 0, 1, ..., (S-1) as follows (*)
//
//   extern __shared__ uint shmem[];
//   uint bf = load_bad_feed_mask(bf_mask, shmem, S);
//
//   for (int s = threadIdx.x; s < S; s += blockDim.x) {
//       bool valid = bf & 1;
//          ...
//       bf >>= 1;
//   }
//
// Note that threadIdx.y and threadIdx.z can be > 1. In that case, these indices are "spectators"
// to the above loop over stations.
//
// Assumptions (caller must check):
//
//   - Number of stations S is a multiple of 128.
//   - Number of stations S is <= 32 * blockDim.x.
//
// The 'sp' pointer should point to a shared memory buffer of length max(S/32,32).
// WARNING: caller must call __syncthreads() before shared memory 'sp' can be re-used!


__host__ __device__ inline uint bf_mask_shmem_nelts(int S)
{
    return (S > 1024) ? (S >> 5) : 32;
}


__host__ inline uint bf_mask_shmem_nbytes(int S, int Wx)
{
    assert(S > 0);
    assert(S <= 1024*Wx);
    assert((S % 128) == 0);
    
    return 4 * bf_mask_shmem_nelts(S);
}


template<bool Debug = false>
__device__ inline uint load_bad_feed_mask(const uint *bf_mask, uint *sp, int S)
{
    uint t0 = threadIdx.z;
    t0 = (t0 * blockDim.y) + threadIdx.y;
    t0 = (t0 * blockDim.x) + threadIdx.x;

    uint nt = blockDim.x * blockDim.y * blockDim.z;
    
    // In each iteration of the loop, we read 128 stations.
    
    for (uint t = t0; t < (S>>2); t += nt) {
	uint x = __vcmpne4(bf_mask[t], 0);       // s0 s1 s2 s3 s4 s5 s6 <-> b3 b4 t0 t1 t2 t3 t4
	x = transpose_bit_with_lane<1> (x, 1);   // s0 s1 s2 s3 s4 s5 s6 <-> b3 b4 b0 t1 t2 t3 t4
	x = transpose_bit_with_lane<2> (x, 2);   // s0 s1 s2 s3 s4 s5 s6 <-> b3 b4 b0 b1 t2 t3 t4
	x = transpose_bit_with_lane<4> (x, 4);   // s0 s1 s2 s3 s4 s5 s6 <-> b3 b4 b0 b1 b2 t3 t4

	// Shared memory layout:
	// Define s = 32*shi + slo, and store at shmem[shi].	
	// FIXME (low-priority) creates a bank conflict if S > 1024, but I doubt this is ever a bottleneck.

	uint shi = t >> 3;
	
	if constexpr (Debug)
	    assert(shi < bf_mask_shmem_nelts(S));
	
	if ((threadIdx.x & 7) == 0)
	    sp[shi] = x;

	__syncwarp();
    }

    __syncthreads();

    // Define s = (blockDim.x*j) + (32*k) + i
    // In the loop (*) above, each warp will process 32 values of i, and (S / blockDim.x) values of j.
    // We read the mask from shared memory as: (bits <-> i), (threads <-> j)
    // And then transpose to: (bits <-> j), (threads <-> i).
    
    uint t01 = threadIdx.x & 3;
    uint t234 = threadIdx.x & 28;
    uint j = (t01 << 3) | (t234 >> 2);         // j0 j1 j2 j3 j4 <-> t2 t3 t4 t0 t1
    uint s = (j * blockDim.x) + threadIdx.x;
    uint shi = s >> 5;
    shi = (s < S) ? shi : (shi & 31);

    if constexpr (Debug)
	assert(shi < bf_mask_shmem_nelts(S));
    
    // FIXME (low-priority) optimize for case where we don't need all the transposes.
    
    uint y = sp[shi];
    y = transpose_bit_with_lane<1> (y, 4);     // i0 i1 i2 i3 i4 <-> b3 b4 t2 b1 b2     j0 j1 j2 j3 j4 <-> b0 t3 t4 t0 t1
    y = transpose_bit_with_lane<2> (y, 8);     // i0 i1 i2 i3 i4 <-> b3 b4 t2 t3 b2     j0 j1 j2 j3 j4 <-> b0 b1 t4 t0 t1
    y = transpose_bit_with_lane<4> (y, 16);    // i0 i1 i2 i3 i4 <-> b3 b4 t2 t3 t4     j0 j1 j2 j3 j4 <-> b0 b1 b2 t0 t1
    y = transpose_bit_with_lane<8> (y, 1);     // i0 i1 i2 i3 i4 <-> t0 b4 t2 t3 t4     j0 j1 j2 j3 j4 <-> b0 b1 b2 b3 t1
    y = transpose_bit_with_lane<16> (y, 2);    // i0 i1 i2 i3 i4 <-> t0 t1 t2 t3 t4     j0 j1 j2 j3 j4 <-> b0 b1 b2 b3 b4
    
    return y;
}


}  // namespace n2k

#endif // _N2K_BAD_FEED_MASK_HPP
