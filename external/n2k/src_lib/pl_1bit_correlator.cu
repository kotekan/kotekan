#include "../include/n2k/pl_kernels.hpp"
#include "../include/n2k/internals/internals.hpp"
#include "../include/n2k/internals/device_inlines.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/device_mma.hpp>

using namespace std;
using namespace gputils;

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// Array layouts.
//
// We use the m8n8k128 int1 tensor core MMA. The input fragments are 8-by-128 int1 matrices,
// represented by an 'int' register. The output fragment is an 8-by-8 int32 matrix, represented
// by int[2]. The tensor core register assignments are:
//
//   A_{ij}    b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4     t0 t1 t2 t3 t4 <-> j5 j6 i0 i1 i2
//   B_{jk}    b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4     t0 t1 t2 t3 t4 <-> j5 j6 k0 k1 k2
//   C_{ik}    r0 <-> k0                             t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2
//
// The input array layouts are:
//
//   uint1 pl_mask[T/64][F][Sds][64];    // where (F,Sds) may be downsampled by (2,4,8)
//   uint1 rfimask[F][T];


// -------------------------------------------------------------------------------------------------


// Reads a 16-by-128 submatrix, as two 8-by-128 fragments in tensor core ordering (see above).
// The global memory pointer 'pl_adjusted' should include threadblock/warp offsets to point
// at the appropriate 16-by-128 submatrix, plus the 'pl_lane_offset' (see below) so that
// each lane points to a distinct 64-bit part of the 16-by-128 submatrix.

__device__ inline void read_pl_16_128(int pl[2][1], const ulong *pl_adjusted, uint rfi, uint n128)
{
    // pl_lane_offset() has been written so that dereferencing the pointer gives
    // the following register assigment:
    //
    // b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4    t0 t1 t2 t3 t4 <-> i3 j6 i0 i1 i2    r <-> j5
    
    int2 t = *((const int2 *) pl_adjusted);
    
    // Now we're one warp_transpose() away from the desired tensor core assignment:
    //
    // b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4    t0 t1 t2 t3 t4 <-> j5 j6 i0 i1 i2    r <-> i3

    warp_transpose(t.x, t.y, 1);

    // Get RFI mask from appropriate thread.
    uint dst_lane = (threadIdx.x & 3) + (n128 << 2);
    rfi = __shfl_sync(FULL_MASK, rfi, dst_lane);
    
    pl[0][0] = t.x & rfi;
    pl[1][0] = t.y & rfi;
}

// The following per-lane offset is applied to the 'pl' global memory pointer,
// so that each lane points to a distinct 64-bit part of a 16-by-128 submatrix.
//
// The pl_lane_offset is constructed so that dereferencing the pointer gives
// the following register assignment:
//
// b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4    t0 t1 t2 t3 t4 <-> i3 j6 i0 i1 i2    r <-> j5
//
// Reminder: offsets/strides are 64-bit, since pl is (const ulong *).
// (In particular, the 't64_stride' argument should be F*Sds.)

__device__ inline int pl_lane_offset(uint t64_stride)
{
    // t0 t1 t2 t3 t4 <-> i3 j6 i0 i1 i2
    
    int ilo = (threadIdx.x >> 2) & 0x7;
    int ihi = (threadIdx.x & 0x1) << 3;
    int j = (threadIdx.x & 0x2) ? t64_stride : 0;
    return ilo + ihi + j;
}


// -------------------------------------------------------------------------------------------------


// Write one 8-by-8 tile of the 'counts' array C_{ik} to global memory.
// The tile is stored in tensor core ordering:
//
//    r <-> k0      t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2

__device__ inline void write_8_8(int *counts, int v[2])
{
    int2 t;
    t.x = v[0];
    t.y = v[1];

    int2 *p = (int2 *) (counts);
    p[threadIdx.x & 31] = t;
}


// Write two 8-by-8 tiles of the 'counts' array C_{ik} to global memory.
//
// This needs a little explanation! We assume that the two tiles differ by k=16
// (not k=8 as might be expected), and that both tiles are in the lower triangle.

__device__ inline void write_8_16(int *counts, int v[2][2])
{
    write_8_8(counts, v[0]);
    write_8_8(counts + 128, v[1]);   // pointer offset is 128 if tiles differ by k=16
}


// -------------------------------------------------------------------------------------------------
//
// Parallelization:
//
//   blockIdx.z, threadIdx.z <-> downsampled (output) time
//   blockIdx.y, threadIdx.y <-> frequency channel
//   threadIdx.x <-> sub-tile of Sds-by-Sds matrix
//     (blockIdx.x not currently used, but will be used when Sds > 128 is implemented)


__global__ void __launch_bounds__(128,8)
correlate_pl_kernel_S16(int *counts, const ulong *pl_mask, const uint *rfimask, int rfimask_fstride, int Tout, int F, uint N128)
{
    // Assumes blockDim = {32,Wy,Wz} where (Wy*Wz)=4.
    
    constexpr int S = 16;    
    const uint tout = blockIdx.z * blockDim.z + threadIdx.z;
    const uint f = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tout >= Tout) || (f >= F))
	return;   // okay since this kernel never calls __syncthreads()

    // ulong pl_mask[T/64][F][S];
    const uint t128_stride = (2*F) * S;  // 64-bit (ulong) stride
    pl_mask += ulong(tout) * ulong(N128) * ulong(t128_stride);
    pl_mask += (f * S) + pl_lane_offset(F*S);

    // uint rfimask[F][T/32], note 'rfimask_fstride'.
    rfimask += ulong(f) * ulong(rfimask_fstride);
    rfimask += ((tout * N128) << 2) + (threadIdx.x & 31);
    
    uint rfi = 0;
    int pl[2][1];
    int v[3][2];

    v[0][0] = v[0][1] = v[1][0] = v[1][1] = v[2][0] = v[2][1] = 0;

    for (uint n128 = 0; n128 < N128; n128++) {
	if ((n128 & 7) == 0) {
	    rfi = *rfimask;
	    rfimask += 32;
	}

	// Read 16-by-128 PL-submatrix.
	read_pl_16_128(pl, pl_mask, rfi, n128);

	// Accumulate 16-by-16 V-matrix.
	mma_b1_m8_n8_k128(v[0], pl[0], pl[0], v[0]);
	mma_b1_m8_n8_k128(v[1], pl[1], pl[0], v[1]);
	mma_b1_m8_n8_k128(v[2], pl[1], pl[1], v[2]);
	
	pl_mask += t128_stride;
    }
    
    // int counts[Tout][F][3][8][8]
    counts += ulong(tout) * ulong(192*F);
    counts += (192 * f);

    // Write to global memory.
    write_8_8(counts, v[0]);
    write_8_8(counts + 64, v[1]);
    write_8_8(counts + 128, v[2]);
}


// -------------------------------------------------------------------------------------------------


__device__ inline void multiply_8_16(int v[2][2], int plx[1], int ply[2][1])
{
    mma_b1_m8_n8_k128(v[0], plx, ply[0], v[0]);
    mma_b1_m8_n8_k128(v[1], plx, ply[1], v[1]);
}


__global__ void __launch_bounds__(256,2)
correlate_pl_kernel_S128(int *counts, const ulong *pl_mask, const uint *rfimask, int rfimask_fstride, uint N128)
{
    constexpr int S = 128;
    const uint tout = blockIdx.z;
    const uint f = blockIdx.y;
    const uint F = gridDim.y;
    
    // Assumes blockDim = {256,1,1}.
    const int w = threadIdx.x >> 5;    // 0 <= w < 8
    const int wx = (w & 3);            // 0 <= wx < 4
    const int wy = (w >> 2);           // 0 <= wy < 2
    const int laneId = (threadIdx.x & 31);

    __shared__ int shmem[16*32];  // 2 KB
    
    // ulong pl_mask[T/64][F][S];
    // We assign a 16-by-128 contiguous submatrix to each warp.
    const uint t128_stride = (2*F) * S;       // 64-bit (int2) stride
    pl_mask += ulong(tout) * ulong(N128) * ulong(t128_stride);
    pl_mask += (f * S) + (w * 16) + pl_lane_offset(F*S);

    // uint rfimask[F][T/32], note 'rfimask_fstride'.
    rfimask += ulong(f) * ulong(rfimask_fstride);
    rfimask += ((tout * N128) << 2) + (threadIdx.x & 31);

    // MMA inputs/outputs (55 registers/thread)
    uint rfi = 0;
    int pl_in[2][1];
    int plx[4][1];
    int ply[4][2][1];
    int v[10][2][2];

    #pragma unroll
    for (int i = 0; i < 10; i++)
	v[i][0][0] = v[i][0][1] = v[i][1][0] = v[i][1][1] = 0;
    
    for (uint n128 = 0; n128 < N128; n128++) {
	if ((n128 & 7) == 0) {
	    rfi = *rfimask;
	    rfimask += 32;
	}

	// Read 16-by-128 submatrix from global memory.
	read_pl_16_128(pl_in, pl_mask, rfi, n128);

	// Write 16-by-128 submatrix to shared memory.
	shmem[64*w + laneId] = pl_in[0][0];
	shmem[64*w + laneId + 32] = pl_in[1][0];
	
	__syncthreads();

	// Read from shared memory.
	#pragma unroll
	for (int i = 0; i < 4; i++) {
	    plx[i][0] = shmem[32*wx + laneId + 128*i];
	    ply[i][0][0] = shmem[32*wy + laneId + 128*i];
	    ply[i][1][0] = shmem[32*wy + laneId + 128*i + 64];
	}
	
	__syncthreads();
	
	// Do matrix multiplications (20 int1 m8n8k128 MMAs).

	#pragma unroll
	for (int i = 0; i < 4; i++)
	    #pragma unroll
	    for (int j = 0; j <= i; j++)
		multiply_8_16(v[(i*(i+1))/2 + j], plx[i], ply[j]);
	
	pl_mask += t128_stride;
    }

    // Write to global memory.
    //   i = 4*I + wx          0 <= wx < 4
    //   j = 4*J + 2*R + wy    0 <= wy < 2, 0 <= R < 2
    //
    // Offset = 64*i(i+1)/2  + 64*j
    //        = 512*I^2 + a*I + 256*J + 128*R + b
    //
    //  where a = 256*wx + 128
    //        b = 32*wx*(wx+1) + 64*wy

    // int counts[Tout][F][8*17][8][8]
    int a = 256*wx + 128;
    int b = 32*wx*(wx+1) + 64*wy;
    ulong tf = ulong(tout)*ulong(F) + f;
    counts += tf*(8*17*64);
    counts += b;
    
    // Off-diagonals (I > J).
    #pragma unroll
    for (int i = 1; i < 4; i++)
	#pragma unroll
	for (int j = 0; j < i; j++)
	    write_8_16(counts + a*i + 512*i*i + 256*j, v[(i*(i+1))/2 + j]);
    
    // Diagonals (I = J).
    // We only write if (wx >= wy+2R).

    if (wx >= wy) {   // write R=0
	#pragma unroll
	for (int i = 0; i < 4; i++)
	    write_8_8(counts + a*i + 512*i*i + 256*i, v[(i*(i+1))/2 + i][0]);
    }

    if (wx >= wy+2) {  // write R=1
	#pragma unroll
	for (int i = 0; i < 4; i++)
	    write_8_8(counts + a*i + 512*i*i + 256*i + 128, v[(i*(i+1))/2 + i][1]);
    }
}


// -------------------------------------------------------------------------------------------------


void launch_pl_1bit_correlator(int *counts, const ulong *pl_mask, const uint *rfimask, long rfimask_fstride, long T, long F, long Sds, long Nds, cudaStream_t stream)
{
    if (!counts)
	throw runtime_error("launch_pl_1bit_correlator: 'counts' must be non-NULL");
    if (!pl_mask)
	throw runtime_error("launch_pl_1bit_correlator: 'pl_mask' must be non-NULL");
    if (!rfimask)
	throw runtime_error("launch_pl_1bit_correlator: 'rfimask' must be non-NULL");
    if (rfimask_fstride < (T/32))	
	throw runtime_error("launch_pl_1bit_correlator: expected rfimask_fstride >= T/32");
    if (T <= 0)
	throw runtime_error("launch_pl_1bit_correlator: expected T > 0");
    if (F <= 0)
	throw runtime_error("launch_pl_1bit_correlator: expected F > 0");
    if (Sds <= 0)
	throw runtime_error("launch_pl_1bit_correlator: expected Sds > 0");
    if (Nds <= 0)
	throw runtime_error("launch_pl_1bit_correlator: expected Nds > 0");
    if (Nds & 127)
	throw runtime_error("launch_pl_1bit_correlator: expected Nds to be a multiple of 128 (could be relaxed)");

    long Tout = T / Nds;
    uint N128 = Nds >> 7;
    
    if (T % Nds)
	throw runtime_error("launch_pl_1bit_correlator: expected T to be a multiple of Nds");
    if ((Tout >= INT_MAX) || (N128 >= INT_MAX) || (2*F*Sds >= INT_MAX))	
	throw runtime_error("launch_pl_1bit_correlator: 32-bit overflow");

    if (Sds == 16) {
	dim3 nthreads = {32, 2, 2};
	dim3 nblocks = { 1, uint(F+1)/2, uint(Tout+1)/2 };
	
	correlate_pl_kernel_S16
	    <<< nblocks, nthreads, 0, stream >>>
	    (counts, pl_mask, rfimask, rfimask_fstride, Tout, F, N128);
    }
    else if (Sds == 128) {
	dim3 nblocks = { 1, uint(F), uint(Tout) };
	
	correlate_pl_kernel_S128
	    <<< nblocks, 256, 0, stream >>>
	    (counts, pl_mask, rfimask, rfimask_fstride, N128);
    }
    else {
	throw runtime_error("launch_pl_1bit_correlator: Currently, only Sds=16 and Sds=128 are implemented."
			    " These values correspond to the CHORD pathfinder, and full CHORD. Let me know"
			    " if you need more generality");
    }
    
    CUDA_PEEK("correlate_pl_kernel");
}


void launch_pl_1bit_correlator(Array<int> &counts, const Array<ulong> &pl_mask, const Array<uint> &rfimask, long Nds, cudaStream_t stream)
{
    // pl_mask shape = (T/64, F, Sds)
    // counts shape = (T/Nds, F, ntiles, 8, 8)
    // rfimask shape = (F, T/32)
    
    check_array(counts, "launch_pl_1bit_correlator", "counts", 5, true);     // contiguous=true
    check_array(pl_mask, "launch_pl_1bit_correlator", "pl_mask", 3, true);   // contiguous=true
    check_array(rfimask, "launch_pl_1bit_correlator", "rfimask", 2, false);  // contiguous=false
    
    long T = 64 * pl_mask.shape[0];
    long F = pl_mask.shape[1];
    long Sds = pl_mask.shape[2];
    long ntiles = ((Sds/8) * ((Sds/8) + 1)) / 2;

    if (Nds <= 0)
	throw runtime_error("launch_pl_1bit_correlator: expected Nds > 0");
    if (Nds & 127)
	throw runtime_error("launch_pl_1bit_correlator: expected Nds to be a multiple of 128 (could be relaxed)");
    if (Sds & 7)
	throw runtime_error("launch_pl_1bit_correlator: expected Sds to be a multiple of 8");;
    if (T % Nds)
	throw runtime_error("launch_pl_1bit_correlator: expected T to be a multiple of Nds");

    if (!counts.shape_equals({T/Nds,F,ntiles,8,8})) {
	stringstream ss;
	ss << "launch_pl_1bit_correlator: counts.shape (=" << counts.shape_str() << ")."
	   << " Based on pl_mask.shape (=" << pl_mask.shape_str() << ") and Nds=" << Nds
	   << ", expected shape (" << (T/Nds) << "," << F << "," << ntiles << ",8,8)";
	throw runtime_error(ss.str());
    }

    if (!rfimask.shape_equals({F,T/32})) {
	stringstream ss;
	ss << "launch_pl_1bit_correlator: rfimask.shape (=" << rfimask.shape_str() << ")."
	   << " Based on pl_mask.shape (=" << pl_mask.shape_str()
	   << ", expected shape (" << F << "," << (T/32) << ")";
	throw runtime_error(ss.str());
    }

    if (rfimask.strides[1] != 1)
	throw runtime_error("launch_pl_1bit_correlator: expected inner (time) axis of rfimask to be contiguous");
    
    launch_pl_1bit_correlator(counts.data, pl_mask.data, rfimask.data, rfimask.strides[0], T, F, Sds, Nds, stream);
}


}  // namespace n2k
