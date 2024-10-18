#include "../include/n2k/rfi_kernels.hpp"
#include "../include/n2k/internals/internals.hpp"

#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;


// This is a good place to describe the array layout for PL mask.
//
// (See also software overleaf, section "Data sent from CPU to GPU
// in each kotekan frame".)
//
//   - The PL mask is logically an array bool[T/2][F/4][S/8], where
//     T is the number of time samples, F is number of freq channels,
//     and S is the number of "stations" (i.e. dish+pol pairs).
//     The array layout is non-trivial and described as follows.
//
//   - Separate time index into indices (t128, t2, t1):
//        t = 128*t128 + 2*t2 + t1    (where 0 <= t2 < 64 and 0 <= t1 < 2)
//
//   - Separate freq index into indices (f4, f1)
//        f = 4*t4 + f1               (where 0 <= f1 < 4)
//
//   - Separate station index into indices (s8, s1)
//        s = 8*s8 + s1               (where 0 <= s1 < 8)
//
//   - Then the PL mask uses this array layout:
//
//        // Usage: mask[t128][f4][s8][t2]
//        int1 mask[T/128][F/4][S/8][64]
//
//      where indices are ordered "C-style" from slowest to fastest varying,
//      and little-endian bit ordering is assumed when interpreting int1[...].
//
//    - An equivalent layout which we usually use in code:
//
//        // Usage: mask[t128][f4][s8] & (1UL << t2)
//        uint64 mask[T/128][F/4][S/8]
//
//    - The total memory footprint of the PL mask is 512 times smaller than
//      the corresponding electric field array. E.g. in CHORD, the electric
//      field bandwidth is 10 GB/s/GPU, and the PL mask bandwidth is
//      20 MB/s/GPU.
//
//    - This layout assumes that T is a multiple of 128. I assume this is okay,
//      since we can choose the kotekan frame size to be a multiple of 128.
//
//    - In order for this layout to have good cache alignment properties, we
//      assume that the number of stations S is a multiple of 128. This is the case
//      for all of our current projects (CHIME, HIRAX, CHORD, CHORD pathfinder).
//
//    - If the number of frequency channels F is not a multiple of 4, we can
//      generalize the array layouts to:
//
//        // Usage: mask[t128][f//4][s8][t2]
//        int1 mask[T/128][(F+3)//4][S/8][64]
//
//      (I don't know whether this generalization will be useful, but it doesn't
//       seem to be extra work to implement it in the GPU kernels, so I'm planning
//       to implement it.)


namespace n2k {
#if 0
}  // editor auto-indent
#endif


__device__ uint _cmask(int b)
{
    b = (b >= 0) ? b : 0;
    return (b < 32) ? (1U << b) : 0;
}


// s0_kernel() arguments:
//
//   uint4 s0[T/Nds][F][S/4];                 // output array, (downsampled time index, freq channel, station)
//   uint pl_mask[T/128][(F+3)/4][S/8][2];    // input array, packet loss mask
//   long T;                                  // number of time samples
//   long F;                                  // number of freq channels
//   long S;                                  // number of stations (= dish+pol pairs)
//   long Nds;                                // time downsampling factor
//   long out_fstride4;                       // freq stride of s0 array (ulong4 stride, not ulong stride)
//
// Constraints (checked in launch_s0_kernel() below)
//
//   - Nds must be even.
//   - S must be a multiple of 128.
//   - T must be a multiple of 128.
//   - T must be a multiple of Nds.
//   - out_fstride must be a multiple of 4.
//   - out_fstride must be >= S.
//
// Notes on parallelization:
//
//   - Each warp independently processes one tds index, 4 freqs, and 128 stations.
//     Each thread processes one (t/32) index, 4 freqs, and 8 stations.
//
//   - Within the warp, the thread mapping is
//       t4 t3 t2 t1 t0 <-> (s/64) (s/32) (s/16) (s/8) (time/32)
//
//   - Within the larger kernel, the warp mapping is:
//       wz wy wx <-> (tds) (f/4) (s/128)
//
// FIXME think carefully about int32 overflows!!

__global__ void s0_kernel(ulong4 *s0, const uint *pl, int T, int F, int S, int Nds, int out_fstride4)
{
    static constexpr uint ALL_LANES = 0xffffffffU;
    
    // Warp location within larger kerenl
    int tds = (blockIdx.z * blockDim.z + threadIdx.z);              // output time
    int fds = (blockIdx.y * blockDim.y + threadIdx.y);              // input (f/4)
    int sds = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) << 4;  // input (s/8), laneId not included, multiple of 16

    int Fds = (F+3) >> 2;      // number of downsampled freq channels in 'pl_mask'.
    int nf = min(4, F-4*fds);  // number of frequency channels to write

    // These tests guarante that we don't write past the edge of memory.
    
    if (tds*Nds >= T)
	return;
    if (4*fds >= F)
	return;
    if (8*sds >= S)
	return;

    // Shift input pointer. Note that no time shift is not applied, but laneId is applied.
    // Before the shifts, 'pl' has shape uint[T/128, Fds, S/8, 2].
    // After these shifts, 'pl' has shape uint[T/128] and stride Fds * (S/4).
    
    pl += (sds << 1);
    pl += fds * (S >> 2);
    pl += (threadIdx.x & 31);  // laneId
    long pl_stride = long(Fds) * long(S >> 2);

    // Shift output pointer, including time and laneId.
    // Before the shifts, 's0' has shape ulong4[T/Nds, F, S/4].
    // After the shifts, 's0' has shape ulong4[4] and stride 'out_fstride4'.
    
    s0 += (sds << 1);
    s0 += (fds << 2) * out_fstride4;
    s0 += tds * F * out_fstride4;
    s0 += (threadIdx.x & 31);  // laneId
    
    // [t2_lo:t2_hi) = range of t2 values processed on this warp.
    int t2_lo = tds * (Nds >> 1);
    int t2_hi = t2_lo + (Nds >> 1);
    
    // [t128_lo:128_hi) = range of t128 values processed on this warp.
    int t128_lo = (t2_lo >> 6);
    int t128_hi = ((t2_hi-1) >> 6) + 1;

    uint s0_accum = 0;
    
    for (int t128 = t128_lo; t128 < t128_hi; t128++) {
	uint x = pl[t128 * pl_stride];
	int t2 = (t128 << 6) + ((threadIdx.x & 1) << 5);
	uint mask = _cmask(t2_hi - t2) - _cmask(t2_lo - t2);
	s0_accum += __popc(x & mask);
    }

    s0_accum <<= 1;
    s0_accum += __shfl_sync(ALL_LANES, s0_accum, threadIdx.x ^ 0x1);

    ulong4 s0_x4;
    s0_x4.x = s0_accum;
    s0_x4.y = s0_accum;
    s0_x4.z = s0_accum;
    s0_x4.w = s0_accum;    

    for (int f = 0; f < nf; f++)
	s0[f * out_fstride4] = s0_x4;
}


// launch_s0_kernel() arguments, bare pointer version:
//
//   ulong s0[T/Nds][F][S];                // output array, (downsampled time index, freq channel, station)
//   ulong pl_mask[T/128][(F+3)/4][S/8];   // input array, packet loss mask
//   long T;                               // number of time samples
//   long F;                               // number of freq channels
//   long S;                               // number of stations (= dish+pol pairs)
//   long Nds;                             // time downsampling factor
//   long out_fstride;                     // frequency stride in 'S0' array
//
// Constraints (checked here)
//
//   - Nds must be even.
//   - S must be a multiple of 128.
//   - T must be a multiple of 128.
//   - T must be a multiple of Nds.
//   - out_fstride must be a multiple of 4.
//   - out_fstride must be >= S.
//
// Notes on parallelization:
//
//   - Each warp independently processes one tds index, 4 freqs, and 128 stations.
//     Each thread processes one (t/32) index, 4 freqs, and 8 stations.
//
//   - Within the warp, the thread mapping is
//       t4 t3 t2 t1 t0 <-> (s/64) (s/32) (s/16) (s/8) (time/32)
//
//   - Within the larger kernel, the warp mapping is:
//       wz wy wx <-> (tds) (f/4) (s/128)

void launch_s0_kernel(ulong *s0, const ulong *pl_mask, long T, long F, long S, long Nds, long out_fstride, cudaStream_t stream)
{
    if (T <= 0)
	throw runtime_error("launch_s0_kernel: number of time samples T must be > 0");
    if (T & 127)
	throw runtime_error("launch_s0_kernel: number of time samples T must be a multiple of 128");
    if (F <= 0)
	throw runtime_error("launch_s0_kernel: number of frequency samples F must be > 0");
    if (S <= 0)
	throw runtime_error("launch_s0_kernel: number of stations S must be > 0");
    if (S & 127)
	throw runtime_error("launch_s0_kernel: number of stations S must be a multiple of 128");
    if (Nds <= 0)
	throw runtime_error("launch_s0_kernel: downsampling factor 'Nds' must be positive");
    if (Nds & 1)
	throw runtime_error("launch_s0_kernel: downsampling factor 'Nds' must be even");
    if (out_fstride < S)
	throw runtime_error("launch_s0_kernel(): out_fstride must be >= S");	
    if (out_fstride % 4)
	throw runtime_error("launch_s0_kernel(): out_fstride must be a multiple of 4");

    long Tds = T / Nds;
    
    if (T != (Tds * Nds))
	throw runtime_error("launch_s0_kernel: number of time samples T must be a multiple of downsampling factor 'Nds'");

    dim3 nblocks, nthreads;
    gputils::assign_kernel_dims(nblocks, nthreads, S >> 2, (F+3) >> 2, Tds);

    s0_kernel <<< nblocks, nthreads, 0, stream >>>
	((ulong4 *) s0, (const uint *) pl_mask, T, F, S, Nds, out_fstride >> 2);
    
    CUDA_PEEK("s0_kernel launch");
}

// Arguments:
//
//  - s0: uint64 array of shape (T/Nds, F, S)
//  - pl_mask: uint64 array of shape (T/128, (F+3)//4, S/8).
//  - Nds: Time downsampling factor. Must be multiple of 2.

void launch_s0_kernel(Array<ulong> &s0, const Array<ulong> &pl_mask, long Nds, cudaStream_t stream)
{
    // S0 shape = (T/Nds, F, S)
    // pl_mask shape = (T/128, (F+3)/4, S/8)
    
    check_array(s0, "launch_s0_kernel", "S0", 3, false);            // ndim=3, contiguous=false
    check_array(pl_mask, "launch_s0_kernel", "pl_mask", 3, true);   // ndim=3, contiguous=true
    
    long Tds = s0.shape[0];
    long F = s0.shape[1];
    long S = s0.shape[2];
    long out_fstride = s0.strides[1];
    
    long T128 = pl_mask.shape[0];
    long Fds = pl_mask.shape[1];
    long Sds = pl_mask.shape[2];

    if ((Tds*Nds != T128*128) || (Fds != ((F+3)/4)) || (S != (Sds*8)))
	throw runtime_error("launch_s0_kernel(): s0.shape=" + s0.shape_str() + " and pl_mask.shape=" + pl_mask.shape_str() + " are inconsistent");

    if (s0.strides[2] != 1)
	throw runtime_error("launch_s0_kernel(): expected innermost (station) axis of S0 to be contiguous");
    if (s0.strides[0] != F*out_fstride)
	throw runtime_error("launch_s0_kernel(): expected time+freq axes of S0 to be contiguous");

    launch_s0_kernel(s0.data, pl_mask.data, 128*T128, F, S, Nds, out_fstride, stream);
}


}  // namespace n2k
