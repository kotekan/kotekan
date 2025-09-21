#include "../include/n2k/rfi_kernels.hpp"
#include "../include/n2k/internals/bad_feed_mask.hpp"
#include "../include/n2k/internals/internals.hpp"

#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// Kernel arguments:
//
//   ulong S012_out[M];    // where M is number of "spectator" indices (3*T*F)
//   ulong S012_in[M][S];  // where S=2*D is number stations
//   uint bf_mask[S/4];
//   long M;
//   long S;
//
// Constraints:
//
//   - S must be a multiple of 128 (from load_bad_feed_mask()).
//   - S must be <= 32 * blockDim.x (from load_bad_feed_mask()).
//   - Wy is a power of two.
//   - No constraint on M.
//
// Parallelization:
//
//   - Each block processes 32 spectator indices (and all S stations).
//
//   - threadIdx.x <-> station
//     threadIdx.y, blockIdx.x <-> spectator index   (note xy mismatch here!)
//
// Shared memory layout:
//
//   uint bf[32*Wx];           // (128 * Wx) bytes.
//   uint red[Wx][Wy][32/Wy];  // (128 * Wx) bytes.



// N = number of spectator indices read (equal to 32/Wy in full kernel).
// L = number of lanes summed over so far

template<int N, int L>
struct template_magic
{
    static_assert((L & (L-1)) == 0, "L must be a power of two");
    static_assert((L % N) == 0, "L must be a multiple of N");

    // Default template reduces case (L > N) to case (L == N).
    static __device__ ulong load_and_sum(const ulong *Sin, uint bf, int m, int T, int Tmin, int Tsize, int F, int S)
    {
	constexpr uint bit = (L/2);
	ulong x = template_magic<N,(L/2)>::load_and_sum(Sin, bf, m, T, Tmin, Tsize, F, S);
	return x + __shfl_sync(FULL_MASK, x, threadIdx.x ^ bit);
    }
};

template<int N>
struct template_magic<N,N>
{
    // This template reduces case (L == N > 1) to case (L == N == 1)
    static __device__ ulong load_and_sum(const ulong *Sin, uint bf, int m, int T, int Tmin, int Tsize, int F, int S)
    {
	constexpr uint bit = (N/2);
	
	ulong x = template_magic<(N/2),(N/2)>::load_and_sum(Sin, bf, m, T, Tmin, Tsize, F, S);
	ulong y = template_magic<(N/2),(N/2)>::load_and_sum(Sin, bf, m+(N/2), T, Tmin, Tsize, F, S);
        ulong src = (threadIdx.x & bit) ? x : y;
        ulong dst = (threadIdx.x & bit) ? y : x;
	return dst + __shfl_sync(FULL_MASK, src, threadIdx.x ^ bit);
    }
};

template<>
struct template_magic<1,1>
{
    // Base case: L == N == 1.
    static __device__ ulong load_and_sum(const ulong *Sin, uint bf, int m0, int T, int Tmin, int Tsize, int F, int S)
    {
        const int M = T * F * 3;
	m0 = (m0 < M) ? m0 : (M-1);
	
        // Calculate ringbuffer address
        int t0 = m0 / (F * 3);
        int f = m0 % (F * 3);
        int t = (Tmin + t0) & (Tsize - 1);
        int m = t * (F * 3) + f;

	uint b = bf;
	ulong ret = 0;
	
	for (int s = threadIdx.x; s < S; s += blockDim.x) {
	    ulong x = Sin[m*S + s];
	    ret += ((b & 1) ? x : 0);   // don't sum bad feeds
	    b >>= 1;
	}
	
	return ret;
    }
};


template<int Wy>
__global__ void s012_station_downsample_kernel(ulong* Sout, const ulong* Sin, const uint* bf_mask,
                                               int T, int Tmin, int Tsize, int F, int S)
{
    // Spectator indices per y-warp.
    static constexpr int N = (32 / Wy);
    const uint laneId = threadIdx.x & 31;
    const uint warpId = ((threadIdx.x >> 5) * blockDim.y) + threadIdx.y;

    // Shared memory layout.
    extern __shared__ uint shmem[];
    uint *shmem_bf = shmem;
    ulong *shmem_red = (ulong *) (shmem + bf_mask_shmem_nelts(S));   // shape (Wx, Wy, N)

    const int M = T * F * 3;

    // No syncthreads() needed after load_bad_feed_mask(), since 'shmem_bf' is not re-used.
    int m = 32*blockIdx.x + N*threadIdx.y;
    uint bf = load_bad_feed_mask(bf_mask, shmem_bf, S);
    ulong x = template_magic<N,32>::load_and_sum(Sin, bf, m, T, Tmin, Tsize, F, S);  // handles case m >= M
	
    if (laneId < N)
	shmem_red[warpId*N + laneId] = x;
    
    __syncthreads();
    
    if (warpId != 0)
	return;

    x = 0;
    for (int i = laneId; i < blockDim.x; i += 32)
	x += shmem_red[i];

    m = 32*blockIdx.x + laneId;
    if (m < M)
	Sout[m] = x;
}


// ulong S012_out[M];    // where M is number of "spectator" indices (3*T*F)
// ulong S012_in[M][S];  // where S=2*D is number stations
// uint bf_mask[S/4];

void launch_s012_station_downsample_kernel(ulong* Sout, const ulong* Sin, const uint8_t* bf_mask,
                                           long T, long Tmin, long Tsize, long F, long S,
                                           cudaStream_t stream)
{
    const int M = T * F * 3;

    if (!Sout || !Sin || !bf_mask)
	throw runtime_error("launch_s012_station_downsample_kernel(): pointer was NULL");
    if (T <= 0)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected T > 0");
    if (Tmin < 0)
	throw runtime_error("launch_s012_station_downsample_kernel(): first sample Tmin must be >= 0");
    if (Tsize <= 0)
	throw runtime_error("launch_s012_station_downsample_kernel(): ringbuffer size Tsize be > 0");
    if (F <= 0)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected F > 0");
    if (S <= 0)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected S > 0");
    if (S & 127)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected S to be a multple of 128");
    if (S > rfi_max_stations)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected S to be <= max_rfi_stations");
    if ((M*S) > INT_MAX)
	throw runtime_error("launch_s012_station_downsample_kernel(): 32-bit overflow");

    uint Wx = (S+1023) / 1024;
    int nblocks = (M+31) / 32;
    int shmem_nbytes = bf_mask_shmem_nbytes(S,Wx);    // asserts that Wx is chosen correctly
    shmem_nbytes += (8*32*Wx);  // 'shmem_red' array has shape (Wx,Wy,32/Wy) and dtype ulong

    if (Wx == 1) // use Wy=4
        s012_station_downsample_kernel<4><<<nblocks, {32 * Wx, 4}, shmem_nbytes, stream>>>(
            Sout, Sin, (const uint*)bf_mask, T, Tmin, Tsize, F, S);
    else if (Wx <= 3) // use Wy=2
        s012_station_downsample_kernel<2><<<nblocks, {32 * Wx, 2}, shmem_nbytes, stream>>>(
            Sout, Sin, (const uint*)bf_mask, T, Tmin, Tsize, F, S);
    else // use Wy=1
        s012_station_downsample_kernel<1><<<nblocks, {32 * Wx, 1}, shmem_nbytes, stream>>>(
            Sout, Sin, (const uint*)bf_mask, T, Tmin, Tsize, F, S);

    CUDA_PEEK("launch s012_station_downsample");
}


// ulong S012_out[M];    // where M is number of "spectator" indices (3*T*F)
// ulong S012_in[M][S];  // where S=2*D is number stations
// uint bf_mask[S/4];

void launch_s012_station_downsample_kernel(ulong *Sout, const ulong *Sin, const uint8_t *bf_mask, long M, long S, cudaStream_t stream)
{
    if (!Sout || !Sin || !bf_mask)
	throw runtime_error("launch_s012_station_downsample_kernel(): pointer was NULL");
    if (M <= 0)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected M > 0");
    if (M % 3 != 0)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected M to be a multiple of 3");
    if (S <= 0)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected S > 0");
    if (S & 127)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected S to be a multple of 128");
    if (S > rfi_max_stations)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected S to be <= max_rfi_stations");
    if ((M*S) > INT_MAX)
	throw runtime_error("launch_s012_station_downsample_kernel(): 32-bit overflow");

    const long T = M / 3;
    const long Tmin = 0;
    const long Tsize = 1 << 30;
    const long F = 1;

    launch_s012_station_downsample_kernel(Sout, Sin, bf_mask, T, Tmin, Tsize, F, S, stream);
}


// ulong S012_out[T][F][3];
// ulong S012_in[T][F][3][S];
// uint8_t bf_mask[S];

void launch_s012_station_downsample_kernel(Array<ulong> &Sout, const Array<ulong> &Sin, const Array<uint8_t> &bf_mask, cudaStream_t stream)
{
    check_array(Sout, "launch_s012_station_downsample_kernel", "Sout", 3, true);          // ndim=3, contiguous=true
    check_array(Sin, "launch_s012_station_downsample_kernel", "Sin", 4, true);            // ndim=4, contiguous=true
    check_array(bf_mask, "launch_s012_station_downsample_kernel", "bf_mask", 1, true);    // ndim=1, contiguous=true

    if (Sin.shape[0] != Sout.shape[0])
	throw runtime_error("launch_s012_station_downsample_kernel(): inconsistent number of time samples in input/output S012 arrays");
    if (Sin.shape[1] != Sout.shape[1])
	throw runtime_error("launch_s012_station_downsample_kernel(): inconsistent number of frequency channels in input/output S012 arrays");
    if (Sout.shape[2] != 3)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected Sout.shape[2] == 3");
    if (Sin.shape[2] != 3)
	throw runtime_error("launch_s012_station_downsample_kernel(): expected Sin.shape[2] == 3");
    if (Sin.shape[3] != bf_mask.shape[0])
	throw runtime_error("launch_s012_station_downsample_kernel(): inconsistent number of stations in S012 arrays and bad feed mask");

    long M = 3 * Sin.shape[0] * Sin.shape[1];
    long S = Sin.shape[3];
    
    launch_s012_station_downsample_kernel(Sout.data, Sin.data, bf_mask.data, M, S, stream);
}


}  // namespace n2k
