#include "../include/ksgpu/memcpy_kernels.hpp"
#include "../include/ksgpu/xassert.hpp"

#include <iostream>
#include <stdexcept>


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


__global__ void memcpy_kernel(int *dst, const int *src, long nelts, int nelts_per_block)
{
    long n0 = long(blockIdx.x) * long(nelts_per_block);
    dst += n0;
    src += n0;

    long nn = nelts - n0;
    if (nelts_per_block > nn)
	nelts_per_block = nn;
	
    for (int i = threadIdx.x; i < nelts_per_block; i += blockDim.x)
	dst[i] = src[i];
}


void launch_memcpy_kernel(void *dst, const void *src, long nbytes, cudaStream_t stream)
{
    static constexpr long nthreads_per_block = 128;
    static constexpr long nelts_per_block = 128;
    static constexpr long max_nblocks = (1L << 31) - 1;  // cuda limit
    static constexpr long max_nbytes = 256L * 1024L * 1024L * 1024L;

    static_assert(max_nbytes <= max_nblocks * nelts_per_block * 4);
    
    xassert(dst != nullptr);
    xassert(src != nullptr);
    xassert(nbytes >= 0);
    xassert(nbytes <= max_nbytes);
    xassert((nbytes % 128) == 0);

    if (nbytes == 0)
	return;

    long nelts = nbytes >> 2;
    long nblocks = (nelts + nelts_per_block - 1) / nelts_per_block;

    // Should never fail
    xassert((nblocks > 0) && (nblocks <= max_nblocks));
    
    memcpy_kernel<<< nblocks, nthreads_per_block, 0, stream >>>
	(reinterpret_cast<int *> (dst),
	 reinterpret_cast<const int *> (src),
	 nelts, nelts_per_block);
}


// -------------------------------------------------------------------------------------------------


__global__ void memcpy_2d_kernel(int *dst, long dp_nelts, const int *src, long sp_nelts, long nrows, long ncols, long nrows_per_block)
{
    long r0 = long(blockIdx.y) * nrows_per_block;
    long c0 = long(blockIdx.x) * long(blockDim.x) + threadIdx.x;

    if (c0 >= ncols)
	return;
	
    dst += (dp_nelts * r0) + c0;
    src += (sp_nelts * r0) + c0;

    nrows_per_block = min(nrows_per_block, nrows-r0);
    
    for (long r = threadIdx.y; r < nrows_per_block; r += blockDim.y)
	dst[r * dp_nelts] = src[r * sp_nelts];
}


void launch_memcpy_2d_kernel(void *dst, long dpitch, const void *src, long spitch, long width, long height, cudaStream_t stream)
{
    static constexpr long max_nblocks_x = (1L << 31) - 1;  // cuda limit
    static constexpr long max_nblocks_y = 65535;           // cuda limit

    xassert(dst != nullptr);
    xassert(src != nullptr);
    xassert(width >= 0);
    xassert(height >= 0);

    // Note: could be generalized to allow negative pitches.
    xassert(dpitch >= width);
    xassert(spitch >= width);
    xassert((width % 128) == 0);
    xassert((dpitch % 128) == 0);
    xassert((spitch % 128) == 0);

    if ((width == 0) || (height == 0))
	return;

    long nrows = height;
    long ncols = width >> 2;
    long dp_nelts = dpitch >> 2;
    long sp_nelts = spitch >> 2;
    
    dim3 nthreads;
    nthreads.z = 1;

    if (ncols > 128*nrows) {
	nthreads.x = 128; // cols
	nthreads.y = 1;   // rows
    }
    else if (ncols < 32*nrows) {
	nthreads.x = 32;  // cols
	nthreads.y = 4;   // rows
    }
    else {
	nthreads.x = 64;  // cols
	nthreads.y = 2;   // rows
    }

    long nr0 = nthreads.y * max_nblocks_y;
    long n = (nrows + nr0 - 1) / nr0;
    
    long nrows_per_block = nthreads.y * n;
    long nblocks_y = (nrows + nrows_per_block - 1) / nrows_per_block;  // rows
    long nblocks_x = (ncols + nthreads.x - 1) / nthreads.x;            // cols

    // FIXME add more checks up-front, to ensure that these never fail
    xassert((nblocks_x > 0) && (nblocks_x <= max_nblocks_x));
    xassert((nblocks_y > 0) && (nblocks_y <= max_nblocks_y));

    dim3 nblocks;
    nblocks.x = nblocks_x;
    nblocks.y = nblocks_y;
    nblocks.z = 1;
    
    memcpy_2d_kernel<<< nblocks, nthreads, 0, stream >>>
	(reinterpret_cast<int *> (dst), dp_nelts,
	 reinterpret_cast<const int *> (src), sp_nelts,
	 nrows, ncols, nrows_per_block);
}


} // namespace ksgpu
