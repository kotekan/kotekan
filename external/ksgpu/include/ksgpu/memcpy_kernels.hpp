#ifndef _KSGPU_MEMCPY_KERNELS_HPP
#define _KSGPU_MEMCPY_KERNELS_HPP

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// Drop-in replacement for cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice),
// but uses SMs instead of a copy engine.

extern void launch_memcpy_kernel(
    void *dst,        // pointer to GPU global memory
    const void *src,  // pointer to GPU global memory
    long nbytes,      // must be a multiple of 128
    cudaStream_t stream = nullptr
);


// Drop-in replacement for cudaMemcpy2DAsync(..., cudaMemcpyDeviceToDevice),
// but uses SMs instead of a copy engine.

extern void launch_memcpy_2d_kernel(
    void *dst,        // pointer to GPU global memory
    long dpitch,      // memory offset in bytes between rows in destination array, must be multiple of 128
    const void *src,  // pointer to GPU global memory
    long spitch,      // memory offset in bytes between rows in source array, must be multiple of 128
    long width,       // size of each row in bytes (where a "row" is memory-contiguous), must be multiple of 128
    long height,      // number of rows
    cudaStream_t stream = nullptr
);


// Note: both memcpy kernels use 4-warp threadblocks with no shared memory and few registers.
// This should ensure that the memcpy kernels can run in parallel with other kernels.
//
// Timings (bin/time-memcpy-kernels.cu) show that both memcpy kernels have good performance
// (500-600 GB/s on an A40) in a variety of situations, including very "lopsided" 2-d memcopies.
// This could probably be improved further using 128-bit load/store instructions.


}  // namespace ksgpu

#endif  // _KSGPU_MEMCPY_KERNELS_HPP
