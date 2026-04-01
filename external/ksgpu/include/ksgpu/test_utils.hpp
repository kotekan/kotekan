#ifndef _KSGPU_TEST_UTILS_HPP
#define _KSGPU_TEST_UTILS_HPP

#include "Array.hpp"


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Helper functions for making random array shapes/strides (intended for unit tests)


// If ndim=0, then number of dimensions will be random.
extern std::vector<long> make_random_shape(int ndim=0, long maxaxis=20, long maxsize=10000);


// If ncontig > 0, then the last 'ncontig' axes are guaranteed contiguous.
// If nalign > 1, then all strides besides the last 'ncontig' are guaranteed multiples of 'nalign'.
extern std::vector<long> make_random_strides(int ndim, const long *shape, int ncontig=0, long nalign=1);
extern std::vector<long> make_random_strides(const std::vector<long> &shape, int ncontig=0, long nalign=1);


extern std::vector<long> make_contiguous_strides(int ndim, const long *shape);
extern std::vector<long> make_contiguous_strides(const std::vector<long> &shape);


// This specialized function is intended for testing Array<T>::reshape() (see tests/test-array.cu).
extern void make_random_reshape_compatible_shapes(
    std::vector<long> &dst_shape,
    std::vector<long> &src_shape,
    std::vector<long> &src_strides
);



// -------------------------------------------------------------------------------------------------


// Launches a "busy wait" kernel with one threadblock and 32 threads.
// Useful for testing stream synchronization.
//
// The 'arr' argument is a caller-allocated length-32 array.
// The 'a40_seconds' arg determines the amount of work done by the kernel,
// normalized to "seconds on an NVIDIA A40".

extern void launch_busy_wait_kernel(Array<uint> &arr, double a40_seconds, cudaStream_t s);


}  // namespace ksgpu

#endif // _KSGPU_TEST_UTILS_HPP
