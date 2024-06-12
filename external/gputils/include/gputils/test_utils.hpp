#ifndef _GPUTILS_TEST_UTILS_HPP
#define _GPUTILS_TEST_UTILS_HPP

#include <iostream>
#include "Array.hpp"

// is_complex_v<T>, decomplexify_type<T>::type
#include "complex_type_traits.hpp"


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// Very boneheaded function which prints an array.
// Currently one line per array element -- could be improved!
// Instantiated for T = __half, float, double, (u)int, (u)long, (u)short, (u)char, complex<float>, complex<double>.

template<typename T>
extern void
print_array(const Array<T> &arr,
	    const std::vector<std::string> &axis_names = {},
	    std::ostream &os = std::cout);


// Instantiated for T = float, double, (u)int, (u)long, (u)short, (u)char, complex<float>, complex<double>.
// For non floating point types, the 'epsabs' and 'epsrel' arguments are ignored.
//
// Returns the max difference between arrays. The return value is:
//    T    if instantiated for a non-complex type T
//    R    if instantiated for T= std::complex<R>
//
// FIXME to do: implement assert_arrays_equal<__half> ().
// Temporary workaround:
//    (1) convert both arrays to float with Array<__half>:convert_dtype<float>()
//    (2) call assert_arrays_equal(..., epsabs=0.01, epsrel=0.005);

template<typename T>
extern typename gputils::decomplexify_type<T>::type
assert_arrays_equal(const Array<T> &arr1,
		    const Array<T> &arr2,
		    const std::string &name1,
		    const std::string &name2,
		    const std::vector<std::string> &axis_names,
		    float epsabs = 3.0e-5,
		    float epsrel = 1.0e-5,
		    ssize_t max_display = 15,
		    bool verbose = false);


// -------------------------------------------------------------------------------------------------
//
// Helper functions for making random array shapes/strides (intended for unit tests)


// If ndim=0, then number of dimensions will be random.
extern std::vector<ssize_t> make_random_shape(int ndim=0, ssize_t maxaxis=20, ssize_t maxsize=10000);


// If ncontig > 0, then the last 'ncontig' axes are guaranteed contiguous.
// If nalign > 1, then all strides besides the last 'ncontig' are guaranteed multiples of 'nalign'.
extern std::vector<ssize_t> make_random_strides(int ndim, const ssize_t *shape, int ncontig=0, int nalign=1);
extern std::vector<ssize_t> make_random_strides(const std::vector<ssize_t> &shape, int ncontig=0, int nalign=1);


// This specialized function is intended for testing Array<T>::reshape_ref() (see tests/test-array.cu).
extern void make_random_reshape_compatible_shapes(std::vector<ssize_t> &dst_shape,
						  std::vector<ssize_t> &src_shape,
						  std::vector<ssize_t> &src_strides,
						  int maxaxis = 20,
						  ssize_t maxsize = 10000);


}  // namespace test_utils

#endif // _GPUTILS_TEST_UTILS_HPP
