#ifndef _KSGPU_RAND_UTILS_HPP
#define _KSGPU_RAND_UTILS_HPP

#include <vector>
#include <random>
#include <complex>
#include <type_traits>
#include <cuda_fp16.h>

#include "xassert.hpp"
#include "complex_type_traits.hpp"  // is_complex_v<T>, decomplexify_type<T>::type

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif

extern std::mt19937 default_rng;


// -------------------------------------------------------------------------------------------------


inline long rand_int(long lo, long hi, std::mt19937 &rng = default_rng)
{
    xassert(lo < hi);
    return std::uniform_int_distribution<long>(lo,hi-1)(rng);   // note hi-1 here!
}


inline float rand_uniform(float lo=0.0, float hi=1.0, std::mt19937 &rng = default_rng)
{
    return std::uniform_real_distribution<float>(lo,hi) (rng);
}


// -------------------------------------------------------------------------------------------------


// Version of randomize() for floating-point types
template<typename T>
inline void randomize_f(T *buf, long nelts, std::mt19937 &rng = default_rng)
{
    static_assert(std::is_floating_point_v<T>);

    auto dist = std::uniform_real_distribution<T>(-1.0, 1.0);    
    for (long i = 0; i < nelts; i++)
	buf[i] = dist(rng);
}


// Version of randomize() for integral types
template<typename T>
inline void randomize_i(T *buf, long nelts, std::mt19937 &rng = default_rng)
{
    static_assert(std::is_integral_v<T>);

    long nbytes = nelts * sizeof(T);
    long nints = nbytes / sizeof(int);
    
    for (long i = 0; i < nints; i++)
	((int *)buf)[i] = rng();
    for (long i = nints * sizeof(int); i < nbytes; i++)
	((char *)buf)[i] = rng();
}


// General randomize() template, for built-in C++ int/float types.
// For CUDA __half and __half2, we need specializations (see below).
template<typename T>
inline void randomize(T *buf, long nelts, std::mt19937 &rng = default_rng)
{
    xassert(nelts >= 0);

    if constexpr (ksgpu::is_complex_v<T>) {
	using Tr = typename ksgpu::decomplexify_type<T>::type;
	randomize<Tr> (reinterpret_cast<Tr*> (buf), 2*nelts, rng);
    }
    else if constexpr (std::is_floating_point_v<T>)
	randomize_f(buf, nelts, rng);
    else {
	static_assert(std::is_integral_v<T>, "randomize() array must be either integral, floating-point, or complex type");
	randomize_i(buf, nelts, rng);
    }
}


// __half randomize() template specialization.
template<>
inline void randomize(__half *buf, long nelts, std::mt19937 &rng)
{
    xassert(nelts >= 0);    
    auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    
    for (long i = 0; i < nelts; i++)
	buf[i] = __float2half_rn(dist(rng));
}


// __half2 randomize() template specialization.
template<>
inline void randomize(__half2 *buf, long nelts, std::mt19937 &rng)
{
    xassert(nelts >= 0);
    auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    
    for (long i = 0; i < nelts; i++) {
	float x = dist(rng);
	float y = dist(rng);
	buf[i] = __floats2half2_rn(x,y);
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T>
inline void randomly_permute(std::vector<T> &v, std::mt19937 &rng = default_rng)
{
    for (ulong i = 1; i < v.size(); i++) {
	long j = rand_int(0, i+1, rng);
	std::swap(v[i], v[j]);
    }
}


// Returns a random permutation of {0,1,...,(n-1)}
inline std::vector<long> rand_permutation(long nelts, std::mt19937 &rng = default_rng)
{
    xassert(nelts >= 0);
    
    std::vector<long> v(nelts);
    for (long i = 0; i < nelts; i++)
	v[i] = i;

    randomly_permute(v, rng);
    return v;
}


template<typename T>
inline T rand_element(const std::vector<T> &v, std::mt19937 &rng = default_rng)
{
    xassert(v.size() > 0);
    long ix = rand_int(0, v.size(), rng);
    return v[ix];
}

template<typename T>
inline T rand_element(const std::initializer_list<T> v, std::mt19937 &rng = default_rng)
{
    xassert(v.size() > 0);
    int ix = rand_int(0, v.size(), rng);
    return std::data(v)[ix];
}


extern std::vector<double> random_doubles_with_fixed_sum(int nelts, double sum);

// Useful in unit tests, when generating randomly-sized arrays.
extern std::vector<long> random_integers_with_bounded_product(int nelts, long bound);

    
} // namespace ksgpu

#endif // _KSGPU_RAND_UTILS_HPP
