#ifndef _KSGPU_RAND_UTILS_HPP
#define _KSGPU_RAND_UTILS_HPP

#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "Dtype.hpp"
#include "xassert.hpp"

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif

extern std::mt19937 default_rng;


// -------------------------------------------------------------------------------------------------


// _randomize(): randomizes a buffer whose dtype is specified at runtime.
extern void _randomize(Dtype dtype, void *buf, long nelts, std::mt19937 &rng = default_rng);


// randomize(): randomize a buffer whose type T is specified at compile time.
template<typename T>
inline void randomize(T *buf, long nelts, std::mt19937 &rng = default_rng)
{
    static_assert(!std::is_void_v<T>, "randomize(): if T=void, then you must call _randomize() instead");
    _randomize(Dtype::native<T>(), buf, nelts, rng);
}


// -------------------------------------------------------------------------------------------------


inline long rand_int(long lo, long hi, std::mt19937 &rng = default_rng)
{
    xassert(lo < hi);
    return std::uniform_int_distribution<long>(lo,hi-1)(rng);   // note hi-1 here!
}

inline double rand_uniform(double lo=0.0, double hi=1.0, std::mt19937 &rng = default_rng)
{
    xassert(lo < hi);
    return std::uniform_real_distribution<double>(lo,hi) (rng);
}

inline bool rand_bool(double p=0.5, std::mt19937 &rng = default_rng)
{
    return std::uniform_real_distribution<double>(0.0,1.0)(rng) < p;
}


inline std::vector<long> rand_int_vec(long size, long lo, long hi, std::mt19937 &rng = default_rng)
{
    xassert(size >= 0);
    xassert(lo < hi);
    auto dist = std::uniform_int_distribution<long>(lo,hi-1);   // note hi-1 here!

    std::vector<long> ret(size);
    for (long i = 0; i < size; i++)
        ret[i] = dist(rng);

    return ret;
}

inline std::vector<double> rand_uniform_vec(long size, double lo=0.0, double hi=1.0, std::mt19937 &rng = default_rng)
{
    xassert(size >= 0);
    xassert(lo <= hi);
    auto dist = std::uniform_real_distribution<double>(lo,hi);

    std::vector<double> ret(size);
    for (long i = 0; i < size; i++)
        ret[i] = dist(rng);

    return ret;
}

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


// Intended as a helper for random_integers_with_bounded_product(), but may be independently useful.
extern std::vector<double> random_doubles_with_fixed_sum(int nelts, double sum);

// Useful in unit tests, when generating randomly-sized arrays.
extern std::vector<long> random_integers_with_bounded_product(int nelts, long bound);

// Generate a random hex string of the specified length.
extern std::string make_random_hex_string(int len);

    
} // namespace ksgpu

#endif // _KSGPU_RAND_UTILS_HPP
