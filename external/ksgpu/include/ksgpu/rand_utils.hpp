#ifndef _KSGPU_RAND_UTILS_HPP
#define _KSGPU_RAND_UTILS_HPP

#include <cstdint>
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


// default_rng(): returns a reference to the calling thread's default std::mt19937.
//
// The RNG is constructed lazily on first call from each thread, seeded from
// std::random_device (via a std::seed_seq of 8 draws, for 256 bits of seed
// entropy; see comments in src_lib/rand_utils.cpp). Each thread has its own
// independent RNG, so it is safe to call rand_int(), rand_uniform(), etc.
// concurrently from multiple threads.
//
// Use seed_default_rng() to override the auto-seed for the calling thread
// (e.g. to make a test driver reproducible).
extern std::mt19937 &default_rng();

// seed_default_rng(): reseed the calling thread's default RNG.
//
// Only affects the calling thread. Other threads keep their own seeds. This
// is intended for test drivers that want reproducible random sequences --
// call once at the top of main() with a fixed seed.
extern void seed_default_rng(uint32_t seed);


// -------------------------------------------------------------------------------------------------


// _randomize(): randomizes a buffer whose dtype is specified at runtime.
extern void _randomize(Dtype dtype, void *buf, long nelts, std::mt19937 &rng = default_rng());


// randomize(): randomize a buffer whose type T is specified at compile time.
template<typename T>
inline void randomize(T *buf, long nelts, std::mt19937 &rng = default_rng())
{
    static_assert(!std::is_void_v<T>, "randomize(): if T=void, then you must call _randomize() instead");
    _randomize(Dtype::native<T>(), buf, nelts, rng);
}


// -------------------------------------------------------------------------------------------------


inline long rand_int(long lo, long hi, std::mt19937 &rng = default_rng())
{
    xassert(lo < hi);
    return std::uniform_int_distribution<long>(lo,hi-1)(rng);   // note hi-1 here!
}

inline double rand_uniform(double lo=0.0, double hi=1.0, std::mt19937 &rng = default_rng())
{
    xassert(lo < hi);
    return std::uniform_real_distribution<double>(lo,hi) (rng);
}

inline bool rand_bool(double p=0.5, std::mt19937 &rng = default_rng())
{
    return std::uniform_real_distribution<double>(0.0,1.0)(rng) < p;
}


inline std::vector<long> rand_int_vec(long size, long lo, long hi, std::mt19937 &rng = default_rng())
{
    xassert(size >= 0);
    xassert(lo < hi);
    auto dist = std::uniform_int_distribution<long>(lo,hi-1);   // note hi-1 here!

    std::vector<long> ret(size);
    for (long i = 0; i < size; i++)
        ret[i] = dist(rng);

    return ret;
}

inline std::vector<double> rand_uniform_vec(long size, double lo=0.0, double hi=1.0, std::mt19937 &rng = default_rng())
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
inline void randomly_permute(std::vector<T> &v, std::mt19937 &rng = default_rng())
{
    for (ulong i = 1; i < v.size(); i++) {
        long j = rand_int(0, i+1, rng);
        std::swap(v[i], v[j]);
    }
}


// Returns a random permutation of {0,1,...,(n-1)}
inline std::vector<long> rand_permutation(long nelts, std::mt19937 &rng = default_rng())
{
    xassert(nelts >= 0);

    std::vector<long> v(nelts);
    for (long i = 0; i < nelts; i++)
        v[i] = i;

    randomly_permute(v, rng);
    return v;
}


template<typename T>
inline T rand_element(const std::vector<T> &v, std::mt19937 &rng = default_rng())
{
    xassert(v.size() > 0);
    long ix = rand_int(0, v.size(), rng);
    return v[ix];
}

template<typename T>
inline T rand_element(const std::initializer_list<T> v, std::mt19937 &rng = default_rng())
{
    xassert(v.size() > 0);
    int ix = rand_int(0, v.size(), rng);
    return std::data(v)[ix];
}


// Intended as a helper for random_integers_with_bounded_product(), but may be independently useful.
extern std::vector<double> random_doubles_with_fixed_sum(int nelts, double sum, std::mt19937 &rng = default_rng());

// Useful in unit tests, when generating randomly-sized arrays.
extern std::vector<long> random_integers_with_bounded_product(int nelts, long bound, std::mt19937 &rng = default_rng());

// Generate a random hex string of the specified length.
extern std::string make_random_hex_string(int len);


} // namespace ksgpu

#endif // _KSGPU_RAND_UTILS_HPP
