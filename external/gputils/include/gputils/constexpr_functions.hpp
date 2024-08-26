#ifndef _GPUTILS_CONSTEXPR_FUNCTIONS_HPP
#define _GPUTILS_CONSTEXPR_FUNCTIONS_HPP

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// Frequently used in static_assert().
constexpr __host__ __device__ bool constexpr_is_divisible(int m, int n)
{
    return (m >= 0) && (n > 0) && ((m % n) == 0);
}

// I wanted the syntax: constexpr_idiv(m,n)
// The best I could do was: constexpr_idiv<m,n>:::value
template<int m, int n>
struct constexpr_idiv
{
    static_assert(constexpr_is_divisible(m,n));
    static constexpr int value = m/n;
};

// Frequently used in static_assert().
constexpr __host__ __device__ bool constexpr_is_pow2(int n)
{
    return (n >= 1) && ((n & (n-1)) == 0);
}


constexpr __host__ __device__ int constexpr_ilog2(int n)
{
    // static_assert() is not allowed in constexpr-functions, so
    // caller should call static_assert(constexpr_is_pow2(n));
    
    return (n > 1) ? (constexpr_ilog2(n/2)+1) : 0;
}


template<typename T>
constexpr __host__ __device__ T constexpr_min(T m, T n)
{
    return (m < n) ? m : n;
}

template<typename T>
constexpr __host__ __device__ T constexpr_max(T m, T n)
{
    return (m > n) ? m : n;
}


} // namespace gputils

#endif // _GPUTILS_CONSTEXPR_FUNCTIONS_HPP
