#ifndef _GPUTILS_COMPLEX_TYPE_TRAITS_HPP
#define _GPUTILS_COMPLEX_TYPE_TRAITS_HPP

#include <complex>
#include <type_traits>

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// is_complex_v<T>  -> true   if T is complex
//                     false  otherwise

template<typename T>
struct is_complex : public std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

template<typename T>
static constexpr bool is_complex_v = is_complex<T>::value;


// -------------------------------------------------------------------------------------------------
//
// decomplexify_type<T>::type  -> R   if T=complex<R>
//                                T   otherwise

template<typename T>
struct decomplexify_type
{
    using type = T;
};

template<typename T>
struct decomplexify_type<std::complex<T>>
{
    using type = T;
};


} // namespace gputils

#endif  // _GPUTILS_COMPLEX_TYPE_TRAITS_HPP
