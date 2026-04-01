#ifndef _KSGPU_DTYPE_HPP
#define _KSGPU_DTYPE_HPP

#include <string>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <cuda_fp16.h>  // __half


// Define operator<<() for CUDA __half.
// Note: this has nothing to do with 'class Dtype', but I couldn't find a better place to put it!
inline std::ostream &operator<<(std::ostream &os, __half x)
{
    os << __half2float(x);
    return os;
}


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// Flags for use in Dtype::flags.
// Precisely one of the first 3 flags must be specified.
// In particular, floating-point complex is (df_complex | df_float), not (df_complex).
// (The flag combinations (df_complex | df_int) or (df_complex | df_uint) are also valid.)
static constexpr unsigned short df_int = 0x1;
static constexpr unsigned short df_uint = 0x2;
static constexpr unsigned short df_float = 0x4;
static constexpr unsigned short df_complex = 0x8;


struct Dtype
{
    // Note: we don't currently support simd types (e.g. cuda __half2).
    unsigned short flags = 0;
    unsigned short nbits = 0;   // for complex types, includes factor 2
    
    // This constructor does no error-checking.
    Dtype() { }
    
    // This constructor throws an exception if arguments are invalid.
    // Example usage:
    //
    //    Dtype dt(df_int, 32)     // int32
    //    Dtype dt(df_uint, 16)    // uint16
    //    Dtype dt(df_float, 64)   // float64
    //    Dtype dt(df_complex | df_float, 64)    // float32+32
    
    Dtype(unsigned short flags, unsigned short nbits);

    // Returns Dtype object corresponding to given native C++ type.
    // Can be used with operator==() to verify that dtype matches a native C++ type.
    template<typename T> static inline Dtype native();

    inline bool is_valid() const;
    inline bool is_empty() const { return (flags == 0) && (nbits == 0); }
    
    inline bool operator==(const Dtype &x) const { return (flags == x.flags) && (nbits == x.nbits); }
    inline bool operator!=(const Dtype &x) const { return (flags != x.flags) || (nbits != x.nbits); }
    
    std::string str() const;

    // (float64, float32, float16) -> (1.0e-15, 1.0e-6, 1.0e-3).
    // (all integer types) -> 0
    double precision() const;

    Dtype real() const;     // makes real type from complex type (no-ops if type is already real)
    Dtype complex() const;  // makes complex type from real type (no-ops if type is already complex)

    // If 'throw_exception_on_failure' is false, then an empty Dtype is returned on parse failure.
    static Dtype from_str(const std::string &s, bool throw_exception_on_failure=true);
};


extern std::ostream &operator<<(std::ostream &os, Dtype dt);


inline void _check_dtype_valid(Dtype dtype, const char *where)
{
    if (!dtype.is_valid()) {
        std::stringstream ss;
        ss << where << ": got " << dtype;
        throw std::runtime_error(ss.str());
    }
}

// Checks for consistency between 'dtype' and native C++ type T.
// (If T==void, then checks validity of 'dtype'.)
template<typename T>
inline void _check_dtype(Dtype dtype, const char *where)
{
    if constexpr (!std::is_void_v<T>) {
        Dtype dt_expected = Dtype::native<T>();
        if (dtype != dt_expected) {
            std::stringstream ss;
            ss << where << ": got dtype " << dtype << ", expected dtype " << dt_expected;
            throw std::runtime_error(ss.str());
        }
    }
    else  // T==void
        _check_dtype_valid(dtype, where);
}


// -------------------------------------------------------------------------------------------------


inline Dtype::Dtype(unsigned short flags_, unsigned short nbits_) :
    flags(flags_), nbits(nbits_)
{
    if (!is_valid())
        throw std::runtime_error("Dtype constructor: " + str());
}

inline bool Dtype::is_valid() const
{
    unsigned short f = flags & ~df_complex;

    if ((f != df_int) && (f != df_uint) && (f != df_float))
        return false;
    if (nbits == 0)
        return false;
    if ((flags & df_complex) && (nbits & 1))
        return false;

    return true;
}


// -------------------------------------------------------------------------------------------------


// Helper class for Dtype::native<T>().
template<typename T, class Enable = void>
struct _native_dtype_helper
{
    static constexpr unsigned short flags = 0;
};

template<typename T>
struct _native_dtype_helper<T, typename std::enable_if_t<std::is_integral_v<T>>>
{
    static constexpr unsigned short flags = std::is_unsigned_v<T> ? df_uint : df_int;
};

template<typename T>
struct _native_dtype_helper<T, typename std::enable_if_t<std::is_floating_point_v<T>>>
{
    static constexpr unsigned short flags = df_float;
};

template<>
struct _native_dtype_helper<__half, void>
{
    static constexpr unsigned short flags = df_float;
};
    

template<typename T>
struct _native_dtype_helper<std::complex<T>, void>
{
    static constexpr unsigned short f = _native_dtype_helper<T>::flags;
    static constexpr bool valid = (f != 0) && ((f & df_complex) == 0);
    static constexpr unsigned short flags = valid ? (f | df_complex) : 0;
};

// Static member function
template<typename T>
inline Dtype Dtype::native()
{
    using C = _native_dtype_helper<T>;
    static_assert(C::flags, "can't construct ksgpu::Dtype object from specified C++ datatype");
    
    Dtype ret;
    ret.flags = C::flags;
    ret.nbits = 8 * sizeof(T);
    return ret;
}


} // namespace ksgpu

#endif  // _KSGPU_DTYPE_HPP
