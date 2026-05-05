#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/string_utils.hpp"   // tuple_str()

using namespace std;


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// _isfinite(): like std::isfinite(), but defined for more general types (__half, complex<T>)


// Default _isfinite() for integer types
template<typename T>
inline bool _isfinite(T x)
{
    static_assert(std::is_integral_v<T>);
    return true;
}

inline bool _isfinite(float x) { return std::isfinite(x); }
inline bool _isfinite(double x) { return std::isfinite(x); }

// Note: I tested that __half2float() converts (inf -> inf) as expected.
inline bool _isfinite(__half x) { return std::isfinite(__half2float(x)); }

template<typename T>
inline bool _isfinite(std::complex<T> x)
{
    return _isfinite(x.real()) && _isfinite(x.imag());
}


// -------------------------------------------------------------------------------------------------


// Default l1_norm(), works for most types
template<typename T>
inline double l1_norm(T x)
{
    if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>)
        return x;
    else
        return std::abs(x);
}

template<>
inline double l1_norm(__half x)
{
    return std::abs(double(x));
}

template<typename T>
inline double l1_norm(complex<T> x)
{
    return l1_norm(x.real()) + l1_norm(x.imag());
}


// -------------------------------------------------------------------------------------------------


// Default absdiff(), works for most dtypes.
template<typename T1, typename T2>
inline double default_absdiff(T1 x, T2 y)
{
    // Equivalent to std::abs(x-y), but guaranteed to work if T1 and/or T2 is an unsigned type.
    return (x > y) ? (x-y) : (y-x);
}

template<typename T1, typename T2>
struct absdiff
{
    static inline double eval(T1 x, T2 y) { return default_absdiff(x,y); }
};

// The compiler has trouble with absdiff() calls involving __half.
template<typename T> struct absdiff<T,__half> { static inline double eval(T x, __half y) { return default_absdiff<double,double>(x,y); } };
template<typename T> struct absdiff<__half,T> { static inline double eval( __half x, T y) { return default_absdiff<double,double>(x,y); } };
template<> struct absdiff<__half,__half> { static inline double eval( __half x, __half y) { return default_absdiff<__half,__half>(x,y); } };

// Version of absdiff() that works if both T1 and T2 are complex.
template<typename T1, typename T2>
struct absdiff<complex<T1>, complex<T2>>
{
    static inline double eval(complex<T1> x, complex<T2> y)
    {
        using C = absdiff<T1,T2>;
        return C::eval(x.real(), y.real()) + C::eval(x.imag(), y.imag());
    }
};


// -------------------------------------------------------------------------------------------------


template<typename T1, typename T2>
static __attribute__((noinline))
double _assert_arrays_equal(
    const Array<void> &arr1_,
    const Array<void> &arr2_,
    const string &name1,
    const string &name2,
    const vector<string> &axis_names,
    double epsabs,
    double epsrel,
    long max_display,
    bool verbose)
{
    Array<T1> arr1 = arr1_.cast<T1> ("assert_arrays_equal");
    Array<T2> arr2 = arr2_.cast<T2> ("assert_arrays_equal");
    
    int nfail = 0;
    double maxdiff = 0;

    for (auto ix = arr1.ix_start(); arr1.ix_valid(ix); arr1.ix_next(ix)) {
        T1 x = arr1.at(ix);
        T2 y = arr2.at(ix);
        
        double delta = absdiff<T1,T2>::eval(x,y);
        double thresh = epsabs + epsrel * (l1_norm(x) + l1_norm(y));

        maxdiff = max(maxdiff, delta);
        bool failed = (delta > thresh) || !_isfinite(x) || !_isfinite(y);
        
        if (!failed && !verbose)
            continue;

        if (failed && (nfail == 0))
            cout << "\nassert_arrays_equal() failed [shape=" << arr1.shape_str() << "]\n";

        if (failed)
            nfail++;
        
        if (nfail > max_display)
            continue;
        
        cout << "   ";
        for (int d = 0; d < arr1.ndim; d++)
            cout << " " << axis_names[d] << "=" << ix[d];

        cout << ": " << name1 << "=" << x << ", " << name2
             << "=" << y << "  [delta=" << delta << ", thresh=" << thresh << "]";

        if (failed)
            cout << " FAILED";

        cout << "\n";
    }

    if ((nfail > max_display) && (max_display > 0))
        cout << "        (+ " << (nfail-max_display) << " more failures)\n";
    
    cout.flush();
    
    if (nfail > 0)
        throw runtime_error("Arrays were not equal as expected");
    
    return maxdiff;    
}


// -------------------------------------------------------------------------------------------------
//
// Instantiate _assert_arrays_equal() for:
//  - all pairs (F1,F2) of floating-point types
//  - all pairs (complex<F1>, complex<F2>)
//  - all pairs (T,T).
//
// (This is the same scope as array_convert().)


// Same signature as _assert_arrays_equal<T1,T2> ()
using assert_func = double (*)(const Array<void> &, const Array<void> &,
                               const string &, const string &, const vector<string> &,
                               double, double, long, bool);

// Helper for get_assert_func().
// Matches <T1,T2> or <complex<T1>,complex<T2>>
template<typename T1, typename T2>
static assert_func get_doubly_templated_assert_func(Dtype dt1, Dtype dt2)
{
    if ((dt1 == Dtype::native<T1>()) && (dt2 == Dtype::native<T2>()))
        return _assert_arrays_equal<T1, T2>;
    
    if ((dt1 == Dtype::native<complex<T1>>()) && (dt2 == Dtype::native<complex<T2>>()))
        return _assert_arrays_equal<complex<T1>, complex<T2>>;

    return nullptr;
}

// Helper for get_assert_func().
// Matches (Tdst,Fsrc) or (complex<Tdst>, complex<Fsrc>), where Fsrc is any floating-point type.
template<typename T1>
static assert_func get_singly_templated_assert_func(Dtype dt1, Dtype dt2)
{
    Dtype dt = dt2.real();

    if (dt == Dtype::native<__half>())
        return get_doubly_templated_assert_func<T1, __half> (dt1, dt2);
    
    if (dt == Dtype::native<float>())
        return get_doubly_templated_assert_func<T1, float> (dt1, dt2);
    
    if (dt == Dtype::native<double>())
        return get_doubly_templated_assert_func<T1, double> (dt1, dt2);

    return nullptr;
}

// Matches (T,T), (F1,F2), or (complex<F1>, complex<F2>),
// where T denotes an arbitrary type, and F1/F2 denote floating-point types.
static assert_func get_assert_func(Dtype dt1, Dtype dt2)
{
    Dtype dt = dt1.real();

    if (dt == Dtype::native<__half>())
        return get_singly_templated_assert_func<__half> (dt1, dt2);
    
    if (dt == Dtype::native<float>())
        return get_singly_templated_assert_func<float> (dt1, dt2);
    
    if (dt == Dtype::native<double>())
        return get_singly_templated_assert_func<double> (dt1, dt2);

    if (dt == Dtype::native<int>())
        return get_doubly_templated_assert_func<int,int> (dt1, dt2);

    if (dt == Dtype::native<uint>())
        return get_doubly_templated_assert_func<uint,uint> (dt1, dt2);

    if (dt == Dtype::native<long>())
        return get_doubly_templated_assert_func<long,long> (dt1, dt2);

    if (dt == Dtype::native<ulong>())
        return get_doubly_templated_assert_func<ulong,ulong> (dt1, dt2);

    if (dt == Dtype::native<short>())
        return get_doubly_templated_assert_func<short,short> (dt1, dt2);

    if (dt == Dtype::native<ushort>())
        return get_doubly_templated_assert_func<ushort,ushort> (dt1, dt2);

    if (dt == Dtype::native<char>())
        return get_doubly_templated_assert_func<char,char> (dt1, dt2);

    if (dt == Dtype::native<unsigned char>())
        return get_doubly_templated_assert_func<unsigned char, unsigned char> (dt1, dt2);

    return nullptr;
}


// -------------------------------------------------------------------------------------------------


double assert_arrays_equal(
    const Array<void> &arr1,
    const Array<void> &arr2,
    const string &name1,
    const string &name2,
    const vector<string> &axis_names,
    double epsabs,
    double epsrel,
    long max_display,
    bool verbose)
{
    if (!arr1.shape_equals(arr2)) {
        stringstream ss;
        ss << "assert_arrays_equal(): arrays have different shapes: " 
            << name1 << ".shape=" << arr1.shape_str() << " and "
            << name2 << ".shape=" << arr2.shape_str();
        throw runtime_error(ss.str());
    }

    if (axis_names.size() != uint(arr1.ndim)) {
        stringstream ss;
        ss << "assert_arrays_equal(" << name1 << "," << name2 << "): axis_names=" << tuple_str(axis_names) 
           << " must have the same length as the array shape " << arr1.shape_str();
        throw runtime_error(ss.str());
    }

    xassert(max_display > 0);

    assert_func afunc = get_assert_func(arr1.dtype, arr2.dtype);
    
    if (!afunc) {
        stringstream ss;
        ss << "assert_arrays_equal() is not implemented for this dtype pair: (" << arr1.dtype << ", " << arr2.dtype << ")";
        throw runtime_error(ss.str());
    }
    
    double eps = 10.0 * max(arr1.dtype.precision(), arr2.dtype.precision());
    epsabs = (epsabs >= 0) ? epsabs : eps;
    epsrel = (epsrel >= 0) ? epsrel : eps;

    Array<void> harr1 = arr1.to_host(false);  // page_locked=false
    Array<void> harr2 = arr2.to_host(false);  // page_locked=false

    return afunc(harr1, harr2, name1, name2, axis_names, epsabs, epsrel, max_display, verbose);
}


}  // namespace ksgpu

