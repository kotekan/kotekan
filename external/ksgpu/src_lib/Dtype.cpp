#include "../include/ksgpu/Dtype.hpp"
#include <sstream>

using namespace std;


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// Caller will probably want to wrap output with parentheses.
static string dflag_str(unsigned short f)
{
    const char *sep = "";
    static const char *bar = " | ";
    stringstream ss;
    
    if (!f) {
        ss << "0";
    }
    if (f & df_int) {
        ss << "df_int"; sep = bar;
    }
    if (f & df_uint) {
        ss << sep << "df_uint"; sep = bar;
    }
    if (f & df_float) {
        ss << sep << "df_float"; sep = bar;
    }
    if (f & df_complex) {
        ss << sep << "df_complex"; sep = bar;
    }

    constexpr unsigned short df_all = df_int | df_uint | df_float | df_complex;
    f &= ~df_all;

    if (f)
        ss << sep << "invalid flags 0x" << hex << f;
    return ss.str();
}


// Pretty-print a dtype.
ostream &operator<<(ostream &os, Dtype dt)
{
    if (dt.is_empty()) {
        os << "empty_dtype";
        return os;
    }
    
    if (!dt.is_valid()) {
        os << "invalid_dtype(flags = (" << dflag_str(dt.flags) << "), nbits = " << dt.nbits;
        if (dt.flags == df_complex)
            os << ", maybe df_complex | df_float was intended?";
        os << ")";
        return os;
    }

    if (dt.flags & df_complex)
        os << ((dt.flags & df_float) ? "complex" : "complex_");
    else if (dt.flags & df_float)
        os << "float";

    if (dt.flags & df_int)
        os << "int";
    else if (dt.flags & df_uint)
        os << "uint";

    if (dt.flags & df_complex)
        os << (dt.nbits/2) << "+" << (dt.nbits/2);
    else
        os << dt.nbits;

    return os;
}


string Dtype::str() const
{
    stringstream ss;
    ss << (*this);
    return ss.str();
}


double Dtype::precision() const
{
    ushort f = (flags & ~df_complex);
    int c = (flags & df_complex) ? 2 : 1;

    if (f == df_float) {
        if (nbits == 64*c) return 1.0e-15;
        if (nbits == 32*c) return 1.0e-6;
        if (nbits == 16*c) return 1.0e-3;
    }
        
    else if ((f == df_int) || (f == df_uint))
        return 0;

    stringstream ss;
    ss << "Dtype::precision(): invalid dtype: " << (*this);
    throw runtime_error(ss.str());
}


Dtype Dtype::real() const
{
    if ((flags & df_complex) == 0)
        return *this;

    if ((nbits & 1) == 0)
        return Dtype(flags & ~df_complex, nbits >> 1);

    stringstream ss;
    ss << "Dtype::real(): invalid dtype: " << (*this);
    throw runtime_error(ss.str());
}


Dtype Dtype::complex() const
{
    if (flags & df_complex)
        return *this;
    else
        return Dtype(flags | df_complex, nbits << 1);
}


// Helper for Dtype::from_str()
static long _scan_digits(const char *s)
{
    long n = 0;
    while (s[n] && isdigit(s[n]))
        n++;
    return n;
}

// Static member function
Dtype Dtype::from_str(const string &s, bool throw_exception_on_failure)
{
    if (throw_exception_on_failure) {
        Dtype d = Dtype::from_str(s, false);
        if (!d.is_valid())
            throw runtime_error("Dtype::from_str(): invalid argument '" + s + "'");
        return d;
    }
    
    const char *p = s.c_str();
    ushort flags = 0;
    long nbits = 0;

    long n = 0;
    while (p[n] && !isdigit(p[n]))
        n++;

    if ((n == 3) && !memcmp(p,"int",3))
        flags = df_int;
    else if ((n == 4) && !memcmp(p,"uint",4))
        flags = df_uint;
    else if ((n == 5) && !memcmp(p,"float",5))
        flags = df_float;
    else if ((n == 7) && !memcmp(p,"complex",7))
        flags = (df_complex | df_float);
    else if ((n == 11) && !memcmp(p,"complex_int",11))
        flags = (df_complex | df_int);
    else if ((n == 12) && !memcmp(p,"complex_uint",12))
        flags = (df_complex | df_uint);
    else
        return Dtype();

    p += n;

    if (flags & df_complex) {
        const char *p0 = p;
        long n0 = _scan_digits(p0);
        if ((n0 == 0) || (p0[n0] != '+'))
            return Dtype();

        p += (n0+1);
        n = _scan_digits(p);
            
        if ((n0 != n) || memcmp(p,p0,n))
            return Dtype();
    }
    else
        n = _scan_digits(p);

    if ((n == 0) || p[n])
        return Dtype();

    nbits = atol(p);
    if (flags & df_complex)
        nbits *= 2;

    if ((nbits <= 0) || (nbits >= 65536L))
        return Dtype();
    
    return Dtype(flags, nbits);
}


} // namespace ksgpu
