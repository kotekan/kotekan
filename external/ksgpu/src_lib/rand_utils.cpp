#include "../include/ksgpu/rand_utils.hpp"

using namespace std;

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


std::mt19937 default_rng(137);


// -------------------------------------------------------------------------------------------------


// Version of randomize() for floating-point types T=float, T=double.
template<typename T>
inline void _randomize_f(T *buf, long nelts, std::mt19937 &rng = default_rng)
{
    static_assert(std::is_floating_point_v<T>);

    auto dist = std::uniform_real_distribution<T>(-1.0, 1.0);    
    for (long i = 0; i < nelts; i++)
        buf[i] = dist(rng);
}

// Version of randomize() for T=__half.
template<>
inline void _randomize_f(__half *buf, long nelts, std::mt19937 &rng)
{
    auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);    
    for (long i = 0; i < nelts; i++)
        buf[i] = __float2half_rn(dist(rng));
}


void _randomize(Dtype dtype, void *buf, long nelts, std::mt19937 &rng)
{
    _check_dtype_valid(dtype, "randomize");
    
    xassert(nelts >= 0);
    xassert((nelts == 0) || (buf != nullptr));

    if (nelts == 0)
        return;

    if (dtype.flags & (df_int | df_uint)) {
        // Ignore details of dtype, and generate the correct number of random bits.
        // Note: this code is still correct if dtype is a complex type.
        
        long nbits = nelts * dtype.nbits;
        long nbytes = (nbits + 7) >> 3;
        long nints = nbytes / sizeof(int);

        for (long i = 0; i < nints; i++)
            ((int *)buf)[i] = rng();
        for (long i = nints * sizeof(int); i < nbytes; i++)
            ((char *)buf)[i] = rng();
        
        return;
    }
    
    xassert(dtype.flags & df_float);

    bool is_complex = (dtype.flags & df_complex);
    long re_nelts = is_complex ? (nelts << 1) : (nelts);
    long re_nbits = is_complex ? (dtype.nbits >> 1) : (dtype.nbits);

    if (re_nbits == (8 * sizeof(float)))
        _randomize_f<float> ((float *) buf, re_nelts, rng);
    else if (re_nbits == (8 * sizeof(double)))
        _randomize_f<double> ((double *) buf, re_nelts, rng);
    else if (re_nbits == (8 * sizeof(__half)))
        _randomize_f<__half> ((__half *) buf, re_nelts, rng);
    else
        throw std::runtime_error("randomize() not implemented for " + dtype.str());
}


// -------------------------------------------------------------------------------------------------


vector<double> random_doubles_with_fixed_sum(int n, double sum)
{
    if (n < 1)
        throw runtime_error("random_doubles_with_fixed_sum(): 'n' argument must be >= 1");
    if (sum <= 0.0)
        throw runtime_error("random_doubles_with_fixed_sum(): 'sum' argument must be > 0");

    vector<double> ret(n);

    for (int i = 0; i < n-1; i++) {
        double t = rand_uniform();
        double p = 1.0 / double(n-1-i);
        ret[i] = sum * (1 - pow(t,p));
        sum -= ret[i];
    }

    ret[n-1] = sum;
    return ret;
}


vector<long> random_integers_with_bounded_product(int n, long bound)
{    
    if (n < 1)
        throw runtime_error("random_integers_with_bounded_product(): 'n' argument must be >= 1");
    if (bound < 1)
        throw runtime_error("random_integers_with_bounded_product(): 'bound' argument must be >= 1");

    double target_log = log(rand_uniform(1.01, bound+0.99));
    vector<double> x = random_doubles_with_fixed_sum(n, target_log);
    
    vector<long> ret(n);    
    for (int i = 0; i < n; i++)
        ret[i] = long(exp(x[i]) + 0.5);
    
    int i0 = rand_int(0, n);

    for (int i1 = 0; i1 < n; i1++) {
        int i = (i0 + i1) % n;
        
        long p = 1;
        for (int j = 0; j < n; j++)
            if (j != i)
                p *= ret[j];

        ret[i] = long(exp(target_log-log(p)) + 0.5);
        ret[i] = min(ret[i], bound/p);
        ret[i] = max(ret[i], 1L);
    }

    long p = 1;
    for (int j = 0; j < n; j++)
        p *= ret[j];

    xassert(p > 0);
    xassert(p <= bound);
    return ret;
}


// -------------------------------------------------------------------------------------------------


string make_random_hex_string(int len)
{
    static const char hex_chars[] = "0123456789abcdef";
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 15);
    
    string s;
    for (int i = 0; i < len; i++)
        s += hex_chars[dist(gen)];
    
    return s;
}


} // namespace ksgpu
