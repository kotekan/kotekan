#include "../include/ksgpu/test_utils.hpp"
#include "../include/ksgpu/complex_type_traits.hpp"  // is_complex_v<T>, decomplexify_type<T>::type
#include "../include/ksgpu/rand_utils.hpp"
#include "../include/ksgpu/xassert.hpp"

#include <cmath>
#include <complex>
#include <iostream>

using namespace std;

namespace ksgpu {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// Helper for make_random_shape() and make_random_reshape_compatible_shapes().
inline long make_random_axis(long maxaxis, long &maxsize)
{
    xassert(maxaxis > 0);
    xassert(maxsize > 0);
    
    long n = std::min(maxaxis, maxsize);
    double t = rand_uniform(1.0e-6, log(n+1.0) - 1.0e-6);
    long ret = long(exp(t));  // round down
    maxsize = maxsize / ret;        // round down
    return ret;
}


vector<long> make_random_shape(int ndim, long maxaxis, long maxsize)
{
    if (ndim == 0)
	ndim = rand_int(1, ArrayMaxDim+1);
    
    xassert(ndim > 0);
    xassert(ndim <= ArrayMaxDim);
    xassert(maxsize > 0);

    vector<long> shape(ndim);
    for (int d = 0; d < ndim; d++)
	shape[d] = make_random_axis(maxaxis, maxsize);  // modifies 'maxsize'

    randomly_permute(shape);
    return shape;
}


// -------------------------------------------------------------------------------------------------


vector<long> make_random_strides(int ndim, const long *shape, int ncontig, int nalign)
{
    xassert(ndim <= ArrayMaxDim);
    xassert(ncontig >= 0);
    xassert(ncontig <= ndim);
    xassert(nalign >= 1);

    int nd_strided = ndim - ncontig;
    vector<long> axis_ordering = rand_permutation(nd_strided);
    
    vector<long> strides(ndim);
    long min_stride = 1;

    // These strides are contiguous
    for (int d = ndim-1; d >= nd_strided; d--) {
	xassert(shape[d] > 0);
	strides[d] = min_stride;
	min_stride += (shape[d]-1) * strides[d];
    }

    // These strides are not necessarily contiguous
    for (int i = 0; i < nd_strided; i++) {
	int d = axis_ordering[i];
	xassert(shape[d] > 0);

	// Assign stride (as multiple of nalign)
	long smin = (min_stride + nalign - 1) / nalign;
	long smax = std::max(smin+1, (2*min_stride)/nalign);
	long s = (rand_uniform() < 0.33) ? smin : rand_int(smin,smax+1);
	
	strides[d] = s * nalign;
	min_stride += (shape[d]-1) * strides[d];
    }

    return strides;
}


vector<long> make_random_strides(const vector<long> &shape, int ncontig, int nalign)
{
    return make_random_strides(shape.size(), &shape[0], ncontig, nalign);
}


// -------------------------------------------------------------------------------------------------


// Helper for make_random_reshape_compatible_shapes()
struct RcBlock
{
    bool dflag;
    vector<long> bshape;
    vector<long> bstrides;  // contiguous
    long bsize;
};


void make_random_reshape_compatible_shapes(vector<long> &dshape,
					   vector<long> &sshape,
					   vector<long> &sstrides,
					   int maxaxis, long maxsize)
{
    xassert(maxaxis > 0);
    xassert(maxsize > 0);
    
    vector<long> dstrides;
    dshape.clear();
    sshape.clear();
    sstrides.clear();
    
    vector<RcBlock> blocks;
    uint ddims = 0;
    uint sdims = 0;

    while (blocks.size() < ArrayMaxDim) {
	RcBlock block;
	block.dflag = rand_int(0,2);
	
	uint &fdims = block.dflag ? ddims : sdims;   // "factored" dims
	uint &udims = block.dflag ? sdims : ddims;   // "unfactored" dims

	if (udims == ArrayMaxDim)
	    break;
	
	int nb_max = ArrayMaxDim - fdims;
	int nb = 0;
	
	if ((nb_max > 0) && (rand_uniform() < 0.95)) {
	    nb = rand_int(1, nb_max+1);
	    block.bshape = make_random_shape(nb, maxaxis, maxsize);  // modifies 'maxsize'
	}
	
	block.bstrides.resize(nb);
	block.bsize = 1;

	for (int i = nb-1; i >= 0; i--) {
	    block.bstrides[i] = block.bsize;
	    block.bsize *= block.bshape[i];
	}

	blocks.push_back(block);
	fdims += nb;
	udims += 1;

	if (rand_uniform() < 0.2)
	    break;
    }

    randomly_permute(blocks);

    int nblocks = blocks.size();
    vector<long> block_sizes(nblocks);
    
    for (int i = 0; i < nblocks; i++)
	block_sizes[i] = blocks[i].bsize;
    
    vector<long> block_strides = make_random_strides(block_sizes);    

    for (int i = 0; i < nblocks; i++) {
	const RcBlock &block = blocks[i];
	int block_stride = block_strides[i];
	
	vector<long> &fshape = block.dflag ? dshape : sshape;   // "factored" shape
	vector<long> &ushape = block.dflag ? sshape : dshape;   // "unfactored" shape
	vector<long> &fstrides = block.dflag ? dstrides : sstrides;   // "factored" strides
	vector<long> &ustrides = block.dflag ? sstrides : dstrides;   // "unfactored" strides

	ushape.push_back(block.bsize);
	ustrides.push_back(block_stride);

	for (uint j = 0; j < block.bshape.size(); j++) {
	    fshape.push_back(block.bshape[j]);
	    fstrides.push_back(block.bstrides[j] * block_stride);
	}
    }

    xassert(dshape.size() == ddims);
    xassert(sshape.size() == sdims);
    xassert(dstrides.size() == ddims);
    xassert(sstrides.size() == sdims);

    if (ddims == 0)
	dshape.push_back(1);
    if (sdims == 0)
	sshape.push_back(1);
    if (sdims == 0)
	sstrides.push_back(1);
}

		 
// -------------------------------------------------------------------------------------------------


inline ostream &operator<<(ostream &os, __half x)
{
    os << __half2float(x);
    return os;
}


template<typename T>
void print_array(const Array<T> &arr, const vector<string> &axis_names, std::ostream &os)
{
    xassert((axis_names.size() == 0) || (axis_names.size() == uint(arr.ndim)));

    int nd = arr.ndim;
    
    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix)) {
	if (axis_names.size() == 0) {
	    os << "    (";
	    for (int d = 0; d < nd; d++)
		os << (d ? "," : "") << ix[d];
	    os << ((nd <= 1) ? ",)" : ")");
	}
	else {
	    os << "   ";
	    for (int d = 0; d < nd; d++)
		os << " " << axis_names[d] << "=" << ix[d];
	}

	os << ": " << arr.at(ix) << "\n";
    }

    os.flush();
}


// -------------------------------------------------------------------------------------------------

    
template<typename T>
typename ksgpu::decomplexify_type<T>::type
assert_arrays_equal(const Array<T> &arr1,
		    const Array<T> &arr2,
		    const string &name1,
		    const string &name2,
		    const vector<string> &axis_names,
		    float epsabs,
		    float epsrel,
		    long max_display,
		    bool verbose)
{
    using Tr = typename decomplexify_type<T>::type;
    
    xassert(arr1.shape_equals(arr2));
    xassert(axis_names.size() == uint(arr1.ndim));
    xassert(max_display > 0);
    xassert(epsabs >= 0.0);
    xassert(epsrel >= 0.0);

    Array<T> harr1 = arr1.to_host(false);  // page_locked=false
    Array<T> harr2 = arr2.to_host(false);  // page_locked=false
    int nfail = 0;
    Tr maxdiff = 0;

    for (auto ix = arr1.ix_start(); arr1.ix_valid(ix); arr1.ix_next(ix)) {
	T x = harr1.at(ix);
	T y = harr2.at(ix);

	Tr delta;
	if constexpr (!is_unsigned_v<T>)
	    delta = std::abs(x-y);
	else
	    delta = (x > y) ? (x-y) : (y-x);
	
	Tr thresh = 0;
	if constexpr (!is_integral_v<T>)
	    thresh = epsabs + 0.5*epsrel * (std::abs(x) + std::abs(y));

	maxdiff = max(maxdiff, delta);
	bool failed = (delta > thresh);

	// Automatically fail if either array contains NaN/Inf.
	// (Could introduce a flag to toggle this behavior on/off.)
	if constexpr (is_complex_v<T>) {
	    if (!std::isfinite(x.real()) || !std::isfinite(x.imag()) || !std::isfinite(y.real()) || !std::isfinite(y.imag()))
		failed = true;
	}	    
	else if constexpr (!is_integral_v<T>) {
	    if (!std::isfinite(x) || !std::isfinite(y))
		failed = true;
	}
	
	if (!failed && !verbose)
	    continue;

	if (failed && (nfail == 0))
	    cout << "\nassert_arrays_equal() failed [shape=" << arr1.shape_str() << "]\n";

	if (failed)
	    nfail++;
	
	if (nfail >= max_display)
	    continue;
	
	cout << "   ";
	for (int d = 0; d < arr1.ndim; d++)
	    cout << " " << axis_names[d] << "=" << ix[d];

	cout << ": " << name1 << "=" << x << ", " << name2
	     << "=" << y << "  [delta=" << delta << "]";

	if (failed)
	    cout << " FAILED";

	cout << "\n";
    }
    
    if ((nfail > max_display) && !verbose)
	cout << "        [ + " << (nfail-max_display) << " more failures]\n";

    cout.flush();
    
    if (nfail > 0)
	exit(1);
    
    return maxdiff;
}


// -------------------------------------------------------------------------------------------------


__global__ void busy_wait_kernel(uint *arr, long niter)
{
    uint x = arr[threadIdx.x];
    for (long i = 0; i < niter; i++) {
	x = (x ^ 0x12345678U);
	x = (x << 5) ^ (x >> 10);
    }
    arr[threadIdx.x] = x;
}

void launch_busy_wait_kernel(Array<uint> &arr, double a40_seconds, cudaStream_t s)
{
    xassert(arr.ndim == 1);
    xassert(arr.size == 32);
    xassert(arr.strides[0] == 1);
    xassert(arr.on_gpu());

    long niter = 1.4e8 * a40_seconds;
    
    busy_wait_kernel <<< 1, 32, 0, s >>>
	(arr.data, niter);
}


// -------------------------------------------------------------------------------------------------


#define INSTANTIATE_PRINT_ARRAY(T)	    \
    template void print_array(              \
	const Array<T> &arr,                \
	const vector<string> &axis_names,   \
	ostream &os)

#define INSTANTIATE_ASSERT_ARRAYS_EQUAL(T)  \
    template				    \
    ksgpu::decomplexify_type<T>::type	    \
    assert_arrays_equal(		    \
	const Array<T> &arr1,	            \
	const Array<T> &arr2,		    \
	const string &name1,		    \
	const string &name2,		    \
	const vector<string> &axis_names,   \
	float epsabs,                       \
	float epsrel,                       \
	long max_display, 	            \
	bool verbose)

#define INSTANTIATE_TEMPLATES(T) \
    INSTANTIATE_PRINT_ARRAY(T); \
    INSTANTIATE_ASSERT_ARRAYS_EQUAL(T)


INSTANTIATE_TEMPLATES(float);
INSTANTIATE_TEMPLATES(double);
INSTANTIATE_TEMPLATES(int);
INSTANTIATE_TEMPLATES(long);
INSTANTIATE_TEMPLATES(short);
INSTANTIATE_TEMPLATES(char);
INSTANTIATE_TEMPLATES(uint);
INSTANTIATE_TEMPLATES(ulong);
INSTANTIATE_TEMPLATES(ushort);
INSTANTIATE_TEMPLATES(unsigned char);
INSTANTIATE_TEMPLATES(complex<float>);
INSTANTIATE_TEMPLATES(complex<double>);
INSTANTIATE_TEMPLATES(complex<int>);

// FIXME implement assert_arrays_equal<__half>().
// In the meantime, I'm instantiating print_array<__half>(), but not assert_arrays_equal<__half>().
INSTANTIATE_PRINT_ARRAY(__half);


}  // namespace ksgpu
