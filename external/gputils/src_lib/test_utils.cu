#include "../include/gputils/test_utils.hpp"
#include "../include/gputils/rand_utils.hpp"

// is_complex_v<T>, decomplexify_type<T>::type
#include "../include/gputils/complex_type_traits.hpp"

#include <cmath>
#include <cassert>
#include <complex>
#include <iostream>

using namespace std;

namespace gputils {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// Helper for make_random_shape() and make_random_reshape_compatible_shapes().
inline ssize_t make_random_axis(ssize_t maxaxis, ssize_t &maxsize)
{
    assert(maxaxis > 0);
    assert(maxsize > 0);
    
    ssize_t n = std::min(maxaxis, maxsize);
    double t = rand_uniform(1.0e-6, log(n+1.0) - 1.0e-6);
    ssize_t ret = ssize_t(exp(t));  // round down
    maxsize = maxsize / ret;        // round down
    return ret;
}


vector<ssize_t> make_random_shape(int ndim, ssize_t maxaxis, ssize_t maxsize)
{
    if (ndim == 0)
	ndim = rand_int(1, ArrayMaxDim+1);
    
    assert(ndim > 0);
    assert(ndim <= ArrayMaxDim);
    assert(maxsize > 0);

    vector<ssize_t> shape(ndim);
    for (int d = 0; d < ndim; d++)
	shape[d] = make_random_axis(maxaxis, maxsize);  // modifies 'maxsize'

    randomly_permute(shape);
    return shape;
}


// -------------------------------------------------------------------------------------------------


vector<ssize_t> make_random_strides(int ndim, const ssize_t *shape, int ncontig, int nalign)
{
    assert(ndim <= ArrayMaxDim);
    assert(ncontig >= 0);
    assert(ncontig <= ndim);
    assert(nalign >= 1);

    int nd_strided = ndim - ncontig;
    vector<ssize_t> axis_ordering = rand_permutation(nd_strided);
    
    vector<ssize_t> strides(ndim);
    ssize_t min_stride = 1;

    // These strides are contiguous
    for (int d = ndim-1; d >= nd_strided; d--) {
	assert(shape[d] > 0);
	strides[d] = min_stride;
	min_stride += (shape[d]-1) * strides[d];
    }

    // These strides are not necessarily contiguous
    for (int i = 0; i < nd_strided; i++) {
	int d = axis_ordering[i];
	assert(shape[d] > 0);

	// Assign stride (as multiple of nalign)
	ssize_t smin = (min_stride + nalign - 1) / nalign;
	ssize_t smax = std::max(smin+1, (2*min_stride)/nalign);
	ssize_t s = (rand_uniform() < 0.33) ? smin : rand_int(smin,smax+1);
	
	strides[d] = s * nalign;
	min_stride += (shape[d]-1) * strides[d];
    }

    return strides;
}


vector<ssize_t> make_random_strides(const vector<ssize_t> &shape, int ncontig, int nalign)
{
    return make_random_strides(shape.size(), &shape[0], ncontig, nalign);
}


// -------------------------------------------------------------------------------------------------


// Helper for make_random_reshape_compatible_shapes()
struct RcBlock
{
    bool dflag;
    vector<ssize_t> bshape;
    vector<ssize_t> bstrides;  // contiguous
    ssize_t bsize;
};


void make_random_reshape_compatible_shapes(vector<ssize_t> &dshape,
					   vector<ssize_t> &sshape,
					   vector<ssize_t> &sstrides,
					   int maxaxis, ssize_t maxsize)
{
    assert(maxaxis > 0);
    assert(maxsize > 0);
    
    vector<ssize_t> dstrides;
    dshape.clear();
    sshape.clear();
    sstrides.clear();
    
    vector<RcBlock> blocks;
    unsigned int ddims = 0;
    unsigned int sdims = 0;

    while (blocks.size() < ArrayMaxDim) {
	RcBlock block;
	block.dflag = rand_int(0,2);
	
	unsigned int &fdims = block.dflag ? ddims : sdims;   // "factored" dims
	unsigned int &udims = block.dflag ? sdims : ddims;   // "unfactored" dims

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
    vector<ssize_t> block_sizes(nblocks);
    
    for (int i = 0; i < nblocks; i++)
	block_sizes[i] = blocks[i].bsize;
    
    vector<ssize_t> block_strides = make_random_strides(block_sizes);    

    for (int i = 0; i < nblocks; i++) {
	const RcBlock &block = blocks[i];
	int block_stride = block_strides[i];
	
	vector<ssize_t> &fshape = block.dflag ? dshape : sshape;   // "factored" shape
	vector<ssize_t> &ushape = block.dflag ? sshape : dshape;   // "unfactored" shape
	vector<ssize_t> &fstrides = block.dflag ? dstrides : sstrides;   // "factored" strides
	vector<ssize_t> &ustrides = block.dflag ? sstrides : dstrides;   // "unfactored" strides

	ushape.push_back(block.bsize);
	ustrides.push_back(block_stride);

	for (unsigned int j = 0; j < block.bshape.size(); j++) {
	    fshape.push_back(block.bshape[j]);
	    fstrides.push_back(block.bstrides[j] * block_stride);
	}
    }

    assert(dshape.size() == ddims);
    assert(sshape.size() == sdims);
    assert(dstrides.size() == ddims);
    assert(sstrides.size() == sdims);

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
    assert((axis_names.size() == 0) || (axis_names.size() == arr.ndim));

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
typename gputils::decomplexify_type<T>::type
assert_arrays_equal(const Array<T> &arr1,
		    const Array<T> &arr2,
		    const string &name1,
		    const string &name2,
		    const vector<string> &axis_names,
		    float epsabs,
		    float epsrel,
		    ssize_t max_display,
		    bool verbose)
{
    using Tr = typename decomplexify_type<T>::type;
    
    assert(arr1.shape_equals(arr2));
    assert(axis_names.size() == arr1.ndim);
    assert(max_display > 0);
    assert(epsabs >= 0.0);
    assert(epsrel >= 0.0);

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


#define INSTANTIATE_PRINT_ARRAY(T)	    \
    template void print_array(              \
	const Array<T> &arr,                \
	const vector<string> &axis_names,   \
	ostream &os)

#define INSTANTIATE_ASSERT_ARRAYS_EQUAL(T)  \
    template				    \
    gputils::decomplexify_type<T>::type	    \
    assert_arrays_equal(		    \
	const Array<T> &arr1,	            \
	const Array<T> &arr2,		    \
	const string &name1,		    \
	const string &name2,		    \
	const vector<string> &axis_names,   \
	float epsabs,                       \
	float epsrel,                       \
	ssize_t max_display, 	            \
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
INSTANTIATE_TEMPLATES(unsigned int);
INSTANTIATE_TEMPLATES(unsigned long);
INSTANTIATE_TEMPLATES(unsigned short);
INSTANTIATE_TEMPLATES(unsigned char);
INSTANTIATE_TEMPLATES(complex<float>);
INSTANTIATE_TEMPLATES(complex<double>);

// FIXME implement assert_arrays_equal<__half>().
// In the meantime, I'm instantiating print_array<__half>(), but not assert_arrays_equal<__half>().
INSTANTIATE_PRINT_ARRAY(__half);


}  // namespace gputils
