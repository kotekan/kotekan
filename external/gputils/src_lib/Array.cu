#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/cuda_utils.hpp"    // CUDA_CALL()
#include "../include/gputils/string_utils.hpp"  // tuple_str()

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


ssize_t compute_size(int ndim, const ssize_t *shape)
{
    ssize_t ret = ndim ? 1 : 0;
    for (int d = 0; d < ndim; d++)
	ret *= shape[d];
    return ret;
}


int compute_ncontig(int ndim, const ssize_t *shape, const ssize_t *strides)
{
    for (int d = 0; d < ndim; d++)
	if (shape[d] == 0)
	    return ndim;

    ssize_t s = 1;
    for (int d = ndim-1; d >= 0; d--) {
	if ((shape[d] > 1) && (strides[d] != s))
	    return ndim-1-d;
	s *= shape[d];
    }

    return ndim;
}


string shape_str(int ndim, const ssize_t *shape)
{
    // Avoids #include-ing string_utils.hpp in Array.hpp
    return tuple_str(ndim, shape);
}


bool shape_eq(int ndim1, const ssize_t *shape1, int ndim2, const ssize_t *shape2)
{
    if (ndim1 != ndim2)
	return false;
    
    for (int i = 0; i < ndim1; i++)
	if (shape1[i] != shape2[i])
	    return false;

    return true;
}


// -------------------------------------------------------------------------------------------------


// Helper for check_array_invariants()
struct stride_checker {
    ssize_t axis_length;
    ssize_t axis_stride;

    bool operator<(const stride_checker &x)
    {
	return this->axis_stride < x.axis_stride;
    }
};


void check_array_invariants(const void *data, int ndim, const ssize_t *shape,
			    ssize_t size, const ssize_t *strides, int aflags)
{
    assert(ndim >= 0 && ndim <= ArrayMaxDim);

    for (int d = 0; d < ndim; d++) {
	assert(shape[d] >= 0);
	assert(strides[d] >= 0);
    }

    assert(size == compute_size(ndim, shape));
    assert(!((data == nullptr) && (size != 0)));
    assert(!((data != nullptr) && (size == 0)));    
    check_aflags(aflags);

    if (size <= 1)
	return;

    // Stride checks follow
    
    stride_checker sc[ndim];
    int n = 0;
    
    for (int d = 0; d < ndim; d++) {
	// Length-1 axes can have arbitrary strides
	if (shape[d] == 1)
	    continue;
	
	sc[n].axis_length = shape[d];
	sc[n].axis_stride = strides[d];
	n++;
    }

    assert(n > 0);  // should never fail
    std::sort(sc, sc+n);

    ssize_t min_stride = 1;
    for (int i = 0; i < n; i++) {
	assert(sc[i].axis_stride >= min_stride);
	min_stride += (sc[i].axis_length - 1) * sc[i].axis_stride;
    }
}


// -------------------------------------------------------------------------------------------------


// reshape_helper2()
//
//   - assumes src_ndim, src_shape, src_strides, dst_ndim have been validated by caller
//   - validates dst_shape
//   - initializes dst_strides
//
// Return value:
//   0 = success
//   1 = dst_shape is incompatible with src_shape (or dst_shape is invalid)
//   2 = src and dst shapes are compatible, but src_strides don't allow axes to be combined

static int reshape_helper2(int src_ndim, const ssize_t *src_shape, const ssize_t *src_strides,
			   int dst_ndim, const ssize_t *dst_shape, ssize_t *dst_strides)
{
    // If we detect shape incompatbility, we "return 1" immediately.
    // If we detect bad src_strides, we set ret=2, rather than "return 2" immediately.
    // This is so shape incompatibility takes precedence over bad strides.
    int ret = 0;
    
    for (int d = 0; d < dst_ndim; d++)
	if (dst_shape[d] < 0)
	    return 1;  // invalid dst_shape
    
    ssize_t src_size = compute_size(src_ndim, src_shape);
    ssize_t dst_size = compute_size(dst_ndim, dst_shape);
    if (src_size != dst_size)
	return 1;      // catches empty-array corner cases

    if (src_size == 0) {
	// Both arrays are empty
	for (int d = 0; d < dst_ndim; d++)
	    dst_strides[d] = 0;  // arbitrary
	return 0;
    }

    int is = 0;
    int id = 0;
    
    for (;;) {
	// At top of loop, src indices <is and dst_indices <id
	// have been "consumed", and verified to be compatible.
	
	// Advance until non-1 is reached
	while ((is < src_ndim) && (src_shape[is] == 1))
	    is++;
	
	// Advance until non-1 is reached
	while ((id < dst_ndim) && (dst_shape[id] == 1))
	    dst_strides[id++] = 0;  // arbitrary

	if ((is == src_ndim) && (id == dst_ndim))
	    return ret;  // shapes are compatible (return value may be 0 or 2)

	if ((is == src_ndim) || (id == dst_ndim))
	    return 1;    // should never happen, thanks to "if (src_size != dst_size) ..." above

	ssize_t ss = src_shape[is];
	ssize_t sd = dst_shape[id];
	assert((ss >= 2) && (sd >= 2));  // should never fail

	if ((ss % sd) == 0) {
	    // Split source axis across one or more destination axes.
	    // In the loop below, source axis parameters (is,ss) are fixed.
	    // At top, dst axes <= id have "consumed" sd elements, where sd | ss.
	    
	    for (;;) {
		dst_strides[id] = (ss/sd) * src_strides[is];
		id++;
		if (ss == sd)
		    break;
		if (id == dst_ndim)
		    return 1;
		sd *= dst_shape[id];
		if ((ss % sd) != 0)
		    return 1;
	    }
	    
	    is++;
	}
	else if ((sd % ss) == 0) {
	    // Combine multiple source axes into single destination axis.
	    // In the loop below, destination axis parameters (id,sd) are fixed.
	    // At top, src axes <= is have "consumed" ss elements, where ss | sd.
	    
	    ssize_t tot_stride = (src_strides[is] * ss);
	    if (tot_stride % sd != 0)
		ret = 2;   // not "return 2"
	    
	    dst_strides[id] = tot_stride / sd;

	    for (;;) {
		is++;
		if (ss == sd)
		    break;
		if (is == src_ndim)
		    return 1;
		ss *= src_shape[is];
		if ((sd % ss) != 0)
		    return 1;
		if ((src_strides[is] * ss) != tot_stride)
		    ret = 2;  // not "return 2"
	    }

	    id++;
	}
	else
	    return 1;
    }
}
	    

// reshape_ref_helper()
//
//   - assumes src_ndim, src_shape, src_strides, dst_ndim have been validated by caller
//   - validates dst_shape
//   - initializes dst_strides

void reshape_ref_helper(int src_ndim, const ssize_t *src_shape, const ssize_t *src_strides,
			int dst_ndim, const ssize_t *dst_shape, ssize_t *dst_strides)
{
    int status = reshape_helper2(src_ndim, src_shape, src_strides, dst_ndim, dst_shape, dst_strides);

    if (status == 1) {
	stringstream ss;
	ss << "Array::reshape_ref(): src_shape=" << shape_str(src_ndim, src_shape)
	   << " is incompatible with dst_shape=" << shape_str(dst_ndim, dst_shape);
	throw runtime_error(ss.str());
	
    }
    else if (status == 2) {
	stringstream ss;
	ss << "Array::reshape_ref(): src_shape=" << shape_str(src_ndim, src_shape)
	   << " and dst_shape=" << shape_str(dst_ndim, dst_shape)
	   << " are compatible, but src_strides=" << shape_str(src_ndim, src_strides)
	   << " don't allow axes to be combined";
	throw runtime_error(ss.str());
    }
}


// -------------------------------------------------------------------------------------------------


struct fill_axis {
    ssize_t length;
    ssize_t dstride;  // in bytes
    ssize_t sstride;  // in bytes

    inline bool operator<(const fill_axis &a) const
    {
	// FIXME put more thought into this
	return dstride < a.dstride;
    }

    void show() const
    {
	cout << "    fill_axis(len=" << length << ", dstride=" << dstride << ", sstride=" << sstride << ")\n";
    }
};


static void fill_helper2(char *dst, const char *src, int ndim, const fill_axis *axes)
{
    // Caller guarantees the following:
    //   ndim > 0
    //   axes[0].dstride == 1
    //   axes[0].sstride == 1
    
    if (ndim <= 1) {
	CUDA_CALL(cudaMemcpy(dst, src, axes[0].length, cudaMemcpyDefault));
	return;
    }

    if (ndim == 2) {
	// These two asserts are required by cudaMemcpy2D().
	// In particular, strides must be positive.
	assert(axes[1].dstride >= axes[0].length);
	assert(axes[1].sstride >= axes[0].length);
	CUDA_CALL(cudaMemcpy2D(dst, axes[1].dstride, src, axes[1].sstride, axes[0].length, axes[1].length, cudaMemcpyDefault));
	return;
    }

    // Note: there is a cudaMemcpy3D(), but it requires that either the
    // data be in a cudaArray, or that strides are contiguous (so that
    // it's a cudaMemcpy2D in disguise), so we can't use it here.
    
    for (int i = 0; i < axes[ndim-1].length; i++) {
	fill_helper2(dst + i * axes[ndim-1].dstride,
		     src + i * axes[ndim-1].sstride,
		     ndim-1, axes);
    }
}



void fill_helper(void *dst, int dst_ndim, const ssize_t *dst_shape, const ssize_t *dstride,
		 const void *src, int src_ndim, const ssize_t *src_shape, const ssize_t *sstride,
		 ssize_t itemsize, bool noisy)
{
    // Check that dst/src shapes match.
    
    if (!shape_eq(dst_ndim, dst_shape, src_ndim, src_shape)) {
	stringstream ss;
	ss << "gputils::Array::fill(): dst_shape=" << shape_str(dst_ndim,dst_shape)
	   << " and src_shape=" << shape_str(src_ndim,src_shape) << " are unequal";
	throw runtime_error(ss.str());
    }

    int ndim = dst_ndim;
    const ssize_t *shape = dst_shape;

    // If empty array, then return early
    
    if (dst_ndim == 0)
	return;
    
    for (int d = 0; d < ndim; d++) {
	if (shape[d] == 0)
	    return;
    }

    // Uncoalesced axes
    fill_axis axes_u[ndim];
    int nax_u = 0;

    for (int d = 0; d < ndim; d++) {
	if (shape[d] <= 1)
	    continue;

	axes_u[nax_u].length = shape[d];
	axes_u[nax_u].dstride = dstride[d] * itemsize;
	axes_u[nax_u].sstride = sstride[d] * itemsize;
	nax_u++;
    }

    // Sort by increasing dstride
    std::sort(axes_u, axes_u + nax_u);

    // Coalece axes, and represent itemsize by a new axis
    fill_axis axes_c[ndim+1];
    int nax_c = 1;

    axes_c[0].length = itemsize;
    axes_c[0].dstride = 1;
    axes_c[0].sstride = 1;

    // (Length * stride) of last coalesced axis.
    ssize_t dlen = itemsize;
    ssize_t slen = itemsize;
    
    for (int d = 0; d < nax_u; d++) {
	if ((axes_u[d].dstride == dlen) && (axes_u[d].sstride == slen)) {
	    // Can coalesce
	    ssize_t s = axes_u[d].length;
	    axes_c[nax_c-1].length *= s;
	    dlen *= s;
	    slen *= s;
	}
	else {
	    // Can't coalesce
	    axes_c[nax_c] = axes_u[d];
	    dlen = axes_u[d].length * axes_u[d].dstride;
	    slen = axes_u[d].length * axes_u[d].sstride;
	    nax_c++;
	}
    }

    if (noisy) {
	cout << "fill: input shape=" << tuple_str(ndim, shape)
	     << ", dstride_nelts=" << tuple_str(ndim, dstride)
	     << ", sstride_nelts=" << tuple_str(ndim, sstride)
	     << ", itemsize=" << itemsize << "\n";

	for (int d = 0; d < nax_c; d++)
	    axes_c[d].show();	
    }
    
    fill_helper2((char *)dst, (const char *)src, nax_c, axes_c);
}


}  // namespace gputils
