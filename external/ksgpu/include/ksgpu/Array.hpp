#ifndef _KSGPU_ARRAY_HPP
#define _KSGPU_ARRAY_HPP

#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_fp16.h>    // __half

#include "mem_utils.hpp"  // af_alloc() and flags
#include "xassert.hpp"    // af_alloc() and flags

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


static constexpr int ArrayMaxDim = 8;


template<typename T>
struct Array {
    T *data = nullptr;
    
    int ndim = 0;
    long shape[ArrayMaxDim];
    long size = 0;

    long strides[ArrayMaxDim];  // in units sizeof(T), not bytes

    std::shared_ptr<void> base;
    int aflags = 0;

    // "Empty" arrays are size-zero objects containing null pointers.
    // All Arrays obey the rules:
    //
    //  (data == nullptr)
    //      iff (size == 0)
    //      iff ((ndim==0) || (shape[i]==0 for some 0 <= i < ndim))
    //
    // Note that zero-dimensional arrays are empty (unlike numpy,
    // where zero-dimensional arrays have size 1).

    Array();
    
    // Allocator flags ('aflags') are defined in mem_utils.hpp
    // Flags can be used to allocate memory on CPU/GPU, zero memory after allocation, etc.
    Array(int ndim, const long *shape, int aflags);
    Array(const std::vector<long> &shape, int aflags);
    
    // Syntactic sugar for constructing array with "inline" dimensions, e.g.
    //    Array<float> arr({m,n}, af_gpu);
    Array(std::initializer_list<long> shape, int aflags);

    // These constructors allow explicit strides.
    // Often used in unit tests, with make_random_strides() in test_utils.hpp.
    Array(int ndim, const long *shape, const long *strides, int aflags);
    Array(const std::vector<long> &shape, const std::vector<long> &strides, int aflags);
    Array(std::initializer_list<long> shape, std::initializer_list<long> strides, int aflags);

    
    // Is array addressable on GPU? On host?
    inline bool on_gpu() const { return !data || af_on_gpu(aflags); }
    inline bool on_host() const { return !data || af_on_host(aflags); }

    // Copies data from 'src' to 'this'. Arrays must have the same shape, but need not have the same strides.
    // FIXME currently using cudaMemcpy() even if both arrays are on host -- is this slow?
    inline void fill(const Array<T> &src);
    
    inline Array<T> clone(int aflags) const;
    inline Array<T> clone() const;  // retains location_flags of source array
    
    // Returns an Array addressable on GPU (or host), making a copy if necessary.
    inline Array<T> to_gpu() const;
    inline Array<T> to_host(bool registered=true) const;

    // Returns number of contiguous dimensions, assuming indices are ordered
    // from slowest to fastest varying. Returns 'ndim' for an empty array.
    int get_ncontig() const;
    bool is_fully_contiguous() const { return get_ncontig() == ndim; }
    
    // at(): range-checked accessor
    // (I'm reserving operator[] for an unchecked accessor.)
    
    inline T& at(int ndim, const long *ix);
    inline T& at(const std::vector<long> &ix);
    inline T& at(std::initializer_list<long> ix);
    
    inline const T& at(int ndim, const long *ix) const;
    inline const T& at(const std::vector<long> &ix) const;
    inline const T& at(std::initializer_list<long> ix) const;

    
    // The new Arrays returned by slice() contain references
    // (not copies) to the data in the original Array.
    inline Array<T> slice(int axis, int start, int stop) const;
    inline Array<T> slice(int axis, int ix) const;   // returns array of dimension (ndim-1)


    // Reshape-by-reference. Throws an exception if either (1) requested shape
    // is incompatible with the current shape; (2) current strides don't permit
    // axes to be combined without copying.
    //
    // FIXME reshape_ref() hasn't been systematically tested, and should be!

    inline Array<T> reshape_ref(int ndim, const long *shape) const;
    inline Array<T> reshape_ref(const std::vector<long> &shape) const;
    inline Array<T> reshape_ref(std::initializer_list<long> shape) const;

    // Converts (array of type T) -> (array of type T2).
    // FIXME for now, both arrays must be on host.
    //
    // Element conversion is done using C++ type conversion, except in cases
    // (__half) <-> (float or double), when we call cuda intrinsics such as
    // __float2half().
    
    template<typename Tdst> inline Array<Tdst> convert_dtype() const;
    template<typename Tdst> inline Array<Tdst> convert_dtype(int aflags) const;

    
    // For looping over array indices (not high-performance):
    //
    //    Array<T> arr;
    //    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix)) {
    //        T x = arr.at(ix);
    //        // ...
    //    }
    //    
    // Warning: ix_valid() is not a general-purpose index validator!
    // It only works on the output of ix_start() -> ix_next() -> ...
    
    inline std::vector<long> ix_start() const;
    inline bool ix_valid(const std::vector<long> &ix) const;
    inline void ix_next(std::vector<long> &ix) const;
    
    inline bool shape_equals(int ndim, const long *shape) const;
    inline bool shape_equals(const std::vector<long> &shape) const;
    inline bool shape_equals(std::initializer_list<long> shape) const;
    template<typename T2> inline bool shape_equals(const Array<T2> &a) const;    

    inline std::string shape_str() const;
    inline std::string stride_str() const;

    // Throws exception on failure.
    void check_invariants() const;
    
    // "Cheat" accessor, which gives a non-const reference to a const Array
    inline T& _at(int ndim, const long *ix) const;
};


// -------------------------------------------------------------------------------------------------
//
// Non-inline functions defined elsewhere


extern void check_array_invariants(const void *data, int ndim, const long *shape,
				   long size, const long *strides, int aflags);

extern int compute_ncontig(int ndim, const long *shape, const long *strides);

extern long compute_size(int ndim, const long *shape);

extern bool shape_eq(int ndim1, const long *shape1, int ndim2, const long *shape2);

extern std::string shape_str(int ndim, const long *shape);

extern void reshape_ref_helper(int src_ndim, const long *src_shape, const long *src_strides,
			       int dst_ndim, const long *dst_shape, long *dst_strides);

extern void fill_helper(void *dst, int dst_ndim, const long *dst_shape, const long *dst_strides,
			const void *src, int src_ndim, const long *src_shape, const long *src_strides,
			long itemsize, bool noisy=false);


// -------------------------------------------------------------------------------------------------
//
// Inline implementations


template<typename T>
Array<T>::Array()
    : Array(0, nullptr, 0) { }

template<typename T>
Array<T>::Array(const std::vector<long> &shape_, int aflags)
    : Array(shape_.size(), &shape_[0], aflags) { }

template<typename T>
Array<T>::Array(std::initializer_list<long> shape_, int aflags)
    : Array(shape_.size(), shape_.begin(), aflags) { }


template<typename T>
Array<T>::Array(int ndim_, const long *shape_, int aflags_)
    : ndim(ndim_), aflags(aflags_)
{
    xassert(ndim >= 0);
    xassert(ndim <= ArrayMaxDim);

    for (int i = ndim; i < ArrayMaxDim; i++)
	shape[i] = strides[i] = 0;

    if (ndim == 0) {
	data = nullptr;
	size = 0;
	return;
    }

    size = 1;    
    for (int i = ndim-1; i >= 0; i--) {
	strides[i] = size;
	shape[i] = shape_[i];
	size *= shape_[i];
	xassert(shape[i] >= 0);
    }

    if (size != 0) {
	std::shared_ptr<T> p = af_alloc<T> (size, aflags);
	data = p.get();
	base = p;  // implicit conversion shared_ptr<T> -> shared_ptr<void>
    }
    else
	data = nullptr;
}


template<typename T>
Array<T>::Array(int ndim_, const long *shape_, const long *strides_, int aflags_)
    : ndim(ndim_), aflags(aflags_)
{
    xassert(ndim > 0);
    xassert(ndim <= ArrayMaxDim);

    for (int i = ndim; i < ArrayMaxDim; i++)
	shape[i] = strides[i] = 0;

    if (ndim == 0) {
	data = nullptr;
	size = 0;
	return;
    }

    size = 1;
    long nalloc = 1;
	
    for (int d = 0; d < ndim; d++) {
	shape[d] = shape_[d];
	strides[d] = strides_[d];
	
	xassert(shape[d] >= 0);
	xassert(strides[d] >= 0);
	size *= shape[d];
	nalloc += (shape[d]-1) * strides[d];
    }

    if (size != 0) {
	std::shared_ptr<T> p = af_alloc<T> (nalloc, aflags);
	data = p.get();
	base = p;  // implicit conversion shared_ptr<T> -> shared_ptr<void>
    }
    else
	data = nullptr;

    this->check_invariants();
}


// This little device is useful in chaining Array constructors
template<class C> int ndim_ss(const C &shape, const C &strides)
{
    if (shape.size() != strides.size())
	throw std::runtime_error("shape/strides length mismatch in Array constructor");
    return shape.size();
}

template<typename T>
Array<T>::Array(const std::vector<long> &shape_, const std::vector<long> &strides_, int aflags)
    : Array(ndim_ss(shape_,strides_), &shape_[0], &strides_[0], aflags) { }

template<typename T>
Array<T>::Array(std::initializer_list<long> shape_, std::initializer_list<long> strides_, int aflags)
    : Array(ndim_ss(shape_,strides_), shape_.begin(), strides_.begin(), aflags) { }


template<typename T>
void Array<T>::fill(const Array<T> &src)
{
    fill_helper(data, ndim, shape, strides,
		src.data, src.ndim, src.shape, src.strides,
		sizeof(T), false);   // noisy=false
}


template<typename T>
Array<T> Array<T>::clone(int aflags_) const
{
    Array<T> ret(this->ndim, this->shape, aflags_ & ~af_initialization_flags);
    ret.fill(*this);
    return ret;
}

template<typename T>
Array<T> Array<T>::clone() const
{
    return this->clone(this->aflags & af_location_flags);
}

template<typename T>
Array<T> Array<T>::to_gpu() const
{
    return this->on_gpu() ? (*this) : this->clone(af_gpu);
}

template<typename T>
Array<T> Array<T>::to_host(bool registered) const
{
    int dst_flags = registered ? af_rhost : af_uhost;
    return this->on_host() ? (*this) : this->clone(dst_flags);
}

template<typename T>
int Array<T>::get_ncontig() const
{
    return compute_ncontig(ndim, shape, strides);
}

template<typename T>
T& Array<T>::_at(int nd, const long *ix) const
{
    xassert(on_host());
    xassert(this->ndim == nd);
    
    long pos = 0;
    for (int d = 0; d < nd; d++) {
	xassert(ix[d] >= 0 && ix[d] < shape[d]);
	pos += ix[d] * strides[d];
    }
    
    return data[pos];
}

template<typename T> T& Array<T>::at(int nd, const long *ix)          { return _at(nd, ix); }
template<typename T> T& Array<T>::at(const std::vector<long> &ix)     { return _at(ix.size(), &ix[0]); }
template<typename T> T& Array<T>::at(std::initializer_list<long> ix)  { return _at(ix.size(), ix.begin()); }

template<typename T> const T& Array<T>::at(int nd, const long *ix) const          { return _at(nd, ix); }
template<typename T> const T& Array<T>::at(const std::vector<long> &ix) const     { return _at(ix.size(), &ix[0]); }
template<typename T> const T& Array<T>::at(std::initializer_list<long> ix) const  { return _at(ix.size(), ix.begin()); }


template<typename T>
Array<T> Array<T>::slice(int axis, int ix) const
{
    xassert((axis >= 0) && (axis < ndim));
    xassert((ix >= 0) && (ix < shape[axis]));

    // Slicing (1-dim -> 0-dim) doesn't make sense,
    // since our zero-dimensional Arrays are empty.
    xassert(ndim > 1);
    
    Array<T> ret;
    ret.ndim = ndim-1;
    ret.aflags = aflags & af_location_flags;
    ret.base = base;
    ret.size = 1;

// Suppress spurious GCC warning in loop that follows.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
    
    for (int i = 0; i < ndim-1; i++) {
	int j = (i < axis) ? i : (i+1);
	ret.shape[i] = shape[j];
	ret.strides[i] = strides[j];
	ret.size *= shape[j];
    }
    
#pragma GCC diagnostic pop

    if (ret.size == 0) {
	ret.data = nullptr;
	return ret;
    }

    ret.data = data + (ix * strides[axis]);    
    return ret;
}


template<typename T>
Array<T> Array<T>::slice(int axis, int start, int stop) const
{
    // Currently we allow slices like arr[2:10] but not arr[2:-10] or arr[2:10:2].
    // This would be easy to generalize!
    xassert((axis >= 0) && (axis < ndim));
    xassert((start >= 0) && (start <= stop) && (stop <= shape[axis]));
    
    Array<T> ret;
    ret.ndim = ndim;
    ret.aflags = aflags & af_location_flags;
    ret.base = base;
    ret.size = 1;

    for (int i = 0; i < ndim; i++) {
	ret.shape[i] = (i == axis) ? (stop-start) : shape[i];
	ret.strides[i] = strides[i];
	ret.size *= ret.shape[i];
    }

    if (ret.size == 0) {
	ret.data = nullptr;
	return ret;
    }

    ret.data = data + (start * strides[axis]);
    return ret;
}


template<typename T>
Array<T> Array<T>::reshape_ref(int ndim_, const long *shape_) const
{
    xassert(ndim_ >= 0);
    xassert(ndim_ <= ArrayMaxDim);
	   
    Array<T> ret;
    ret.ndim = ndim_;
    ret.data = this->data;
    ret.size = this->size;
    ret.base = this->base;
    ret.aflags = this->aflags;
    
    for (int d = 0; d < ret.ndim; d++)
	ret.shape[d] = shape_[d];

    // reshape_ref_helper()
    //  - assumes src_ndim, src_shape, src_strides, dst_ndim have been validated by caller
    //  - validates dst_shape
    //  - initializes dst_strides
    //  - throws exception if shapes are incompatible, or src_strides are bad.
    reshape_ref_helper(ndim, shape, strides, ret.ndim, ret.shape, ret.strides);
    
    return ret;
}

template<typename T>
inline Array<T> Array<T>::reshape_ref(const std::vector<long> &shape) const
{
    return reshape_ref(shape.size(), &shape[0]);
}

template<typename T>
inline Array<T> Array<T>::reshape_ref(std::initializer_list<long> shape) const
{
    return reshape_ref(shape.size(), shape.begin());
}


template<typename T>
std::vector<long> Array<T>::ix_start() const
{
    return std::vector<long> (ndim, 0);
}

template<typename T>
bool Array<T>::ix_valid(const std::vector<long> &ix) const
{
    // Warning: ix_valid() is not a general-purpose index validator!
    // Only works on the output of ix_start() -> ix_next() -> ...
    return (size > 0) && (ix[0] < shape[0]);
}

template<typename T>
void Array<T>::ix_next(std::vector<long> &ix) const
{
    for (int d = ndim-1; d >= 1; d--) {
	if (ix[d] < shape[d]-1) {
	    ix[d]++;
	    return;
	}
	ix[d] = 0;
    }
    
    if (ndim > 0)
	ix[0]++;
}


template<typename T>
bool Array<T>::shape_equals(int ndim_, const long *shape_) const
{
    return shape_eq(this->ndim, this->shape, ndim_, shape_);
}

template<typename T>
bool Array<T>::shape_equals(const std::vector<long> &shape) const
{
    return shape_eq(this->ndim, this->shape, shape.size(), &shape[0]);
}

template<typename T>
bool Array<T>::shape_equals(std::initializer_list<long> shape_) const
{
    return shape_eq(this->ndim, this->shape, shape_.size(), shape_.begin());
}

template<typename T> template<typename T2>
bool Array<T>::shape_equals(const Array<T2> &a) const
{
    return shape_eq(this->ndim, this->shape, a.ndim, a.shape);
}

template<typename T> std::string Array<T>::shape_str() const
{
    return ::ksgpu::shape_str(ndim, shape);
}

template<typename T> std::string Array<T>::stride_str() const
{
    return ::ksgpu::shape_str(ndim, strides);
}

template<typename T> void Array<T>::check_invariants() const
{
    check_array_invariants(data, ndim, shape, size, strides, aflags);
}


// -------------------------------------------------------------------------------------------------
//
// Array<T>::convert_dtype()


// Default dtype converter (just use C++ type conversion)
template<typename Tdst, typename Tsrc>
struct DtypeConverter
{
    static inline Tdst convert(Tsrc x) { return x; }
};

// Dtype conversion float -> __half
template<> struct DtypeConverter<__half, float>
{
    static inline __half convert(float x) { return __float2half(x); }
};

// Dtype conversion double -> __half
template<> struct DtypeConverter<__half, double>
{
    static inline __half convert(double x) { return __double2half(x); }
};


// Dtype conversion __half -> float
template<> struct DtypeConverter<float, __half>
{
    static inline float convert(__half x) { return __half2float(x); }
};

// Dtype conversion __half -> double
template<> struct DtypeConverter<double, __half>
{
    // CUDA doesn't define __half2double()
    static inline double convert(__half x) { return __half2float(x); }
};


template<typename Tdst, typename Tsrc>
inline void convert_dtype_helper(Tdst *dst,
				 const Tsrc *src,
				 int nouter_dims,
				 const long *outer_shape,			    
				 const long *outer_dst_strides,
				 const long *outer_src_strides,
				 long nelts_contig)
{
    if (nouter_dims > 0) {
	for (int i = 0; i < outer_shape[0]; i++) {
	    convert_dtype_helper(dst + i * outer_dst_strides[0],
				 src + i * outer_src_strides[0],
				 nouter_dims - 1,
				 outer_shape + 1,
				 outer_dst_strides + 1,
				 outer_src_strides + 1,
				 nelts_contig);
	}
    }
    else {
	for (int i = 0; i < nelts_contig; i++)
	    dst[i] = DtypeConverter<Tdst,Tsrc>::convert(src[i]);
    }
}


template<typename Tsrc> template<typename Tdst>
inline Array<Tdst> Array<Tsrc>::convert_dtype(int aflags) const
{
    xassert(on_host());           // src array must be on host
    xassert(af_on_host(aflags));  // dst array must be on host
    Array<Tdst> dst(ndim, shape, aflags);

    int ncontig = get_ncontig();
    int nouter = ndim - ncontig;

    long nelts_contig = 1;
    for (int i = nouter; i < ndim; i++)
	nelts_contig *= shape[i];

    convert_dtype_helper(dst.data, this->data, nouter, shape, dst.strides, strides, nelts_contig);
    return dst;
}


template<typename Tsrc> template<typename Tdst>
inline Array<Tdst> Array<Tsrc>::convert_dtype() const
{
    return this->convert_dtype<Tdst> (aflags & af_location_flags);
}

    
} // namespace ksgpu

#endif // _KSGPU_ARRAY_HPP
