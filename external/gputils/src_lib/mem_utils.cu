#include <sstream>
#include <iostream>

#include "../include/gputils/mem_utils.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/string_utils.hpp"          // nbytes_to_str()
#include "../include/gputils/constexpr_functions.hpp"   // constexpr_is_pow2()


using namespace std;

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif

    
// Helper for check_aflags().
inline bool multiple_bits(int x)
{
    return (x & (x-1)) != 0;
}

inline bool non_single_bit(int x)
{
    return (x==0) || multiple_bits(x);
}

void check_aflags(int flags, const char *where)
{
    if (!where)
	where = "gputils::check_aflags()";

    if (_unlikely(flags == 0))
	throw runtime_error(string(where) + ": af_flags==0 (probably uninitialized)");
    
    if (_unlikely(flags & ~af_all_flags))
	throw runtime_error(string(where) + ": unrecognized af_flags were specified");

    if (_unlikely(non_single_bit(flags & af_location_flags)))
	throw runtime_error(string(where) + ": must specify precisely one of " + aflag_str(af_location_flags));

    if (_unlikely(multiple_bits(flags & af_initialization_flags)))
	throw runtime_error(string(where) + ": can specify at most one of " + aflag_str(af_initialization_flags));

    if (_unlikely(multiple_bits(flags & af_mmap_flags)))
	throw runtime_error(string(where) + ": can specify at most one of " + aflag_str(af_mmap_flags));

    if (_unlikely((flags & af_mmap_flags) && !(flags & (af_uhost | af_rhost))))
	throw runtime_error(string(where) + ": if af_mmap_* flag is specified, then one of {af_uhost,af_rhost} must also be specified");
}

// Helper for aflag_str().
inline void _aflag_str(stringstream &ss, int &count, bool pred, const char *name)
{
    if (!pred)
	return;
    if (count > 0)
	ss << " | ";
    ss << name;
    count++;
}

string aflag_str(int flags)
{
    stringstream ss;
    int count = 0;
    
    _aflag_str(ss, count, flags & af_gpu, "af_gpu");
    _aflag_str(ss, count, flags & af_gpu, "af_uhost");
    _aflag_str(ss, count, flags & af_gpu, "af_rhost");
    _aflag_str(ss, count, flags & af_unified, "af_unified");    
    _aflag_str(ss, count, flags & af_zero, "af_zero");
    _aflag_str(ss, count, flags & af_random, "af_random");
    _aflag_str(ss, count, flags & af_mmap_small, "af_mmap_small");
    _aflag_str(ss, count, flags & af_mmap_huge, "af_mmap_huge");
    _aflag_str(ss, count, flags & af_mmap_try_huge, "af_mmap_try_huge");
    _aflag_str(ss, count, flags & af_guard, "af_guard");
    _aflag_str(ss, count, flags & af_verbose, "af_verbose");
    _aflag_str(ss, count, flags & ~af_all_flags, "(unrecognized flags)");

    if (count == 0)
	return "0";
    if (count == 1)
	return ss.str();
    return "(" + ss.str() + ")";
}


// -------------------------------------------------------------------------------------------------


struct alloc_helper {
    static constexpr ssize_t nguard = 4096;

    const ssize_t nbytes_requested;
    const int flags;

    ssize_t nbytes_allocated;
    char *base = nullptr;    
    char *data = nullptr;
    char *gcopy = nullptr;
    

    string mmap_error_message(ssize_t nbytes, bool hugepage_flag)
    {
	stringstream ss;
	
	ss << "mmap(" << nbytes_to_str(nbytes)
	   << (hugepage_flag ? ", MAP_HUGETLB" : "")
	   << ") failed: " << strerror(errno)
	   << (hugepage_flag ? ". Try this: echo [NNN] > /proc/sys/vm/nr_hugepages" : "");

	return ss.str();
    }

    
    // _mmap(): helper method called by constructor.
    // Only called if 'flags' contains an af_mmap_* flag, and check_aflags(flags) passes.
    // Initializes this->base, this->nbytes_allocated (including padding).
    // Does not call cudaRegisterMemory() -- that's done by the caller!
    
    inline void _mmap(ssize_t nbytes_unpadded)
    {
	static constexpr int page_size = 4 * 1024;
	static constexpr int hugepage_size = 2 * 1024 * 1024;
	
	int pflags = PROT_READ | PROT_WRITE;
	int mflags = MAP_PRIVATE | MAP_ANONYMOUS;  // no MAP_HUGETLB

	assert(flags & af_mmap_flags);
	
	if (flags & (af_mmap_huge | af_mmap_try_huge)) {
	    static_assert(constexpr_is_pow2(hugepage_size));
	    static constexpr ssize_t mask = hugepage_size - 1;
	    
	    this->nbytes_allocated = (nbytes_unpadded + mask) & (~mask);
	    this->base = (char *) mmap(NULL, nbytes_allocated, pflags, mflags | MAP_HUGETLB, -1, 0);  // note MAP_HUGETLB
	    
	    if (base != MAP_FAILED)
		return;

	    if (flags & af_mmap_try_huge) {
		// Temporarily buffer in stringstream, to reduce probability of interleaved output in multithreaded programs.
		stringstream ss;
		ss << "Warning: " << mmap_error_message(nbytes_allocated,true) << "\n";
		cout << ss.str() << flush;
	    }
	}

	if (flags & (af_mmap_small | af_mmap_try_huge)) {
	    static_assert(constexpr_is_pow2(page_size));
	    static constexpr ssize_t mask = page_size - 1;
	    
	    this->nbytes_allocated = (nbytes_unpadded + mask) & (~mask);
	    this->base = (char *) mmap(NULL, nbytes_allocated, pflags, mflags, -1, 0);   // note no MAP_HUGETLB
	    
	    if (base != MAP_FAILED)
		return;
	}

	bool hugepage_flag = flags & af_mmap_huge;
	string err_msg = mmap_error_message(nbytes_allocated, hugepage_flag);
	throw runtime_error(err_msg);
    }


    // Helper for constructor
    void fill(void *dst, const void *src, ssize_t nbytes)
    {
	if (flags & af_gpu)
	    CUDA_CALL(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
	else
	    memcpy(dst, src, nbytes);
    }
    
    
    alloc_helper(ssize_t nbytes, int flags_) :
	nbytes_requested(nbytes), flags(flags_)
    {
	check_aflags(flags, "af_alloc");
	
	ssize_t g = (flags & af_guard) ? nguard : 0;
	this->nbytes_allocated = nbytes_requested + 2*g;

	// Step 1: allocate memory, initializing this->base, this->nbytes_allocated, this->data.
	// This handles all flags except af_verbose, af_zero, af_guard.
	
	if (flags & af_mmap_flags) {
	    // Note: this->_mmap() initializes this->base, and updates the value of this->nbytes_allocated.
	    this->_mmap(nbytes_allocated);
	    if (flags & af_rhost)
		CUDA_CALL(cudaHostRegister(base, nbytes_allocated, cudaHostRegisterDefault));
	}
	else if (flags & af_gpu)
	    CUDA_CALL(cudaMalloc((void **) &this->base, this->nbytes_allocated));
	else if (flags & af_unified)
	    CUDA_CALL(cudaMallocManaged((void **) &this->base, this->nbytes_allocated, cudaMemAttachGlobal));
	else if (flags & af_rhost)
	    CUDA_CALL(cudaHostAlloc((void **) &this->base, this->nbytes_allocated, 0));
	else if (posix_memalign((void **) &this->base, 128, this->nbytes_allocated))
	    throw std::runtime_error("gputils::alloc(): couldn't allocate " + to_string(nbytes_allocated) + " bytes");
	
	assert(base != nullptr);
	assert(base != MAP_FAILED);
	assert(nbytes_allocated >= nbytes_requested);
	
	this->data = base + g;

	// Step 2: if verbose, announce that memory has been allocated (keep in sync with Step 1 by hand).
	
	if (flags & af_verbose) {
	    // Temporarily buffer in stringstream, to reduce probability of interleaved output in multithreaded programs.
	    stringstream ss;
	    
	    if (flags & af_mmap_flags)
		ss << "mmap";
	    else if (flags & af_gpu)
		ss << "cudaMalloc";
	    else if (flags & af_unified)
		ss << "cudaMallocManaged";
	    else if (flags & af_rhost)
		ss << "cudaHostAlloc";
	    else
		ss << "posix_memalign";
	    
	    ss << "(" << nbytes_requested;
	    if (nbytes_requested < nbytes_allocated)
		ss << " -> " << nbytes_allocated;

	    ss << ")";
	    if ((flags & af_mmap_flags) && (flags & af_rhost))
		ss << " -> cudaHostRegister()";

	    ss << ": " << ((void *) base);
	    if (base != data)
		ss << " [data=" << ((void *) data) << "]";

	    ss << "\n";
	    cout << ss.str() << flush;
	}

	// Step 3: handle af_zero, af_guard.
	
	if ((flags & af_zero) && !(flags & af_mmap_flags)) {
	    // Note: if memory is allocated with mmap(), then it is automatically zeroed.
	    // (At least on Linux, the man page states that MAP_ANONYMOUS mappings are zeroed.)
	    if (flags & af_gpu)
		CUDA_CALL(cudaMemset(data, 0, nbytes));
	    else
		memset(data, 0, nbytes);
	}

	if (flags & af_guard) {
	    this->gcopy = (char *) malloc(2*nguard);
	    assert(gcopy != nullptr);
	    randomize(gcopy, 2*nguard);
	    
	    fill(base, gcopy, nguard);
	    fill(base + nguard + nbytes_requested, gcopy + nguard, nguard);
	}
    }

    
    // check_guard() is called when the shared_ptr is deleted, in the
    // case where one the flags 'af_guard' is specified.
    //
    // Note: we call assert() instead of throwing exceptions, since
    // shared_ptr deleters aren't supposed to throw exceptions.

    void check_guard()
    {
	assert(flags & af_guard);
	assert(gcopy != nullptr);

	if (flags & af_gpu) {
	    char *p = (char *) malloc(2*nguard);
	    assert(p != nullptr);

	    CUDA_CALL_ABORT(cudaMemcpy(p, base, nguard, cudaMemcpyDeviceToHost));
	    CUDA_CALL_ABORT(cudaMemcpy(p + nguard, base + nguard + nbytes_requested, nguard, cudaMemcpyDeviceToHost));

	    // If this fails, buffer overflow occurred.
	    assert(memcmp(p, gcopy, 2*nguard) == 0);
	    free(p);
	}
	else {
	    // If these fail, buffer overflow occurred.
	    assert(memcmp(base, gcopy, nguard) == 0);
	    assert(memcmp(base + nguard + nbytes_requested, gcopy + nguard, nguard) == 0);
	}	
    }
    

    // operator(): called by shared_ptr deleter, works for any combination of af_flags.
    //
    // Note: we call assert() instead of throwing exceptions, since
    // shared_ptr deleters aren't supposed to throw exceptions.
    
    void operator()(void *ptr)
    {
	if (flags & af_guard) {
	    check_guard();
	    free(gcopy);
	}

	// Keep this part in sync with "Deallocate memory" just below.
	if (flags & af_verbose) {
	    // Temporarily buffer in stringstream, to reduce probability of interleaved output in multithreaded programs.
	    stringstream ss;

	    if ((flags & af_mmap_flags) && (flags & af_rhost))
		ss << "cudaHostUnregister() -> munmap";
	    else if (flags & af_mmap_flags)
		ss << "munmap";
	    else if (flags & (af_gpu | af_unified))
		ss << "cudaFree";
	    else if (flags & af_rhost)
		ss << "cudaFreeHost";
	    else
		ss << "free";

	    ss << "(" << ((void *) base) << ")\n";
	    cout << ss.str() << flush;
	}

	// Deallocate memory. Keep this part in sync with "Step 1: allocate memory..." in constructor.
	
	if (flags & af_mmap_flags) {
	    if (flags & af_rhost)
		CUDA_CALL_ABORT(cudaHostUnregister(base));
	    int munmap_err = munmap(base, nbytes_allocated);
	    assert(munmap_err == 0);
	}
	else if (flags & (af_gpu | af_unified))
	    CUDA_CALL_ABORT(cudaFree(base));
	else if (flags & af_rhost)
	    CUDA_CALL_ABORT(cudaFreeHost(base));
	else
	    free(base);
	
	this->nbytes_allocated = 0;
	this->base = this->data = this->gcopy = nullptr;
    }
};


// Handles all flags except 'af_random'
shared_ptr<void> _af_alloc(ssize_t nbytes, int flags)
{
    alloc_helper h(nbytes, flags);

    // Keep this part in sync with "Step 1: allocate memory..."
    // in 'struct alloc_helper' above.
    
    if (flags & (af_guard | af_verbose | af_mmap_flags)) {
	
	// In these cases, shared_ptr deletion logic is complicated enough that
	// we need alloc_helper::operator(). In remaining cases, we can use a
	// simple deleter (cudaFree(), cudaFreeHost(), free()).
	
	return shared_ptr<void> (h.data, h);
    }
    else if (flags & (af_gpu | af_unified))
	return shared_ptr<void> (h.data, cudaFree);
    else if (flags & af_rhost)
	return shared_ptr<void> (h.data, cudaFreeHost);
    else
	return shared_ptr<void> (h.data, free);
}


// -------------------------------------------------------------------------------------------------


void _af_copy(void *dst, int dst_flags, const void *src, int src_flags, ssize_t nbytes)
{
    if (nbytes == 0)
	return;

    bool host_to_host = ((src_flags | dst_flags) & (af_gpu | af_unified)) == 0;
    
    if (host_to_host)
	memcpy(dst, src, nbytes);
    else
	CUDA_CALL(cudaMemcpy(dst, src, nbytes, cudaMemcpyDefault));
}


}  // namespace gputils
