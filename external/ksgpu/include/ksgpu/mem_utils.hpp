#ifndef _KSGPU_MEM_UTILS_HPP
#define _KSGPU_MEM_UTILS_HPP

#include <string>
#include <memory>
#include <type_traits>

#include "rand_utils.hpp"     // randomize()
#include "xassert.hpp"

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Core functions, defined later in this file ("Implementation" below)


// See below for a complete list of flags.
template<typename T>
inline std::shared_ptr<T> af_alloc(long nelts, int flags);


template<typename T>
inline void af_copy(T *dst, int dst_flags, const T *src, int src_flags, long nelts);


template<typename T>
inline std::shared_ptr<T> af_clone(int dst_flags, const T *src, long nelts);


// -------------------------------------------------------------------------------------------------
//
// Flags for use in af_alloc().
// Note: anticipate refining 'af_unified', to toggle cudaMemAttachHost vs cudaMemAttachGlobal.
// Note: anticipate refining 'af_rhost' (e.g. cudaHostAllocWriteCombined).


// Location flags: where is memory allocated?
// Precisely one of these should be specified.
static constexpr int af_gpu = 0x01;      // gpu memory
static constexpr int af_uhost = 0x02;    // host memory, not registered with cuda runtime or page-locked
static constexpr int af_rhost = 0x04;    // host memory, registered with cuda runtime and page-locked
static constexpr int af_unified = 0x08;  // unified host+gpu memory (slow and never useful?)
static constexpr int af_location_flags = af_gpu | af_uhost | af_rhost | af_unified;

// Initialization flags
static constexpr int af_zero = 0x10;    // zero allocated memory
static constexpr int af_random = 0x20;  // randomize allocated memory
static constexpr int af_initialization_flags = af_zero | af_random;

// Mmap flags: if specified (with af_uhost or af_rhost), then mmap/munmap will be used instead of malloc/free.
static constexpr int af_mmap_small = 0x100;      // 4KB standard pages
static constexpr int af_mmap_huge = 0x200;       // 2MB huge pages
static constexpr int af_mmap_try_huge = 0x400;   // try huge pages, if fails then fall back to 4K pages, and print warning.
static constexpr int af_mmap_flags = af_mmap_small | af_mmap_huge | af_mmap_try_huge;

// Debugging flags
static constexpr int af_guard = 0x1000;     // creates "guard" region before/after allocated memory
static constexpr int af_verbose = 0x2000;   // prints messages on alloc/free
static constexpr int af_debug_flags = af_guard | af_verbose;
static constexpr int af_all_flags = af_location_flags | af_initialization_flags | af_mmap_flags | af_debug_flags;


// Throws exception if aflags are uninitialized or invalid.
extern void check_aflags(int aflags, const char *where = nullptr);

// Utility function for printing flags.
extern std::string aflag_str(int flags);

// Is memory addressable on GPU? On host?
inline bool af_on_gpu(int flags) { return (flags & (af_gpu | af_unified)) != 0; }
inline bool af_on_host(int flags) { return (flags & af_gpu) == 0; }


// -------------------------------------------------------------------------------------------------
//
// Implementation.


// Handles all flags except 'af_random'.
extern std::shared_ptr<void> _af_alloc(long nbytes, int flags);

// Uses location flags, but ignores initialization and debug flags.
extern void _af_copy(void *dst, int dst_flags, const void *src, int src_flags, long nbytes);


template<typename T>
inline std::shared_ptr<T> af_alloc(long nelts, int flags)
{
    // FIXME should have some static_asserts here, to ensure
    // that 'T' doesn't have constructors/destructors.

    xassert(nelts >= 0);
    long nbytes = nelts * sizeof(T);

    // _af_alloc() handles all flags except 'af_random'.
    std::shared_ptr<T> ret = std::reinterpret_pointer_cast<T> (_af_alloc(nbytes, flags));

    if (!(flags & af_random))
	return ret;

    // FIXME should use "if constexpr" for more graceful handling
    // of non-integral and non-floating-point types.
        
    if (!(flags & af_gpu)) {
	randomize(ret.get(), nelts);
	return ret;
    }

    // FIXME slow, memory-intensive way of randomizing array on GPU, by randomizing on CPU
    // and copying. It would be better to launch a kernel to randomize directly on GPU.

    int src_flags = af_rhost;
    std::shared_ptr<T> host = std::reinterpret_pointer_cast<T> (_af_alloc(nbytes, src_flags));
    randomize(host.get(), nelts);
    af_copy(ret.get(), flags, host.get(), src_flags, nelts);
    return ret;
}


template<typename T>
inline void af_copy(T *dst, int dst_flags, const T *src, int src_flags, long nelts)
{
    // FIXME should have some static_asserts here, to ensure
    // that 'T' doesn't have constructors/destructors.

    xassert(nelts >= 0);
    long nbytes = nelts * sizeof(T);
    
    _af_copy(dst, dst_flags, src, src_flags, nbytes);
}


template<typename T>
inline std::shared_ptr<T> af_clone(int dst_flags, const T *src, long nelts)
{
    dst_flags &= ~af_initialization_flags;
    std::shared_ptr<T> ret = af_alloc<T> (nelts, dst_flags);
    af_copy(ret.get(), src, nelts);
}


} // namespace ksgpu

#endif  // _KSGPU_MEM_UTILS_HPP
