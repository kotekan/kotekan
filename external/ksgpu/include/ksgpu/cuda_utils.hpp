#ifndef _KSGPU_CUDA_UTILS_HPP
#define _KSGPU_CUDA_UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>


// Note: CUDA_CALL(), CUDA_PEEK(), and CUDA_CALL_ABORT() are implemented with #define,
// and therefore are outside the ksgpu namespace.

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Misc utilities


extern void assign_kernel_dims(dim3 &nblocks, dim3 &nthreads, long nx, long ny, long nz, int threads_per_block=128, bool noisy=false);

extern double get_sm_cycles_per_second(int device=0);


// -------------------------------------------   Macros   ------------------------------------------
//
// CUDA_CALL(f()): wrapper for CUDA API calls which return cudaError_t.
//
// CUDA_PEEK("label"): throws exception if (cudaPeekAtLastError() != cudaSuccess).
// The argument is a string label which appears in the error message.
// Can be used anywhere, but intended to be called immediately after kernel launches.
//
// CUDA_CALL_ABORT(f()): infrequently-used version of CUDA_CALL() which aborts instead
// of throwing an exception (for use in contexts where exception-throwing is not allowed,
// e.g. shared_ptr deleters).
//
// Example:
//
//    CUDA_CALL(cudaMalloc(&ptr, size));
//    mykernel<<B,T>> (ptr);
//    CUDA_PEEK("mykernel launch")


// Branch predictor hint
#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif

#define CUDA_CALL(x) _CUDA_CALL(x, __STRING(x), __FILE__, __LINE__)
#define CUDA_PEEK(x) _CUDA_CALL(cudaPeekAtLastError(), x, __FILE__, __LINE__)
#define CUDA_CALL_ABORT(x) _CUDA_CALL_ABORT(x, __STRING(x), __FILE__, __LINE__)

#define _CUDA_CALL(x, xstr, file, line) \
    do { \
        cudaError_t xerr = (x); \
        if (_unlikely(xerr != cudaSuccess)) \
            throw ::ksgpu::make_cuda_exception(xerr, xstr, file, line); \
    } while (0)

#define _CUDA_CALL_ABORT(x, xstr, file, line) \
    do { \
        cudaError_t xerr = (x); \
        if (_unlikely(xerr != cudaSuccess)) { \
            fprintf(stderr, "CUDA call '%s' returned %d (%s) [%s:%d]\n", \
                    xstr, int(xerr), cudaGetErrorString(xerr), file, line); \
            exit(1); \
        } \
    } while (0)

// Helper for CUDA_CALL().
std::runtime_error make_cuda_exception(cudaError_t xerr, const char *xstr, const char *file, int line);


// -----------------------------  RAII wrapper for cudaSetDevice()  --------------------------------
//
// Constructor sets current cuda device to 'new_dev'.
// Destructor restores the original cuda_device (at the time the constructor was called).
// If new_dev < 0, then the constructor/destructor will no-op.


struct CudaSetDevice {
    const int new_dev;
    int old_dev = -1;

    // Noncopyable.
    CudaSetDevice() = delete;
    CudaSetDevice(const CudaSetDevice &) = delete;
    CudaSetDevice& operator=(const CudaSetDevice &) = delete;

    CudaSetDevice(int new_dev_) : new_dev(new_dev_)
    {
        if (new_dev < 0)
            return;

        // Save current device in 'old_dev'.
        // Note: we use CUDA_CALL_ABORT() instead of CUDA_CALL(), since CudaSetDevice is
        // used in a context (shared_ptr deleter) where it is not safe to throw an exception.
        CUDA_CALL_ABORT(cudaGetDevice(&old_dev));
        
        if (old_dev != new_dev)
            CUDA_CALL_ABORT(cudaSetDevice(new_dev));
    }

    ~CudaSetDevice()
    {
        if ((old_dev >= 0) && (new_dev >= 0) && (old_dev != new_dev))
            CUDA_CALL_ABORT(cudaSetDevice(old_dev));
    }
};


// ------------------------------  RAII wrapper for cudaEvent_t  -----------------------------------
//
// Reminder: cuda defines the following flags:
//
//   cudaEventDefault = 0
//   cudaEventBlockingSync: callers of cudaEventSynchronize() will block instead of busy-waiting
//   cudaEventDisableTiming: event does not need to record timing data
//   cudaEventInterprocess: event may be used as an interprocess event by cudaIpcGetEventHandle()
//
// I felt compelled to define a different set of flags!
//
//   - ev_time: enables timing (I thought that disabled-timing should be the default)
//
//   - ev_spin, ev_block: selects between busy-wait and blocking. (Here, I wanted to avoid
//      assigning a default, since there are situations where one or the other strongly
//      makes sense, and specifying the wrong one would be a performance-killer.)
//
// Usage reminder:
//   CUDA_CALL(cudaEventRecord(event, stream));   // submits event to stream
//   CUDA_CALL(cudaEventSynchronize(event));      // waits for event


static constexpr int ev_time = 0x100;
static constexpr int ev_spin = 0x200;
static constexpr int ev_block = 0x400;


struct CudaEventWrapper {
    // Reminder: cudaEvent_t is a typedef for (CUevent_st *)
    std::shared_ptr<CUevent_st> p;

    // Constructor flags:
    //   cudaEventDefault = 0
    //   cudaEventBlockingSync: callers of cudaEventSynchronize() will block instead of busy-waiting
    //   cudaEventDisableTiming: event does not need to record timing data
    //   cudaEventInterprocess: event may be used as an interprocess event by cudaIpcGetEventHandle()
    
    CudaEventWrapper() { }

    static CudaEventWrapper create(int ev_flags)
    {
        CudaEventWrapper ret;
        cudaEvent_t e = nullptr;
        CUDA_CALL(cudaEventCreateWithFlags(&e, _cuda_flags_from_ev_flags(ev_flags)));
        ret.p = std::shared_ptr<CUevent_st> (e, cudaEventDestroy);
        return ret;
    }

    static std::vector<CudaEventWrapper> create_vector(long size, int ev_flags)
    {
        if (_unlikely(size < 0))
            throw std::runtime_error("CudaStreamWrapper::create_vector() called with size < 0");

        std::vector<CudaEventWrapper> v(size);
        for (long i = 0; i < size; i++)
            v[i] = CudaEventWrapper::create(ev_flags);
        return v;
    }

    // A CudaEventWrapper can be used anywhere a cudaEvent_t can be used
    // (e.g. in a kernel launch, or elsewhere in the CUDA API), via this
    // conversion operator.
    
    operator cudaEvent_t() const { return p.get(); }


    static uint _cuda_flags_from_ev_flags(int ev_flags)
    {
        bool spin = (ev_flags & ev_spin) != 0;
        bool block = (ev_flags & ev_block) != 0;

        if (_unlikely(ev_flags & ~(ev_time | ev_spin | ev_block)))
            throw std::runtime_error("CudaEventWrapper: invalid flags");
        if (_unlikely(spin == block))
            throw std::runtime_error("CudaEventWrapper: invalid flags: precisely one of ev_spin or ev_block must be specified");

        uint cuda_flags = block ? cudaEventBlockingSync : 0;

        if (!(ev_flags & ev_time))
            cuda_flags |= cudaEventDisableTiming;

        return cuda_flags;
    }
};


// ------------------------------  RAII wrapper for cudaStream_t  ----------------------------------


class CudaStreamWrapper {
public:
    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    std::shared_ptr<CUstream_st> p;

    // Note: default constructor makes an empty shared_ptr, representing the default stream.
    // To get a new stream, use CudaStreamWrapper::create().

    CudaStreamWrapper() { }

    // Create cudaStream with priority. CUDA priorities follow a convention where lower numbers represent
    // higher priorities. '0' represents default priority. The range of meaningful numerical priorities can
    // be queried using cudaDeviceGetStreamPriorityRange(). On an A40, the allowed range is [-5,0].
    
    static CudaStreamWrapper create(int priority=0)
    {
        CudaStreamWrapper ret;
        cudaStream_t s = nullptr;
        CUDA_CALL(cudaStreamCreateWithPriority(&s, cudaStreamDefault, priority));
        ret.p = std::shared_ptr<CUstream_st> (s, cudaStreamDestroy);
        return ret;
    }

    static std::vector<CudaStreamWrapper> create_vector(long size, int priority=0)
    {
        if (_unlikely(size < 0))
            throw std::runtime_error("CudaStreamWrapper::create_vector() called with size < 0");

        std::vector<CudaStreamWrapper> v(size);
        for (long i = 0; i < size; i++)
            v[i] = CudaStreamWrapper::create(priority);
        return v;
    }

    // A CudaStreamWrapper can be used anywhere a cudaStream_t can be used
    // (e.g. in a kernel launch, or elsewhere in the CUDA API), via this
    // conversion operator.
    
    operator cudaStream_t() const { return p.get(); }

    // Commonly occuring sequence (EventCreate) -> (EventRecord)
    CudaEventWrapper record_event(int ev_flags)
    {
        CudaEventWrapper e = CudaEventWrapper::create(ev_flags);
        CUDA_CALL(cudaEventRecord(e, p.get()));
        return e;
    }

    // Commonly occuring sequence (EventCreate) -> (EventRecord) -> (StreamWaitEvent) -> (EventDestroy)
    void synchronize(cudaStream_t src_stream)
    {
        cudaEvent_t e = nullptr;
        CUDA_CALL(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        std::unique_ptr<CUevent_st, decltype(&cudaEventDestroy)> eref(e, cudaEventDestroy);

        CUDA_CALL(cudaEventRecord(e, src_stream));
        CUDA_CALL(cudaStreamWaitEvent(p.get(), e, 0));
        // Note: the event is destroyed when synchronize() returns, even if the
        // stream has not "consumed" the event. This is okay since cuda has its
        // own reference-counting mechanism.
    }
};


} // namespace ksgpu


#endif // _KSGPU_CUDA_UTILS_HPP
