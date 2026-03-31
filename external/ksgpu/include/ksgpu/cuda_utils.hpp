#ifndef _KSGPU_CUDA_UTILS_HPP
#define _KSGPU_CUDA_UTILS_HPP

#include <memory>
#include <stdexcept>


// Note: CUDA_CALL(), CUDA_PEEK(), and CUDA_CALL_ABORT() are implemented with #define,
// and therefore are outside the ksgpu namespace.

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


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
	    fprintf(stderr, "CUDA call '%s' failed at %s:%d\n", xstr, file, line); \
	    exit(1); \
	} \
    } while (0)

// Helper for CUDA_CALL().
std::runtime_error make_cuda_exception(cudaError_t xerr, const char *xstr, const char *file, int line);


// ------------------------------  RAII wrapper for cudaStream_t  ----------------------------------
//
// Note: you can also get RAII semantics for streams by working with shared_ptrs directly, e.g.
//   shared_ptr<CUstream_st> stream = CudaStreamWrapper().p;


struct CudaStreamWrapper {
    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    std::shared_ptr<CUstream_st> p;

    CudaStreamWrapper()
    {
	cudaStream_t s;
	CUDA_CALL(cudaStreamCreate(&s));
	this->p = std::shared_ptr<CUstream_st> (s, cudaStreamDestroy);
    }

    // Create cudaStream with priority. CUDA priorities follow a convention where lower numbers represent
    // higher priorities. '0' represents default priority. The range of meaningful numerical priorities can
    // be queried using cudaDeviceGetStreamPriorityRange(). On an A40, the allowed range is [-5,0].
    
    CudaStreamWrapper(int priority)
    {
	cudaStream_t s;
	CUDA_CALL(cudaStreamCreateWithPriority(&s, cudaStreamDefault, priority));
	this->p = std::shared_ptr<CUstream_st> (s, cudaStreamDestroy);
    }

    // A CudaStreamWrapper can be used anywhere a cudaStream_t can be used
    // (e.g. in a kernel launch, or elsewhere in the CUDA API), via this
    // conversion operator.
    
    operator cudaStream_t() { return p.get(); }
};


// ------------------------------  RAII wrapper for cudaEvent_t  -----------------------------------
//
// Note: you can also get RAII semantics for events by working with shared_ptrs directly, e.g.
//   shared_ptr<CUevent_st> event = CudaEventWrapper(flags).p;
//
// Usage reminder:
//   CUDA_CALL(cudaEventRecord(event, stream));   // submits event to stream
//   CUDA_CALL(cudaEventSynchronize(event));      // waits for event


struct CudaEventWrapper {
    // Reminder: cudaEvent_t is a typedef for (CUevent_st *)
    std::shared_ptr<CUevent_st> p;

    // Constructor flags:
    //   cudaEventDefault = 0
    //   cudaEventBlockingSync: callers of cudaEventSynchronize() will block instead of busy-waiting
    //   cudaEventDisableTiming: event does not need to record timing data
    //   cudaEventInterprocess: event may be used as an interprocess event by cudaIpcGetEventHandle()
    
    CudaEventWrapper(uint flags = cudaEventDefault)
    {
	cudaEvent_t e;
	CUDA_CALL(cudaEventCreateWithFlags(&e, flags));
	this->p = std::shared_ptr<CUevent_st> (e, cudaEventDestroy);
    }

    // A CudaEventWrapper can be used anywhere a cudaEvent_t can be used
    // (e.g. in a kernel launch, or elsewhere in the CUDA API), via this
    // conversion operator.
    
    operator cudaEvent_t() { return p.get(); }
};


// ---------------------------------------   CudaTimer   -------------------------------------------
//
// A very simple class for timing cuda kernels.
//
// Usage:
//
//   // Timer is running when constructed.
//   CudaTimer t;   // or specify optional stream argument
//
//   // Run one or more kernels.
//   kernel1<<<...>>> (...);
//   kernel2<<<...>>> (...);
//
//   // CudaTimer::stop() synchronizes the stream.
//   float elapsed_time = t.stop();


struct CudaTimer {
protected:
    CudaEventWrapper start;
    CudaEventWrapper end;
    cudaStream_t stream;
    bool running = true;

public:
    CudaTimer(cudaStream_t stream_ = nullptr)
    {
	stream = stream_;
	CUDA_CALL(cudaEventRecord(start, stream));
    }

    float stop()
    {
	if (!running)
	    throw std::runtime_error("double call to CudaTimer::stop()");

	running = false;
	CUDA_CALL(cudaEventRecord(end, stream));
	CUDA_CALL(cudaEventSynchronize(end));

	float milliseconds = 0.0;
	CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, end));
	return milliseconds / 1000.;
    }
};


// -------------------------------------------------------------------------------------------------
//
// Misc


extern void assign_kernel_dims(dim3 &nblocks, dim3 &nthreads, long nx, long ny, long nz, int threads_per_block=128, bool noisy=false);
    
// Implements command-line usage: program [device].
extern void set_device_from_command_line(int argc, char **argv);

extern double get_sm_cycles_per_second(int device=0);


} // namespace ksgpu


#endif // _KSGPU_CUDA_UTILS_HPP
