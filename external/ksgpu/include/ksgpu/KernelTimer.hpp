#ifndef _KSGPU_KERNEL_TIMER_HPP
#define _KSGPU_KERNEL_TIMER_HPP

#include "cuda_utils.hpp"  // CudaStreamWrapper, CUDA_CALL()

#include <chrono>
#include <string>
#include <vector>


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// KernelTimer: a simple header-only class for timing cuda kernels.
//
// Usage:
//
//   KernelTimer kt(niterations, nstreams);
//
//   while (kt.next()) {}
//       // NOTE: kernels should be launched on "kt.stream".
//       // NOTE: corresponding stream index is "kt.istream" (e.g. for pointer offsets)
//
//       mykernel <<< B, W, 0, kt.stream >>> (base_ptr + kt.istream * offset);
//       CUDA_PEEK("mykernel");
//
//       if (kt.warmed_up)
//           cout << "average time/kernel = " << kt.dt << ", sec" << endl;
//   }


struct KernelTimer {
    // State that the caller will want to reference (see above).
    double dt = 0.0;
    long istream = 0;
    bool warmed_up = false;
    cudaStream_t stream = nullptr;

    // "Internals"

    int niterations = 0;
    int nstreams = 0;
    long curr_iteration = -1;

    std::vector<CudaStreamWrapper> streams;  // length (nstreams)
    std::vector<CudaEventWrapper> events;    // length (nstreams)
    std::vector<double> timestamps;          // length (niterations+1)

    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint start_time;

    KernelTimer(long niterations_, long nstreams_ = 1) :
        niterations(niterations_), nstreams(nstreams_)
    {
        if (nstreams <= 0)
            throw std::runtime_error("KernelTimer constructor called with nstreams <= 0");
        if (niterations < nstreams+1)
            throw std::runtime_error("KernelTimer constructor called with niterations < nstreams+1");

        streams = CudaStreamWrapper::create_vector(nstreams);
        events = CudaEventWrapper::create_vector(nstreams, ev_spin);
        timestamps.resize(niterations+1, 0.0);
    }

    bool next() 
    {
        if (curr_iteration < 0) {
            // Drain any pending kernels, before starting timing run.
            CUDA_CALL(cudaDeviceSynchronize());
            start_time = Clock::now();
        }
        else if (curr_iteration >= niterations)
            throw std::runtime_error("KernelTimer::next() called after timer has completed");

        curr_iteration++;
        istream = curr_iteration % nstreams;
        stream = streams[istream];

        if (curr_iteration >= nstreams) {
            // Busy-wait on event 'istream'.
            for (;;) {
                int status = cudaEventQuery(events[istream]);
                if (status == cudaSuccess)
                    break;
                if (status != cudaErrorNotReady)
                    throw std::runtime_error("cudaEventQuery() failed");
            }
        }

        timestamps.at(curr_iteration) = std::chrono::duration<double>(Clock::now() - start_time).count();

        int it_min = nstreams+1;

        if (curr_iteration > it_min) {
            int i0 = it_min + (curr_iteration-it_min)/2;
            dt = (timestamps.at(curr_iteration) - timestamps.at(i0)) / (curr_iteration-i0);
            warmed_up = true;
        }

        CUDA_CALL(cudaEventRecord(events[istream], stream));
        return (curr_iteration < niterations);
    }
};


} // namespace ksgpu

#endif // _KSGPU_KERNEL_TIMER_HPP
