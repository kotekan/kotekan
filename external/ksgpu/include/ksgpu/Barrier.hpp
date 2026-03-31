#ifndef _KSGPU_BARRIER_HPP
#define _KSGPU_BARRIER_HPP

#include <mutex>
#include <string>
#include <condition_variable>

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// Barrier: synchronization point between N threads

struct Barrier
{        
    std::mutex lock;
    std::condition_variable cv;

    // Protected by lock.
    int nthreads = 0;
    int nthreads_waiting = 0;
    int wait_count = 0;
    bool aborted = false;
    std::string abort_msg;

    // If constructor is called with nthreads=0, then 'nthreads' must be
    // set later, with a call to initialize().
    Barrier(int nthreads);

    void wait();
    void abort(const std::string &msg);
    void initialize(int nthreads);
    
    // Noncopyable
    Barrier(const Barrier &) = delete;
    Barrier &operator=(const Barrier &) = delete;
};


}  // namespace ksgpu

#endif // _KSGPU_BARRIER_HPP
