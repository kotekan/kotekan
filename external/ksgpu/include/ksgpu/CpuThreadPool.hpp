#ifndef _KSGPU_CPUTHREADPOOL_HPP
#define _KSGPU_CPUTHREADPOOL_HPP

#include <string>
#include <vector>
#include <mutex>
#include <functional>
#include <sys/time.h>


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// CpuThreadPool: run multiple CPU threads with dynamic load-balancing, intended for timing
//
// Example:
// 
//   int nthreads = 16;
//   int callbacks_per_thread = 100;
//
//   auto callback = [&](const CpuThreadPool &pool, int ithread)
//       {
//          // do some work
//	 };
//
//   CpuThreadPool tp(callback, nthreads, callbacks_per_thread, "kernel name");
//
//   // Example timing monitor: suppose each callback uses 2 GB of memory BW
//   tp.monitor_throughput("Memory BW (GB/s)", 2.0);
//
//   // Example timing monitor: suppose each callback processes 0.5 sec of real-time data
//   tp.monitor_timing("Real-time fraction", 0.5);
//
//   tp.run();


class CpuThreadPool {
public:
    // callback(pool, ithread)
    using callback_t = std::function<void(const CpuThreadPool &, int)>;

    // If max_callbacks_per_thread=0, then CpuThreadPool.run() will run forever.
    CpuThreadPool(const callback_t &callback, int nthreads,
		  int max_callbacks_per_thread_per_thread=0,
		  const std::string &name="CpuThreadPool");

    // Runs stream pool to completion.
    void run();

    // These functions define "timing monitors".
    // To monitor continuously as stream runs, call between constructor and run().
    // To show once at the end, call after run(), then call show_timings().
    void monitor_throughput(const std::string &label = "callbacks/sec", double coeff=1.0);
    void monitor_time(const std::string &label = "seconds/callback", double coeff=1.0);
    
    // Show all timing monitors.
    void show_timings();
    
protected:
    // Constant after construction, not protected by lock.
    const callback_t callback;
    const int nthreads;
    const int max_callbacks_per_thread;
    const std::string name;
    
    mutable std::mutex main_lock;
    mutable std::mutex tm_lock;
        
    // Protected by main_lock.
    struct timeval start_time;
    double elapsed_time = 0.0;
    int num_callbacks = 0;
    bool is_started = false;
    
    struct TimingMonitor {
	std::string label;
	double coeff;
	bool thrflag;
    };

    // Protected by tm_lock.
    std::vector<TimingMonitor> timing_monitors;

    void _add_timing_monitor(const std::string &label, double coeff, bool thrflag);   // call without tm_lock held
    void _show_timings(int ncb, double ttot);    // call with tm_lock held.
    
    static void worker_thread_body(CpuThreadPool *pool, int ithread);
};
    

} // namespace ksgpu

#endif // _KSGPU_CPUTHREADPOOL_HPP
