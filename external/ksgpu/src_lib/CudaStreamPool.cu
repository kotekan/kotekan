#include "../include/ksgpu/CudaStreamPool.hpp"
#include "../include/ksgpu/cuda_utils.hpp"    // CUDA_CALL(), CudaStreamWrapper
#include "../include/ksgpu/time_utils.hpp"    // get_time(), time_since()
#include "../include/ksgpu/xassert.hpp"

#include <thread>
#include <iostream>

using namespace std;

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


CudaStreamPool::CudaStreamPool(const callback_t &callback_, int max_callbacks_, int nstreams_, const string &name_)
    : callback(callback_), max_callbacks(max_callbacks_), nstreams(nstreams_), name(name_)
{
    xassert(max_callbacks >= 0);
    xassert(nstreams > 0);

    CUDA_CALL(cudaGetDevice(&this->cuda_device));
    xassert(cuda_device >= 0);
    
    // Streams will be created by cudaStreamWrapper constructor.
    this->streams.resize(nstreams);
    this->sstate.resize(nstreams);
    
    for (int i = 0; i < nstreams; i++) {
	this->sstate[i].pool = this;
	this->sstate[i].state = 0;
	this->sstate[i].istream = i;
    }
}

void CudaStreamPool::run()
{
    unique_lock ulock(lock);
    
    if (is_started)
	throw runtime_error("CudaStreamPool::run() called twice");
    if (timing_monitors.size() == 0)
	throw runtime_error("CudaStreamPool: run() was called without adding any timing monitors first");
    
    is_started = true;
    ulock.unlock();
    
    std::thread t(manager_thread_body, this);
    t.join();
}

void CudaStreamPool::monitor_throughput(const string &label, double coeff)
{
    _add_timing_monitor(label, coeff, true);
}

void CudaStreamPool::monitor_time(const string &label, double coeff)
{
    _add_timing_monitor(label, coeff, false);
}

void CudaStreamPool::show_timings()
{
    TimingMonitor tm_default;
    tm_default.label = "callbacks/sec";
    tm_default.coeff = 1.0;
    tm_default.thrflag = true;

    lock_guard<mutex> lg(lock);
    xassert(is_started);

    double t = time_per_callback;
    double trec = (t > 0.0) ? (1.0/t) : 0.0;
    
    int ntm = timing_monitors.size();
    const TimingMonitor *tm = &timing_monitors[0];

    if (ntm == 0) {
	ntm = 1;
	tm = &tm_default;
    }

    if (!is_done)
	cout << "    ";

    cout << name << " [" << num_callbacks;

    if (max_callbacks > 0)
	cout << "/" << max_callbacks;

    cout << "]";

    for (int i = 0; i < ntm; i++) {
	double x = tm[i].thrflag ? (tm[i].coeff * trec) : (t / tm[i].coeff);
	cout << ((i > 0) ? ", " : ": ") << tm[i].label << " = " << x;
    }

    cout << "\n";
}

void CudaStreamPool::_add_timing_monitor(const string &label, double coeff, bool thrflag)
{
    TimingMonitor tm;
    tm.label = label;
    tm.coeff = coeff;
    tm.thrflag = thrflag;

    lock_guard<mutex> lg(lock);
    
    xassert(is_done || !is_started);
    timing_monitors.push_back(tm);
}

void CudaStreamPool::manager_thread_body(CudaStreamPool *pool)
{
    auto start_time = get_time();
    unique_lock ulock(pool->lock);

    xassert(pool->cuda_device >= 0);
    CUDA_CALL(cudaSetDevice(pool->cuda_device));
    
    for (;;) {
	bool dropped_lock = false;

	for (int istream = 0; istream < pool->nstreams; istream++) {
	    // At top of loop, lock is held.
	    StreamState &ss = pool->sstate[istream];

	    if (ss.state == 1)  // kernel running on stream
		continue;

	    if (ss.state == 2) {  // kernel finished
		pool->num_callbacks++;
		pool->elapsed_time = time_since(start_time);
		pool->time_per_callback = pool->elapsed_time / pool->num_callbacks;
		pool->is_done = (pool->max_callbacks > 0) && (pool->num_callbacks >= pool->max_callbacks);

		if (pool->is_done || (pool->timing_monitors.size() > 0)) {
		    dropped_lock = true;
		    ulock.unlock();
		    pool->show_timings();
		    ulock.lock();
		}
		
		if (pool->is_done) {
		    ulock.unlock();
		    pool->synchronize();
		    return;   // this is where the manager thread exits!
		}
	    }

	    // Call callback function without holding lock.
	    ulock.unlock();
	    pool->callback(*pool, pool->streams[istream], istream);
	    dropped_lock = true;

	    // Reacquire lock to set state, before queueing cuda_callback.
	    ulock.lock();
	    ss.state = 1;
	    
	    // Queue cuda_callback without holding lock.
	    ulock.unlock();
	    CUDA_CALL(cudaLaunchHostFunc(pool->streams[istream], cuda_callback, &pool->sstate[istream]));

	    // Reacquire lock before proceeding with loop.
	    ulock.lock();
	}

	if (!dropped_lock)
	    pool->cv.wait(ulock);
    }
}

void CudaStreamPool::cuda_callback(void *up)
{
    StreamState *u = reinterpret_cast<StreamState *> (up);
    CudaStreamPool *pool = u->pool;
    int istream = u->istream;

    xassert((istream >= 0) && (istream < pool->nstreams));
    xassert(&pool->sstate[istream] == u);

    unique_lock<mutex> ulock(pool->lock);
    u->state = 2;
    pool->cv.notify_all();
}

void CudaStreamPool::synchronize()
{
    for (int istream = 0; istream < nstreams; istream++)
	CUDA_CALL(cudaStreamSynchronize(streams[istream]));
}


}  // namespace ksgpu
