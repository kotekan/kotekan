#include "../include/ksgpu/CpuThreadPool.hpp"
#include "../include/ksgpu/time_utils.hpp"    // get_time(), time_since()
#include "../include/ksgpu/xassert.hpp"

#include <thread>
#include <iostream>

using namespace std;

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


CpuThreadPool::CpuThreadPool(const callback_t &callback_, int nthreads_, int max_callbacks_per_thread_, const string &name_)
    : callback(callback_), nthreads(nthreads_), max_callbacks_per_thread(max_callbacks_per_thread_), name(name_)
{
    xassert(nthreads > 0);
    xassert(nthreads <= 256);
    xassert(max_callbacks_per_thread >= 0);
}

void CpuThreadPool::run()
{
    vector<std::thread> threads(nthreads);
    
    unique_lock ulock(main_lock);
    if (is_started)
	throw runtime_error("CpuThreadPool::run() called twice");

    start_time = get_time();
    is_started = true;
    ulock.unlock();

    for (int i = 0; i < nthreads; i++)
	threads[i] = std::thread(worker_thread_body, this, i);
    
    for (int i = 0; i < nthreads; i++)
	threads[i].join();
}

void CpuThreadPool::monitor_throughput(const string &label, double coeff)
{
    _add_timing_monitor(label, coeff, true);
}

void CpuThreadPool::monitor_time(const string &label, double coeff)
{
    _add_timing_monitor(label, coeff, false);
}

// Called without any locks held.
void CpuThreadPool::show_timings()
{
    unique_lock<mutex> ul(main_lock);
    int n = num_callbacks;
    double t = elapsed_time;
    ul.unlock();
    
    lock_guard<mutex> lg(tm_lock);
    _show_timings(n, t);
}



void CpuThreadPool::worker_thread_body(CpuThreadPool *pool, int ithread)
{
    int nthreads = pool->nthreads;
    int max_callbacks = nthreads * pool->max_callbacks_per_thread;
    auto callback = pool->callback;
    
    unique_lock<mutex> ul(pool->main_lock);
    auto start_time = pool->start_time;
    ul.unlock();
    
    for (;;) {
	// Run callback without lock held.
	pool->callback(*pool, ithread);
	double t = time_since(start_time);

	ul.lock();
	int n = pool->num_callbacks;
	
	if ((max_callbacks > 0) && (n >= max_callbacks))
	    return;

	n++;
	pool->num_callbacks = n;
	pool->elapsed_time = t;
	
	ul.unlock();

	if (n % nthreads)
	    continue;
	
	unique_lock<mutex> tl(pool->tm_lock);
	
	if ((n == max_callbacks) || (pool->timing_monitors.size() > 0))
	    pool->_show_timings(n, t);
    }
}


void CpuThreadPool::_add_timing_monitor(const string &label, double coeff, bool thrflag)
{
    TimingMonitor tm;
    tm.label = label;
    tm.coeff = coeff;
    tm.thrflag = thrflag;

    lock_guard<mutex> lg(tm_lock);
    timing_monitors.push_back(tm);
}


// Called with tm_lock held
void CpuThreadPool::_show_timings(int ncb, double ttot)
{
    int max_callbacks = nthreads * max_callbacks_per_thread;
	
    TimingMonitor tm_default;
    tm_default.label = "callbacks/sec";
    tm_default.coeff = 1.0;
    tm_default.thrflag = true;

    double t = (ncb > 0) ? (ttot/ncb) : 0.0;
    double trec = (ttot > 0.0) ? (ncb/ttot) : 0.0;
    
    int ntm = timing_monitors.size();
    const TimingMonitor *tm = &timing_monitors[0];

    if (ntm == 0) {
	ntm = 1;
	tm = &tm_default;
    }

    if ((max_callbacks == 0) || (ncb < max_callbacks))
	cout << "    ";

    cout << name << " [" << ncb;

    if (max_callbacks > 0)
	cout << "/" << max_callbacks;

    cout << "]";

    for (int i = 0; i < ntm; i++) {
	double x = tm[i].thrflag ? (tm[i].coeff * trec) : (t / tm[i].coeff);
	cout << ((i > 0) ? ", " : ": ") << tm[i].label << " = " << x;
    }

    cout << "\n";
}


}  // namespace ksgpu
