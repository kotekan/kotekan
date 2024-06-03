#include "../include/gputils/Barrier.hpp"

#include <cassert>
#include <stdexcept>

// Branch predictor hint
#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


Barrier::Barrier(int nthreads_)
{
    if (_unlikely(nthreads_ < 0))
	throw runtime_error("Barrier constructor: expected nthreads <= 0");

    this->nthreads = nthreads_;
}


void Barrier::initialize(int nthreads_)
{
    assert(nthreads_ > 0);

    std::unique_lock ul(lock);
    
    if (_unlikely(this->nthreads > 0))
	throw runtime_error("Barrier::initialize() called on already-initialized Barrier");
    
    if (_unlikely(this->aborted))
	throw runtime_error(this->abort_msg);

    this->nthreads = nthreads_;
}


void Barrier::wait()
{
    std::unique_lock ul(lock);

    if (_unlikely(nthreads <= 0))
	throw runtime_error("Barrier::wait() called on uninitialized Barrier");
    
    if (_unlikely(aborted))
	throw runtime_error(abort_msg);

    if (nthreads_waiting == nthreads-1) {
	this->nthreads_waiting = 0;
	this->wait_count++;
	ul.unlock();
	cv.notify_all();
	return;
    }
	
    this->nthreads_waiting++;
    
    int wc = this->wait_count;
    cv.wait(ul, [this,wc] { return (this->aborted || (this->wait_count > wc)); });
    
    if (_unlikely(aborted))
	throw runtime_error(abort_msg);
}


void Barrier::abort(const string &msg)
{
    std::unique_lock ul(lock);
    
    if (_unlikely(aborted))
	return;
    
    this->aborted = true;
    this->abort_msg = msg;
    ul.unlock();
    cv.notify_all();
};


} // namespace gputils
