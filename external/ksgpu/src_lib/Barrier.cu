#include "../include/ksgpu/Barrier.hpp"
#include "../include/ksgpu/xassert.hpp"

#include <stdexcept>

using namespace std;


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


Barrier::Barrier(int nthreads_)
{
    xassert_msg(nthreads_ > 0, "Barrier constructor: expected nthreads > 0");
    this->nthreads = nthreads_;
}


void Barrier::initialize(int nthreads_)
{
    xassert(nthreads_ > 0);
    std::unique_lock ul(lock);

    xassert_msg(this->nthreads <= 0, "Barrier::initialize() called on already-initialized Barrier");
    xassert_msg(!this->aborted, this->abort_msg);
    this->nthreads = nthreads_;
}


void Barrier::wait()
{
    std::unique_lock ul(lock);

    xassert_msg(nthreads > 0, "Barrier::wait() called on uninitialized Barrier");
    xassert_msg(!aborted, abort_msg);

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


} // namespace ksgpu
