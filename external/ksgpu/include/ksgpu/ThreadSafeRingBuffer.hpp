#ifndef _KSGPU_THREAD_SAFE_RING_BUFFER_HPP
#define _KSGPU_THREAD_SAFE_RING_BUFFER_HPP

#include <mutex>
#include <vector>
#include <condition_variable>

#include "xassert.hpp"

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// ThreadSafeRingBuffer<T>: A simple ring buffer which allows producer threads to "put" items of type T
// (blocking if ring buffer is full), and consumer threads to "get" items (blocking if ring buffer is empty).
//
// Typical use cases are T=int (representing iterations of a loop), or T=shared_ptr<...> (representing
// work units defined by a more complex data structure)


template<class T>
struct ThreadSafeRingBuffer
{
    const long capacity;

    std::mutex lock;
    std::condition_variable get_notifier;  // this cv gets notified on calls to get()
    std::condition_variable put_notifier;  // this cv gets notified on calls to put() or set_done()

    // Protected by 'lock'.
    std::vector<T> ringbuf;    
    bool is_done = false;
    long ix0 = 0;
    long ix1 = 0;

    
    ThreadSafeRingBuffer(long capacity_) :
	capacity(capacity_)
    {
	xassert(capacity > 0);
	ringbuf.resize(capacity);
    }

    // Noncopyable (intended to be shared between threads with shared_ptr<ThreadSafeRingBuffer<T>>.
    ThreadSafeRingBuffer(const ThreadSafeRingBuffer<T> &) = delete;
    ThreadSafeRingBuffer<T> &operator=(const ThreadSafeRingBuffer<T> &) = delete;

    
    // put(): called on producer thread, to put item of type T.
    // If ring buffer is full, blocks until space is available.
    
    void put(const T &t)
    {
	std::unique_lock lk(lock);

	// If there is no space in ring buffer, wait for a call to get().
	get_notifier.wait(lk, [this]{ return ((this->ix1 - this->ix0) < capacity); });

	xassert(!is_done);
	ringbuf[ix1 % capacity] = t;
	ix1++;

	lk.unlock();
	put_notifier.notify_one();
    }


    // get(): called on consumer thread, to get item of type T.
    // Returns 'true' on success, 'false' if a producer has called set_done().
    
    bool get(T &t)
    {
	std::unique_lock lk(lock);

	// If ring buffer is empty (and set_done() has not been called), wait for a call to put().
	put_notifier.wait(lk, [this]{ return (this->is_done || (this->ix0 < this->ix1)); });

	if (ix0 >= ix1) {
	    xassert(is_done);
	    return false;
	}

	// Note std::swap() here, instead of the simpler assignment statement:
	//   t = ringbuf[ix0 % capacity];    (*)
	//
	// To see why this is useful, consider a case where T is a shared_ptr to a "heavyweight" object.
	// Then (*) would leave a stray reference in the ring buffer, whereas std::swap() allows the
	// consumer (i.e. caller of get()) to "swap in" an empty pointer.
	
	std::swap(t, ringbuf[ix0 % capacity]);
	ix0++;

	lk.unlock();
	get_notifier.notify_one();
	return true;
    }


    // set_done(): called on producer thread, to mark ring buffer as 'done'.
    // After set_done() has been called, calls to get() return 'false', and calls to put() throw an exception.
    
    void set_done()
    {
	std::unique_lock lk(lock);
	is_done = true;
	lk.unlock();
	put_notifier.notify_all();
    }


    // wait_until_empty(): called on producer thread.
    // Blocks until ring buffer has been emptied (by consumers calling get()).
    
    void wait_until_empty()
    {
	std::unique_lock lk(lock);
	get_notifier.wait(lk, [this]{ return (this->ix0 >= this->ix1); });
	lk.unlock();

	// This call to notify_one() handles the case where there are multiple producer threads
	// concurrently calling wait_until_empty().
	
	get_notifier.notify_one();
    }
};



}  // namespace ksgpu

#endif // _KSGPU_THREAD_SAFE_RING_BUFFER_HPP
