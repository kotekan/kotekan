/**
 * @file
 * @brief A generic queue with synchronized access
 * - SynchronizedQueue
 */
#ifndef SYNCHRONIZED_QUEUE_HPP
#define SYNCHRONIZED_QUEUE_HPP

#include <condition_variable>
#include <deque>
#include <optional>
#include <mutex>


/**
 * @class SynchronizedQueue
 * @brief Generic FIFO that protects accesses with locks
 *
 * This class wraps std::deque and synchronizes access and modification methods using a mutex.
 * All blocking callers can be interrupted by calling the `cancel` method, after which the queue
 * will not produce any more results.
 *
 * @author Davor Cubranic
 */
template<typename T>
class SynchronizedQueue {
public:
    SynchronizedQueue() = default;

    /**
     * @brief Appends an element at the back of the queue
     *
     * Synchronizes access to prevent multiple threads modifying the list at the same time, and
     * blocks until an element is available. Blocked callers will be interrupted when the `cancel`
     * method is called on the queue.
     */
    void put(const T& v) {
        {
            if (stop) {
                return;
            };
            std::lock_guard<std::mutex> lock(mtx);

            queue.push_back(v);
        }

        cv.notify_one();
    }

    /**
     * @brief Removes the first element from the front of the queue and returns it
     *
     * Synchronizes access to prevent multiple threads modifying the list at the same time, and
     * blocks until an element is available. Blocked callers will be interrupted when the `cancel`
     * method is called on the queue.
     *
     * @returns std::nullopt if the queue is cancelled, and the element otherwise
     */
    std::optional<T> get() {

        std::unique_lock<std::mutex> lock(mtx);

        while (queue.empty() && !stop) {
            cv.wait(lock);
        }
        if (stop) {
            return std::nullopt;
        }

        auto v = queue.front();
        queue.pop_front();
        return v;
    }

    /**
     * @brief Interrupts all blocked callers and prevents further modification.
     */
    void cancel() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
    }

private:
    /// backing queue that stores the actual elements
    std::deque<T> queue;

    /// disables all queue operations when true
    bool stop = false;

    /// synchronizes access to the queue
    std::mutex mtx;

    /// notifying blocked callers
    std::condition_variable cv;
};

#endif // SYNCHRONIZED_QUEUE_HPP
