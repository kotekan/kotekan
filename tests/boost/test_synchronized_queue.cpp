#define BOOST_TEST_MODULE "test_SynchronizedQueue"

#include "SynchronizedQueue.hpp" // for SynchronizedQueue

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_CHECK, BOOST_PP_BOOL_2
#include <optional>                          // for optional
#include <thread>                            // for thread

/*
 * A basic check that you `get` what you `put` into the queue.
 */
BOOST_AUTO_TEST_CASE(put_get) {
    SynchronizedQueue<int> q;
    q.put(42);
    q.put(41);
    BOOST_CHECK(*q.get() == 42);
    BOOST_CHECK(*q.get() == 41);
}

/*
 * Start a producer and a consumer on the queue. The producer `put`s numbers [10..0] into the queue.
 * The consumer `get`s from the queue and expects the same sequence, returning from the thread on
 * `0`. It is an error if the queue does not return a value (since it is not going to be cancelled),
 * and if the lambda does not return out of its receiving loop.
 */
BOOST_AUTO_TEST_CASE(put_get_with_threads) {
    SynchronizedQueue<int> q;
    std::thread t1([&q]() {
        for (int i = 10; i >= 0; --i) {
            q.put(i);
        }
    });
    std::thread t2([&q]() {
        for (int expected = 10; expected >= 0; --expected) {
            auto v = q.get();
            BOOST_CHECK(v);
            BOOST_CHECK(*v == expected);
            if (*v == 0) {
                return;
            }
        }
        // We should never end up here, because of the `return` on `get`ting a zero value from the
        // queue
        BOOST_CHECK(false);
    });
    t1.join();
    BOOST_CHECK(!t1.joinable());
    t2.join();
    BOOST_CHECK(!t2.joinable());
}

/*
 * A `cancel`led queue does not produce any more values, even if it still has unconsumed ones.
 */
BOOST_AUTO_TEST_CASE(cancel) {
    SynchronizedQueue<int> q;
    q.put(42);
    q.cancel();
    BOOST_CHECK(!q.get());
}

/*
 * Same with multiple threads waiting to consume from a `cancel`led queue.
 */
BOOST_AUTO_TEST_CASE(cancel_with_threads) {
    SynchronizedQueue<int> q;
    std::thread t1([&q]() { BOOST_CHECK(!q.get()); });
    std::thread t2([&q]() { BOOST_CHECK(!q.get()); });
    q.cancel();

    t1.join();
    BOOST_CHECK(!t1.joinable());
    t2.join();
    BOOST_CHECK(!t2.joinable());
}
