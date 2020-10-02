/*
 * Boost tests for BipBuffer
 */
#define BOOST_TEST_MODULE "test_BipBuffer"

#include "BipBuffer.hpp"         // for BipWriteReservation, BipBuffer, BipBufferRe...
#include "SynchronizedQueue.hpp" // for SynchronizedQueue
#include "kotekanLogging.hpp"    // for DEBUG_NON_OO, DEBUG2_NON_OO

#include "gsl-lite.hpp" // for span

#include <algorithm>                         // for min
#include <atomic>                            // for atomic_size_t
#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_CHECK, BOOST_PP_BOOL_2
#include <chrono>                            // for milliseconds
#include <memory>                            // for unique_ptr
#include <optional>                          // for optional
#include <random>                            // for mt19937, uniform_int_distribution, random_d...
#include <stddef.h>                          // for size_t
#include <stdint.h>                          // for uint8_t
#include <thread>                            // for sleep_for, thread
#include <vector>                            // for vector, operator==

/*
 * To see log output, assign 0 to `__enable_syslog` so the messages go to
 * stderr.
 *
 * The starting log level is INFO (3). To change it, assign the desired level to
 * `_global_log_level` (ERROR=1, DEBUG2=5)
 */
// __enable_syslog = 0;
// DEFAULT: _global_log_level = 3;

/*
 * A basic check of creating the buffer.
 */
BOOST_AUTO_TEST_CASE(buffer_api) {
    BipBuffer buffer(10);
    BOOST_CHECK(buffer.len == 10);
    BOOST_CHECK(buffer.read.load() == 0);
    BOOST_CHECK(buffer.write.load() == 0);
    BOOST_CHECK(buffer.watermark.load() == 0);
}


/*
 * Tests alternate writes and reads from the buffer, including the cases where the writer wraps
 * around, and when there isn't a large enough segment to accommodate the requested size.
 */
BOOST_AUTO_TEST_CASE(buffer_write_read_loop) {
    BipBuffer buffer(10);
    BipBufferWriter writer(buffer);
    BipBufferReader reader(buffer);

    std::vector<size_t> expected = {0, 1, 2, 3, 4, 5};
    std::vector<size_t> actual;
    for (size_t v = 0; v < 10; ++v) {
        auto reservation = writer.reserve(v);
        if (reservation) {
            actual.push_back(v);
            writer.commit(*reservation);
            auto read_res = reader.access(reservation->length);
            BOOST_CHECK(read_res);
            reader.advance(*read_res);
        }
    }
    BOOST_CHECK(expected == actual);
}

/*
 * Tests reading and writing from separate threads, with both sides using the `_max` version of the
 * reservation to accept smaller reservation sizes. The writer writes a sequence of numbers into the
 * reserved cells, so that the reader can easily check that each read cell constains the expected
 * value.
 */
BOOST_AUTO_TEST_CASE(buffer_max_write_read_independent_threads) {
    BipBuffer buffer(100);
    constexpr int count = 300;
    std::thread t1([&buffer]() {
        // std::cout << "writer\n";
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(3, 40);

        BipBufferWriter writer(buffer);
        for (int v = 0, size = dis(gen); v < count; size = std::min(dis(gen), count - v)) {
            auto reservation = writer.reserve_max(size);
            if (reservation) {
                BOOST_CHECK(reservation->length <= size_t(size));
                DEBUG_NON_OO("Writing {} bytes starting at {} (value {}).", reservation->length,
                             reservation->data.data() - buffer.data.get(), uint8_t(v));
                std::this_thread::sleep_for(std::chrono::milliseconds(size * 5));
                for (size_t i = 0; i < reservation->length; ++i) {
                    reservation->data[i] = uint8_t(v);
                    v++;
                }
                writer.commit(*reservation);
                DEBUG_NON_OO("Written {} out of {} bytes. {} bytes remaining.", v, count,
                             count - v);
            } else {
                DEBUG_NON_OO("Reserve not successful: {}", size);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen) * 5));
        }
    });
    std::thread t2([&buffer]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(3, 30);

        BipBufferReader reader(buffer);
        for (int v = 0, size = dis(gen); v < count; size = std::min(dis(gen), count - v)) {
            auto reservation = reader.access_max(size);
            if (reservation) {
                DEBUG_NON_OO("Reading {} bytes from {} (expecting value {}).", reservation->length,
                             reservation->data.data() - buffer.data.get(), uint8_t(v));
                std::this_thread::sleep_for(std::chrono::milliseconds(size * 10));
                for (size_t i = 0; i < reservation->length; ++i) {
                    DEBUG2_NON_OO("Read: {} (exp. {})", int(reservation->data[i]), (v & 0xff));
                    BOOST_CHECK(reservation->data[i] == uint8_t(v));
                    v++;
                }
                reader.advance(*reservation);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen) * 5));
        }
    });
    t1.join();
    BOOST_CHECK(!t1.joinable());
    t2.join();
    BOOST_CHECK(!t2.joinable());
}


/*
 * Auxiliary structure used by the next test case
 */
struct TestTask {
    // value written into all cells of this segment
    uint8_t value;
    // segment's start (== BipWriteReservation.data)
    uint8_t const* start;
    // segment's length (== BipWriteReservation.length)
    size_t length;
};

/*
 * Tests reading and writing from separate threads, with both sides using the exact-size version of
 * the reservations, with the writer communicating the reservation sizes as `TestTask` instances in
 * a `SynchronizedQueue` that the reader uses to determine the size of the requested read
 * reservation sizes. (This roughly approximates how the basebandReadout threads are going to
 * coordinate handling baseband dumps between reading out the baseband buffer and writing out to the
 * HDF file).
 *
 * The writer writes a sequence of numbers into the reserved cells, so that the reader can easily
 * check that each read cell constains the expected value.
 */
BOOST_AUTO_TEST_CASE(buffer_write_read_task_threads) {
    SynchronizedQueue<TestTask> q;
    BipBuffer buffer(100);
    constexpr int count = 300;
    std::vector<int> expected, actual;
    std::thread t1([&q, &buffer, &expected]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(10, 40);

        BipBufferWriter writer(buffer);
        for (int v = 0, size = dis(gen); v < count; size = std::min(dis(gen), count - v)) {
            auto reservation = writer.reserve(size);
            if (reservation) {
                BOOST_CHECK(reservation->length == size_t(size));
                DEBUG_NON_OO("Writing {} bytes from {} (value {}). {} bytes remaining.",
                             reservation->length, reservation->data.data() - buffer.data.get(),
                             uint8_t(v), count - v - size);
                std::this_thread::sleep_for(std::chrono::milliseconds(size * 5));
                for (size_t i = 0; i < reservation->length; ++i) {
                    reservation->data[i] = uint8_t(v);
                }
                writer.commit(*reservation);
                expected.push_back(uint8_t(v));
                q.put(TestTask{uint8_t(v), reservation->data.data(), reservation->length});
                v += size;
            } else {
                DEBUG_NON_OO("Reserve not successful: {}", size);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen) * 5));
        }
        q.put(TestTask{0, nullptr, 0});
    });
    std::thread t2([&q, &buffer, &actual]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(10, 40);

        BipBufferReader reader(buffer);
        while (auto t = q.get()) {
            if (t->start == nullptr) {
                return;
            }
            auto reservation = reader.access(t->length);
            BOOST_CHECK(reservation && reservation->length == t->length
                        && reservation->data.data() == t->start);
            DEBUG_NON_OO("Reading {} bytes from {} (expecting value {})", t->length,
                         t->start - buffer.data.get(), t->value);
            for (size_t i = 0; i < reservation->length; i++) {
                BOOST_CHECK(reservation->data[i] == t->value);
            }
            actual.push_back(t->value);
            reader.advance(*reservation);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen) * 10));
    });

    t1.join();
    BOOST_CHECK(!t1.joinable());

    t2.join();
    BOOST_CHECK(!t2.joinable());

    BOOST_CHECK(expected == actual);
    DEBUG_NON_OO("successful tasks: {}", expected.size());
}
