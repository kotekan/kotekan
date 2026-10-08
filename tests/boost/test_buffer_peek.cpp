#define BOOST_TEST_MODULE "test_buffer_peek"

#include "Config.hpp"          // for Config
#include "buffer.hpp"          // for Buffer
#include "errors.h"            // for __enable_syslog
#include "metadata.hpp"        // for metadataObject, metadataPool
#include "metadataFactory.hpp" // for metadataFactory
#include "test_logging.hpp"    // for SigtermGuard
#include "test_utils.hpp"      // for GlobalFixture_Locale

#include "json.hpp" // for json

#include <atomic> // for atomic
#include <boost/test/included/unit_test.hpp>
#include <memory>    // for shared_ptr
#include <sched.h>   // for sched_getaffinity, cpu_set_t, CPU_ISSET
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint8_t, SIZE_MAX
#include <string.h>  // for memset
#include <thread>    // for thread
#include <time.h>    // for clock_gettime, timespec
#include <vector>    // for vector

using kotekan::Config;
using json = nlohmann::json;

// Lets peek_hold_single_frame_rejected catch the FATAL_ERROR, which calls
// exit_kotekan and raises SIGTERM before throwing FatalError.
static kotekan_test_logging::SigtermGuard g_sigterm_guard;

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

// Build the chordMetadata pool the peek tests allocate metadata from.
static std::shared_ptr<metadataPool> make_pool(Config& config) {
    json json_config = json::parse(
        R"({"type": "config", "log_level": "info", "main_pool": {"kotekan_metadata_pool": "chordMetadata", "num_metadata_objects": 10}})");
    config.update_config(json_config);
    kotekan::metadataFactory mfac(config);
    return mfac.build_pools()["main_pool"];
}

// The frame helpers report failure through their return value rather than a
// Boost assert: they also run on worker threads, and Boost.Test is not
// thread-safe. The main thread wraps them in BOOST_REQUIRE; workers record a
// failure for the main thread to check after joining.

// Fill the frame `id` with `value` and mark it full.
static bool produce_frame(Buffer& buf, int id, uint8_t value) {
    uint8_t* frame = buf.wait_for_empty_frame("prod", id);
    if (frame == nullptr)
        return false;
    memset(frame, value, buf.frame_size);
    buf.allocate_new_metadata_object(id);
    buf.mark_frame_full("prod", id);
    return true;
}

// Consume (and discard) the full frame `id`.
static bool consume_frame(Buffer& buf, int id) {
    uint8_t* frame = buf.wait_for_full_frame("con", id);
    if (frame == nullptr)
        return false;
    buf.mark_frame_empty("con", id);
    return true;
}

// The byte checks below run over 16 MiB frames in the unoptimized Test build,
// so they lean on memcmp/memchr rather than byte loops to stay off the CI
// timeout.

// True if every byte of a peeked frame carries `value`: the frame equals
// itself shifted by one byte exactly when all its bytes are equal.
static bool all_bytes_are(const std::vector<uint8_t>& data, uint8_t value) {
    if (data.empty())
        return true;
    return data[0] == value && memcmp(data.data(), data.data() + 1, data.size() - 1) == 0;
}

// True if any byte of a peeked frame carries `value`.
static bool any_byte_is(const std::vector<uint8_t>& data, uint8_t value) {
    return !data.empty() && memchr(data.data(), value, data.size()) != nullptr;
}

// Churn frames through `buf` on a worker thread, consuming each frame as it
// is produced, while peeking from this one. This is a contention check, not a
// tearing check -- the producer needs a whole lap of the ring to return to
// the frame being copied, so it rarely gets there first either way. What it
// does cover is that peeks, deferred empty transitions and producer wake-ups
// interleave without losing a frame or deadlocking the ring. Frames carry a
// single repeated byte, so a badly torn frame would still show.
static void peek_against_producer(Buffer& buf, int iterations) {
    std::atomic<bool> torn(false);
    std::atomic<bool> worker_failed(false);

    std::thread worker([&]() {
        uint8_t value = 1;
        for (int i = 0; i < iterations; ++i) {
            const int id = i % buf.num_frames;
            if (!produce_frame(buf, id, value) || !consume_frame(buf, id)) {
                worker_failed = true;
                return;
            }
            // 0 is never written, so a stale frame cannot pass as a fresh one.
            if (++value == 0)
                value = 1;
        }
    });

    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> meta;
    for (int i = 0; i < iterations; ++i) {
        if (buf.peek_newest_full_frame(data, SIZE_MAX, meta) < 0)
            continue;
        if (!all_bytes_are(data, data[0]))
            torn = true;
    }

    worker.join();
    BOOST_CHECK(!worker_failed);
    BOOST_CHECK(!torn);
}

BOOST_AUTO_TEST_CASE(peek_newest_full_frame) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    const size_t frame_size = 16;
    Buffer buf(3, frame_size, pool, "peek_buf", "standard", 0, false, false, {}, false);
    buf.register_producer("prod");
    buf.register_consumer("con");

    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> meta;

    // Nothing produced yet.
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), -1);

    // One full frame: peek returns its data and metadata without consuming it.
    BOOST_REQUIRE(produce_frame(buf, 0, 0xA0));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data.size(), frame_size);
    for (uint8_t b : data)
        BOOST_CHECK_EQUAL(b, 0xA0);
    BOOST_CHECK(meta == buf.get_metadata(0));
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);

    // A second frame becomes the newest.
    BOOST_REQUIRE(produce_frame(buf, 1, 0xA1));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);
    BOOST_CHECK_EQUAL(data[0], 0xA1);

    // max_len caps the copy length.
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, 4, meta), 1);
    BOOST_CHECK_EQUAL(data.size(), 4u);

    // max_len 0 is the endpoint's metadata-only request: the frame is found
    // and its metadata returned, with nothing copied.
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, 0, meta), 1);
    BOOST_CHECK_EQUAL(data.size(), 0u);
    BOOST_CHECK(meta == buf.get_metadata(1));

    // Consuming the older frame leaves the newest in place.
    BOOST_REQUIRE(consume_frame(buf, 0));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);

    // Once all frames are consumed the buffer has nothing to peek.
    BOOST_REQUIRE(consume_frame(buf, 1));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), -1);

    // Ring wrap-around: frame 2 then frame 0 again; frame 0 is the newest.
    BOOST_REQUIRE(produce_frame(buf, 2, 0xA2));
    BOOST_REQUIRE(produce_frame(buf, 0, 0xA3));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data[0], 0xA3);

    // Consume the newest frame out of order: peek scans back to the older
    // full frame.
    BOOST_REQUIRE(consume_frame(buf, 0));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 2);
    BOOST_CHECK_EQUAL(data[0], 0xA2);

    BOOST_REQUIRE(consume_frame(buf, 2));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), -1);
}

BOOST_AUTO_TEST_CASE(peek_hold_keeps_newest_frame) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    const size_t frame_size = 16;
    Buffer buf(3, frame_size, pool, "hold_buf", "standard", 0, false, false, {}, false);
    buf.register_producer("prod");
    buf.register_consumer("con");
    buf.enable_peek_hold();

    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> meta;

    // Nothing produced yet: nothing to hold.
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), -1);

    // A fully consumed frame stays peekable: the hold defers its recycling.
    BOOST_REQUIRE(produce_frame(buf, 0, 0xB0));
    BOOST_REQUIRE(consume_frame(buf, 0));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data.size(), frame_size);
    for (uint8_t b : data)
        BOOST_CHECK_EQUAL(b, 0xB0);
    BOOST_CHECK(meta != nullptr);
    BOOST_CHECK(meta == buf.get_metadata(0));
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);

    // The held frame is not re-delivered to the consumer that finished it.
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += 50000000;
    if (deadline.tv_nsec >= 1000000000) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000;
    }
    BOOST_CHECK_EQUAL(buf.wait_for_full_frame_timeout("con", 0, deadline), 1);

    // A newer fill releases the hold: frame 0 empties, frame 1 is held.
    BOOST_REQUIRE(produce_frame(buf, 1, 0xB1));
    BOOST_REQUIRE(consume_frame(buf, 1));
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);
    BOOST_CHECK_EQUAL(data[0], 0xB1);

    // Two full laps of the ring: the producer is never blocked by the
    // hold (a hang here is the deadlock this test guards against), and
    // exactly one frame stays occupied throughout.
    uint8_t value = 0xC0;
    for (int id : {2, 0, 1, 2, 0, 1}) {
        BOOST_REQUIRE(produce_frame(buf, id, value));
        BOOST_REQUIRE(consume_frame(buf, id));
        BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), id);
        BOOST_CHECK_EQUAL(data[0], value);
        BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
        ++value;
    }

    // A frame consumed after a newer one has already filled empties
    // normally; only the newest frame is ever held.
    BOOST_REQUIRE(produce_frame(buf, 2, 0xD0));
    BOOST_REQUIRE(produce_frame(buf, 0, 0xD1));
    BOOST_REQUIRE(consume_frame(buf, 2));
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
    BOOST_REQUIRE(consume_frame(buf, 0));
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data[0], 0xD1);
}

BOOST_AUTO_TEST_CASE(peek_hold_no_consumers) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    const size_t frame_size = 8;
    Buffer buf(2, frame_size, pool, "nc_buf", "standard", 0, false, false, {}, false);
    buf.register_producer("prod");
    buf.enable_peek_hold();

    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> meta;

    // Without the hold a consumer-less buffer drops each frame at
    // mark-full; with it the frame stays peekable, metadata included.
    BOOST_REQUIRE(produce_frame(buf, 0, 0xE0));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data[0], 0xE0);
    BOOST_CHECK(meta != nullptr);

    // The next fill takes over the hold and releases the previous frame.
    BOOST_REQUIRE(produce_frame(buf, 1, 0xE1));
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);
    BOOST_CHECK_EQUAL(data[0], 0xE1);

    // Lap the two-frame ring: the producer never stalls on the hold.
    BOOST_REQUIRE(produce_frame(buf, 0, 0xE2));
    BOOST_REQUIRE(produce_frame(buf, 1, 0xE3));
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);
    BOOST_CHECK_EQUAL(data[0], 0xE3);
}

BOOST_AUTO_TEST_CASE(peek_hold_survives_consumer_unregister) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    Buffer buf(2, 8, pool, "unreg_buf", "standard", 0, false, false, {}, false);
    buf.register_producer("prod");
    buf.register_consumer("con");
    buf.enable_peek_hold();

    // Unregistering the last consumer of a full frame routes through the
    // same deferral: the newest frame is held rather than emptied.
    BOOST_REQUIRE(produce_frame(buf, 0, 0xF0));
    buf.unregister_consumer("con", false);

    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> meta;
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data[0], 0xF0);
}

BOOST_AUTO_TEST_CASE(peek_hold_single_frame_rejected) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    // A single-frame ring would deadlock its producer: refused up front with a
    // FATAL_ERROR, so kotekan comes down rather than running with a buffer
    // nothing can be produced into. FatalError derives from std::runtime_error.
    Buffer buf(1, 8, pool, "one_buf", "standard", 0, false, false, {}, false);
    BOOST_CHECK_THROW(buf.enable_peek_hold(), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(peek_hold_shallow_buffer_warns_but_holds) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    // Two frames is the shallowest depth the hold works at: it warns (the slot
    // is half the buffer), but the hold itself still has to behave.
    Buffer buf(2, 8, pool, "shallow_buf", "standard", 0, false, false, {}, false);
    buf.register_producer("prod");
    buf.register_consumer("con");
    BOOST_CHECK_NO_THROW(buf.enable_peek_hold());

    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> meta;

    // The producer still gets round the ring with one frame permanently held.
    uint8_t value = 0x10;
    for (int id : {0, 1, 0, 1}) {
        BOOST_REQUIRE(produce_frame(buf, id, value));
        BOOST_REQUIRE(consume_frame(buf, id));
        BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), id);
        BOOST_CHECK_EQUAL(data[0], value);
        BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
        ++value;
    }
}

// Frames big enough that a peek of one is still copying when this thread gets
// back to the buffer, so a producer really does arrive mid-copy.
static const size_t slow_frame_size = 16 * 1024 * 1024;

// Produce frame 0, peek it from this thread, and have a worker consume it and
// take it straight back as a producer, overwriting it. If the copy can be
// refilled underneath, the peek comes back holding a mix of the two values;
// nothing here depends on when either thread is scheduled, only on what the
// copy contains. Spawning the worker is itself a race -- it can win, leaving
// the peek nothing to read and reporting -1 -- so the check is made on the
// iterations the peek won, with `peeks_won` making sure that is not vacuously
// none of them. With `hold`, filling frame 1 passes frame 0 from the peek_hold
// deferral to the peek's, exercising both deferrals on one frame.
static void check_peek_blocks_refill(Buffer& buf, uint8_t value, int iterations, bool hold) {
    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> meta;
    int peeks_won = 0;
    std::atomic<bool> worker_failed(false);

    // Fault the destination's pages in before racing anything: a peek that
    // finds nothing clears the vector, but its capacity -- and the mapped
    // pages -- persist across iterations.
    data.assign(slow_frame_size, 0);

    for (int i = 0; i < iterations && !worker_failed; ++i) {
        BOOST_REQUIRE(produce_frame(buf, 0, value));

        std::thread worker([&]() {
            if (!consume_frame(buf, 0)) {
                worker_failed = true;
                return;
            }
            if (hold) {
                // Frame 1 is filled but not written, so releasing the hold on
                // frame 0 lands well inside the copy of it.
                if (buf.wait_for_empty_frame("prod", 1) == nullptr) {
                    worker_failed = true;
                    return;
                }
                buf.allocate_new_metadata_object(1);
                buf.mark_frame_full("prod", 1);
                if (!consume_frame(buf, 1)) {
                    worker_failed = true;
                    return;
                }
            }
            uint8_t* frame = buf.wait_for_empty_frame("prod", 0);
            if (frame == nullptr) {
                worker_failed = true;
                return;
            }
            // Stamp the far end before the bulk write: a copy still running
            // works forwards, so the tail is the part it cannot already
            // have read, and a refill that slipped in always shows there.
            frame[buf.frame_size - 1] = ~value;
            memset(frame, ~value, buf.frame_size);
        });

        const int peeked = buf.peek_newest_full_frame(data, SIZE_MAX, meta);
        worker.join();

        if (peeked != 0)
            continue;
        peeks_won++;
        BOOST_CHECK(all_bytes_are(data, value));
    }

    BOOST_CHECK(!worker_failed);
    BOOST_CHECK(peeks_won > 0);
}

BOOST_AUTO_TEST_CASE(peek_blocks_producer_refill) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    Buffer buf(2, slow_frame_size, pool, "refill_buf", "standard", 0, false, false, {}, false);
    buf.register_producer("prod");
    buf.register_consumer("con");

    check_peek_blocks_refill(buf, 0xB0, 25, false);

    // The deferred transitions ran rather than being dropped.
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 0);
}

BOOST_AUTO_TEST_CASE(peek_blocks_producer_refill_with_hold) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    Buffer buf(2, slow_frame_size, pool, "refill_hold_buf", "standard", 0, false, false, {}, false);
    buf.register_producer("prod");
    buf.register_consumer("con");
    buf.enable_peek_hold();

    check_peek_blocks_refill(buf, 0xB1, 25, true);
}

BOOST_AUTO_TEST_CASE(peek_blocks_producer_refill_while_zeroing) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    // The zeroing threads inherit the buffer's cpu_affinity, and an empty
    // affinity set is an error, so hand them the first CPU this process may
    // run on.
    cpu_set_t cpus;
    BOOST_REQUIRE_EQUAL(sched_getaffinity(0, sizeof(cpus), &cpus), 0);
    int zero_cpu = 0;
    while (!CPU_ISSET(zero_cpu, &cpus))
        ++zero_cpu;

    Buffer buf(2, slow_frame_size, pool, "zero_buf", "standard", 0, false, false, {zero_cpu},
               false);
    buf.register_producer("prod");
    buf.register_consumer("con");
    buf.zero_frames();

    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> meta;
    const uint8_t value = 0xB2;
    int peeks_won = 0;
    std::atomic<bool> worker_failed(false);

    // With zero_frames the empty transition runs on a detached zeroing
    // thread, and the frame stays full until it finishes. Consuming here
    // starts that thread, so the peek lands inside the zeroing window and it
    // is the zeroing thread, not mark_frame_empty(), that has to defer to the
    // pin. The copy may come back half zeroed -- that tearing is documented --
    // but the worker's refill has to stay out of it until the copy is done.
    // Fault the destination's pages in once, as in check_peek_blocks_refill.
    data.assign(slow_frame_size, 0);

    for (int i = 0; i < 10 && !worker_failed; ++i) {
        BOOST_REQUIRE(produce_frame(buf, 0, value));
        BOOST_REQUIRE(consume_frame(buf, 0));

        std::thread worker([&]() {
            uint8_t* frame = buf.wait_for_empty_frame("prod", 0);
            if (frame == nullptr) {
                worker_failed = true;
                return;
            }
            frame[buf.frame_size - 1] = static_cast<uint8_t>(~value);
            memset(frame, ~value, buf.frame_size);
        });

        const int peeked = buf.peek_newest_full_frame(data, SIZE_MAX, meta);
        worker.join();

        if (peeked != 0)
            continue;
        peeks_won++;
        BOOST_CHECK(!any_byte_is(data, static_cast<uint8_t>(~value)));
    }

    BOOST_CHECK(!worker_failed);
    BOOST_CHECK(peeks_won > 0);
}

BOOST_AUTO_TEST_CASE(peek_under_churn) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    Buffer buf(4, 64 * 1024, pool, "churn_buf", "standard", 0, false, false, {}, false);
    buf.register_producer("prod");
    buf.register_consumer("con");
    buf.enable_peek_hold();

    // Consuming each frame as it is produced leaves the hold covering every
    // peek, so every iteration runs a peek and the hold over the same frame's
    // deferred empty transition and laps the ring back onto held frames.
    peek_against_producer(buf, 2000);
}
