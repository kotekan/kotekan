#define BOOST_TEST_MODULE "test_buffer_swap"

#include "Config.hpp"          // for Config
#include "buffer.hpp"          // for Buffer
#include "errors.h"            // for __enable_syslog
#include "metadata.hpp"        // for metadataObject, metadataPool
#include "metadataFactory.hpp" // for metadataFactory
#include "test_utils.hpp"      // for GlobalFixture_Locale

#include "json.hpp" // for json

#include <atomic> // for atomic
#include <boost/test/included/unit_test.hpp>
#include <memory>   // for shared_ptr
#include <stdint.h> // for uint8_t
#include <string.h> // for memset, memcmp
#include <thread>   // for thread
#include <vector>   // for vector

using kotekan::Config;
using json = nlohmann::json;

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

static std::shared_ptr<metadataPool> make_pool(Config& config) {
    json json_config = json::parse(
        R"({"type": "config", "log_level": "info", "main_pool": {"kotekan_metadata_pool": "chordMetadata", "num_metadata_objects": 10}})");
    config.update_config(json_config);
    kotekan::metadataFactory mfac(config);
    return mfac.build_pools()["main_pool"];
}

// The frame helpers report failure rather than asserting, so they can also be
// called from the worker thread below: Boost.Test is not thread safe.

// Fill frame `id` of `buf` with `value` and mark it full.
static bool produce_frame(Buffer& buf, const std::string& name, int id, uint8_t value) {
    uint8_t* frame = buf.wait_for_empty_frame(name, id);
    if (frame == nullptr)
        return false;
    memset(frame, value, buf.frame_size);
    buf.mark_frame_full(name, id);
    return true;
}

// True if every byte of frame `id` of `buf` carries `value`. The frame equals
// itself shifted by one byte exactly when all of its bytes are equal.
static bool frame_is(Buffer& buf, int id, uint8_t value) {
    const uint8_t* frame = buf.frames[id];
    return frame[0] == value && memcmp(frame, frame + 1, buf.frame_size - 1) == 0;
}

BOOST_AUTO_TEST_CASE(swap_frame_hands_over_memory) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    const size_t frame_size = 1024;
    Buffer src(2, frame_size, pool, "swap_src", "standard", 0, false, false, {}, false);
    Buffer dest(2, frame_size, pool, "swap_dest", "standard", 0, false, false, {}, false);
    src.register_producer("prod");
    src.register_consumer("split");
    dest.register_producer("split");

    // With a single consumer on the source the frames are exchanged rather than
    // copied, so the source is left holding whatever the destination had.
    BOOST_REQUIRE(produce_frame(src, "prod", 0, 0xA0));
    memset(dest.frames[0], 0xD0, frame_size);

    BOOST_REQUIRE(src.wait_for_full_frame("split", 0) != nullptr);
    BOOST_REQUIRE(dest.wait_for_empty_frame("split", 0) != nullptr);
    src.safe_swap_frame(0, &dest, 0);

    BOOST_CHECK(frame_is(dest, 0, 0xA0));
    BOOST_CHECK(frame_is(src, 0, 0xD0));

    src.mark_frame_empty("split", 0);
    dest.mark_frame_full("split", 0);
}

BOOST_AUTO_TEST_CASE(copy_frame_leaves_source_intact) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    const size_t frame_size = 1024;
    Buffer src(2, frame_size, pool, "copy_src", "standard", 0, false, false, {}, false);
    Buffer dest(2, frame_size, pool, "copy_dest", "standard", 0, false, false, {}, false);
    src.register_producer("prod");
    src.register_consumer("split");
    // A second consumer still needs the source data, so the frame must be
    // copied and the source left as it was.
    src.register_consumer("other");
    dest.register_producer("split");

    BOOST_REQUIRE(produce_frame(src, "prod", 0, 0xB0));
    memset(dest.frames[0], 0xD0, frame_size);

    BOOST_REQUIRE(src.wait_for_full_frame("split", 0) != nullptr);
    BOOST_REQUIRE(dest.wait_for_empty_frame("split", 0) != nullptr);
    src.safe_swap_frame(0, &dest, 0);

    BOOST_CHECK(frame_is(dest, 0, 0xB0));
    BOOST_CHECK(frame_is(src, 0, 0xB0));

    // The other consumer reads the source frame after the copy.
    BOOST_REQUIRE(src.wait_for_full_frame("other", 0) != nullptr);
    BOOST_CHECK(frame_is(src, 0, 0xB0));

    src.mark_frame_empty("split", 0);
    src.mark_frame_empty("other", 0);
    dest.mark_frame_full("split", 0);
}

BOOST_AUTO_TEST_CASE(copy_frame_runs_beside_the_buffers) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);

    // Frames big enough that a copy of one is still running while the worker
    // below is using the same two buffers.
    const size_t frame_size = 8 * 1024 * 1024;
    Buffer src(4, frame_size, pool, "churn_src", "standard", 0, false, false, {}, false);
    Buffer dest(4, frame_size, pool, "churn_dest", "standard", 0, false, false, {}, false);
    src.register_producer("prod");
    src.register_consumer("split");
    src.register_consumer("other");
    dest.register_producer("split");
    dest.register_consumer("sink");

    // Copies run on frame 0 while the worker fills, drains and inspects the
    // other frames of both buffers. Nothing here is timing dependent: the test
    // is that the two sides interleave without deadlocking or losing data.
    std::atomic<bool> stop(false);
    std::atomic<bool> worker_failed(false);
    std::thread worker([&]() {
        uint8_t value = 1;
        int id = 1;
        while (!stop) {
            if (!produce_frame(src, "prod", id, value)) {
                worker_failed = true;
                return;
            }
            if (src.wait_for_full_frame("split", id) == nullptr
                || src.wait_for_full_frame("other", id) == nullptr) {
                worker_failed = true;
                return;
            }
            if (!frame_is(src, id, value))
                worker_failed = true;
            src.get_num_full_frames();
            dest.get_num_full_frames();
            src.mark_frame_empty("split", id);
            src.mark_frame_empty("other", id);
            id = 1 + (id % 3);
            if (++value == 0)
                value = 1;
        }
    });

    for (int i = 0; i < 20 && !worker_failed; ++i) {
        const uint8_t value = 0xC0 + i;
        BOOST_REQUIRE(produce_frame(src, "prod", 0, value));
        BOOST_REQUIRE(src.wait_for_full_frame("split", 0) != nullptr);
        BOOST_REQUIRE(src.wait_for_full_frame("other", 0) != nullptr);
        BOOST_REQUIRE(dest.wait_for_empty_frame("split", 0) != nullptr);

        src.safe_swap_frame(0, &dest, 0);

        BOOST_CHECK(frame_is(dest, 0, value));
        BOOST_CHECK(frame_is(src, 0, value));

        src.mark_frame_empty("split", 0);
        src.mark_frame_empty("other", 0);
        dest.mark_frame_full("split", 0);
        BOOST_REQUIRE(dest.wait_for_full_frame("sink", 0) != nullptr);
        dest.mark_frame_empty("sink", 0);
    }

    stop = true;
    worker.join();
    BOOST_CHECK(!worker_failed);
}
