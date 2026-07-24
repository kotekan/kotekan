#define BOOST_TEST_MODULE "test_buffer_peek"

#include "Config.hpp"          // for Config
#include "buffer.hpp"          // for Buffer
#include "errors.h"            // for __enable_syslog
#include "metadata.hpp"        // for metadataObject, metadataPool
#include "metadataFactory.hpp" // for metadataFactory
#include "test_utils.hpp"      // for GlobalFixture_Locale

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <memory>    // for shared_ptr
#include <stdexcept> // for invalid_argument
#include <stdint.h>  // for uint8_t, SIZE_MAX
#include <string.h>  // for memset
#include <time.h>    // for clock_gettime, timespec
#include <vector>    // for vector

using kotekan::Config;
using json = nlohmann::json;

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

// Build the chordMetadata pool the peek tests allocate metadata from.
static std::shared_ptr<metadataPool> make_pool(Config& config) {
    json json_config = json::parse(
        R"({"type": "config", "log_level": "info", "main_pool": {"kotekan_metadata_pool": "chordMetadata", "num_metadata_objects": 10}})");
    config.update_config(json_config);
    kotekan::metadataFactory mfac(config);
    return mfac.build_pools()["main_pool"];
}

// Fill the frame `id` with `value` and mark it full.
static void produce_frame(Buffer& buf, int id, uint8_t value) {
    uint8_t* frame = buf.wait_for_empty_frame("prod", id);
    BOOST_REQUIRE(frame != nullptr);
    memset(frame, value, buf.frame_size);
    buf.allocate_new_metadata_object(id);
    buf.mark_frame_full("prod", id);
}

// Consume (and discard) the full frame `id`.
static void consume_frame(Buffer& buf, int id) {
    uint8_t* frame = buf.wait_for_full_frame("con", id);
    BOOST_REQUIRE(frame != nullptr);
    buf.mark_frame_empty("con", id);
}

BOOST_AUTO_TEST_CASE(peek_newest_full_frame) {
    __enable_syslog = 0;

    json json_config = json::parse(
        R"({"type": "config", "log_level": "info", "main_pool": {"kotekan_metadata_pool": "chordMetadata", "num_metadata_objects": 10}})");
    Config config;
    config.update_config(json_config);
    kotekan::metadataFactory mfac(config);
    std::shared_ptr<metadataPool> pool = mfac.build_pools()["main_pool"];
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
    produce_frame(buf, 0, 0xA0);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data.size(), frame_size);
    for (uint8_t b : data)
        BOOST_CHECK_EQUAL(b, 0xA0);
    BOOST_CHECK(meta == buf.get_metadata(0));
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);

    // A second frame becomes the newest.
    produce_frame(buf, 1, 0xA1);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);
    BOOST_CHECK_EQUAL(data[0], 0xA1);

    // max_len caps the copy length.
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, 4, meta), 1);
    BOOST_CHECK_EQUAL(data.size(), 4u);

    // Consuming the older frame leaves the newest in place.
    consume_frame(buf, 0);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);

    // Once all frames are consumed the buffer has nothing to peek.
    consume_frame(buf, 1);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), -1);

    // Ring wrap-around: frame 2 then frame 0 again; frame 0 is the newest.
    produce_frame(buf, 2, 0xA2);
    produce_frame(buf, 0, 0xA3);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data[0], 0xA3);

    // Consume the newest frame out of order: peek scans back to the older
    // full frame.
    consume_frame(buf, 0);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 2);
    BOOST_CHECK_EQUAL(data[0], 0xA2);

    consume_frame(buf, 2);
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
    produce_frame(buf, 0, 0xB0);
    consume_frame(buf, 0);
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
    produce_frame(buf, 1, 0xB1);
    consume_frame(buf, 1);
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);
    BOOST_CHECK_EQUAL(data[0], 0xB1);

    // Two full laps of the ring: the producer is never blocked by the
    // hold (a hang here is the deadlock this test guards against), and
    // exactly one frame stays occupied throughout.
    uint8_t value = 0xC0;
    for (int id : {2, 0, 1, 2, 0, 1}) {
        produce_frame(buf, id, value);
        consume_frame(buf, id);
        BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), id);
        BOOST_CHECK_EQUAL(data[0], value);
        BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
        ++value;
    }

    // A frame consumed after a newer one has already filled empties
    // normally; only the newest frame is ever held.
    produce_frame(buf, 2, 0xD0);
    produce_frame(buf, 0, 0xD1);
    consume_frame(buf, 2);
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
    consume_frame(buf, 0);
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
    produce_frame(buf, 0, 0xE0);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 0);
    BOOST_CHECK_EQUAL(data[0], 0xE0);
    BOOST_CHECK(meta != nullptr);

    // The next fill takes over the hold and releases the previous frame.
    produce_frame(buf, 1, 0xE1);
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 1);
    BOOST_CHECK_EQUAL(buf.peek_newest_full_frame(data, SIZE_MAX, meta), 1);
    BOOST_CHECK_EQUAL(data[0], 0xE1);

    // Lap the two-frame ring: the producer never stalls on the hold.
    produce_frame(buf, 0, 0xE2);
    produce_frame(buf, 1, 0xE3);
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
    produce_frame(buf, 0, 0xF0);
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

    // A single-frame ring would deadlock its producer: refused up front.
    Buffer buf(1, 8, pool, "one_buf", "standard", 0, false, false, {}, false);
    BOOST_CHECK_THROW(buf.enable_peek_hold(), std::invalid_argument);
}
