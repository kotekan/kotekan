#define BOOST_TEST_MODULE "test_buffer_peek"

#include "Config.hpp"          // for Config
#include "buffer.hpp"          // for Buffer
#include "errors.h"            // for __enable_syslog
#include "metadata.hpp"        // for metadataObject, metadataPool
#include "metadataFactory.hpp" // for metadataFactory
#include "test_utils.hpp"      // for GlobalFixture_Locale

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <memory>   // for shared_ptr
#include <stdint.h> // for uint8_t, SIZE_MAX
#include <string.h> // for memset
#include <vector>   // for vector

using kotekan::Config;
using json = nlohmann::json;

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

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
