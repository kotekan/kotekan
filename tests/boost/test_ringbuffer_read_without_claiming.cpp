#define BOOST_TEST_MODULE "test_ringbuffer_read_without_claiming"

// Tests for the ringbuffer semantics that let a consumer read data without claiming them,
// which is what a slowly varying input (e.g. a bad feed mask) needs: it is read on every
// iteration but advances only rarely.
//
// These tests drive `RingBuffer` directly. They cover the primitives the RFI stages rely on
// -- that a non-claiming read leaves the read head where it is, that the data stay protected
// from the producer while they do, and that a single consumer name is shared by all of a
// command's instances. They do not exercise `NDArrayRingBuffer` itself (constructing one
// needs a `cudaCommand`, hence a GPU) nor the stages built on top of it.

#include "Config.hpp" // for Config
#include "metadataFactory.hpp"
#include "ringbuffer.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

#undef WITH_CUDA

#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value...

#include <boost/test/included/unit_test.hpp>

using kotekan::Config;
using json = nlohmann::json;
using namespace kotekan;

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

// One "element" of the simulated slowly varying input, in bytes
static constexpr std::ptrdiff_t G = 8;
// Ring holding four such elements, like `host_bf_mask_ringbuffer`
static constexpr std::ptrdiff_t RING = 4 * G;

static std::shared_ptr<metadataPool> make_pool() {
    json json_config = json::parse(
        R"({"type": "config", "log_level": "info", "main_pool": {"kotekan_metadata_pool": "oneHotMetadata", "num_metadata_objects": 100}})");
    Config conf;
    conf.update_config(json_config);
    metadataFactory mfac(conf);
    auto facs = mfac.build_pools();
    return facs["main_pool"];
}

// Reading without claiming must leave the read head alone, so the same element is readable
// again on the next iteration.
BOOST_AUTO_TEST_CASE(read_without_claiming_does_not_advance_the_read_head) {
    __enable_syslog = 0;
    std::shared_ptr<metadataPool> pool = make_pool();
    BOOST_REQUIRE(pool != nullptr);

    RingBuffer rb(RING, pool, "rb", "ring");
    rb.register_producer("P");
    rb.register_consumer("C");

    // Fill the ring
    BOOST_REQUIRE(rb.wait_for_writable("P", 0, RING).value_or(-1) == 0);
    rb.finish_write("P", 0, RING);

    // Several non-claiming reads in a row all see the same element
    for (int i = 0; i < 5; ++i) {
        const auto waited = rb.wait_without_claiming("C", 0, G);
        BOOST_REQUIRE(waited.has_value());
        BOOST_CHECK(waited.value() == RING); // all four elements still available
        const auto peeked = rb.peek_readable("C", 0);
        BOOST_REQUIRE(peeked.has_value());
        BOOST_CHECK(peeked.value().first == 0); // read head has not moved
        BOOST_CHECK(peeked.value().second == RING);
    }

    // Claiming does move it
    BOOST_CHECK(rb.wait_and_claim_readable("C", 0, G).value_or(-1) == 0);
    const auto peeked = rb.peek_readable("C", 0);
    BOOST_REQUIRE(peeked.has_value());
    BOOST_CHECK(peeked.value().first == G);
    BOOST_CHECK(peeked.value().second == RING - G);
}

// While a consumer has claimed but not finished, its read tail pins the producer -- which is
// what keeps an element readable across the frames that only read it.
BOOST_AUTO_TEST_CASE(unfinished_claim_protects_the_element_from_the_producer) {
    __enable_syslog = 0;
    std::shared_ptr<metadataPool> pool = make_pool();

    RingBuffer rb(RING, pool, "rb", "ring");
    rb.register_producer("P");
    rb.register_consumer("C");

    BOOST_REQUIRE(rb.wait_for_writable("P", 0, RING).value_or(-1) == 0);
    rb.finish_write("P", 0, RING);

    // Ring is full: no space for the producer
    BOOST_CHECK(rb.get_writable("P", 0).value().second == 0);

    // Read three elements without claiming: still no space, and nothing was consumed
    for (int i = 0; i < 3; ++i)
        BOOST_REQUIRE(rb.wait_without_claiming("C", 0, G).has_value());
    BOOST_CHECK(rb.get_writable("P", 0).value().second == 0);

    // Claiming alone does not free space either -- only finishing the read does
    BOOST_REQUIRE(rb.wait_and_claim_readable("C", 0, G).has_value());
    BOOST_CHECK(rb.get_writable("P", 0).value().second == 0);
    rb.finish_read("C", 0, G);
    BOOST_CHECK(rb.get_writable("P", 0).value().second == G);
}

// `RingBuffer` keys its cursors on the consumer name alone; the instance argument is
// ignored. All instances of a command therefore share one read head, which is why an
// instance's own bookkeeping cannot be used as its stream position.
BOOST_AUTO_TEST_CASE(instances_share_one_read_head_per_consumer_name) {
    __enable_syslog = 0;
    std::shared_ptr<metadataPool> pool = make_pool();

    RingBuffer rb(RING, pool, "rb", "ring");
    rb.register_producer("P");
    rb.register_consumer("C");

    BOOST_REQUIRE(rb.wait_for_writable("P", 0, RING).value_or(-1) == 0);
    rb.finish_write("P", 0, RING);

    // Four "instances" of the same consumer claim one element each, in turn
    for (int inst = 0; inst < 4; ++inst) {
        const auto claimed = rb.wait_and_claim_readable("C", inst, G);
        BOOST_REQUIRE(claimed.has_value());
        // Each instance continues where the previous one left off
        BOOST_CHECK(claimed.value() == inst * G);
        // ... and any instance peeking sees the shared position, not its own
        BOOST_CHECK(rb.peek_readable("C", 0).value().first == (inst + 1) * G);
    }
}

// The pattern the RFI stages use: N frames read element k, the Nth also claims it. The read
// head must advance exactly once per N frames, and the producer must regain space only then.
BOOST_AUTO_TEST_CASE(read_many_claim_once_advances_one_element_per_cycle) {
    __enable_syslog = 0;
    std::shared_ptr<metadataPool> pool = make_pool();

    RingBuffer rb(RING, pool, "rb", "ring");
    rb.register_producer("P");
    rb.register_consumer("C");

    constexpr int frames_per_element = 20;
    constexpr int num_elements = 6; // more than the ring holds, so it must recycle

    // Prime the ring
    BOOST_REQUIRE(rb.wait_for_writable("P", 0, 2 * G).value_or(-1) == 0);
    rb.finish_write("P", 0, 2 * G);

    for (int element = 0; element < num_elements; ++element) {
        for (int frame = 0; frame < frames_per_element; ++frame) {
            const bool last_use = frame == frames_per_element - 1;

            // The element we are about to read is always the one at the read head
            const auto peeked = rb.peek_readable("C", frame % 4);
            BOOST_REQUIRE(peeked.has_value());
            BOOST_CHECK_MESSAGE(peeked.value().first == element * G,
                                "element " << element << " frame " << frame << ": read head is "
                                           << peeked.value().first << ", expected " << element * G);
            BOOST_REQUIRE(peeked.value().second >= G); // element is available

            if (last_use) {
                BOOST_REQUIRE(rb.wait_and_claim_readable("C", frame % 4, G).value_or(-1)
                              == element * G);
                rb.finish_read("C", frame % 4, G);
            } else {
                BOOST_REQUIRE(rb.wait_without_claiming("C", frame % 4, G).has_value());
            }
        }
        // Exactly one element was consumed, so the producer has room for exactly one more
        BOOST_CHECK(rb.get_writable("P", 0).value().second >= G);
        if (element + 2 < num_elements) {
            BOOST_REQUIRE(rb.wait_for_writable("P", 0, G).has_value());
            rb.finish_write("P", 0, G);
        }
    }

    // Every element was consumed exactly once
    BOOST_CHECK(rb.peek_readable("C", 0).value().first == num_elements * G);
}

// The premise of the bad feed mask fix: an instance's own previous read position is not its
// current stream position, so a stage must ask the ringbuffer where its next read begins.
//
// This is what the RFI stages used to get wrong. With the CHARTS numbers -- 1280 rfi_S012
// elements per bad feed mask element, 64 elements read per frame, 4 instances -- instance 0's
// own position reached 1088 while the shared read head was at 1280. Clamping the read against
// 1088 allowed 64 more elements, so the read ran to 1344 and past the end of mask element 0.
BOOST_AUTO_TEST_CASE(an_instances_own_position_lags_the_shared_read_head) {
    __enable_syslog = 0;
    std::shared_ptr<metadataPool> pool = make_pool();

    constexpr std::ptrdiff_t R = 8; // bytes claimed per frame
    constexpr int instances = 4;    // == buffer_depth
    RingBuffer rb(instances * R, pool, "rb", "ring");
    rb.register_producer("P");
    rb.register_consumer("C");

    // Each instance remembers where its own previous read ended, the way
    // `NDArrayRingBuffer::read_valid` does after `finish_read`.
    std::ptrdiff_t own_end[instances] = {};

    for (int frame = 0; frame < 3 * instances; ++frame) {
        const int inst = frame % instances;

        if (rb.get_writable("P", 0).value().second >= R) {
            BOOST_REQUIRE(rb.wait_for_writable("P", 0, R).has_value());
            rb.finish_write("P", 0, R);
        }

        const auto peeked = rb.peek_readable("C", inst);
        BOOST_REQUIRE(peeked.has_value());
        const std::ptrdiff_t shared_head = peeked.value().first;

        // The claim really does start at the shared head -- so that is the position a stage
        // must clamp against.
        const auto claimed = rb.wait_and_claim_readable("C", inst, R);
        BOOST_REQUIRE(claimed.has_value());
        BOOST_CHECK(claimed.value() == shared_head);

        // Once every instance has had a turn, its own position is stale by exactly what the
        // other instances claimed in between.
        if (frame >= instances)
            BOOST_CHECK(shared_head - own_end[inst] == (instances - 1) * R);

        rb.finish_read("C", inst, R);
        own_end[inst] = shared_head + R;
    }
}

// A consumer waiting for an element that never arrives must be released by shutdown, not
// hang.
BOOST_AUTO_TEST_CASE(shutdown_releases_a_waiting_non_claiming_read) {
    __enable_syslog = 0;
    std::shared_ptr<metadataPool> pool = make_pool();

    RingBuffer rb(RING, pool, "rb", "ring");
    rb.register_producer("P");
    rb.register_consumer("C");

    // Nothing has been produced, so this blocks
    std::atomic<bool> returned(false);
    std::atomic<bool> had_value(true);
    std::thread consumer([&]() {
        const auto waited = rb.wait_without_claiming("C", 0, G);
        had_value = waited.has_value();
        returned = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    BOOST_CHECK(!returned);

    rb.send_shutdown_signal();
    for (int i = 0; i < 200 && !returned; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    BOOST_CHECK(returned);
    BOOST_CHECK(!had_value); // shutdown is reported as "no value"
    consumer.join();
}
