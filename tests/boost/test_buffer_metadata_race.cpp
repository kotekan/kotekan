#define BOOST_TEST_MODULE "test_buffer_metadata_race"

#include "Config.hpp"          // for Config
#include "buffer.hpp"          // for Buffer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "errors.h"            // for __enable_syslog
#include "metadata.hpp"        // for metadataObject, metadataPool
#include "metadataFactory.hpp" // for metadataFactory
#include "test_utils.hpp"      // for GlobalFixture_Locale

#include "json.hpp" // for json

#include <atomic> // for atomic
#include <boost/test/included/unit_test.hpp>
#include <cstdint> // for int64_t
#include <memory>  // for shared_ptr, static_pointer_cast
#include <thread>  // for thread

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

// One thread replaces a frame's metadata with a fresh object over and over while another
// copies it out through both accessors and reads it. Every copy must be a live object whose
// counter never goes backwards. The accessors copy the slot under the buffer mutex; without
// that, a copy racing a replacement can revive an object whose count already reached zero,
// and it is then destroyed (and its json freed) twice.
BOOST_AUTO_TEST_CASE(metadata_copy_races_replacement) {
    __enable_syslog = 0;

    Config config;
    std::shared_ptr<metadataPool> pool = make_pool(config);
    BOOST_REQUIRE(pool != nullptr);
    Buffer buf(2, 64, pool, "metadata_race", "standard", 0, false, false, {}, false);
    buf.set_metadata(0, pool->request_metadata_object());

    constexpr int64_t iterations = 200000;
    std::atomic<bool> writer_done{false};

    std::thread writer([&]() {
        for (int64_t i = 1; i <= iterations; ++i) {
            // From the pool, like every object a buffer holds: get_chord_metadata needs the
            // object's parent_pool to know its type.
            std::shared_ptr<metadataObject> obj = pool->request_metadata_object();
            std::static_pointer_cast<chordMetadata>(obj)->set_fpga_seq_num(i);
            buf.set_metadata(0, obj);
        }
        writer_done = true;
    });

    // Boost.Test is not thread safe, so the reader runs here and only counts failures.
    int64_t last = 0, reads = 0, missing = 0, backwards = 0;
    while (!writer_done.load() || reads < 1000) {
        const std::shared_ptr<chordMetadata> meta =
            (reads % 2) ? get_chord_metadata(&buf, 0)
                        : std::dynamic_pointer_cast<chordMetadata>(buf.get_metadata(0));
        ++reads;
        if (!meta) {
            ++missing;
            continue;
        }
        const int64_t seq = meta->has_fpga_seq_num() ? meta->get_fpga_seq_num() : 0;
        if (seq < last)
            ++backwards;
        last = seq;
    }
    writer.join();

    BOOST_CHECK_EQUAL(missing, 0);
    BOOST_CHECK_EQUAL(backwards, 0);
    BOOST_CHECK_GE(reads, 1000);
    const std::shared_ptr<chordMetadata> final_meta = get_chord_metadata(&buf, 0);
    BOOST_REQUIRE(final_meta);
    BOOST_CHECK_EQUAL(final_meta->get_fpga_seq_num(), iterations);
}
