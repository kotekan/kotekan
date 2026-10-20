#define BOOST_TEST_MODULE "test_bufferBadInputs"

#include "Config.hpp"                // for Config
#include "NDArray.hpp"               // for NDArray
#include "Telescope.hpp"             // for Telescope
#include "buffer.hpp"                // for Buffer
#include "bufferBadInputs.hpp"       // for bufferBadInputs
#include "bufferContainer.hpp"       // for bufferContainer
#include "chordMetadata.hpp"         // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"         // for configUpdater
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "metadataFactory.hpp"       // for metadataFactory
#include "prometheusMetrics.hpp"     // for Metrics
#include "test_utils.hpp"            // for add_test_telescope_config
#include "visUtil.hpp"               // for ts_to_double

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK_EQUAL, BOOST_REQUIRE
#include <memory>                            // for unique_ptr, make_unique, shared_ptr
#include <stdint.h>                          // for int64_t, int8_t
#include <string>                            // for string
#include <vector>                            // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::metadataFactory;
using kotekan::prometheus::Metrics;
using json = nlohmann::json;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

// The test telescope has 2 dishes, both in the main array, so the baseline mask is all good.
static const int num_polarizations = 2;
static const int num_dishes = 2;
static const int num_elements = num_polarizations * num_dishes;

static const int64_t seq0 = 1000000;
static const int64_t T = 100;
static const int64_t alignment = 3 * T;

// A stream of the bad feed mask, read as its consumer would.
struct maskStream {
    std::unique_ptr<Buffer> buf;
    int64_t lifetime;
    int64_t num_consumed = 0;

    maskStream(const std::shared_ptr<metadataPool>& pool, const std::string& name, int num_frames,
               int64_t lifetime) : lifetime(lifetime) {
        buf = std::make_unique<Buffer>(num_frames, num_elements, pool, name, "ndarray", 0, false,
                                       false, std::vector<int>{}, true);
        buf->ensure_frame_desc(kotekan::NDArray<int8_t, 3>::describe(
            "bf_mask", {1, num_polarizations, num_dishes}, {"Tbf", "P", "D"}, {lifetime, 1, 1}));
        buf->register_consumer("test");
    }

    // Consume the next frame and check that it holds `expected` and has the next seq.
    void consume(const std::vector<int8_t>& expected) {
        const int frame_id = num_consumed % buf->num_frames;
        const int8_t* frame = (const int8_t*)buf->wait_for_full_frame("test", frame_id);
        BOOST_REQUIRE(frame != nullptr);
        const int64_t seq = get_chord_metadata(buf.get(), frame_id)->get_fpga_seq_num();
        BOOST_CHECK_EQUAL(seq, seq0 + num_consumed * lifetime);
        BOOST_CHECK_MESSAGE(std::vector<int8_t>(frame, frame + num_elements) == expected,
                            buf->buffer_name << ": wrong mask at seq " << seq);
        buf->mark_frame_empty("test", frame_id);
        ++num_consumed;
    }

    // Wait until the producer has filled the buffer, so that exactly num_frames frames past the
    // consumed ones have been produced.
    void wait_until_full() {
        const int frame_id = (num_consumed + buf->num_frames - 1) % buf->num_frames;
        BOOST_REQUIRE(buf->wait_for_full_frame("test", frame_id) != nullptr);
    }
};

static json update(const std::vector<int>& bad_inputs, int64_t start_seq, const std::string& id) {
    return {{"bad_inputs", bad_inputs},
            {"start_time", ts_to_double(Telescope::instance().to_time(start_seq))},
            {"update_id", id}};
}

static bool metric_has(const std::string& line) {
    return Metrics::instance().serialize().find(line) != std::string::npos;
}

BOOST_AUTO_TEST_CASE(test_switch_by_fpga_time) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    add_test_telescope_config(cfg);
    cfg["gps_time"] = {{"frame0_nano", 1767225600000000000}};
    cfg["chord_pool"] = {{"kotekan_metadata_pool", "chordMetadata"}, {"num_metadata_objects", 32}};
    cfg["updatable_config"]["bad_inputs"] = {{"kotekan_update_endpoint", "json"},
                                             {"bad_inputs", json::array()},
                                             {"start_time", 0.},
                                             {"update_id", "initial"}};
    cfg["num_polarizations"] = num_polarizations;
    cfg["num_dishes"] = num_dishes;
    cfg["num_elements"] = num_elements;
    cfg["input_order"] = "CHORDBeamformer";
    cfg["output_order"] = "CHORDBeamformer";
    cfg["bf_mask_update_alignment_in_samples"] = alignment;
    for (const char* name : {"bad_a", "bad_b"}) {
        cfg[name]["kotekan_stage"] = "bufferBadInputs";
        cfg[name]["updatable_config"]["bad_inputs"] = "/updatable_config/bad_inputs";
        cfg[name]["metadata_source"] = "clock";
        cfg[name]["out_buf"] = name;
    }
    cfg["bad_a"]["bf_mask_lifetime_in_samples"] = T;
    cfg["bad_b"]["bf_mask_lifetime_in_samples"] = alignment;

    Config config;
    config.update_config(cfg);
    configUpdater::instance().apply_config(config);
    Telescope::instance(config);

    metadataFactory mfac(config);
    std::shared_ptr<metadataPool> pool = mfac.build_pools()["chord_pool"];

    // The clock: one frame at seq0, which is not a multiple of the alignment
    Buffer clock(1, 1, pool, "clock", "standard", 0, false, false, std::vector<int>{}, true);
    clock.register_producer("test");
    // The streams run ahead of their consumers by different numbers of frames and samples
    maskStream a(pool, "bad_a", 8, T);
    maskStream b(pool, "bad_b", 4, alignment);

    bufferContainer bc;
    bc.add_buffer("clock", &clock);
    bc.add_buffer("bad_a", a.buf.get());
    bc.add_buffer("bad_b", b.buf.get());
    bufferBadInputs stage_a(config, "/bad_a", bc);
    bufferBadInputs stage_b(config, "/bad_b", bc);
    stage_a.start();
    stage_b.start();

    BOOST_REQUIRE(clock.wait_for_empty_frame("test", 0) != nullptr);
    clock.allocate_new_metadata_object(0);
    get_chord_metadata(&clock, 0)->set_fpga_seq_num(seq0);
    clock.mark_frame_full("test", 0);

    const std::vector<int8_t> all_good = {1, 1, 1, 1};
    const std::vector<int8_t> mask1 = {1, 0, 1, 1};
    const std::vector<int8_t> mask2 = {1, 1, 0, 0};

    // An update for a future sample between two boundaries applies from the next boundary in
    // both streams.
    for (bufferBadInputs* stage : {&stage_a, &stage_b}) {
        json u1 = update({1}, seq0 + 3050, "u1");
        BOOST_CHECK(stage->update_bad_inputs_callback(u1));
    }
    while (a.num_consumed < 36)
        a.consume(seq0 + a.num_consumed * T < seq0 + 3300 ? all_good : mask1);
    while (b.num_consumed < 12)
        b.consume(seq0 + b.num_consumed * alignment < seq0 + 3300 ? all_good : mask1);

    // A late update, for a boundary (seq0 + 3900) both streams have already produced, applies
    // from each stream's next frame: a has produced up to seq0 + 4300, b up to seq0 + 4500.
    a.wait_until_full();
    b.wait_until_full();
    BOOST_CHECK(!metric_has("kotekan_bufferbadinputs_late_update_count"));
    for (bufferBadInputs* stage : {&stage_a, &stage_b}) {
        json u2 = update({2, 3}, seq0 + 3650, "u2");
        BOOST_CHECK(stage->update_bad_inputs_callback(u2));
    }
    while (a.num_consumed < 50)
        a.consume(seq0 + a.num_consumed * T < seq0 + 4400 ? mask1 : mask2);
    while (b.num_consumed < 18)
        b.consume(seq0 + b.num_consumed * alignment < seq0 + 4800 ? mask1 : mask2);
    BOOST_CHECK(metric_has("kotekan_bufferbadinputs_late_update_count{stage_name=\"/bad_a\"} 1"));
    BOOST_CHECK(metric_has("kotekan_bufferbadinputs_late_update_count{stage_name=\"/bad_b\"} 1"));

    a.buf->send_shutdown_signal();
    b.buf->send_shutdown_signal();
    stage_a.stop();
    stage_b.stop();
    stage_a.join();
    stage_b.join();
}
