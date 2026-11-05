#define BOOST_TEST_MODULE "test_bufferGains"

#include "Config.hpp"                // for Config
#include "NDArray.hpp"               // for NDArray
#include "Telescope.hpp"             // for Telescope
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "bufferGains.hpp"           // for bufferGains
#include "chordMetadata.hpp"         // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"         // for configUpdater
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "metadataFactory.hpp"       // for metadataFactory
#include "modp_b64.hpp"              // for modp_b64_encode, modp_b64_encode_len
#include "prometheusMetrics.hpp"     // for Metrics
#include "test_utils.hpp"            // for add_test_telescope_config
#include "visUtil.hpp"               // for current_time, double_to_ts

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK_EQUAL, BOOST_REQUIRE
#include <complex>                           // for complex
#include <memory>                            // for unique_ptr, make_unique, shared_ptr
#include <stdexcept>                         // for runtime_error
#include <string>                            // for string
#include <vector>                            // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::metadataFactory;
using kotekan::prometheus::Metrics;
using json = nlohmann::json;
using cfloat = std::complex<float>;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

// The test telescope has 2 dishes.
static const int num_polarizations = 2;
static const int num_dishes = 2;
static const int num_elements = num_polarizations * num_dishes;
static const std::vector<int> local_freq = {5, 7, 9};
static const int num_local_freq = 3;
static const cfloat unit = {1.0f, 0.0f};

// A gains array as the calibration service posts it: base64 of the complex64 values.
static json gain_json(const std::vector<cfloat>& gains, const std::vector<size_t>& shape) {
    const size_t len = gains.size() * sizeof(cfloat);
    std::string encoded(modp_b64_encode_len(len), '\0');
    encoded.resize(modp_b64_encode(&encoded[0], reinterpret_cast<const char*>(gains.data()), len));
    return {{"shape", shape}, {"dtype", "complex64"}, {"encoding", "base64"}, {"data", encoded}};
}

// An update with gains[f * num_elements + e] for freq_id[f].
static json update(const std::string& id, const std::vector<int>& freq_id,
                   const std::vector<cfloat>& gains) {
    return {{"update_id", id},
            {"freq_id", freq_id},
            {"gain", gain_json(gains, {freq_id.size(), size_t(num_elements)})}};
}

// gains[f * num_elements + e] = freq_id[f] + i e, so a frame shows where each row came from.
static std::vector<cfloat> labelled(const std::vector<int>& freq_id) {
    std::vector<cfloat> gains;
    for (int f : freq_id)
        for (int e = 0; e < num_elements; ++e)
            gains.emplace_back(float(f), float(e));
    return gains;
}

static bool metric_has(const std::string& line) {
    return Metrics::instance().serialize().find(line) != std::string::npos;
}

// The gains buffer, read as a consumer would.
struct gainStream {
    std::unique_ptr<Buffer> buf;
    int num_consumed = 0;

    gainStream(const std::shared_ptr<metadataPool>& pool, const std::string& name, int num_frames) {
        buf = std::make_unique<Buffer>(num_frames, num_local_freq * num_elements * sizeof(cfloat),
                                       pool, name, "ndarray", 0, false, false, std::vector<int>{},
                                       true);
        buf->ensure_frame_desc(kotekan::NDArray<float, 4>::describe(
            "gain", {num_local_freq, num_polarizations, num_dishes, 2}, {"F", "P", "D", "C"},
            {1, 1, 1, 1}));
        buf->register_consumer("test");
    }

    // Consume the next frame and check it holds `expected`, [F][E] in local frequency order.
    void consume(const std::vector<cfloat>& expected) {
        const int frame_id = num_consumed % buf->num_frames;
        const cfloat* frame = (const cfloat*)buf->wait_for_full_frame("test", frame_id);
        BOOST_REQUIRE(frame != nullptr);
        BOOST_CHECK(get_chord_metadata(buf.get(), frame_id)->get_coarse_freq() == local_freq);
        const std::vector<cfloat> got(frame, frame + num_local_freq * num_elements);
        BOOST_CHECK_MESSAGE(got == expected,
                            buf->buffer_name << ": wrong gains in frame " << num_consumed);
        buf->mark_frame_empty("test", frame_id);
        ++num_consumed;
    }

    // True if no frame arrives within `seconds`.
    bool quiet(double seconds) {
        return buf->wait_for_full_frame_timeout("test", num_consumed % buf->num_frames,
                                                double_to_ts(current_time() + seconds))
               == 1;
    }
};

// The config of one bufferGains stage `name`, fed by the updatable block of the same name and
// the clock buffer "clock". Each test uses its own name: metrics and endpoints are global.
static json make_config(const std::string& name, const json& initial_block) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    add_test_telescope_config(cfg);
    cfg["chord_pool"] = {{"kotekan_metadata_pool", "chordMetadata"}, {"num_metadata_objects", 32}};
    cfg["updatable_config"][name] = initial_block;
    cfg["updatable_config"][name]["kotekan_update_endpoint"] = "json";
    cfg["num_local_freq"] = num_local_freq;
    cfg["num_polarizations"] = num_polarizations;
    cfg["num_dishes"] = num_dishes;
    cfg[name]["kotekan_stage"] = "bufferGains";
    cfg[name]["updatable_config"]["gains"] = "/updatable_config/" + name;
    cfg[name]["metadata_source"] = "clock";
    cfg[name]["out_buf"] = "gains_out";
    return cfg;
}

// Everything a test needs to run the stage: the clock, the output stream and the stage itself.
struct harness {
    Config config;
    std::shared_ptr<metadataPool> pool;
    std::unique_ptr<Buffer> clock;
    std::unique_ptr<gainStream> out;
    bufferContainer bc;
    std::unique_ptr<bufferGains> stage;

    harness(const std::string& name, const json& cfg) {
        config.update_config(cfg);
        configUpdater::instance().apply_config(config);
        Telescope::instance(config);
        metadataFactory mfac(config);
        pool = mfac.build_pools()["chord_pool"];
        clock = std::make_unique<Buffer>(1, 1, pool, "clock", "standard", 0, false, false,
                                         std::vector<int>{}, true);
        clock->register_producer("test");
        out = std::make_unique<gainStream>(pool, "gains_out", 2);
        bc.add_buffer("clock", clock.get());
        bc.add_buffer("gains_out", out->buf.get());
        stage = std::make_unique<bufferGains>(config, "/" + name, bc);
    }

    // Start the stage and give it its frequencies.
    void start() {
        stage->start();
        BOOST_REQUIRE(clock->wait_for_empty_frame("test", 0) != nullptr);
        clock->allocate_new_metadata_object(0);
        get_chord_metadata(clock.get(), 0)->set_fpga_seq_num(1);
        get_chord_metadata(clock.get(), 0)->set_coarse_freq(local_freq);
        clock->mark_frame_full("test", 0);
    }

    ~harness() {
        out->buf->send_shutdown_signal();
        stage->stop();
        stage->join();
        configUpdater::instance().reset();
    }
};

BOOST_AUTO_TEST_CASE(test_default_then_updates) {
    harness h("gains", make_config("gains", {{"update_id", "initial"},
                                             {"freq_id", json::array()},
                                             {"gain", json::object()}}));
    h.start();

    // The initial block has no gains, so the first frame holds the default unit gains.
    h.out->consume(std::vector<cfloat>(num_local_freq * num_elements, unit));

    // An update with the frequencies in another order, and one this instance does not have.
    json u1 = update("u1", {9, 100, 5, 7}, labelled({9, 100, 5, 7}));
    BOOST_CHECK(h.stage->receive_gains(u1));
    h.out->consume(labelled(local_freq));

    // Updates missing a frequency, with the wrong element count, or with another dtype are
    // ignored and counted, not fatal, and produce no frame.
    json u2 = update("u2", {5, 9}, labelled({5, 9}));
    BOOST_CHECK(h.stage->receive_gains(u2));
    json u3 = {{"update_id", "u3"},
               {"freq_id", local_freq},
               {"gain", gain_json(std::vector<cfloat>(num_local_freq * 3, unit),
                                  {size_t(num_local_freq), 3})}};
    BOOST_CHECK(h.stage->receive_gains(u3));
    json u4 = update("u4", local_freq, labelled(local_freq));
    u4["gain"]["dtype"] = "complex128";
    BOOST_CHECK(h.stage->receive_gains(u4));
    BOOST_CHECK(h.out->quiet(0.3));
    BOOST_CHECK(metric_has("kotekan_buffergains_invalid_update_count{stage_name=\"/gains\"} 3"));
    BOOST_CHECK(metric_has("kotekan_buffergains_update_count{stage_name=\"/gains\"} 2"));

    // An empty gain object goes back to the default gains.
    json u5 = {{"update_id", "u5"}, {"freq_id", json::array()}, {"gain", json::object()}};
    BOOST_CHECK(h.stage->receive_gains(u5));
    h.out->consume(std::vector<cfloat>(num_local_freq * num_elements, unit));
}

BOOST_AUTO_TEST_CASE(test_initial_block_with_gains) {
    // Gains in the config block, as choco can provide at startup, are the first frame.
    const std::vector<int> freq_id = {7, 9, 5};
    harness h("gains_init",
              make_config("gains_init", update("startup", freq_id, labelled(freq_id))));
    h.start();
    h.out->consume(labelled(local_freq));
}

BOOST_AUTO_TEST_CASE(test_bad_initial_block) {
    // A malformed initial block is a config error and fails construction.
    json block = update("startup", local_freq, labelled(local_freq));
    block["gain"]["dtype"] = "complex128";
    const json cfg = make_config("gains_bad", block);
    BOOST_CHECK_THROW(harness h("gains_bad", cfg), std::runtime_error);
    configUpdater::instance().reset();
}
