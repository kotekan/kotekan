#define BOOST_TEST_MODULE "test_basebandReadout"

#include "BasebandMetadata.hpp"      // for BasebandMetadata
#include "Config.hpp"                // for Config
#include "Telescope.hpp"             // for Telescope
#include "basebandApiManager.hpp"    // for basebandApiManager
#include "basebandReadout.hpp"       // for basebandReadout
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "chordMetadata.hpp"         // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"         // for configUpdater
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "metadataFactory.hpp"       // for metadataFactory
#include "prometheusMetrics.hpp"     // for Metrics
#include "restClient.hpp"            // for restClient
#include "restServer.hpp"            // for restServer
#include "test_utils.hpp"            // for add_test_telescope_config

#include "json.hpp" // for json

#include <algorithm>                         // for max, min
#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK_EQUAL, BOOST_REQUIRE
#include <chrono>                            // for milliseconds
#include <cmath>                             // for abs, round
#include <map>                               // for map
#include <memory>                            // for shared_ptr, static_pointer_cast
#include <stdint.h>                          // for int64_t, uint8_t, uint32_t
#include <string>                            // for string, to_string
#include <thread>                            // for sleep_for
#include <time.h>                            // for timespec
#include <vector>                            // for vector

using kotekan::basebandApiManager;
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::metadataFactory;
using kotekan::restServer;
using kotekan::prometheus::Metrics;
using json = nlohmann::json;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

// Frames laid out as the CRS capture makes them: [time, frequency, element], with the
// frequencies ascending and on both sides of the 400 MHz reference of the trigger.
static const std::vector<int> freq_ids = {1600, 2048, 4096, 7168}; // 312.5, 400, 800, 1400 MHz
static const int num_freq = 4;
static const int num_elements = 128;
static const int64_t samples_per_frame = 256;
static const int64_t dump_frame_samples = 64;
static const int64_t seq0 = 1000000;
static const double sample_period_s = 5.12e-6;

static uint8_t sample(int64_t seq, int freq_idx, int element) {
    return uint8_t(3 * seq + 41 * freq_idx + 7 * element);
}

static void feed_frame(Buffer& in, int64_t frame_index) {
    const int frame_id = frame_index % in.num_frames;
    uint8_t* frame = in.wait_for_empty_frame("test", frame_id);
    BOOST_REQUIRE(frame != nullptr);
    const int64_t seq = seq0 + frame_index * samples_per_frame;
    for (int64_t t = 0; t < samples_per_frame; ++t)
        for (int f = 0; f < num_freq; ++f)
            for (int e = 0; e < num_elements; ++e)
                frame[(t * num_freq + f) * num_elements + e] = sample(seq + t, f, e);
    // No first_packet_recv_time: the CRS capture does not record one.
    in.allocate_new_metadata_object(frame_id);
    auto meta = get_chord_metadata(&in, frame_id);
    meta->set_fpga_seq_num(seq);
    meta->set_coarse_freq(freq_ids);
    in.mark_frame_full("test", frame_id);
}

// Request a dump of `length` samples arriving at seq `start` at the reference frequency, and
// return the start and length the API assigned to each frequency.
static std::map<int, std::pair<int64_t, int64_t>>
request_dump(uint64_t event_id, int64_t start, int64_t length, double dm, double dm_error) {
    const timespec ts = Telescope::instance().to_time(start);
    const json request = {
        {"event_id", event_id},
        {"start_unix_seconds", ts.tv_sec},
        {"start_unix_nano", ts.tv_nsec},
        {"duration_nano", length * int64_t(Telescope::instance().seq_length_nsec())},
        {"dm", dm},
        {"dm_error", dm_error},
        {"file_path", "dumps"}};
    const auto reply = restClient::instance().make_request_blocking(
        "/baseband", request, "127.0.0.1", restServer::instance().port());
    BOOST_REQUIRE(reply.first);
    const json slices = json::parse(reply.second);
    BOOST_REQUIRE_EQUAL(slices.size(), freq_ids.size());
    std::map<int, std::pair<int64_t, int64_t>> dumps;
    for (int freq_id : freq_ids) {
        const json& slice = slices.at(std::to_string(freq_id));
        dumps[freq_id] = {slice.at("start_fpga"), slice.at("length_fpga")};
    }
    return dumps;
}

// Dispersion delay relative to 400 MHz, in samples
static double delay(int freq_id, double dm) {
    const double freq_MHz = Telescope::instance().to_freq_MHz(freq_id_t(freq_id));
    return 4.148808e3 * dm * (1. / (freq_MHz * freq_MHz) - 1. / (400. * 400.)) / sample_period_s;
}

// Read the next dump from `out` and check that it is the one of `freq_idx`, complete and in
// order, with the samples of [start, start + length).
static void check_dump(Buffer& out, int& out_frame_id, uint64_t event_id, int freq_idx,
                       int64_t start, int64_t length) {
    int64_t seq = start;
    while (seq < start + length) {
        const uint8_t* frame = out.wait_for_full_frame("test", out_frame_id);
        BOOST_REQUIRE(frame != nullptr);
        auto meta = std::static_pointer_cast<BasebandMetadata>(out.get_metadata(out_frame_id));
        BOOST_REQUIRE_EQUAL(int64_t(meta->freq_id), freq_ids[freq_idx]);
        BOOST_CHECK_EQUAL(meta->event_id, event_id);
        BOOST_CHECK_EQUAL(int64_t(meta->event_start_fpga), start);
        BOOST_CHECK_EQUAL(int64_t(meta->event_end_fpga), start + length);
        BOOST_CHECK_EQUAL(int64_t(meta->time0_fpga), start);
        BOOST_CHECK_EQUAL(meta->frame_fpga_seq, seq);
        BOOST_CHECK_EQUAL(meta->num_elements, num_elements);
        BOOST_CHECK_EQUAL(meta->first_packet_recv_time, 0.);
        const int64_t valid = std::min(dump_frame_samples, start + length - seq);
        BOOST_REQUIRE_EQUAL(meta->valid_to, valid);

        bool data_ok = true;
        for (int64_t t = 0; t < dump_frame_samples; ++t)
            for (int e = 0; e < num_elements; ++e)
                data_ok &=
                    frame[t * num_elements + e] == (t < valid ? sample(seq + t, freq_idx, e) : 0);
        BOOST_CHECK_MESSAGE(data_ok, "wrong samples for freq_id "
                                         << freq_ids[freq_idx] << " in the frame at seq " << seq);

        out.mark_frame_empty("test", out_frame_id);
        out_frame_id = (out_frame_id + 1) % out.num_frames;
        seq += valid;
    }
}

BOOST_AUTO_TEST_CASE(test_dispersed_multifreq_dump) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    add_test_telescope_config(cfg);
    cfg["gps_time"] = {{"frame0_nano", 1767225600000000000}};
    cfg["chord_pool"] = {{"kotekan_metadata_pool", "chordMetadata"}, {"num_metadata_objects", 64}};
    cfg["baseband_pool"] = {{"kotekan_metadata_pool", "BasebandMetadata"},
                            {"num_metadata_objects", 64}};
    cfg["num_polarizations"] = 2;
    cfg["num_elements"] = num_elements;
    cfg["baseband"] = {{"kotekan_stage", "basebandReadout"},
                       {"in_buf", "voltage"},
                       {"out_buf", "dump"},
                       {"num_local_freq", num_freq},
                       {"samples_per_data_set", samples_per_frame},
                       {"num_frames_buffer", 32},
                       {"max_dump_samples", 1000}};

    Config config;
    config.update_config(cfg);
    configUpdater::instance().apply_config(config);
    Telescope::instance(config);

    restServer::instance().start(std::vector<std::string>{"127.0.0.1"}, 0);
    basebandApiManager::instance().register_with_server(&restServer::instance());

    metadataFactory mfac(config);
    auto pools = mfac.build_pools();
    Buffer in(40, samples_per_frame * num_freq * num_elements, pools["chord_pool"], "voltage",
              "standard", 0, false, false, std::vector<int>{}, true);
    Buffer out(64, dump_frame_samples * num_elements, pools["baseband_pool"], "dump", "standard", 0,
               false, false, std::vector<int>{}, true);
    in.register_producer("test");
    out.register_consumer("test");

    bufferContainer bc;
    bc.add_buffer("voltage", &in);
    bc.add_buffer("dump", &out);
    basebandReadout stage(config, "/baseband", bc);
    stage.start();

    // The stage registers its frequencies with the API on its first frame, the last one last.
    int64_t num_fed = 0;
    while (num_fed < 12)
        feed_frame(in, num_fed++);
    const std::string last_registered =
        "kotekan_baseband_readout_in_progress{stage_name=\"/baseband\",freq_id=\""
        + std::to_string(freq_ids.back()) + "\"";
    while (Metrics::instance().serialize().find(last_registered) == std::string::npos)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // A dispersed event arriving at 400 MHz in frame 10. Each frequency is dumped at its own
    // delay, widened by the DM error: before the reference time above 400 MHz, after it below.
    const int64_t start = seq0 + 10 * samples_per_frame + 37;
    const int64_t length = 300;
    const double dm = 0.1;
    const double dm_error = 0.02;
    auto dumps = request_dump(1, start, length, dm, dm_error);
    for (int freq_id : freq_ids) {
        const double first = std::min(delay(freq_id, dm - dm_error), delay(freq_id, dm + dm_error));
        const double last = std::max(delay(freq_id, dm - dm_error), delay(freq_id, dm + dm_error));
        BOOST_CHECK_LE(std::abs(dumps[freq_id].first - (start + std::round(first))), 1);
        BOOST_CHECK_LE(std::abs(dumps[freq_id].second - (length + std::round(last - first))), 1);
    }
    BOOST_CHECK_LT(dumps[7168].first, dumps[4096].first);
    BOOST_CHECK_LT(dumps[4096].first, start);
    BOOST_CHECK_EQUAL(dumps[2048].first, start);
    BOOST_CHECK_EQUAL(dumps[2048].second, length);
    BOOST_CHECK_GT(dumps[1600].first, start + 200);
    BOOST_CHECK_GT(dumps[1600].second, length + 100);

    // The dumps come out highest frequency first, which is the order the data arrived in.
    int out_frame_id = 0;
    for (int freq_idx : {3, 2, 1})
        check_dump(out, out_frame_id, 1, freq_idx, dumps[freq_ids[freq_idx]].first,
                   dumps[freq_ids[freq_idx]].second);

    // The end of the 312.5 MHz dump is in frame 12, which has not arrived: the readout waits.
    BOOST_REQUIRE_GE(dumps[1600].first + dumps[1600].second, seq0 + 12 * samples_per_frame);
    while (num_fed < 20)
        feed_frame(in, num_fed++);
    check_dump(out, out_frame_id, 1, 0, dumps[1600].first, dumps[1600].second);

    // An undispersed event, after the input ring has wrapped around.
    while (num_fed < 60)
        feed_frame(in, num_fed++);
    dumps = request_dump(2, seq0 + 55 * samples_per_frame + 11, 500, 0., 0.);
    for (int freq_idx : {3, 2, 1, 0}) {
        BOOST_CHECK_EQUAL(dumps[freq_ids[freq_idx]].first, seq0 + 55 * samples_per_frame + 11);
        BOOST_CHECK_EQUAL(dumps[freq_ids[freq_idx]].second, 500);
        check_dump(out, out_frame_id, 2, freq_idx, dumps[freq_ids[freq_idx]].first,
                   dumps[freq_ids[freq_idx]].second);
    }

    in.send_shutdown_signal();
    out.send_shutdown_signal();
    stage.stop();
    stage.join();
}
