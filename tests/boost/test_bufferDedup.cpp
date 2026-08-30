#define BOOST_TEST_MODULE "test_bufferDedup"

#include "Config.hpp"          // for Config
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "bufferDedup.hpp"     // for bufferDedup
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "metadata.hpp"        // for metadataPool
#include "metadataFactory.hpp" // for metadataFactory
#include "test_utils.hpp"      // for GlobalFixture_Locale

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace kotekan;
using json = nlohmann::json;

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

static const size_t frame_size = 64;
static const int num_frames = 4;

static Config make_test_config(const std::string& stage_name, int resend_after_frames) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["chord_pool"] = {{"kotekan_metadata_pool", "chordMetadata"}, {"num_metadata_objects", 32}};

    json stage;
    stage["kotekan_stage"] = "bufferDedup";
    stage["cpu_affinity"] = std::vector<int>{0};
    stage["in_buf"] = "in_buf";
    stage["out_buf"] = "out_buf";
    if (resend_after_frames > 0)
        stage["resend_after_frames"] = resend_after_frames;
    cfg[stage_name.substr(1)] = stage;

    Config conf;
    conf.update_config(cfg);
    return conf;
}

struct TestSetup {
    std::shared_ptr<metadataPool> chord_pool;
    std::unique_ptr<Buffer> in_buf;
    std::unique_ptr<Buffer> out_buf;
    bufferContainer bc;

    TestSetup(Config& config) {
        metadataFactory mfac(config);
        chord_pool = mfac.build_pools()["chord_pool"];

        in_buf = std::make_unique<Buffer>(num_frames, frame_size, chord_pool, "in_buf", "standard",
                                          0, false, false, std::vector<int>{}, true);
        in_buf->register_producer("test_producer");

        out_buf = std::make_unique<Buffer>(num_frames, frame_size, chord_pool, "out_buf",
                                           "standard", 0, false, false, std::vector<int>{}, true);
        out_buf->register_consumer("test_consumer");

        bc.add_buffer("in_buf", in_buf.get());
        bc.add_buffer("out_buf", out_buf.get());
    }
};

// Produce one input frame filled with `value`, seq-stamped so forwarded frames can be
// traced back to the input frame that carried them.
static void produce_frame(Buffer* in_buf, int frame_id, uint8_t value, int64_t seq) {
    uint8_t* frame = in_buf->wait_for_empty_frame("test_producer", frame_id);
    BOOST_REQUIRE(frame != nullptr);
    std::memset(frame, value, frame_size);
    in_buf->allocate_new_metadata_object(frame_id);
    get_chord_metadata(in_buf, frame_id)->set_fpga_seq_num(seq);
    in_buf->mark_frame_full("test_producer", frame_id);
}

// Consume one output frame; check its fill value and the seq it carried.
static void expect_output(Buffer* out_buf, int frame_id, uint8_t value, int64_t seq) {
    const uint8_t* frame = out_buf->wait_for_full_frame("test_consumer", frame_id);
    BOOST_REQUIRE(frame != nullptr);
    for (size_t i = 0; i < frame_size; ++i)
        BOOST_CHECK_EQUAL(frame[i], value);
    BOOST_CHECK_EQUAL(get_chord_metadata(out_buf, frame_id)->get_fpga_seq_num(), seq);
    out_buf->mark_frame_empty("test_consumer", frame_id);
}

// No output frame should be waiting: a zero deadline turns the wait into a poll.
static void expect_no_output(Buffer* out_buf, int frame_id) {
    const timespec poll = {0, 0};
    BOOST_CHECK_EQUAL(out_buf->wait_for_full_frame_timeout("test_consumer", frame_id, poll), 1);
}

// The first frame and every content change are forwarded; repeats are suppressed.
BOOST_AUTO_TEST_CASE(forward_on_change) {
    const std::string stage_name = "/test_dedup";
    Config config = make_test_config(stage_name, 0);
    TestSetup setup(config);

    bufferDedup stage(config, stage_name, setup.bc);
    stage.start();

    // A A A B B C: expect A(seq 0), B(seq 300), C(seq 500) forwarded.
    const uint8_t values[] = {0xa1, 0xa1, 0xa1, 0xb2, 0xb2, 0xc3};
    for (int i = 0; i < 6; ++i)
        produce_frame(setup.in_buf.get(), i % num_frames, values[i], 100 * i);

    expect_output(setup.out_buf.get(), 0, 0xa1, 0);
    expect_output(setup.out_buf.get(), 1, 0xb2, 300);
    expect_output(setup.out_buf.get(), 2, 0xc3, 500);
    expect_no_output(setup.out_buf.get(), 3);

    setup.in_buf->send_shutdown_signal();
    stage.stop();
    stage.join();
}

// With resend_after_frames set, an unchanged frame is forwarded after that many
// suppressed frames, so a lossy consumer recovers the current contents.
BOOST_AUTO_TEST_CASE(periodic_resend) {
    const std::string stage_name = "/test_dedup";
    Config config = make_test_config(stage_name, 2);
    TestSetup setup(config);

    bufferDedup stage(config, stage_name, setup.bc);
    stage.start();

    // A then five repeats: A forwarded (change), two suppressed, resend, two
    // suppressed... only frames 0 and 3 come through.
    for (int i = 0; i < 6; ++i)
        produce_frame(setup.in_buf.get(), i % num_frames, 0xa1, 100 * i);

    expect_output(setup.out_buf.get(), 0, 0xa1, 0);
    expect_output(setup.out_buf.get(), 1, 0xa1, 300);
    expect_no_output(setup.out_buf.get(), 2);

    setup.in_buf->send_shutdown_signal();
    stage.stop();
    stage.join();
}
