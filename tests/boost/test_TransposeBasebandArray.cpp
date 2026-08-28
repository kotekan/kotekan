#define BOOST_TEST_MODULE "test_TransposeBasebandArray"

#include "Config.hpp"
#include "DataType.hpp"
#include "NDArray.hpp"
#include "Stage.hpp"
#include "TransposeBasebandArray.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "metadata.hpp"
#include "metadataFactory.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace kotekan;
using json = nlohmann::json;

// The stage hard-codes these, and fatally checks the config against them.
static const uint32_t num_local_freq = 384;
static const uint32_t num_elements = 128;
static const uint32_t time_short = 16;
static const uint32_t element_short = 8;
static const uint32_t element_long = num_elements / element_short;

// Small enough to keep the test quick, divisible by 64 for the pl_mask layout.
static const uint32_t timesamples_per_frame = 64;
static const uint32_t time_long = timesamples_per_frame / time_short;

static const size_t in_frame_size =
    (size_t)time_long * num_local_freq * element_long * time_short * element_short;
static const size_t out_frame_size = (size_t)timesamples_per_frame * num_local_freq * num_elements;
static const size_t pl_mask_frame_size =
    (size_t)(timesamples_per_frame / 64) * num_local_freq * (num_elements / 8) * sizeof(uint64_t);

static const uint8_t fill_value = 0x88;

static Config make_test_config(const std::string& stage_name, bool disable_avx512) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["chord_pool"] = {{"kotekan_metadata_pool", "chordMetadata"}, {"num_metadata_objects", 16}};

    json stage;
    stage["kotekan_stage"] = "TransposeBasebandArray";
    stage["log_level"] = "INFO";
    stage["cpu_affinity"] = std::vector<int>{0};
    stage["in_buf"] = "in_buf";
    stage["out_buf"] = "out_buf";
    stage["pl_mask_buf"] = "pl_mask_buf";
    stage["timesamples_per_frame"] = timesamples_per_frame;
    stage["num_local_freq"] = num_local_freq;
    stage["num_elements"] = num_elements;
    stage["time_short"] = time_short;
    stage["element_short"] = element_short;
    stage["disable_avx512"] = disable_avx512;

    cfg[stage_name.substr(1)] = stage;

    Config conf;
    conf.update_config(cfg);
    return conf;
}

struct TestSetup {
    std::shared_ptr<metadataPool> chord_pool;
    std::unique_ptr<Buffer> in_buf;
    std::unique_ptr<Buffer> out_buf;
    std::unique_ptr<Buffer> pl_mask_buf;

    TestSetup(Config& config) {
        metadataFactory mfac(config);
        auto pools = mfac.build_pools();
        chord_pool = pools["chord_pool"];

        const int num_frames = 2;
        // Matches the network_input_buffer configuration: unwritten bytes read as 0x88.
        in_buf =
            std::make_unique<Buffer>(num_frames, in_frame_size, chord_pool, "in_buf", "standard", 0,
                                     false, false, std::vector<int>{}, true, fill_value);
        in_buf->register_producer("test_producer");

        out_buf = std::make_unique<Buffer>(num_frames, out_frame_size, chord_pool, "out_buf",
                                           "ndarray", 0, false, false, std::vector<int>{}, true);
        out_buf->ensure_frame_desc(GenericNDArray::describe(kotekan::int4x2_swapped_withoffset, "E",
                                                            {(std::ptrdiff_t)timesamples_per_frame,
                                                             (std::ptrdiff_t)num_local_freq, 2,
                                                             (std::ptrdiff_t)num_elements / 2},
                                                            {"T", "F", "P", "D"}, {1, 1, 1, 1}));
        out_buf->register_consumer("test_consumer");

        pl_mask_buf =
            std::make_unique<Buffer>(num_frames, pl_mask_frame_size, chord_pool, "pl_mask_buf",
                                     "standard", 0, false, false, std::vector<int>{}, true);
        pl_mask_buf->register_producer("test_producer");
    }
};

// Distinct per (source element, time, frequency) so a mix-up on any axis is caught.
// 0x88 is excluded so that fill bytes are never mistaken for real data.
static uint8_t source_value(uint32_t element, uint32_t time, uint32_t freq) {
    uint8_t v = (uint8_t)((element * 31u + time * 17u + freq * 11u) & 0xff);
    return v == fill_value ? (uint8_t)(fill_value + 1) : v;
}

// Fill one input frame, writing only the element_long blocks in live_boards. Blocks for
// boards not listed keep the buffer's 0x88 fill, as an unwired CRS board would.
static void produce_input_frame(Buffer* in_buf, Buffer* pl_mask_buf, int frame_id,
                                const std::vector<uint32_t>& live_boards, bool packet_loss) {
    uint8_t* frame = in_buf->wait_for_empty_frame("test_producer", frame_id);
    BOOST_REQUIRE(frame != nullptr);
    std::memset(frame, fill_value, in_frame_size);

    for (uint32_t t_long = 0; t_long < time_long; t_long++) {
        for (uint32_t freq = 0; freq < num_local_freq; freq++) {
            const size_t block = ((size_t)t_long * num_local_freq + freq) * element_long
                                 * time_short * element_short;
            for (uint32_t board : live_boards) {
                for (uint32_t t_s = 0; t_s < time_short; t_s++) {
                    for (uint32_t e_s = 0; e_s < element_short; e_s++) {
                        const size_t off =
                            block + board * time_short * element_short + t_s * element_short + e_s;
                        frame[off] = source_value(board * element_short + e_s,
                                                  t_long * time_short + t_s, freq);
                    }
                }
            }
        }
    }

    in_buf->allocate_new_metadata_object(frame_id);
    auto meta = get_chord_metadata(in_buf, frame_id);
    meta->set_fpga_seq_num(1000 * frame_id);
    // The stage forwards these to the output frame, so they must be present.
    meta->set_coarse_freq(std::vector<int>(num_local_freq, 0));
    meta->set_freq_upchan_factor(std::vector<int>(num_local_freq, 1));
    meta->set_time_downsampling_fpga(1);
    in_buf->mark_frame_full("test_producer", frame_id);

    uint8_t* mask = pl_mask_buf->wait_for_empty_frame("test_producer", frame_id);
    BOOST_REQUIRE(mask != nullptr);
    // 0xff everywhere is "all data good"; clearing it all marks every block lost.
    std::memset(mask, packet_loss ? 0x00 : 0xff, pl_mask_frame_size);
    pl_mask_buf->mark_frame_full("test_producer", frame_id);
}

// Independent reference transpose, deliberately written as the plainest possible loop.
static std::vector<uint8_t> expected_output(const std::vector<uint32_t>& live_boards,
                                            bool packet_loss) {
    std::vector<uint8_t> out(out_frame_size, 0);
    for (uint32_t time = 0; time < timesamples_per_frame; time++) {
        for (uint32_t freq = 0; freq < num_local_freq; freq++) {
            for (uint32_t out_e = 0; out_e < num_elements; out_e++) {
                const size_t idx = ((size_t)time * num_local_freq + freq) * num_elements + out_e;
                const uint32_t src = out_e;
                const uint32_t board = src / element_short;
                const bool live =
                    std::find(live_boards.begin(), live_boards.end(), board) != live_boards.end();
                out[idx] = (packet_loss || !live) ? fill_value : source_value(src, time, freq);
            }
        }
    }
    return out;
}

// Compare and report the first mismatch, rather than one BOOST assertion per byte.
static void check_frame(const uint8_t* actual, const std::vector<uint8_t>& expected) {
    if (std::memcmp(actual, expected.data(), expected.size()) == 0) {
        BOOST_CHECK(true);
        return;
    }
    for (size_t i = 0; i < expected.size(); i++) {
        if (actual[i] != expected[i]) {
            const uint32_t out_e = i % num_elements;
            const uint32_t freq = (i / num_elements) % num_local_freq;
            const uint32_t time = i / ((size_t)num_local_freq * num_elements);
            BOOST_FAIL("mismatch at time " << time << " freq " << freq << " element " << out_e
                                           << ": got 0x" << std::hex << (int)actual[i]
                                           << ", expected 0x" << (int)expected[i]);
        }
    }
}

// Run one frame through the stage and compare against the reference.
static void run_case(const std::string& name, const std::vector<uint32_t>& live_boards,
                     bool packet_loss, bool disable_avx512) {
    std::cout << "  " << name << (disable_avx512 ? " [scalar]" : " [avx512]") << "\n";

    const std::string stage_name = "/test_transpose";
    Config config = make_test_config(stage_name, disable_avx512);
    TestSetup setup(config);

    bufferContainer bc;
    bc.add_buffer("in_buf", setup.in_buf.get());
    bc.add_buffer("out_buf", setup.out_buf.get());
    bc.add_buffer("pl_mask_buf", setup.pl_mask_buf.get());

    TransposeBasebandArray stage(config, stage_name, bc);
    stage.start();

    produce_input_frame(setup.in_buf.get(), setup.pl_mask_buf.get(), 0, live_boards, packet_loss);

    const uint8_t* out = setup.out_buf->wait_for_full_frame("test_consumer", 0);
    BOOST_REQUIRE(out != nullptr);
    check_frame(out, expected_output(live_boards, packet_loss));
    setup.out_buf->mark_frame_empty("test_consumer", 0);

    setup.in_buf->send_shutdown_signal();
    setup.pl_mask_buf->send_shutdown_signal();
    stage.stop();
    stage.join();
}
// All 16 boards present, and the 4 that are live on the pathfinder today.
static const std::vector<uint32_t> all_boards = {0, 1, 2,  3,  4,  5,  6,  7,
                                                 8, 9, 10, 11, 12, 13, 14, 15};
static const std::vector<uint32_t> live_boards = {0, 1, 2, 3};

BOOST_AUTO_TEST_CASE(test_transpose_all_boards) {
    std::cout << "Testing TransposeBasebandArray: transpose with every board present...\n";
    for (bool scalar : {false, true})
        run_case("all boards wired", all_boards, false, scalar);
    std::cout << "Success.\n";
}

// Source elements no board writes keep the input buffer's fill value, which is the
// encoded zero for int4x2_swapped_withoffset rather than 0x00.
BOOST_AUTO_TEST_CASE(test_unwired_boards_are_encoded_zero) {
    std::cout << "Testing TransposeBasebandArray: unwired boards stay 0x88...\n";
    for (bool scalar : {false, true})
        run_case("4 of 16 boards live", live_boards, false, scalar);
    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_packet_loss_fill) {
    std::cout << "Testing TransposeBasebandArray: packet loss fills 0x88...\n";
    for (bool scalar : {false, true})
        run_case("all blocks lost", all_boards, true, scalar);
    std::cout << "Success.\n";
}
