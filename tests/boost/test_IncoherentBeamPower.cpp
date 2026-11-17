#define BOOST_TEST_MODULE "test_IncoherentBeamPower"

#include "Config.hpp"
#include "NDArray.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLoggingFixture.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

struct PowerPipeline {
    static constexpr int T = 4, F = 3;
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool;
    Buffer input, output;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;

    PowerPipeline() :
        pool(metadataPool::create(32, sizeof(chordMetadata), "pool", "chordMetadata")),
        input(2, T * F * 3 * sizeof(uint64_t), pool, "input", "standard", 0, false, false,
              std::vector<int>{}, true),
        output(2, T * F * sizeof(uint16_t), pool, "output", "standard", 0, false, false,
               std::vector<int>{}, true) {
        // Force metadata registration from the static library.
        chordMetadata metadata;
        config.update_config({{"cpu_affinity", {0}},
                              {"log_level", "ERROR"},
                              {"power", {{"in_buf", "input"}, {"out_buf", "output"}}}});
        input.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint64, "S012tilde", {T, F, 3}, {"Trfi", "F", "S"}, {256, 1, 1}));
        buffers.add_buffer("input", &input);
        buffers.add_buffer("output", &output);
        input.register_producer("test");
        output.register_consumer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("IncoherentBeamPower")->create(config, "/power", buffers));
    }

    ~PowerPipeline() {
        input.send_shutdown_signal();
        output.send_shutdown_signal();
        stage->stop();
        stage->join();
    }
};

BOOST_AUTO_TEST_CASE(power_and_metadata) {
    PowerPipeline p;
    p.stage->start();

    // {S0, S1} for each value: none, exact, rounding down and up, the 4-bit maximum, saturation.
    const uint64_t s01[][2] = {{0, 0},          {0, 99},      {1000, 0},        {32000, 224000},
                               {1000, 7001},    {1000, 7002}, {32000, 4096000}, {1, 1000},
                               {31744, 232000}, {3, 1},       {3, 2},           {32000, 1}};
    const uint16_t expected[] = {0, 0, 0, 1792, 1792, 1793, 32768, 65535, 1871, 85, 171, 0};
    const int64_t seq = (int64_t(1) << 34) + 4096;

    BOOST_REQUIRE(p.input.wait_for_empty_frame("test", 0));
    p.input.allocate_new_metadata_object(0);
    auto meta = get_chord_metadata(&p.input, 0);
    meta->set_from_frame_desc(p.input.get_frame_desc<kotekan::GenericNDArray>());
    meta->set_fpga_seq_num(seq);
    meta->set_time_downsampling_fpga(256);
    meta->set_coarse_freq({10, 11, 12});
    uint64_t* const s012 = (uint64_t*)p.input.frames[0];
    for (int i = 0; i < PowerPipeline::T * PowerPipeline::F; ++i) {
        s012[3 * i] = s01[i][0];
        s012[3 * i + 1] = s01[i][1];
        s012[3 * i + 2] = ~uint64_t(0); // S2 is not used
    }
    p.input.mark_frame_full("test", 0);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (p.output.get_num_full_frames() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    BOOST_REQUIRE_GT(p.output.get_num_full_frames(), 0);
    const uint16_t* const power = (const uint16_t*)p.output.wait_for_full_frame("test", 0);
    BOOST_REQUIRE(power);
    for (int i = 0; i < PowerPipeline::T * PowerPipeline::F; ++i)
        BOOST_CHECK_EQUAL(power[i], expected[i]);

    const auto out_meta = get_chord_metadata(&p.output, 0);
    BOOST_REQUIRE(out_meta);
    BOOST_CHECK_EQUAL(out_meta->get_fpga_seq_num(), seq);
    BOOST_CHECK_EQUAL(out_meta->get_time_downsampling_fpga(), 256);
    BOOST_CHECK(out_meta->get_coarse_freq() == std::vector<int>({10, 11, 12}));
    const auto desc = p.output.get_frame_desc<kotekan::GenericNDArray>();
    BOOST_REQUIRE(desc);
    BOOST_CHECK(desc->get_value_datatype() == kotekan::uint16);
    BOOST_CHECK(desc->get_extents()
                == std::vector<std::ptrdiff_t>({PowerPipeline::T, PowerPipeline::F}));
    BOOST_CHECK_EQUAL(desc->get_dimscaling(0), 256);
    p.output.mark_frame_empty("test", 0);
}
