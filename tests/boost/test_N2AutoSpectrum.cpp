#define BOOST_TEST_MODULE "test_N2AutoSpectrum"

#include "Config.hpp"
#include "DataType.hpp"
#include "N2AutoSpectrum.hpp"
#include "N2FrameDesc.hpp"
#include "N2FrameView.hpp"
#include "N2Layout.hpp"
#include "N2Metadata.hpp"
#include "N2Util.hpp"
#include "NDArray.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "metadata.hpp"
#include "metadataFactory.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace kotekan;
using json = nlohmann::json;

static const uint32_t num_elements = 4;
static const int64_t num_local_freq = 3;

// Config with the stage section plus the two metadata pools the buffers use
static Config make_test_config(const std::string& stage_name, const std::string& in_buf_name,
                               const std::string& out_buf_name, uint32_t num_pending_bins) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["n2_pool"] = {{"kotekan_metadata_pool", "N2Metadata"}, {"num_metadata_objects", 16}};
    cfg["chord_pool"] = {{"kotekan_metadata_pool", "chordMetadata"}, {"num_metadata_objects", 16}};

    json stage;
    stage["kotekan_stage"] = "N2AutoSpectrum";
    stage["log_level"] = "DEBUG";
    stage["cpu_affinity"] = std::vector<int>{0};
    stage["in_buf"] = in_buf_name;
    stage["out_buf"] = out_buf_name;
    stage["num_local_freq"] = num_local_freq;
    stage["freq_id_base"] = 0;
    stage["num_pending_bins"] = num_pending_bins;

    cfg[stage_name] = stage;
    if (!stage_name.empty() && stage_name.front() == '/')
        cfg[stage_name.substr(1)] = stage;

    Config conf;
    conf.update_config(cfg);
    return conf;
}

// Expected autocorrelation value for (bin, freq, element)
static float auto_value(uint64_t bin, uint32_t freq_id, uint32_t element) {
    return 1000.0f * bin + 100.0f * freq_id + element;
}

// Fill and submit one N2 input frame for (bin, freq_id). Autos get
// auto_value() with a nonzero imaginary part (only the real part should
// be extracted); cross products get sentinel values.
static void produce_input_frame(Buffer* buf, int frame_id, uint64_t bin, uint32_t freq_id) {
    BOOST_REQUIRE(buf->wait_for_empty_frame("test_producer", frame_id) != nullptr);

    buf->allocate_new_metadata_object(frame_id);
    auto meta = get_N2_metadata(buf, frame_id);
    meta->freq_id = freq_id;
    meta->abs_time_idx = bin;
    meta->fpga_start_tick = 1000 * bin;
    meta->frame_start_time_ns = 1000000000ULL * bin;
    meta->frame_length_fpga_ticks = 100;
    meta->n_valid_fpga_ticks = 100;
    meta->n_rfi_fpga_ticks = 0;
    meta->n_rfi_only_fpga_ticks = 0;
    meta->n_pl_fpga_ticks = 0;

    N2FrameView fv(buf, frame_id);
    const auto& prods = fv._desc->get_product_list();
    for (size_t k = 0; k < prods.size(); ++k) {
        if (prods[k].input_a == prods[k].input_b)
            fv.vis[k] = N2::cfloat(auto_value(bin, freq_id, prods[k].input_a), 0.5f);
        else
            fv.vis[k] = N2::cfloat(-1.0f, -1.0f);
        fv.weight[k] = 1.0f;
    }

    buf->mark_frame_full("test_producer", frame_id);
}

// Check one output row against auto_value(), or all-NaN for a missing freq
static void check_row(const float* frame, int64_t row, bool present, uint64_t bin,
                      uint32_t freq_id) {
    for (uint32_t e = 0; e < num_elements; ++e) {
        float val = frame[row * num_elements + e];
        if (present)
            BOOST_CHECK_EQUAL(val, auto_value(bin, freq_id, e));
        else
            BOOST_CHECK(std::isnan(val));
    }
}

// Build the pair of buffers and pools shared by the test cases
struct TestSetup {
    std::shared_ptr<metadataPool> n2_pool;
    std::shared_ptr<metadataPool> chord_pool;
    std::unique_ptr<Buffer> in_buf;
    std::unique_ptr<Buffer> out_buf;

    TestSetup(Config& config, const std::string& in_buf_name, const std::string& out_buf_name) {
        metadataFactory mfac(config);
        auto pools = mfac.build_pools();
        n2_pool = pools["n2_pool"];
        chord_pool = pools["chord_pool"];

        const uint32_t num_frames = 4;
        uint32_t num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::FullUpperTri);
        size_t in_frame_size = N2FrameDesc::calculate_frame_size(num_elements, 0, num_prod);
        in_buf = std::make_unique<Buffer>(num_frames, in_frame_size, n2_pool, in_buf_name, "N2", 0,
                                          false, false, std::vector<int>{}, true);
        in_buf->ensure_frame_desc(
            std::make_shared<N2FrameDesc>(num_elements, 0, num_prod, N2Layout::FullUpperTri));
        in_buf->register_producer("test_producer");

        size_t out_frame_size = num_local_freq * num_elements * sizeof(float);
        out_buf = std::make_unique<Buffer>(num_frames, out_frame_size, chord_pool, out_buf_name,
                                           "ndarray", 0, false, false, std::vector<int>{}, true);
        out_buf->ensure_frame_desc(GenericNDArray::describe(
            kotekan::float32, "autospectrum", {num_local_freq, (std::ptrdiff_t)num_elements},
            {"F", "E"}, {1, 1}));
        out_buf->register_consumer("test_consumer");
    }
};

BOOST_AUTO_TEST_CASE(test_gather_flush_and_drops) {
    std::cout << "Testing N2AutoSpectrum: gather, flush, late and out-of-range drops...\n";

    const std::string in_buf_name = "in_buf";
    const std::string out_buf_name = "out_buf";
    const std::string stage_name = "/test_autospectrum";

    Config config = make_test_config(stage_name, in_buf_name, out_buf_name, 1);
    TestSetup setup(config, in_buf_name, out_buf_name);

    bufferContainer bc;
    bc.add_buffer(in_buf_name, setup.in_buf.get());
    bc.add_buffer(out_buf_name, setup.out_buf.get());

    N2AutoSpectrum stage(config, stage_name, bc);
    stage.start();

    // Bin 5 with freqs 0 and 2; bin 6 opens and flushes bin 5
    produce_input_frame(setup.in_buf.get(), 0, 5, 0);
    produce_input_frame(setup.in_buf.get(), 1, 5, 2);
    produce_input_frame(setup.in_buf.get(), 2, 6, 0);

    const float* frame0 = (float*)setup.out_buf->wait_for_full_frame("test_consumer", 0);
    BOOST_REQUIRE(frame0 != nullptr);
    check_row(frame0, 0, true, 5, 0);
    check_row(frame0, 1, false, 0, 0);
    check_row(frame0, 2, true, 5, 2);
    BOOST_CHECK_EQUAL(get_chord_metadata(setup.out_buf.get(), 0)->get_fpga_seq_num(), 5000);

    // A late frame for the flushed bin 5, and an out-of-range freq for bin 6:
    // both dropped. Bin 7 then flushes bin 6.
    produce_input_frame(setup.in_buf.get(), 3, 5, 1);
    produce_input_frame(setup.in_buf.get(), 0, 6, 7);
    produce_input_frame(setup.in_buf.get(), 1, 7, 1);

    const float* frame1 = (float*)setup.out_buf->wait_for_full_frame("test_consumer", 1);
    BOOST_REQUIRE(frame1 != nullptr);
    check_row(frame1, 0, true, 6, 0);
    check_row(frame1, 1, false, 0, 0);
    check_row(frame1, 2, false, 0, 0);
    BOOST_CHECK_EQUAL(get_chord_metadata(setup.out_buf.get(), 1)->get_fpga_seq_num(), 6000);

    setup.out_buf->mark_frame_empty("test_consumer", 0);
    setup.out_buf->mark_frame_empty("test_consumer", 1);

    setup.in_buf->send_shutdown_signal();
    stage.stop();
    stage.join();

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_interleaved_bins) {
    std::cout << "Testing N2AutoSpectrum: interleaved bins with num_pending_bins=2...\n";

    const std::string in_buf_name = "in_buf2";
    const std::string out_buf_name = "out_buf2";
    const std::string stage_name = "/test_autospectrum2";

    Config config = make_test_config(stage_name, in_buf_name, out_buf_name, 2);
    TestSetup setup(config, in_buf_name, out_buf_name);

    bufferContainer bc;
    bc.add_buffer(in_buf_name, setup.in_buf.get());
    bc.add_buffer(out_buf_name, setup.out_buf.get());

    N2AutoSpectrum stage(config, stage_name, bc);
    stage.start();

    // Frequencies of bins 10 and 11 arrive interleaved (sender skew); bin 12
    // forces the flush of bin 10 only.
    produce_input_frame(setup.in_buf.get(), 0, 10, 0);
    produce_input_frame(setup.in_buf.get(), 1, 11, 0);
    produce_input_frame(setup.in_buf.get(), 2, 10, 1);
    produce_input_frame(setup.in_buf.get(), 3, 11, 2);
    produce_input_frame(setup.in_buf.get(), 0, 12, 0);

    const float* frame0 = (float*)setup.out_buf->wait_for_full_frame("test_consumer", 0);
    BOOST_REQUIRE(frame0 != nullptr);
    check_row(frame0, 0, true, 10, 0);
    check_row(frame0, 1, true, 10, 1);
    check_row(frame0, 2, false, 0, 0);
    BOOST_CHECK_EQUAL(get_chord_metadata(setup.out_buf.get(), 0)->get_fpga_seq_num(), 10000);

    setup.out_buf->mark_frame_empty("test_consumer", 0);

    setup.in_buf->send_shutdown_signal();
    stage.stop();
    stage.join();

    std::cout << "Success.\n";
}
