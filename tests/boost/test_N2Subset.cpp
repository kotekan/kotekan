#define BOOST_TEST_MODULE "test_N2Subset"

#include "N2FrameDesc.hpp"
#include "N2FrameView.hpp"
#include "N2Layout.hpp"
#include "N2Metadata.hpp"
#include "N2Subset.hpp"
#include "N2Util.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "metadata.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <complex>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace kotekan;
using json = nlohmann::json;

// Helper to create a minimal config for testing
static Config make_test_config(const std::string& stage_name, const std::string& in_buf_name,
                               const std::string& out_buf_name) {
    json cfg;
    cfg["log_level"] = "WARN";

    json stage;
    stage["kotekan_stage"] = "N2Subset";
    stage["log_level"] = "DEBUG";
    stage["cpu_affinity"] = std::vector<int>{0};
    stage["in_buf"] = in_buf_name;
    stage["out_buf"] = out_buf_name;

    cfg[stage_name] = stage;
    // Also add without leading '/'
    if (!stage_name.empty() && stage_name.front() == '/')
        cfg[stage_name.substr(1)] = stage;

    Config conf;
    conf.update_config(cfg);
    return conf;
}

// Fill an input frame with deterministic vis data: vis[prod_idx] = {input_a, input_b}
static void fill_input_frame(Buffer* buf, int frame_id) {
    buf->allocate_new_metadata_object(frame_id);
    auto meta = get_N2_metadata(buf, frame_id);
    meta->freq_id = 0;
    meta->abs_time_idx = frame_id;
    meta->fpga_start_tick = 1000 * frame_id;
    meta->frame_start_time_ns = 1000000000ULL * frame_id;
    meta->frame_length_fpga_ticks = 100;
    meta->n_valid_fpga_ticks = 100;
    meta->n_rfi_fpga_ticks = 0;

    N2FrameView fv(buf, frame_id);

    // Fill vis with (input_a, input_b) pattern
    const auto& prods = fv._desc->get_product_list();

    for (size_t i = 0; i < prods.size(); ++i) {
        fv.vis[i] = N2::cfloat(float(prods[i].input_a), float(prods[i].input_b));
        fv.weight[i] = float(100 + i); // Unique weight per product
    }

    // Fill other fields
    for (uint32_t i = 0; i < fv.num_elements; ++i) {
        fv.flags[i] = 1.0f;
        fv.gain[i] = N2::cfloat(1.0f, 0.0f);
    }

    for (uint32_t i = 0; i < fv.num_ev; ++i) {
        fv.eval[i] = float(i);
        for (uint32_t j = 0; j < fv.num_elements; ++j) {
            fv.evec[i * fv.num_elements + j] = N2::cfloat(float(i), float(j));
        }
    }

    fv.erms = 1.0f;
    fv.emethod = N2EigenMethod::none;
}

BOOST_AUTO_TEST_CASE(test_fulluppertri_to_autocorrelations) {
    std::cout << "Testing N2Subset: FullUpperTri -> Autocorrelations...\n";

    const uint32_t num_elements = 8;
    const uint32_t num_ev = 2;
    const uint32_t num_frames = 2;
    const std::string in_buf_name = "in_buf";
    const std::string out_buf_name = "out_buf";
    const std::string stage_name = "/test_subset";

    // Create metadata pool
    auto pool = metadataPool::create(2 * num_frames, sizeof(N2Metadata), "test_pool", "N2Metadata");

    // Create input buffer (FullUpperTri)
    uint32_t in_num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::FullUpperTri);
    size_t in_frame_size = N2FrameDesc::calculate_frame_size(num_elements, num_ev, in_num_prod);
    Buffer in_buf(num_frames, in_frame_size, pool, in_buf_name, "N2", 0, false, false,
                  std::vector<int>{}, true);
    in_buf.set_frame_desc(
        std::make_shared<N2FrameDesc>(num_elements, num_ev, in_num_prod, N2Layout::FullUpperTri));
    in_buf.register_producer("test_producer");

    // Create output buffer (Autocorrelations)
    uint32_t out_num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::Autocorrelations);
    size_t out_frame_size = N2FrameDesc::calculate_frame_size(num_elements, num_ev, out_num_prod);
    Buffer out_buf(num_frames, out_frame_size, pool, out_buf_name, "N2", 0, false, false,
                   std::vector<int>{}, true);
    out_buf.set_frame_desc(std::make_shared<N2FrameDesc>(num_elements, num_ev, out_num_prod,
                                                         N2Layout::Autocorrelations));

    // Create buffer container
    bufferContainer bc;
    bc.add_buffer(in_buf_name, &in_buf);
    bc.add_buffer(out_buf_name, &out_buf);

    // Register test as consumer of output buffer
    out_buf.register_consumer("test_consumer");

    // Create config
    Config config = make_test_config(stage_name, in_buf_name, out_buf_name);

    // Create and start stage
    N2Subset stage(config, stage_name, bc);
    stage.start();

    // Wait for empty input frame, fill it, and mark full
    in_buf.wait_for_empty_frame("test_producer", 0);
    fill_input_frame(&in_buf, 0);
    in_buf.mark_frame_full("test_producer", 0);

    // Wait for output frame
    uint8_t* out_frame = out_buf.wait_for_full_frame("test_consumer", 0);
    BOOST_REQUIRE(out_frame != nullptr);

    // Verify output
    N2FrameView out_fv(&out_buf, 0);

    BOOST_CHECK_EQUAL(out_fv.num_prod, num_elements); // Autocorrelations only

    // Check vis values: should be diagonal elements (i, i)
    for (uint32_t i = 0; i < num_elements; ++i) {
        N2::cfloat expected{float(i), float(i)};
        BOOST_CHECK_SMALL(out_fv.vis[i].real() - expected.real(), 1e-5f);
        BOOST_CHECK_SMALL(out_fv.vis[i].imag() - expected.imag(), 1e-5f);
    }

    // Check weights: diagonal elements in FullUpperTri are at cmap(i,i,n)
    for (uint32_t i = 0; i < num_elements; ++i) {
        uint32_t in_idx = N2::cmap(i, i, num_elements);
        float expected_weight = float(100 + in_idx);
        BOOST_CHECK_EQUAL(out_fv.weight[i], expected_weight);
    }

    // Check that non-vis fields were copied
    BOOST_CHECK_EQUAL(out_fv.erms, 1.0f);
    for (uint32_t i = 0; i < num_elements; ++i) {
        BOOST_CHECK_EQUAL(out_fv.flags[i], 1.0f);
    }

    // Check metadata was copied
    BOOST_CHECK_EQUAL(out_fv._metadata->freq_id, 0);
    BOOST_CHECK_EQUAL(out_fv._metadata->abs_time_idx, 0);

    // Mark output frame as empty so stage can exit cleanly
    out_buf.mark_frame_empty("test_consumer", 0);

    // Signal shutdown so stage exits
    in_buf.send_shutdown_signal();

    // Stop stage
    stage.stop();
    stage.join();

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_fulluppertri_to_general_subset) {
    std::cout << "Testing N2Subset: FullUpperTri -> GeneralSubset...\n";

    const uint32_t num_elements = 8;
    const uint32_t num_ev = 0;
    const uint32_t num_frames = 2;
    const std::string in_buf_name = "in_buf2";
    const std::string out_buf_name = "out_buf2";
    const std::string stage_name = "/test_subset2";

    // Define subset: specific products
    std::vector<N2::prod_ctype> product_list = {{0, 0}, {0, 3}, {1, 1}, {2, 5}, {3, 7}, {7, 7}};

    // Create metadata pool
    auto pool =
        metadataPool::create(2 * num_frames, sizeof(N2Metadata), "test_pool2", "N2Metadata");

    // Create input buffer (FullUpperTri)
    uint32_t in_num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::FullUpperTri);
    size_t in_frame_size = N2FrameDesc::calculate_frame_size(num_elements, num_ev, in_num_prod);
    Buffer in_buf(num_frames, in_frame_size, pool, in_buf_name, "N2", 0, false, false,
                  std::vector<int>{}, true);
    in_buf.set_frame_desc(
        std::make_shared<N2FrameDesc>(num_elements, num_ev, in_num_prod, N2Layout::FullUpperTri));
    in_buf.register_producer("test_producer");

    // Create output buffer (GeneralSubset)
    size_t out_frame_size =
        N2FrameDesc::calculate_frame_size(num_elements, num_ev, product_list.size());
    Buffer out_buf(num_frames, out_frame_size, pool, out_buf_name, "N2", 0, false, false,
                   std::vector<int>{}, true);
    out_buf.set_frame_desc(std::make_shared<N2FrameDesc>(num_elements, num_ev, product_list.size(),
                                                         N2Layout::GeneralSubset, product_list));

    // Create buffer container
    bufferContainer bc;
    bc.add_buffer(in_buf_name, &in_buf);
    bc.add_buffer(out_buf_name, &out_buf);

    // Register test as consumer of output buffer
    out_buf.register_consumer("test_consumer");

    // Create config
    Config config = make_test_config(stage_name, in_buf_name, out_buf_name);

    // Create and start stage
    N2Subset stage(config, stage_name, bc);
    stage.start();

    // Wait for empty input frame, fill it, and mark full
    in_buf.wait_for_empty_frame("test_producer", 0);
    fill_input_frame(&in_buf, 0);
    in_buf.mark_frame_full("test_producer", 0);

    // Wait for output frame
    uint8_t* out_frame = out_buf.wait_for_full_frame("test_consumer", 0);
    BOOST_REQUIRE(out_frame != nullptr);

    // Verify output
    N2FrameView out_fv(&out_buf, 0);

    BOOST_CHECK_EQUAL(out_fv.num_prod, product_list.size());

    // Check vis values match the subset
    for (size_t i = 0; i < product_list.size(); ++i) {
        N2::cfloat expected(float(product_list[i].input_a), float(product_list[i].input_b));
        BOOST_CHECK_SMALL(out_fv.vis[i].real() - expected.real(), 1e-5f);
        BOOST_CHECK_SMALL(out_fv.vis[i].imag() - expected.imag(), 1e-5f);
    }

    // Mark output frame as empty so stage can exit cleanly
    out_buf.mark_frame_empty("test_consumer", 0);

    // Signal shutdown so stage exits
    in_buf.send_shutdown_signal();

    // Stop stage
    stage.stop();
    stage.join();

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_autocorrelations_to_autocorrelations) {
    std::cout << "Testing N2Subset: Autocorrelations -> Autocorrelations (identity)...\n";

    const uint32_t num_elements = 8;
    const uint32_t num_ev = 0;
    const uint32_t num_frames = 2;
    const std::string in_buf_name = "in_buf3";
    const std::string out_buf_name = "out_buf3";
    const std::string stage_name = "/test_subset3";

    // Create metadata pool
    auto pool =
        metadataPool::create(2 * num_frames, sizeof(N2Metadata), "test_pool3", "N2Metadata");

    // Both buffers Autocorrelations
    uint32_t num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::Autocorrelations);
    size_t frame_size = N2FrameDesc::calculate_frame_size(num_elements, num_ev, num_prod);

    Buffer in_buf(num_frames, frame_size, pool, in_buf_name, "N2", 0, false, false,
                  std::vector<int>{}, true);
    in_buf.set_frame_desc(
        std::make_shared<N2FrameDesc>(num_elements, num_ev, num_prod, N2Layout::Autocorrelations));
    in_buf.register_producer("test_producer");

    Buffer out_buf(num_frames, frame_size, pool, out_buf_name, "N2", 0, false, false,
                   std::vector<int>{}, true);
    out_buf.set_frame_desc(
        std::make_shared<N2FrameDesc>(num_elements, num_ev, num_prod, N2Layout::Autocorrelations));

    // Create buffer container
    bufferContainer bc;
    bc.add_buffer(in_buf_name, &in_buf);
    bc.add_buffer(out_buf_name, &out_buf);

    // Register test as consumer of output buffer
    out_buf.register_consumer("test_consumer");

    // Create config
    Config config = make_test_config(stage_name, in_buf_name, out_buf_name);

    // Create and start stage
    N2Subset stage(config, stage_name, bc);
    stage.start();

    // Wait for empty input frame, fill it, and mark full
    in_buf.wait_for_empty_frame("test_producer", 0);
    fill_input_frame(&in_buf, 0);
    in_buf.mark_frame_full("test_producer", 0);

    // Wait for output frame
    uint8_t* out_frame = out_buf.wait_for_full_frame("test_consumer", 0);
    BOOST_REQUIRE(out_frame != nullptr);

    // Verify output (should be identical to input)
    N2FrameView out_fv(&out_buf, 0);

    BOOST_CHECK_EQUAL(out_fv.num_prod, num_elements);

    for (uint32_t i = 0; i < num_elements; ++i) {
        N2::cfloat expected{float(i), float(i)};
        BOOST_CHECK_SMALL(out_fv.vis[i].real() - expected.real(), 1e-5f);
        BOOST_CHECK_SMALL(out_fv.vis[i].imag() - expected.imag(), 1e-5f);
    }

    // Mark output frame as empty so stage can exit cleanly
    out_buf.mark_frame_empty("test_consumer", 0);

    // Signal shutdown so stage exits
    in_buf.send_shutdown_signal();

    // Stop stage
    stage.stop();
    stage.join();

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_have_inputs_subset) {
    std::cout << "Testing N2Subset: FullUpperTri -> InputORMasked (have_inputs)...\n";

    const uint32_t num_elements = 4;
    const uint32_t num_ev = 0;
    const uint32_t num_frames = 2;
    const std::string in_buf_name = "in_buf4";
    const std::string out_buf_name = "out_buf4";
    const std::string stage_name = "/test_subset4";

    // have_inputs condition: products containing input 0 OR input 2
    // Products with 0 or 2: (0,0), (0,1), (0,2), (0,3), (1,2), (2,2), (2,3)
    std::vector<N2::prod_ctype> product_list = {{0, 0}, {0, 1}, {0, 2}, {0, 3},
                                                {1, 2}, {2, 2}, {2, 3}};

    // Create metadata pool
    auto pool =
        metadataPool::create(2 * num_frames, sizeof(N2Metadata), "test_pool4", "N2Metadata");

    // Create input buffer (FullUpperTri)
    uint32_t in_num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::FullUpperTri);
    size_t in_frame_size = N2FrameDesc::calculate_frame_size(num_elements, num_ev, in_num_prod);
    Buffer in_buf(num_frames, in_frame_size, pool, in_buf_name, "N2", 0, false, false,
                  std::vector<int>{}, true);
    in_buf.set_frame_desc(
        std::make_shared<N2FrameDesc>(num_elements, num_ev, in_num_prod, N2Layout::FullUpperTri));
    in_buf.register_producer("test_producer");

    // Create output buffer (InputORMasked)
    size_t out_frame_size =
        N2FrameDesc::calculate_frame_size(num_elements, num_ev, product_list.size());
    Buffer out_buf(num_frames, out_frame_size, pool, out_buf_name, "N2", 0, false, false,
                   std::vector<int>{}, true);
    out_buf.set_frame_desc(std::make_shared<N2FrameDesc>(num_elements, num_ev, product_list.size(),
                                                         N2Layout::InputORMasked, product_list));

    // Create buffer container
    bufferContainer bc;
    bc.add_buffer(in_buf_name, &in_buf);
    bc.add_buffer(out_buf_name, &out_buf);

    // Register test as consumer of output buffer
    out_buf.register_consumer("test_consumer");

    // Create config
    Config config = make_test_config(stage_name, in_buf_name, out_buf_name);

    // Create and start stage
    N2Subset stage(config, stage_name, bc);
    stage.start();

    // Wait for empty input frame, fill it, and mark full
    in_buf.wait_for_empty_frame("test_producer", 0);
    fill_input_frame(&in_buf, 0);
    in_buf.mark_frame_full("test_producer", 0);

    // Wait for output frame
    uint8_t* out_frame = out_buf.wait_for_full_frame("test_consumer", 0);
    BOOST_REQUIRE(out_frame != nullptr);

    // Verify output
    N2FrameView out_fv(&out_buf, 0);

    BOOST_CHECK_EQUAL(out_fv.num_prod, product_list.size());

    for (size_t i = 0; i < product_list.size(); ++i) {
        N2::cfloat expected(float(product_list[i].input_a), float(product_list[i].input_b));
        BOOST_CHECK_SMALL(out_fv.vis[i].real() - expected.real(), 1e-5f);
        BOOST_CHECK_SMALL(out_fv.vis[i].imag() - expected.imag(), 1e-5f);
    }

    // Mark output frame as empty so stage can exit cleanly
    out_buf.mark_frame_empty("test_consumer", 0);

    // Signal shutdown so stage exits
    in_buf.send_shutdown_signal();

    // Stop stage
    stage.stop();
    stage.join();

    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_only_inputs_subset) {
    std::cout << "Testing N2Subset: FullUpperTri -> InputANDMasked (only_inputs)...\n";

    const uint32_t num_elements = 4;
    const uint32_t num_ev = 0;
    const uint32_t num_frames = 2;
    const std::string in_buf_name = "in_buf5";
    const std::string out_buf_name = "out_buf5";
    const std::string stage_name = "/test_subset5";

    // only_inputs condition: products with BOTH inputs from {0, 1, 2}
    std::vector<N2::prod_ctype> product_list = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    // Create metadata pool
    auto pool =
        metadataPool::create(2 * num_frames, sizeof(N2Metadata), "test_pool5", "N2Metadata");

    // Create input buffer (FullUpperTri)
    uint32_t in_num_prod = N2FrameDesc::get_num_prod(num_elements, N2Layout::FullUpperTri);
    size_t in_frame_size = N2FrameDesc::calculate_frame_size(num_elements, num_ev, in_num_prod);
    Buffer in_buf(num_frames, in_frame_size, pool, in_buf_name, "N2", 0, false, false,
                  std::vector<int>{}, true);
    in_buf.set_frame_desc(
        std::make_shared<N2FrameDesc>(num_elements, num_ev, in_num_prod, N2Layout::FullUpperTri));
    in_buf.register_producer("test_producer");

    // Create output buffer (InputANDMasked)
    size_t out_frame_size =
        N2FrameDesc::calculate_frame_size(num_elements, num_ev, product_list.size());
    Buffer out_buf(num_frames, out_frame_size, pool, out_buf_name, "N2", 0, false, false,
                   std::vector<int>{}, true);
    out_buf.set_frame_desc(std::make_shared<N2FrameDesc>(num_elements, num_ev, product_list.size(),
                                                         N2Layout::InputANDMasked, product_list));

    // Create buffer container
    bufferContainer bc;
    bc.add_buffer(in_buf_name, &in_buf);
    bc.add_buffer(out_buf_name, &out_buf);

    // Register test as consumer of output buffer
    out_buf.register_consumer("test_consumer");

    // Create config
    Config config = make_test_config(stage_name, in_buf_name, out_buf_name);

    // Create and start stage
    N2Subset stage(config, stage_name, bc);
    stage.start();

    // Wait for empty input frame, fill it, and mark full
    in_buf.wait_for_empty_frame("test_producer", 0);
    fill_input_frame(&in_buf, 0);
    in_buf.mark_frame_full("test_producer", 0);

    // Wait for output frame
    uint8_t* out_frame = out_buf.wait_for_full_frame("test_consumer", 0);
    BOOST_REQUIRE(out_frame != nullptr);

    // Verify output
    N2FrameView out_fv(&out_buf, 0);

    BOOST_CHECK_EQUAL(out_fv.num_prod, product_list.size());

    for (size_t i = 0; i < product_list.size(); ++i) {
        N2::cfloat expected(float(product_list[i].input_a), float(product_list[i].input_b));
        BOOST_CHECK_SMALL(out_fv.vis[i].real() - expected.real(), 1e-5f);
        BOOST_CHECK_SMALL(out_fv.vis[i].imag() - expected.imag(), 1e-5f);
    }

    // Mark output frame as empty so stage can exit cleanly
    out_buf.mark_frame_empty("test_consumer", 0);

    // Signal shutdown so stage exits
    in_buf.send_shutdown_signal();

    // Stop stage
    stage.stop();
    stage.join();

    std::cout << "Success.\n";
}
