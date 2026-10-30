// Boost test for baseband subset files: SubsetBaseband feeding hdf5FileWrite. The files must
// describe the inputs they hold the way hdf5N2Write's files do (input_list, /index_map) and
// carry the digital gains of just those inputs (/digital_gains/*_by_dish).

#define BOOST_TEST_MODULE "test_hdf5FileWrite_subset"

#include "CHORDTelescope.hpp"
#include "Config.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "SubsetBaseband.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "configUpdater.hpp"
#include "metadataFactory.hpp"
#include "test_logging.hpp"
#include "test_utils.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <complex>
#include <cstdint>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace kotekan;
using json = nlohmann::json;

static const uint32_t num_dishes = 16;
static const uint32_t num_elements = 2 * num_dishes;
static const uint32_t timesamples_per_frame = 16;
static const uint32_t num_local_freq = 4;
static const int num_frames = 2;

// Dish block 1 is dishes 8-15. Dishes 12-15 are Missing in the test telescope, and 0-7 are
// connected but not selected, so SubsetBaseband warns about both.
static const std::vector<int> dish_blocks_subset = {1};
static const uint32_t num_dishes_out = 8;
// Slot i of the element axis holds raw board crs_board_remap[i].
static const std::vector<uint32_t> crs_board_remap = {1, 0, 3, 2};

static const size_t num_archive_chan = 4;
static const size_t num_archive_input = num_elements;

static const std::string base_dir = "test_hdf5FileWrite_subset";
static const std::string gains_file = base_dir + "/gains.h5";

static HighFive::CompoundType complex64_type() {
    return HighFive::CompoundType{{"r", HighFive::AtomicType<float>{}},
                                  {"i", HighFive::AtomicType<float>{}}};
}

// The gain of archive input `in` in channel `chan` is in + i * chan, so every value is distinct.
static void write_gains_archive() {
    HighFive::File f(gains_file, HighFive::File::Truncate);
    std::vector<std::complex<float>> coeff(num_archive_chan * num_archive_input);
    for (size_t c = 0; c < num_archive_chan; c++)
        for (size_t in = 0; in < num_archive_input; in++)
            coeff[c * num_archive_input + in] = {float(in), float(c)};
    f.createDataSet("gain_coeff", HighFive::DataSpace({1, num_archive_chan, num_archive_input}),
                    complex64_type())
        .write_raw(coeff.data(), complex64_type());
    std::vector<int32_t> exps(num_archive_input, -1);
    f.createDataSet<int32_t>("gain_exp", HighFive::DataSpace({1, num_archive_input}))
        .write_raw(exps.data());
}

static Config make_config() {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["num_polarizations"] = 2;
    cfg["num_dishes"] = num_dishes;
    cfg["crs_board_remap"] = crs_board_remap;
    cfg["chord_pool"] = {{"kotekan_metadata_pool", "chordMetadata"}, {"num_metadata_objects", 16}};

    add_test_telescope_config(cfg);
    json dishes = json::array();
    for (uint32_t d = 0; d < num_dishes; d++) {
        dishes.push_back({{"dish_idx", d},
                          {"grid_x_idx", d % 8},
                          {"grid_y_idx", d / 8},
                          {"feed_pos_disp_m", {0.0, 0.0, 0.0}},
                          {"coelev_disp_deg", 0.0},
                          {"type", d < 12 ? "ArrayDish" : "Missing"},
                          {"label", fmt::format("D{:02d}", d)}});
    }
    for (const std::string key : {"telescope", "/telescope"}) {
        cfg[key]["num_dishes"] = num_dishes;
        cfg[key]["dish_inputs"] = dishes;
    }

    cfg["subset"] = {{"kotekan_stage", "SubsetBaseband"},
                     {"cpu_affinity", {0}},
                     {"in_buf", "in_buf"},
                     {"out_buf", "subset_buf"},
                     {"timesamples_per_frame", timesamples_per_frame},
                     {"num_local_freq", num_local_freq},
                     {"num_elements", num_elements},
                     {"dish_blocks_subset", dish_blocks_subset}};
    cfg["write"] = {{"kotekan_stage", "hdf5FileWrite"}, {"cpu_affinity", {0}},
                    {"in_buf", "subset_buf"},           {"base_dir", base_dir},
                    {"file_name", "baseband"},          {"prefix_hostname", false},
                    {"use_compression", false},         {"frames_per_file", num_frames},
                    {"baseband_gain_file", gains_file}};

    Config conf;
    conf.update_config(cfg);
    return conf;
}

// Distinct per (time, freq, pol, dish) so that a mix-up on any axis is caught.
static uint8_t input_value(uint32_t t, uint32_t f, uint32_t p, uint32_t d) {
    return (uint8_t)(t * 37u + f * 11u + p * 101u + d * 3u);
}

static void produce_frame(Buffer* in_buf, int frame_id) {
    uint8_t* frame = in_buf->wait_for_empty_frame("test_producer", frame_id);
    BOOST_REQUIRE(frame != nullptr);
    for (uint32_t t = 0; t < timesamples_per_frame; t++)
        for (uint32_t f = 0; f < num_local_freq; f++)
            for (uint32_t p = 0; p < 2; p++)
                for (uint32_t d = 0; d < num_dishes; d++)
                    frame[((t * num_local_freq + f) * 2 + p) * num_dishes + d] =
                        input_value(frame_id * timesamples_per_frame + t, f, p, d);

    in_buf->allocate_new_metadata_object(frame_id);
    auto meta = get_chord_metadata(in_buf, frame_id);
    meta->set_fpga_seq_num(frame_id * timesamples_per_frame);
    meta->set_time_downsampling_fpga(1);
    meta->set_coarse_freq({100, 101, 102, 103});
    in_buf->mark_frame_full("test_producer", frame_id);
}

// Run num_frames frames through SubsetBaseband and hdf5FileWrite, which writes them to one file.
static void run_pipeline(Config& config) {
    metadataFactory mfac(config);
    auto pools = mfac.build_pools();
    Buffer in_buf(num_frames, (size_t)timesamples_per_frame * num_local_freq * num_elements,
                  pools["chord_pool"], "in_buf", "standard", 0, false, false, std::vector<int>{},
                  true);
    in_buf.register_producer("test_producer");
    Buffer subset_buf(
        num_frames, (size_t)timesamples_per_frame * num_local_freq * 2 * num_dishes_out,
        pools["chord_pool"], "subset_buf", "ndarray", 0, false, false, std::vector<int>{}, true);
    subset_buf.register_consumer("test_consumer");

    bufferContainer bc;
    bc.add_buffer("in_buf", &in_buf);
    bc.add_buffer("subset_buf", &subset_buf);

    SubsetBaseband subset(config, "/subset", bc);
    std::unique_ptr<Stage> write(StageFactoryRegistry::get_registered_stages()
                                     .at("hdf5FileWrite")
                                     ->create(config, "/write", bc));
    subset.start();
    write->start();

    for (int i = 0; i < num_frames; i++)
        produce_frame(&in_buf, i);

    // Once this consumer has seen every subset frame, a frame is empty again only after the
    // writer has finished with it too.
    for (int i = 0; i < num_frames; i++) {
        BOOST_REQUIRE(subset_buf.wait_for_full_frame("test_consumer", i) != nullptr);
        subset_buf.mark_frame_empty("test_consumer", i);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!(subset_buf.is_frame_empty(0) && subset_buf.is_frame_empty(1))) {
        BOOST_REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    in_buf.send_shutdown_signal();
    subset_buf.send_shutdown_signal();
    subset.stop();
    write->stop();
    subset.join();
    write->join();
}

BOOST_AUTO_TEST_CASE(test_subset_file_inputs_and_gains) {
    kotekan_test_logging::configure();
    rm_tree_if_exists(base_dir);
    ensure_directory(base_dir);
    write_gains_archive();

    Config config = make_config();
    configUpdater::instance().apply_config(config);
    Telescope::instance(config);
    run_pipeline(config);

    HighFive::File file(base_dir + "/baseband.00000000.h5", HighFive::File::ReadOnly);

    // The elements the file holds, P-major: dishes 8-15 of each polarization.
    std::vector<int32_t> expected_input_list;
    std::vector<std::string> expected_labels;
    std::vector<int64_t> expected_dish;
    std::vector<int32_t> expected_pol, expected_type, expected_gain_input;
    for (uint32_t p = 0; p < 2; p++) {
        for (uint32_t d = 8; d < 16; d++) {
            const uint32_t e = p * num_dishes + d;
            expected_input_list.push_back(e);
            expected_labels.push_back(fmt::format("D{:02d}{:s}", d, p == 0 ? "X" : "Y"));
            expected_dish.push_back(d);
            expected_pol.push_back(p);
            expected_type.push_back(d < 12 ? 0 : -1); // ArrayDish, Missing
            expected_gain_input.push_back(crs_board_remap[e / 8] * 8 + e % 8);
        }
    }

    std::vector<int32_t> input_list;
    file.getAttribute("input_list").read(input_list);
    BOOST_CHECK_EQUAL_COLLECTIONS(input_list.begin(), input_list.end(), expected_input_list.begin(),
                                  expected_input_list.end());
    std::string input_order;
    file.getAttribute("input_order").read(input_order);
    BOOST_CHECK_EQUAL(input_order, "CHORDBeamformer");

    std::vector<std::string> labels;
    std::vector<int64_t> dish_idx;
    std::vector<int32_t> pol, type;
    file.getDataSet("/index_map/label").read(labels);
    file.getDataSet("/index_map/dish_idx").read(dish_idx);
    file.getDataSet("/index_map/pol").read(pol);
    file.getDataSet("/index_map/type").read(type);
    BOOST_CHECK_EQUAL_COLLECTIONS(labels.begin(), labels.end(), expected_labels.begin(),
                                  expected_labels.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(dish_idx.begin(), dish_idx.end(), expected_dish.begin(),
                                  expected_dish.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(pol.begin(), pol.end(), expected_pol.begin(), expected_pol.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(type.begin(), type.end(), expected_type.begin(),
                                  expected_type.end());
    std::vector<std::string> axis;
    file.getDataSet("/index_map/label").getAttribute("axis").read(axis);
    BOOST_CHECK(axis == std::vector<std::string>{"element"});
    BOOST_CHECK(file.getDataSet("/index_map/dish_positions_in_grid_coords").getDimensions()
                == std::vector<size_t>({num_dishes, 3}));

    // The dataset attributes describe the whole telescope; the subset is in /index_map.
    auto dataset = file.getDataSet("baseband");
    BOOST_CHECK(!dataset.hasAttribute("dish_index"));
    BOOST_CHECK(!dataset.hasAttribute("dish_label"));
    uint64_t num_dishes_attr = 0;
    dataset.getAttribute("num_dishes").read(num_dishes_attr);
    BOOST_CHECK_EQUAL(num_dishes_attr, num_dishes);
    BOOST_CHECK(dataset.getAttribute("dish_grid_indices").getSpace().getDimensions()
                == std::vector<size_t>({num_dishes, 2}));

    // The data: both frames, dishes 8-15 of each polarization.
    BOOST_REQUIRE(dataset.getDimensions()
                  == std::vector<size_t>(
                      {num_frames * timesamples_per_frame, num_local_freq, 2, num_dishes_out}));
    std::vector<uint8_t> data(num_frames * timesamples_per_frame * num_local_freq * 2
                              * num_dishes_out);
    dataset.read_raw(data.data(), HighFive::AtomicType<uint8_t>());
    size_t mismatches = 0;
    for (uint32_t t = 0; t < num_frames * timesamples_per_frame; t++)
        for (uint32_t f = 0; f < num_local_freq; f++)
            for (uint32_t p = 0; p < 2; p++)
                for (uint32_t k = 0; k < num_dishes_out; k++)
                    mismatches += data[((t * num_local_freq + f) * 2 + p) * num_dishes_out + k]
                                  != input_value(t, f, p, 8 + k);
    BOOST_CHECK_EQUAL(mismatches, 0u);

    // The gains of just those inputs, located through crs_board_remap.
    std::vector<int32_t> gain_input(2 * num_dishes_out);
    auto ds_index = file.getDataSet("/digital_gains/gain_input_index");
    BOOST_CHECK(ds_index.getDimensions() == std::vector<size_t>({2, num_dishes_out}));
    ds_index.read_raw(gain_input.data());
    BOOST_CHECK_EQUAL_COLLECTIONS(gain_input.begin(), gain_input.end(), expected_gain_input.begin(),
                                  expected_gain_input.end());

    auto ds_coeff = file.getDataSet("/digital_gains/gain_coeff_by_dish");
    BOOST_REQUIRE(ds_coeff.getDimensions()
                  == std::vector<size_t>({1, num_archive_chan, 2, num_dishes_out}));
    std::vector<std::complex<float>> coeff(num_archive_chan * 2 * num_dishes_out);
    ds_coeff.read_raw(coeff.data(), complex64_type());
    for (size_t c = 0; c < num_archive_chan; c++)
        for (size_t s = 0; s < expected_gain_input.size(); s++)
            BOOST_CHECK(coeff[c * expected_gain_input.size() + s]
                        == std::complex<float>(float(expected_gain_input[s]), float(c)));

    uint64_t root_hash = 0, dataset_hash = 0;
    file.getAttribute("digital_gains_hash").read(root_hash);
    dataset.getAttribute("digital_gains_hash").read(dataset_hash);
    BOOST_CHECK_EQUAL(root_hash, dataset_hash);

    rm_tree_if_exists(base_dir);
}
