// Boost tests for the hdf5N2Write stage end-to-end, writing HDF5 files

#define BOOST_TEST_MODULE "test_hdf5N2Write"

#include "CHORDTelescope.hpp"
#include "Config.hpp" // for Config
#include "H5Support.hpp"
#include "N2FrameDesc.hpp" // for N2FrameDesc
#include "N2FrameView.hpp" // for N2FrameView
#include "N2Metadata.hpp"  // for N2Metadata, get_N2_metadata
#include "N2Util.hpp"      // for N2 helpers
#include "Stage.hpp"       // for Stage
#include "Telescope.hpp"
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "configUpdater.hpp"
#include "hdf5N2Write.hpp" // for hdf5N2Write
#include "restServer.hpp"
#include "test_logging.hpp"
#include "test_utils.hpp"

#include "json.hpp"

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h> // for opendir, readdir
#include <highfive/H5File.hpp>
#include <iomanip>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <sys/stat.h> // for stat
#include <sys/wait.h>
#include <thread>
#include <unistd.h> // for gethostname
#include <utility>
#include <vector>

// Lets the tests below catch FATAL_ERROR_NON_OO exceptions, which call
// exit_kotekan and raise SIGTERM before throwing FatalError.
static kotekan_test_logging::SigtermGuard g_sigterm_guard;

using std::string;

using HighFive::File;
using kotekan::N2FrameDesc;

// Absolute path to test gains file (injected by CMake via TEST_DATA_DIR)
static const std::string TEST_GAINS_FILE =
    std::string(TEST_DATA_DIR) + "/baseband_gains/test_gains.h5";

// Install the test telescope: two dishes D00 and D01, with D00 optionally disconnected
// (typed Fake) so that a DishInputs frame is a proper subset of the array.
static void set_test_telescope(bool dish0_connected) {
    nlohmann::json cfg;
    cfg["num_polarizations"] = 2;
    add_test_telescope_config(cfg);
    if (!dish0_connected) {
        cfg["telescope"]["dish_inputs"][0]["type"] = "Fake";
        cfg["/telescope"] = cfg["telescope"];
    }
    kotekan::Config conf;
    conf.update_config(cfg);
    kotekan::configUpdater::instance().apply_config(conf);
    Telescope::instance(conf);
}

static freq_id_t get_abs_freq_id(size_t f_index) {
    const auto& tel = Telescope::instance().cast<CHORDTelescope>();
    return tel.min_science_freq_id() + f_index;
}

static void fill_n2_frame_with_abs_freq(Buffer* buf, int frame_id, size_t num_input, size_t num_ev,
                                        size_t f_index, size_t t_index,
                                        uint64_t frame_start_time_ns, uint64_t frame_length_ticks,
                                        uint64_t abs_time_idx) {
    fill_n2_frame_with_abs(buf, frame_id, num_input, num_ev, f_index, t_index, frame_start_time_ns,
                           frame_length_ticks, abs_time_idx);
    auto meta = get_N2_metadata(buf, frame_id);
    BOOST_REQUIRE(meta);
    meta->freq_id = get_abs_freq_id(f_index);
    // Keep ERAL within the valid bounds enforced by add_frame bounds checks
    meta->bin_end_ERA_deg = 7.89 + double(t_index);
    meta->bin_start_ERAL_deg = 1.23 + double(t_index);
    meta->bin_end_ERAL_deg = 4.56 + double(t_index);
}


/// Simple class to expose N2FileData internals without modifying production code.
class TestVisFileData : public N2FileData {
public:
    TestVisFileData(const N2FrameView& fv, uint64_t num_file_t, double open_wall_s,
                    uint64_t abs_file_idx, std::string base_dir,
                    std::string gains_file = TEST_GAINS_FILE) :
        N2FileData(N2FileData::CHORD, num_file_t, fv, open_wall_s, abs_file_idx,
                   /*input_order*/ ElementOrder::CHORDBeamformer,
                   /*blocksize_f*/ 0,
                   /*blocksize_p*/ 0,
                   /*blocksize_t*/ num_file_t,
                   /*compression*/ "none",
                   /*compression_level*/ 0,
                   /*use_bitshuffle*/ false, std::move(base_dir),
                   /*baseband_gain_file*/ std::move(gains_file)) {}

    N2::cfloat get_vis(size_t f, size_t p, size_t t) const {
        return vis[idx_fpt(f, p, t)];
    }
    float get_weight(size_t f, size_t p, size_t t) const {
        return vis_weight[idx_fpt(f, p, t)];
    }
    float get_eval(size_t f, size_t e, size_t t) const {
        return eval[idx_fet(f, e, t)];
    }
    N2::cfloat get_evec(size_t f, size_t e, size_t i, size_t t) const {
        return evec[idx_feit(f, e, i, t)];
    }
    float get_erms(size_t f, size_t t) const {
        return erms[idx_ft(f, t)];
    }
    N2::cfloat get_gain(size_t f, size_t i, size_t t) const {
        return gain[idx_fit(f, i, t)];
    }
    float get_flags(size_t f, size_t i, size_t t) const {
        return flags[idx_fit(f, i, t)];
    }
    float get_frac_lost(size_t f, size_t t) const {
        return frac_lost[idx_ft(f, t)];
    }
    float get_frac_rfi(size_t f, size_t t) const {
        return frac_rfi[idx_ft(f, t)];
    }
    uint64_t get_fpga_start_tick(size_t t) const {
        return fpga_start_tick.at(t);
    }
    uint64_t get_frame_length_fpga_ticks(size_t t) const {
        return frame_length_fpga_ticks.at(t);
    }
    int64_t get_time_center_ut1(size_t t) const {
        return time_center_ut1_ns.at(t);
    }
    int64_t get_bin_ut1(size_t t) const {
        return bin_ut1_ns.at(t);
    }
    size_t get_added_count() const {
        return added_count;
    }
};

// Force registration of metadata with Metadata factory
static N2Metadata _force_n2meta_registration;

/********************************************************/
/* Re-usable utility functions specific to writer tests */
/********************************************************/

// Build dataset filename to simulate pre-existing final files (mirrors stage logic)
static std::string get_dataset_name(const std::string& base_dir, uint64_t abs_file_idx,
                                    uint64_t file_start_time_ns, const std::string& suffix) {
    std::ostringstream buf;
    buf << base_dir;
    if (!base_dir.empty() && base_dir.back() != '/')
        buf << '/';
    buf << "vis_" << std::setw(10) << std::setfill('0') << abs_file_idx << "_";
    std::time_t tsec = file_start_time_ns / 1'000'000'000ULL;
    const uint64_t nsec = file_start_time_ns % 1'000'000'000ULL;
    buf << std::put_time(std::gmtime(&tsec), "%Y%m%dT%H%M%S") << "_" << std::setw(9)
        << std::setfill('0') << nsec << suffix;
    return buf.str();
}

// The /index_map input tables have one row per element of the frame. A full
// layout holds the first num_input elements of the array in the file's
// input_order: the test telescope has two dishes and two polarizations, and
// CHORDBeamformer order puts element = dish + pol * num_dishes, so the rows are
// D00p1, D01p1, D00p2, D01p2 with dish i in grid column i.
static void validate_index_map_inputs(File& file, size_t num_input) {
    std::vector<std::string> labels;
    std::vector<int64_t> dish_idx;
    std::vector<int32_t> pol;
    std::vector<int64_t> grid_x;
    file.getDataSet("/index_map/label").read(labels);
    file.getDataSet("/index_map/dish_idx").read(dish_idx);
    file.getDataSet("/index_map/pol").read(pol);
    file.getDataSet("/index_map/grid_x_idx").read(grid_x);

    BOOST_REQUIRE_LE(num_input, 4u);
    const std::vector<std::string> all_labels{"D00p1", "D01p1", "D00p2", "D01p2"};
    const std::vector<int64_t> all_dish{0, 1, 0, 1};
    const std::vector<int32_t> all_pol{0, 0, 1, 1};
    const std::vector<std::string> expected_labels(all_labels.begin(),
                                                   all_labels.begin() + num_input);
    const std::vector<int64_t> expected_dish(all_dish.begin(), all_dish.begin() + num_input);
    const std::vector<int32_t> expected_pol(all_pol.begin(), all_pol.begin() + num_input);
    BOOST_CHECK_EQUAL_COLLECTIONS(labels.begin(), labels.end(), expected_labels.begin(),
                                  expected_labels.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(dish_idx.begin(), dish_idx.end(), expected_dish.begin(),
                                  expected_dish.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(pol.begin(), pol.end(), expected_pol.begin(), expected_pol.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(grid_x.begin(), grid_x.end(), expected_dish.begin(),
                                  expected_dish.end());
}

// Read back and validate a few arrays using the known patterns
// Note this is not a full validation of all data, just spot-checks
// This function assumes the fill_n2_frame function has been called
static void validate_dataset_content(File& file, size_t num_input, size_t num_ev, size_t nfreq,
                                     size_t file_nt) {
    const size_t num_prod = N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri);

    // Check one representative frequency (e.g., f=1) across time
    size_t f = std::min<size_t>(1, nfreq - 1);

    // vis + weights
    {
        std::vector<std::vector<std::vector<cfloat>>> vis_out;
        std::vector<std::vector<std::vector<float>>> w_out;
        file.getDataSet("/vis").read(vis_out);
        file.getDataSet("/vis_weight").read(w_out);
        for (size_t t = 0; t < file_nt; ++t) {
            for (size_t p = 0; p < num_prod; ++p) {
                float base = 1000.0f * float(t) + 100.0f * float(f) + 10.0f * float(p);
                BOOST_CHECK(vis_out[f][p][t] == N2::cfloat(base + 1.0f, base + 2.0f));
                BOOST_CHECK_CLOSE_FRACTION(w_out[f][p][t], 1000.0f + float(p), 1e-6f);
            }
        }
    }

    // eval
    {
        std::vector<std::vector<std::vector<float>>> eval_out;
        file.getDataSet("/eval").read(eval_out);
        for (size_t e = 0; e < num_ev; ++e)
            for (size_t t = 0; t < file_nt; ++t)
                BOOST_CHECK_CLOSE_FRACTION(eval_out[f][e][t], 60.0f + float(e), 1e-6f);
    }

    // evec slice at i=0 spot-check
    {
        std::vector<std::vector<std::vector<std::vector<cfloat>>>> out;
        file.getDataSet("/evec").read(out);
        for (size_t e = 0; e < num_ev; ++e) {
            N2::cfloat expected0(100.0f * float(e) + 0.5f, -(100.0f * float(e) + 1.5f));
            for (size_t t = 0; t < file_nt; ++t)
                BOOST_CHECK(out[f][e][0][t] == expected0);
        }
    }

    // erms
    {
        std::vector<std::vector<float>> out;
        file.getDataSet("/erms").read(out);
        for (size_t t = 0; t < file_nt; ++t)
            BOOST_CHECK_CLOSE_FRACTION(out[f][t], 3.14f, 1e-6f);
    }

    // gain + flags spot-check i=0
    {
        std::vector<std::vector<std::vector<cfloat>>> gout;
        std::vector<std::vector<std::vector<float>>> fout;
        file.getDataSet("/gain").read(gout);
        file.getDataSet("/flags").read(fout);
        for (size_t t = 0; t < file_nt; ++t) {
            BOOST_CHECK(gout[f][0][t] == N2::cfloat(200.0f, -200.0f));
            BOOST_CHECK_CLOSE_FRACTION(fout[f][0][t], 300.0f, 1e-6f);
        }
    }

    // per-(freq,time) derived fractions spot-check
    {
        std::vector<std::vector<float>> fl_out, fr_out;
        file.getDataSet("/frac_lost").read(fl_out);
        file.getDataSet("/frac_rfi").read(fr_out);
        for (size_t t = 0; t < file_nt; ++t) {
            BOOST_CHECK_CLOSE_FRACTION(fl_out[f][t], 1.0f - 80.0f / 100.0f, 1e-6f);
            BOOST_CHECK_CLOSE_FRACTION(fr_out[f][t], 5.0f / 100.0f, 1e-6f);
        }
    }

    // per-time arrays shape exists
    {
        std::vector<uint64_t> s0, s2;
        std::vector<int64_t> tcen, bin;
        file.getDataSet("/fpga_start_tick").read(s0);
        file.getDataSet("/frame_length_fpga_ticks").read(s2);
        file.getDataSet("/time_center_ut1_ns").read(tcen);
        file.getDataSet("/bin_ut1_ns").read(bin);
        BOOST_CHECK(!s0.empty() && !s2.empty());
        BOOST_CHECK(!tcen.empty() && tcen.size() == bin.size());
    }
}

/***********************************/
/* Tests for the visFileData class */
/***********************************/

// Test 1: add_frame for a single (f,t) slot, verify data stored correctly in memory
BOOST_AUTO_TEST_CASE(test_visfiledata_add_frame_single_slot) {
    N2Metadata force_link_marker;
    const size_t num_input = 3;
    const size_t num_prod = N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri);
    const size_t num_ev = 2;
    const size_t num_file_t = 2;

    const size_t frame_size = N2FrameDesc::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(1, sizeof(N2Metadata), "test_pool", "N2Metadata");
    Buffer buf(1, frame_size, pool, "n2buf", "N2", 1, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<kotekan::N2FrameDesc>(num_input, num_ev, num_prod,
                                                                 N2Layout::FullUpperTri));

    buf.allocate_new_metadata_object(0);
    auto meta = get_N2_metadata(&buf, 0);
    BOOST_REQUIRE(meta);
    const size_t f_index = 1;
    meta->freq_id = get_abs_freq_id(f_index);
    meta->fpga_start_tick = 111;
    meta->frame_start_time_ns = 222;
    meta->frame_length_fpga_ticks = 100;
    meta->n_valid_fpga_ticks = 80;
    meta->n_rfi_fpga_ticks = 5;
    meta->n_rfi_only_fpga_ticks = 4;
    meta->n_pl_fpga_ticks = 100 - (80 + 4);
    meta->abs_time_idx = 5;
    meta->time_center_eop.t_ut1_ns = 333;
    meta->bin_eop.t_ut1_ns = 444;
    meta->time_center_eop.ERA_deg = 12.34;
    meta->bin_eop.ERA_deg = 56.78;

    N2FrameView fv(&buf, 0);
    fv.zero_frame();
    for (size_t p = 0; p < num_prod; ++p) {
        fv.vis[p] = N2::cfloat(10.0f * p + 1.0f, 10.0f * p + 2.0f);
        fv.weight[p] = float(1000 + p);
    }
    for (size_t e = 0; e < num_ev; ++e) {
        fv.eval[e] = float(60 + e);
        for (size_t i = 0; i < num_input; ++i)
            fv.evec[num_input * e + i] =
                N2::cfloat(100.0f * e + float(i) + 0.5f, -(100.0f * e + float(i) + 1.5f));
    }
    fv.erms = 3.14f;
    for (size_t i = 0; i < num_input; ++i) {
        fv.gain[i] = N2::cfloat(200.0f + float(i), -200.0f - float(i));
        fv.flags[i] = float(300 + i);
    }

    const std::string base_dir = "test_visfiledata_single_slot";
    rm_tree_if_exists(base_dir);
    ensure_directory(base_dir);
    ensure_directory(base_dir + "/.partial");
    TestVisFileData data(fv, num_file_t, 100.0, 0, base_dir);
    const size_t t = 1;

    // Add frame to (f,t) slot
    data.add_frame(fv, t);

    // Check a few expected values in memory
    BOOST_CHECK(data.get_vis(f_index, 0, t) == N2::cfloat(1.0f, 2.0f));
    BOOST_CHECK_CLOSE_FRACTION(data.get_weight(f_index, 0, t), float(1000), 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_eval(f_index, 0, t), float(60), 1e-6f);
    BOOST_CHECK(data.get_evec(f_index, 0, 0, t) == N2::cfloat(0.5f, -1.5f));
    BOOST_CHECK_CLOSE_FRACTION(data.get_erms(f_index, t), 3.14f, 1e-6f);
    BOOST_CHECK(data.get_gain(f_index, 0, t) == N2::cfloat(200.0f, -200.0f));
    BOOST_CHECK_CLOSE_FRACTION(data.get_flags(f_index, 0, t), 300.0f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_lost(f_index, t), 1.0f - 80.0f / 100.0f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_rfi(f_index, t), 5.0f / 100.0f, 1e-6f);
    BOOST_CHECK_EQUAL(data.get_fpga_start_tick(t), uint64_t(111));
    BOOST_CHECK_EQUAL(data.get_frame_length_fpga_ticks(t), uint64_t(100));
    BOOST_CHECK_EQUAL(data.get_time_center_ut1(t), int64_t(333));
    BOOST_CHECK_EQUAL(data.get_bin_ut1(t), int64_t(444));
    BOOST_CHECK_EQUAL(data.get_added_count(), size_t(1));
    rm_tree_if_exists(base_dir);
}

// Test 2: add_frame for the same (f,t) slot twice with differing metadata values
// A DishInputs frame carries only the telescope's connected elements, so the /index_map
// input tables hold those rows in the N2 layout's element order. With D00 disconnected,
// the connected elements (CHORDBeamformer: element = dish + pol * 2) are 1 and 3, the
// two polarizations of D01.
BOOST_AUTO_TEST_CASE(test_visfiledata_index_map_dish_inputs) {
    set_test_telescope(false);
    const size_t num_input = 2;
    const size_t num_prod = N2FrameDesc::get_num_prod(num_input, N2Layout::DishInputs);
    const size_t num_ev = 1;
    const size_t num_file_t = 1;

    const size_t frame_size = N2FrameDesc::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(1, sizeof(N2Metadata), "test_pool_di", "N2Metadata");
    Buffer buf(1, frame_size, pool, "n2buf_di", "N2", 1, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(
        std::make_shared<kotekan::N2FrameDesc>(num_input, num_ev, num_prod, N2Layout::DishInputs));
    buf.allocate_new_metadata_object(0);
    auto meta = get_N2_metadata(&buf, 0);
    BOOST_REQUIRE(meta);
    meta->freq_id = get_abs_freq_id(0);
    N2FrameView fv(&buf, 0);
    fv.zero_frame();

    const std::string base_dir = "test_visfiledata_index_map_dish_inputs";
    rm_tree_if_exists(base_dir);
    ensure_directory(base_dir);
    ensure_directory(base_dir + "/.partial");
    {
        TestVisFileData data(fv, num_file_t, 100.0, 0, base_dir);

        std::vector<std::string> labels;
        std::vector<int64_t> dish_idx;
        std::vector<int32_t> pol;
        std::vector<int32_t> type;
        data.h5_file->getDataSet("/index_map/label").read(labels);
        data.h5_file->getDataSet("/index_map/dish_idx").read(dish_idx);
        data.h5_file->getDataSet("/index_map/pol").read(pol);
        data.h5_file->getDataSet("/index_map/type").read(type);

        const std::vector<std::string> expected_labels{"D01p1", "D01p2"};
        const std::vector<int64_t> expected_dish{1, 1};
        const std::vector<int32_t> expected_pol{0, 1};
        const std::vector<int32_t> expected_type{0, 0}; // ArrayDish
        BOOST_CHECK_EQUAL_COLLECTIONS(labels.begin(), labels.end(), expected_labels.begin(),
                                      expected_labels.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(dish_idx.begin(), dish_idx.end(), expected_dish.begin(),
                                      expected_dish.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(pol.begin(), pol.end(), expected_pol.begin(),
                                      expected_pol.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(type.begin(), type.end(), expected_type.begin(),
                                      expected_type.end());
    }
    rm_tree_if_exists(base_dir);
    set_test_telescope(true);
}

BOOST_AUTO_TEST_CASE(test_visfiledata_era_and_fraction_guards) {
    N2Metadata force_link_marker;
    const size_t num_input = 2;
    const size_t num_prod = N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri);
    const size_t num_ev = 1;
    const size_t num_file_t = 2;

    const size_t frame_size = N2FrameDesc::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_guard", "N2Metadata");
    Buffer buf(2, frame_size, pool, "n2buf_guard", "N2", 1, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<kotekan::N2FrameDesc>(num_input, num_ev, num_prod,
                                                                 N2Layout::FullUpperTri));

    // Prepare frame view and two metadata instances for the same (f,t)
    for (int idx = 0; idx < 2; ++idx)
        buf.allocate_new_metadata_object(idx);
    auto meta1 = get_N2_metadata(&buf, 0);
    auto meta2 = get_N2_metadata(&buf, 1);

    const size_t f_index = 1;
    const size_t t = 1;

    // meta1
    meta1->freq_id = get_abs_freq_id(f_index);
    meta1->fpga_start_tick = 1000;
    meta1->frame_start_time_ns = 2000;
    meta1->frame_length_fpga_ticks = 100;
    meta1->n_valid_fpga_ticks = 80;
    meta1->n_rfi_fpga_ticks = 30;      // sum > frame_len -> should clamp to 20
    meta1->n_rfi_only_fpga_ticks = 30; // sum > frame_len -> should clamp to 20
    meta1->n_pl_fpga_ticks = 30;       // sum > frame_len -> should clamp to 20
    meta1->abs_time_idx = 10;
    meta1->time_center_eop.t_ut1_ns = 10'000;
    meta1->bin_eop.t_ut1_ns = 10'000;
    meta1->time_center_eop.ERA_deg = 0.0; // legitimate 0.0 value

    // meta2 (same slot), differing ERA and pathological counts
    *meta2 = *meta1;
    meta2->n_valid_fpga_ticks = 150; // > frame len -> clamp to 100
    meta2->n_rfi_fpga_ticks = 50;    // will be ignored because slot already set; kept for symmetry
    meta2->n_rfi_only_fpga_ticks =
        50;                      // will be ignored because slot already set; kept for symmetry
    meta2->n_pl_fpga_ticks = 50; // will be ignored because slot already set; kept for symmetry
    meta2->time_center_eop.t_ut1_ns = 11'000; // should not overwrite the first set value
    meta2->bin_eop.t_ut1_ns = 11'000;
    meta2->time_center_eop.ERA_deg = 12.34;

    N2FrameView fv1(&buf, 0);
    fv1.zero_frame();
    N2FrameView fv2(&buf, 1);
    fv2.zero_frame();

    const std::string base_dir = "test_visfiledata_duplicate";
    rm_tree_if_exists(base_dir);
    ensure_directory(base_dir);
    ensure_directory(base_dir + "/.partial");
    TestVisFileData data(fv1, num_file_t, 100.0, 0, base_dir);

    // First write
    BOOST_REQUIRE(data.add_frame(fv1, t) == N2FileData::AddFrameStatus::Success);
    // Verify fractions computed directly from metadata values
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_lost(f_index, t), 0.2f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_rfi(f_index, t), 0.3f, 1e-6f);
    BOOST_CHECK_EQUAL(data.get_time_center_ut1(t), int64_t(10'000));
    BOOST_CHECK_EQUAL(data.get_bin_ut1(t), int64_t(10'000));

    // Second write to same (f,t) with different UT1 values should throw FatalError.
    BOOST_CHECK_THROW(data.add_frame(fv2, t), std::runtime_error);
    // Stored fractions should remain from the first write, unaffected by 2nd input
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_lost(f_index, t), 0.2f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_rfi(f_index, t), 0.3f, 1e-6f);
    // Stored values remain the originals.
    BOOST_CHECK_EQUAL(data.get_time_center_ut1(t), int64_t(10'000));
    BOOST_CHECK_EQUAL(data.get_bin_ut1(t), int64_t(10'000));
    rm_tree_if_exists(base_dir);
}

/*************************************/
/* Tests for the visFileWriter class */
/*************************************/

struct RestServerFixture {
    RestServerFixture() {
        try {
            kotekan::restServer::instance().start("127.0.0.1", 0);
        } catch (...) {
        }
    }
};

BOOST_TEST_GLOBAL_FIXTURE(RestServerFixture);
BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

struct TelescopeFixture {
    TelescopeFixture() {
        set_test_telescope(true);
    }
};

BOOST_TEST_GLOBAL_FIXTURE(TelescopeFixture);

// Use large, nonuniform counts that would lose precision in float32.
// Include a supported visibility with zero weight.
static uint64_t product_support(size_t f, size_t p, size_t t) {
    if (p == 0)
        return 0;
    return (uint64_t(1) << 40) + 17 * f + 31 * p + 7 * t;
}

static void fill_product_support_frame(Buffer* buf, int frame_id, size_t f, size_t t) {
    buf->allocate_new_metadata_object(frame_id);
    auto meta = get_N2_metadata(buf, frame_id);
    meta->freq_id = get_abs_freq_id(f);
    meta->abs_time_idx = t;
    meta->fpga_start_tick = 100 + t * (uint64_t(1) << 42);
    meta->frame_length_fpga_ticks = uint64_t(1) << 42;
    meta->frame_start_time_ns = 1'000'000'000 + t * 1'000'000;
    meta->n_valid_fpga_ticks = 0;
    meta->n_pl_fpga_ticks = 0;
    meta->n_rfi_fpga_ticks = 0;
    meta->n_rfi_only_fpga_ticks = 0;
    meta->bin_eop.ERA_deg = 10.0 + t;
    meta->time_center_eop.ERA_deg = 10.0 + t;
    meta->bin_start_ERA_deg = 10.0 + t;
    meta->bin_end_ERA_deg = 10.5 + t;
    N2FrameView fv(buf, frame_id, true);
    fv.zero_frame();
    for (size_t p = 0; p < fv.num_prod; ++p) {
        fv.valid_fpga_ticks[p] = product_support(f, p, t);
        fv.vis[p] =
            p == 0 ? N2::cfloat(0, 0) : N2::cfloat(float(100 * f + 10 * p + t), -float(2 * p + t));
        fv.weight[p] = p <= 1 ? 0.0f : float(20 * f + p + 2 * t);
    }
}

BOOST_AUTO_TEST_CASE(test_per_product_file_roundtrip_chord_and_chime) {
    using kotekan::N2SupportMode;
    const size_t ni = 3, ne = 1, np = N2FrameDesc::get_num_prod(ni, N2Layout::FullUpperTri);
    const auto mode = N2SupportMode::PerProductV1;
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "product_pool", "N2Metadata");
    Buffer buf(2, N2FrameDesc::calculate_frame_size(ni, ne, np, mode), pool, "product_buf", "N2", 0,
               false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<N2FrameDesc>(ni, ne, np, N2Layout::FullUpperTri,
                                                        std::vector<N2::prod_ctype>{}, mode));
    for (const auto file_mode : {N2FileData::CHORD, N2FileData::CHIME}) {
        const std::string base = file_mode == N2FileData::CHORD ? "product_chord" : "product_chime";
        const std::string prefix = file_mode == N2FileData::CHORD ? "" : "/flags";
        rm_tree_if_exists(base);
        ensure_directory(base);
        ensure_directory(base + "/.partial");
        fill_product_support_frame(&buf, 0, 0, 2);
        N2FrameView initial(&buf, 0, true);
        N2FileData data(file_mode, 3, initial, 0.0, 0, ElementOrder::CHORDBeamformer, 1, 2, 2,
                        "none", 0, false, base, "");
        // Reuse both ring slots, reverse time/frequency order, and leave t=1 empty.
        int frame_id = 0;
        for (size_t t : {size_t(2), size_t(0)}) {
            for (size_t f : {size_t(1), size_t(0)}) {
                fill_product_support_frame(&buf, frame_id, f, t);
                N2FrameView fv(&buf, frame_id, true);
                BOOST_REQUIRE(data.add_frame(fv, t) == N2FileData::AddFrameStatus::Success);
                frame_id = 1 - frame_id;
            }
        }
        BOOST_REQUIRE(data.flush_to_disk());
        data.close();
        {
            File file(base + "/.partial/vis_0.h5", File::ReadOnly);
            std::string version, support, availability, units;
            file.getAttribute("version").read(version);
            file.getAttribute("support_mode").read(support);
            file.getAttribute("scalar_support_availability").read(availability);
            BOOST_CHECK_EQUAL(availability, "unavailable");
            file.getAttribute("loss_reason_availability").read(availability);
            BOOST_CHECK_EQUAL(availability, "unavailable");
            file.getAttribute("support_units").read(units);
            BOOST_CHECK_EQUAL(version, "CHORD_0.1");
            BOOST_CHECK_EQUAL(support, "per_product_v1");
            BOOST_CHECK_EQUAL(units, "fpga_ticks");
            for (const std::string name :
                 {"valid_fpga_count", "rfi_fpga_count", "rfi_only_fpga_count", "pl_fpga_count",
                  "frac_lost", "frac_rfi", "frac_rfi_only", "frac_pl"})
                BOOST_CHECK(!file.exist(prefix + "/" + name));
            auto counts_dataset = file.getDataSet(prefix + "/valid_fpga_count_per_product");
            const auto dims = counts_dataset.getSpace().getDimensions();
            BOOST_REQUIRE(dims == std::vector<size_t>({data.num_file_f, np, 3}));
            BOOST_CHECK_EQUAL(counts_dataset.getDataType().getSize(), sizeof(uint64_t));
            std::vector<std::string> axes;
            counts_dataset.getAttribute("axis").read(axes);
            BOOST_CHECK(axes == std::vector<std::string>({"frequency", "product", "time"}));
            std::vector<uint64_t> counts(data.num_file_f * np * 3);
            std::vector<N2::cfloat> vis(counts.size());
            std::vector<float> weights(counts.size());
            std::vector<uint8_t> added(data.num_file_f * 3);
            counts_dataset.read_raw(counts.data());
            file.getDataSet("/vis").read_raw(vis.data());
            file.getDataSet(prefix + "/vis_weight").read_raw(weights.data());
            file.getDataSet("/frames_added").read_raw(added.data());
            for (size_t f = 0; f < data.num_file_f; ++f) {
                for (size_t t = 0; t < 3; ++t) {
                    const bool present = f < 2 && t != 1;
                    BOOST_CHECK_EQUAL(added[data.idx_ft(f, t)], present ? 1 : 0);
                    for (size_t p = 0; p < np; ++p) {
                        const size_t i = data.idx_fpt(f, p, t);
                        BOOST_CHECK_EQUAL(counts[i], present ? product_support(f, p, t) : 0);
                        const auto expected_vis =
                            present && p > 0
                                ? N2::cfloat(float(100 * f + 10 * p + t), -float(2 * p + t))
                                : N2::cfloat(0, 0);
                        BOOST_CHECK(vis[i] == expected_vis);
                        BOOST_CHECK_EQUAL(weights[i],
                                          present && p > 1 ? float(20 * f + p + 2 * t) : 0.0f);
                    }
                }
            }
        }
        rm_tree_if_exists(base);
    }
}

BOOST_AUTO_TEST_CASE(test_per_product_file_refuses_invalid_support_and_mode_change) {
    using kotekan::N2SupportMode;
    const size_t ni = 2, ne = 1, np = 3;
    const auto mode = N2SupportMode::PerProductV1;
    auto pool = metadataPool::create(1, sizeof(N2Metadata), "product_guard_pool", "N2Metadata");
    Buffer buf(1, N2FrameDesc::calculate_frame_size(ni, ne, np, mode), pool, "product_guard", "N2",
               0, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<N2FrameDesc>(ni, ne, np, N2Layout::FullUpperTri,
                                                        std::vector<N2::prod_ctype>{}, mode));
    fill_product_support_frame(&buf, 0, 0, 0);
    N2FrameView fv(&buf, 0, true);
    const std::string base = "product_guard_file";
    rm_tree_if_exists(base);
    ensure_directory(base);
    ensure_directory(base + "/.partial");
    TestVisFileData data(fv, 2, 0.0, 0, base, "");
    fv.valid_fpga_ticks[1] = fv.frame_length_fpga_ticks + 1;
    BOOST_CHECK_THROW(data.add_frame(fv, 0), std::runtime_error);
    BOOST_CHECK_EQUAL(data.get_added_count(), 0);
    fv.valid_fpga_ticks[1] = product_support(0, 1, 0);
    auto meta = get_N2_metadata(&buf, 0);
    for (auto counter : {&meta->n_valid_fpga_ticks, &meta->n_pl_fpga_ticks, &meta->n_rfi_fpga_ticks,
                         &meta->n_rfi_only_fpga_ticks}) {
        *counter = 1;
        BOOST_CHECK_THROW(data.add_frame(fv, 0), std::runtime_error);
        BOOST_CHECK_EQUAL(data.get_added_count(), 0);
        *counter = 0;
    }
    auto scalar_pool =
        metadataPool::create(1, sizeof(N2Metadata), "scalar_guard_pool", "N2Metadata");
    Buffer scalar(1, N2FrameDesc::calculate_frame_size(ni, ne, np), scalar_pool, "scalar_guard",
                  "N2", 0, false, false, std::vector<int>{}, true);
    scalar.ensure_frame_desc(std::make_shared<N2FrameDesc>(ni, ne, np, N2Layout::FullUpperTri));
    fill_n2_frame_with_abs_freq(&scalar, 0, ni, ne, 0, 0, 1'000'000'000, 100, 0);
    N2FrameView scalar_fv(&scalar, 0);
    BOOST_CHECK_THROW(data.add_frame(scalar_fv, 0), std::runtime_error);
    BOOST_CHECK_EQUAL(data.get_added_count(), 0);
    BOOST_REQUIRE(data.add_frame(fv, 0) == N2FileData::AddFrameStatus::Success);
    BOOST_REQUIRE(data.flush_to_disk());
    data.close();
    rm_tree_if_exists(base);
}

BOOST_AUTO_TEST_CASE(test_per_product_subset_index_order_and_reorder_refusal) {
    const auto mode = kotekan::N2SupportMode::PerProductV1;
    const std::vector<N2::prod_ctype> products = {{2, 2}, {0, 2}, {1, 1}};
    const std::vector<N2::prod_ctype> reordered = {{0, 2}, {2, 2}, {1, 1}};
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "subset_support_pool", "N2Metadata");
    const auto size = N2FrameDesc::calculate_frame_size(3, 1, 3, mode);
    Buffer source(1, size, pool, "subset_support", "N2", 0, false, false, std::vector<int>{}, true);
    Buffer changed(1, size, pool, "subset_reordered", "N2", 0, false, false, std::vector<int>{},
                   true);
    source.ensure_frame_desc(
        std::make_shared<N2FrameDesc>(3, 1, 3, N2Layout::GeneralSubset, products, mode));
    changed.ensure_frame_desc(
        std::make_shared<N2FrameDesc>(3, 1, 3, N2Layout::GeneralSubset, reordered, mode));
    fill_product_support_frame(&source, 0, 0, 0);
    fill_product_support_frame(&changed, 0, 1, 0);
    N2FrameView fv(&source, 0, true), reordered_fv(&changed, 0, true);
    const std::string base = "product_subset_order";
    rm_tree_if_exists(base);
    ensure_directory(base);
    ensure_directory(base + "/.partial");
    TestVisFileData data(fv, 1, 0.0, 0, base, "");
    BOOST_REQUIRE(data.add_frame(fv, 0) == N2FileData::AddFrameStatus::Success);
    BOOST_CHECK_THROW(data.add_frame(reordered_fv, 0), std::runtime_error);
    BOOST_CHECK_EQUAL(data.get_added_count(), 1);
    BOOST_REQUIRE(data.flush_to_disk());
    data.close();
    {
        File file(base + "/.partial/vis_0.h5", File::ReadOnly);
        std::vector<N2::prod_ctype> indices(3);
        file.getDataSet("/index_map/prod").read_raw(indices.data());
        std::vector<uint64_t> counts(data.num_file_f * 3);
        file.getDataSet("/valid_fpga_count_per_product").read_raw(counts.data());
        for (size_t p = 0; p < 3; ++p) {
            BOOST_CHECK_EQUAL(indices[p].input_a, products[p].input_a);
            BOOST_CHECK_EQUAL(indices[p].input_b, products[p].input_b);
            BOOST_CHECK_EQUAL(counts[p], product_support(0, p, 0));
        }
        for (size_t p = 3; p < counts.size(); ++p)
            BOOST_CHECK_EQUAL(counts[p], 0);
    }
    rm_tree_if_exists(base);
}

BOOST_AUTO_TEST_CASE(test_writer_per_product_explicit_mode_and_serialization) {
    using kotekan::N2SupportMode;
    // Reject mismatched support modes; write a file when the modes match.
    for (int scenario = 0; scenario < 3; ++scenario) {
        const bool per_product = scenario != 0;
        const std::string stage_mode = scenario == 1 ? "scalar" : "per_product_v1";
        const auto mode = per_product ? N2SupportMode::PerProductV1 : N2SupportMode::Scalar;
        const std::string base = "product_stage_" + std::to_string(scenario);
        const std::string stage_name = "/product_writer_" + std::to_string(scenario);
        const std::string config_key = stage_name.substr(1);
        rm_tree_if_exists(base);
        auto conf = make_writer_config(stage_name, "product_stage_buf", base, "vis", false, 1);
        auto cfg = conf.get_full_config_json();
        cfg[config_key]["support_mode"] = stage_mode;
        cfg[config_key]["max_frames"] = 1;
        cfg[stage_name] = cfg[config_key];
        conf.update_config(cfg);
        const size_t ni = 2, ne = 1, np = 3;
        auto pool = metadataPool::create(1, sizeof(N2Metadata), "product_stage_pool", "N2Metadata");
        Buffer buf(1, N2FrameDesc::calculate_frame_size(ni, ne, np, mode), pool,
                   "product_stage_buf", "N2", 0, false, false, std::vector<int>{}, true);
        buf.ensure_frame_desc(std::make_shared<N2FrameDesc>(ni, ne, np, N2Layout::FullUpperTri,
                                                            std::vector<N2::prod_ctype>{}, mode));
        buf.register_producer("product-producer");
        kotekan::bufferContainer bc;
        bc.add_buffer("product_stage_buf", &buf);
        hdf5N2Write stage(conf, stage_name, bc);
        if (per_product)
            fill_product_support_frame(&buf, 0, 0, 0);
        else
            fill_n2_frame_with_abs_freq(&buf, 0, ni, ne, 0, 0, 1'000'000'000, 100, 0);
        buf.mark_frame_full("product-producer", 0);
        if (scenario < 2) {
            BOOST_CHECK_THROW(stage.main_thread(), std::runtime_error);
            BOOST_CHECK(list_h5_datasets(base).empty());
        } else {
            stage.main_thread();
            auto files = list_h5_datasets(base);
            BOOST_REQUIRE_EQUAL(files.size(), 1);
            File file(files.front(), File::ReadOnly);
            BOOST_CHECK(file.exist("/valid_fpga_count_per_product"));
            BOOST_CHECK(!file.exist("/frac_lost"));
            std::vector<uint64_t> counts(
                Telescope::instance().cast<CHORDTelescope>().num_science_freqs() * np);
            file.getDataSet("/valid_fpga_count_per_product").read_raw(counts.data());
            for (size_t p = 0; p < np; ++p)
                BOOST_CHECK_EQUAL(counts[p], product_support(0, p, 0));
        }
        buf.send_shutdown_signal();
        rm_tree_if_exists(base);
    }
}

// Test 1: Two hdf5N2Write stage blocks pointing at the same base_dir should
// be rejected at construction. Placed before any test that calls
// kotekan_test_logging::configure() — that helper installs a SIGTERM handler
// that _Exits the process, which would prevent BOOST_CHECK_THROW from catching
// the FATAL_ERROR.
BOOST_AUTO_TEST_CASE(test_writer_base_dir_conflict_detection) {

    const std::string base_dir = "test_hdf5N2Write_conflict";
    const std::string unique_a = "/hdf5_vis_writer_conflict_a";
    const std::string in_buf_name = "n2buf_conflict";
    rm_tree_if_exists(base_dir);

    auto conf = make_writer_config(unique_a, in_buf_name, base_dir, /*file_name*/ "vis",
                                   /*prefix_hostname*/ false, /*num_file_t*/ 2,
                                   /*input_order*/ ElementOrder::CHORDBeamformer,
                                   /*blocksize_f*/ 0, /*blocksize_p*/ 0,
                                   /*blocksize_t*/ 2, /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL, TEST_GAINS_FILE);

    // Inject a peer hdf5N2Write stage block sharing the same base_dir.
    {
        auto cfg = conf.get_full_config_json();
        nlohmann::json peer;
        peer["kotekan_stage"] = "hdf5N2Write";
        peer["base_dir"] = base_dir;
        cfg["hdf5_vis_writer_conflict_b"] = peer;
        conf.update_config(cfg);
    }

    // Minimal buffer setup (the stage's constructor needs in_buf to exist).
    const size_t num_input = 2, num_ev = 1;
    const size_t num_prod = N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri);
    const size_t frame_size = N2FrameDesc::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_conflict", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
    buf.ensure_frame_desc(
        std::make_shared<N2FrameDesc>(num_input, num_ev, num_prod, N2Layout::FullUpperTri));
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    BOOST_CHECK_THROW(hdf5N2Write(conf, unique_a, bc), std::runtime_error);

    rm_tree_if_exists(base_dir);
}

// Test 2: Full-block flush with transpose validation
BOOST_AUTO_TEST_CASE(test_writer_full_block_transpose) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer";
    const std::string in_buf_name = "n2buf";
    const std::string base_dir = "test_hdf5N2Write_full";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    // Dims
    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    // Two 100-second frames per file -> file_nt=2
    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 100; // ensure fractions compute as 80/100
    const uint64_t num_file_t = 2;
    const size_t expected_num_file_t = num_file_t;

    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name,
                                   /*prefix_hostname*/ false, num_file_t,
                                   /*input_order*/ ElementOrder::CHORDBeamformer,
                                   /*blocksize_f (0=all)*/ 0, /*blocksize_p*/ 0,
                                   /*blocksize_t*/ num_file_t, /*grace*/ 60,
                                   /*seq_override*/ dt_ns, TEST_GAINS_FILE);
    set_file_num_t(conf, unique_name, num_file_t);

    // Buffer + container
    const size_t num_prod = N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri);
    const size_t frame_size = N2FrameDesc::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_full", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
    buf.ensure_frame_desc(
        std::make_shared<N2FrameDesc>(num_input, num_ev, num_prod, N2Layout::FullUpperTri));
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    // Create and start stage
    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    // Time logic: keep base_time within the first file window for deterministic naming
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t base_time_ns = 10'000'000'000ULL; // falls within file window

    // Send frames out of time order to exercise t-indexing
    N2::frameID fid(&buf);
    const uint64_t abs_base_idx = 0;
    // Order: all f at t=1 then all f at t=0
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, /*t*/ 1,
                                    base_time_ns + 1 * frame_len_ns, frame_len_ticks,
                                    abs_base_idx + 1);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, /*t*/ 0, base_time_ns,
                                    frame_len_ticks, abs_base_idx + 0);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    // Wait for the last produced frame to be consumed to ensure flush happened
    wait_until_frame_empty(&buf, fid - 1, 30.0);

    // Graceful shutdown
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Find exactly one dataset under base_dir's acquisition subdir
    auto datasets = list_h5_datasets(base_dir);
    BOOST_REQUIRE_MESSAGE(datasets.size() == 1, "Expected 1 dataset, found " << datasets.size());
    const std::string ds_path = datasets[0];

    {
        File f(ds_path, File::ReadOnly);
        validate_dataset_content(f, num_input, num_ev, nfreq, expected_num_file_t);
        validate_index_map_inputs(f, num_input);
    }

    // Cleanup
    rm_tree_if_exists(base_dir);
}

// Test 3: Partial flush triggered on exit (incomplete time block)
BOOST_AUTO_TEST_CASE(test_writer_partial_flush_on_exit) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_partial";
    const std::string in_buf_name = "n2buf_partial";
    const std::string base_dir = "test_hdf5N2Write_partial";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    // Dims
    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    // Use 2 time frames per file so file_nt=2 with 1s frames
    const uint64_t num_file_t = 2;
    const size_t num_prod = N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri);

    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name,
                                   /*prefix_hostname*/ false, num_file_t,
                                   /*input_order*/ ElementOrder::CHORDBeamformer,
                                   /*blocksize_f (0=all)*/ 0, /*blocksize_p*/ 0, /*blocksize_t*/ 1,
                                   /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL, TEST_GAINS_FILE);
    set_file_num_t(conf, unique_name, num_file_t);

    // Buffer + container
    const size_t frame_size = N2FrameDesc::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_partial", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
    buf.ensure_frame_desc(
        std::make_shared<N2FrameDesc>(num_input, num_ev, num_prod, N2Layout::FullUpperTri));
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t frame_len_ticks = 1;
    const uint64_t base_time_ns = 2'000'000'000ULL; // +2 seconds

    // Only produce t=0 for all freqs, leave t=1 missing to force partial flush on exit
    N2::frameID fid(&buf);
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, /*t*/ 0, base_time_ns,
                                    frame_len_ticks, 0);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    // Ensure stage consumed the last produced frame
    wait_until_frame_empty(&buf, fid - 1, 30.0);

    // Trigger shutdown to force partial flush path
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    auto datasets = list_h5_datasets(base_dir);
    BOOST_REQUIRE_MESSAGE(datasets.size() == 1, "Expected 1 dataset, found " << datasets.size());
    const std::string ds_path = datasets[0];

    {
        File f(ds_path, File::ReadOnly);
        std::vector<std::vector<std::vector<cfloat>>> vis_out;
        std::vector<std::vector<std::vector<float>>> w_out;
        f.getDataSet("/vis").read(vis_out);
        f.getDataSet("/vis_weight").read(w_out);
        size_t ff = std::min<size_t>(1, nfreq - 1);
        for (size_t p = 0; p < num_prod; ++p) {
            float base = 100.0f + 10.0f * float(p);
            BOOST_CHECK(vis_out[ff][p][0] == N2::cfloat(base + 1.0f, base + 2.0f));
            BOOST_CHECK_CLOSE_FRACTION(w_out[ff][p][0], 1000.0f + float(p), 1e-6f);
            BOOST_CHECK(vis_out[ff][p][1] == N2::cfloat(0.0f, 0.0f));
            BOOST_CHECK_CLOSE_FRACTION(w_out[ff][p][1], 0.0f, 1e-6f);
        }
    }

    // Cleanup
    rm_tree_if_exists(base_dir);
}

// Test 4: Multi-file rollover when time crosses a file window
BOOST_AUTO_TEST_CASE(test_writer_multi_file_rollover) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_rollover";
    const std::string in_buf_name = "n2buf_rollover";
    const std::string base_dir = "test_hdf5N2Write_rollover";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const uint64_t num_file_t = 2;
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false, num_file_t,
                                   /*input_order*/ ElementOrder::CHORDBeamformer,
                                   /*bs_f (0=all)*/ 0, /*bs_p*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL, TEST_GAINS_FILE);
    set_file_num_t(conf, unique_name, num_file_t);

    const size_t frame_size = N2FrameDesc::calculate_frame_size(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_roll", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<N2FrameDesc>(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri),
        N2Layout::FullUpperTri));
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 100;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t file_len_ns = frame_len_ns * num_file_t;
    const uint64_t baseA = 12'000'000'000ULL;
    const uint64_t baseB = baseA + file_len_ns;

    N2::frameID fid(&buf);
    const uint64_t abs_base_a = 0;
    const uint64_t abs_base_b = num_file_t;
    // File window A (t=0..1)
    for (size_t t = 0; t < num_file_t; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, t,
                                        baseA + t * frame_len_ns, frame_len_ticks, abs_base_a + t);
            buf.mark_frame_full("test-producer", fid);
            fid++;
        }
    // File window B (t=0..1)
    for (size_t t = 0; t < num_file_t; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, t,
                                        baseB + t * frame_len_ns, frame_len_ticks, abs_base_b + t);
            buf.mark_frame_full("test-producer", fid);
            fid++;
        }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    auto datasets = list_h5_datasets(base_dir);
    BOOST_REQUIRE_MESSAGE(datasets.size() == 2, "Expected 2 datasets, found " << datasets.size());
    // Validate each one opens and contents make sense
    for (const auto& p : datasets) {
        File f(p, File::ReadOnly);
        validate_dataset_content(f, num_input, num_ev, nfreq, num_file_t);
    }
    rm_tree_if_exists(base_dir);
}

// Test 5: Adjacent file windows produce distinct dataset names
BOOST_AUTO_TEST_CASE(test_writer_distinct_window_names) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_names";
    const std::string in_buf_name = "n2buf_names";
    const std::string base_dir = "test_hdf5N2Write_names";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t num_file_t = 1; // one frame per file
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false, num_file_t,
                                   /*input_order*/ ElementOrder::CHORDBeamformer,
                                   /*bs_f (0=all)*/ 0, /*bs_p*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ dt_ns, TEST_GAINS_FILE);
    set_file_num_t(conf, unique_name, num_file_t);
    const size_t frame_size = N2FrameDesc::calculate_frame_size(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_subsec", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<N2FrameDesc>(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri),
        N2Layout::FullUpperTri));
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t baseA = 6'000'000'000ULL;
    const uint64_t baseB = baseA + frame_len_ns; // next file window

    N2::frameID fid(&buf);
    const uint64_t abs_base_a = 0;
    const uint64_t abs_base_b = 1;
    // File window A
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, 0, baseA, frame_len_ticks,
                                    abs_base_a);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // File window B
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, 0, baseB, frame_len_ticks,
                                    abs_base_b);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Both datasets should exist under base_dir/acq_*/ and have different names
    auto datasets = list_h5_datasets(base_dir);
    BOOST_REQUIRE_MESSAGE(datasets.size() == 2, "Expected 2 datasets, found " << datasets.size());
    BOOST_CHECK(datasets[0] != datasets[1]);
    rm_tree_if_exists(base_dir);
}

// Test 6: Grace-based finalize of partial dataset (late_frame_grace_seconds=0)
BOOST_AUTO_TEST_CASE(test_writer_timeout_finalize_zero_threshold) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_timeout";
    const std::string in_buf_name = "n2buf_timeout";
    const std::string base_dir = "test_hdf5N2Write_timeout";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const uint64_t num_file_t = 2;
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false, num_file_t,
                                   /*input_order*/ ElementOrder::CHORDBeamformer, 0 /*bs_f*/,
                                   0 /*bs_p*/, 0 /*bs_t*/, 0 /*late_frame_grace_seconds*/,
                                   1'000'000'000ULL, TEST_GAINS_FILE);
    set_file_num_t(conf, unique_name, num_file_t);

    const size_t frame_size = N2FrameDesc::calculate_frame_size(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(8, sizeof(N2Metadata), "pool_timeout", "N2Metadata");
    Buffer buf(8, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<N2FrameDesc>(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri),
        N2Layout::FullUpperTri));
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t baseA = 6'000'000'000ULL;
    const uint64_t baseB = baseA + frame_len_ns * num_file_t; // next file window

    N2::frameID fid(&buf);
    const uint64_t abs_base_a = 0;
    const uint64_t abs_base_b = num_file_t;
    // Produce only t=0 for file window A (all freqs)
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, 0, baseA, frame_len_ticks,
                                    abs_base_a);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // Small pause to allow stage to open/initialize dataset and update last activity
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // Produce t=0 for file window B to trigger timeout scan and finalize A
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, 0, baseB, frame_len_ticks,
                                    abs_base_b);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    auto datasets = list_h5_datasets(base_dir);
    BOOST_REQUIRE_MESSAGE(datasets.size() >= 1, "Expected at least 1 finalized dataset");
    rm_tree_if_exists(base_dir);
}

// Test 7: Late-frame drop when final already exists
BOOST_AUTO_TEST_CASE(test_writer_drop_if_final_exists) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_drop";
    const std::string in_buf_name = "n2buf_drop";
    const std::string base_dir = "test_hdf5N2Write_drop";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 2;
    const uint64_t num_file_t = 1;
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false, num_file_t,
                                   /*input_order*/ ElementOrder::CHORDBeamformer,
                                   /*bs_f (0=all)*/ 0, /*bs_p*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL, TEST_GAINS_FILE);
    set_file_num_t(conf, unique_name, num_file_t);
    set_stage_log_level(conf, unique_name, "ERROR");
    const size_t frame_size = N2FrameDesc::calculate_frame_size(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_drop", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<N2FrameDesc>(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri),
        N2Layout::FullUpperTri));
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t base_time_ns = 8'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t file_start_time_ns = base_time_ns; // aligned to 1-second file window

    // The writer creates an `acq_<timestamp>` subdir at startup; pre-create the
    // final dataset there so the writer's drop-on-existing-final logic triggers.
    const std::string acq_dir = wait_for_acq_dir(base_dir);
    BOOST_REQUIRE_MESSAGE(!acq_dir.empty(),
                          "Writer did not create an acq_* subdir under " << base_dir);
    const std::string ds_final = get_dataset_name(acq_dir, 0, file_start_time_ns, suffix);
    {
        FILE* fp = std::fopen(ds_final.c_str(), "wb");
        BOOST_REQUIRE(fp != nullptr);
        std::fclose(fp);
    }

    N2::frameID fid(&buf);
    const uint64_t abs_base_a = 0;
    const uint64_t abs_base_b = 1;
    // Attempt to produce file window A (should be dropped)
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, 0, base_time_ns,
                                    frame_len_ticks, abs_base_a);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // Produce next file window (should write)
    const uint64_t next_time = base_time_ns + frame_len_ns;
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, 0, next_time, frame_len_ticks,
                                    abs_base_b);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Expect exactly one new dataset in addition to the pre-existing marker
    auto datasets = list_h5_datasets(base_dir);
    BOOST_REQUIRE_MESSAGE(datasets.size() == 2,
                          "Expected 2 dataset entries (pre-existing + new), found "
                              << datasets.size());

    // Cleanup
    rm_tree_if_exists(base_dir);
}

/// Test 8: Basic geometry write test
BOOST_AUTO_TEST_CASE(test_writer_geometry_basic) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_geom";
    const std::string in_buf_name = "n2buf_geom";
    const std::string base_dir = "test_hdf5N2Write_geom";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const uint64_t num_file_t = 2;

    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false, num_file_t,
                                   /*input_order*/ ElementOrder::CHORDBeamformer,
                                   /*bs_f (0=all)*/ 0, /*bs_p*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL, TEST_GAINS_FILE);
    set_file_num_t(conf, unique_name, num_file_t);
    const size_t frame_size = N2FrameDesc::calculate_frame_size(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(4, sizeof(N2Metadata), "pool_geom", "N2Metadata");
    Buffer buf(4, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
    buf.ensure_frame_desc(std::make_shared<N2FrameDesc>(
        num_input, num_ev, N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri),
        N2Layout::FullUpperTri));
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t frame_len_ticks = 1;
    const uint64_t base_time_ns = 9'000'000'000ULL;

    N2::frameID fid(&buf);
    const uint64_t abs_base = 0;
    // Produce t=0 for f=0 with normal geometry
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs_freq(&buf, fid, num_input, num_ev, f, 0, base_time_ns,
                                    frame_len_ticks, abs_base);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Dataset should still exist and be readable
    auto datasets = list_h5_datasets(base_dir);
    BOOST_REQUIRE(!datasets.empty());
    const std::string ds_path = datasets[0];
    {
        File f(ds_path, File::ReadOnly);
        // Quick sanity: check arrays exist
        auto ds_vis = f.getDataSet("/vis");
        BOOST_REQUIRE(ds_vis.getElementCount() > 0);
    }
    rm_tree_if_exists(ds_path);
    rm_tree_if_exists(base_dir);
}
