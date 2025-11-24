// Boost tests for the hdf5VisWrite stage end-to-end, writing HDF5 files

#define BOOST_TEST_MODULE "test_hdf5VisWrite"

#include "Config.hpp" // for Config
#include "H5Support.hpp"
#include "N2FrameView.hpp"     // for N2FrameView
#include "N2Metadata.hpp"      // for N2Metadata, get_N2_metadata
#include "N2Util.hpp"          // for N2 helpers
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "hdf5N2Write.hpp"    // for hdf5N2Write
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

using std::string;

using HighFive::File;

namespace {

/// Simple test double to expose N2FileData internals without modifying production code.
class TestVisFileData : public N2FileData {
public:
    TestVisFileData(const N2FrameView& fv, uint64_t num_file_t, double open_wall_s,
                    uint64_t abs_file_idx, std::string base_dir) :
        N2FileData(N2FileData::CHORD, num_file_t, fv, open_wall_s, abs_file_idx,
                   /*blocksize_f*/ 0,
                   /*blocksize_t*/ 1,
                   /*compression*/ "none",
                   /*compression_level*/ 0,
                   /*use_bitshuffle*/ false, std::move(base_dir)) {}

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
        return time_center_ut1.at(t);
    }
    int64_t get_bin_ut1(size_t t) const {
        return bin_ut1.at(t);
    }
    size_t get_added_count() const {
        return added_count;
    }
};

static void ensure_directory(const std::string& path) {
    if (path.empty())
        return;
    if (::mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
        BOOST_FAIL("Failed to create directory " << path << " errno=" << errno);
    }
}

static void set_file_num_t(kotekan::Config& conf, const std::string& unique_name,
                           uint64_t file_num_t) {
    nlohmann::json cfg = conf.get_full_config_json();
    auto update_stage = [&](const std::string& key) {
        if (!key.empty() && cfg.contains(key) && cfg[key].is_object())
            cfg[key]["file_num_t"] = file_num_t;
    };
    update_stage(unique_name);
    if (!unique_name.empty() && unique_name.front() == '/')
        update_stage(unique_name.substr(1));
    conf.update_config(cfg);
}

static void set_stage_log_level(kotekan::Config& conf, const std::string& unique_name,
                                const std::string& level) {
    nlohmann::json cfg = conf.get_full_config_json();
    auto update_stage = [&](const std::string& key) {
        if (!key.empty() && cfg.contains(key) && cfg[key].is_object())
            cfg[key]["log_level"] = level;
    };
    update_stage(unique_name);
    if (!unique_name.empty() && unique_name.front() == '/')
        update_stage(unique_name.substr(1));
    conf.update_config(cfg);
}

static void fill_n2_frame_with_abs(Buffer* buf, int frame_id, size_t num_input, size_t num_ev,
                                   size_t nfreq, size_t f_index, size_t t_index,
                                   uint64_t frame_start_time_ns, uint64_t frame_length_ticks,
                                   uint64_t abs_time_idx) {
    fill_n2_frame(buf, frame_id, num_input, num_ev, nfreq, f_index, t_index, frame_start_time_ns,
                  frame_length_ticks);
    auto meta = get_N2_metadata(buf, frame_id);
    BOOST_REQUIRE(meta);
    meta->abs_time_idx = abs_time_idx;
    meta->time_center_eop.t_ut1 = static_cast<int64_t>(frame_start_time_ns);
    meta->bin_eop.t_ut1 = static_cast<int64_t>(frame_start_time_ns);
}

} // namespace

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
    buf << "vis_" << abs_file_idx << "_";
    std::time_t tsec = file_start_time_ns / 1'000'000'000ULL;
    const uint64_t nsec = file_start_time_ns % 1'000'000'000ULL;
    buf << std::put_time(std::gmtime(&tsec), "%Y%m%dT%H%M%S") << "_" << std::setw(9)
        << std::setfill('0') << nsec << suffix;
    return buf.str();
}

// Read back and validate a few arrays using the known patterns
// Note this is not a full validation of all data, just spot-checks
// This function assumes the fill_n2_frame function has been called
static void validate_dataset_content(File& file, size_t num_input, size_t num_ev, size_t nfreq,
                                     size_t file_nt) {
    const size_t num_prod = N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri);

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
        file.getDataSet("/time_center_ut1").read(tcen);
        file.getDataSet("/bin_ut1").read(bin);
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
    const size_t num_prod = N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri);
    const size_t num_ev = 2;
    const size_t num_freq = 3;
    const size_t num_file_t = 2;

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(1, sizeof(N2Metadata), "test_pool", "N2Metadata");
    Buffer buf(1, frame_size, pool, "n2buf", "N2", 1, false, false, std::vector<int>{}, true);

    buf.allocate_new_metadata_object(0);
    auto meta = get_N2_metadata(&buf, 0);
    BOOST_REQUIRE(meta);
    meta->num_elements = num_input;
    meta->num_prod = num_prod;
    meta->num_ev = num_ev;
    meta->nfreq = num_freq;
    meta->freq_id = 1;
    meta->fpga_start_tick = 111;
    meta->frame_start_time_ns = 222;
    meta->frame_length_fpga_ticks = 100;
    meta->n_valid_fpga_ticks = 80;
    meta->n_rfi_fpga_ticks = 5;
    meta->abs_time_idx = 5;
    meta->time_center_eop.t_ut1 = 333;
    meta->bin_eop.t_ut1 = 444;
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
    const size_t f = meta->freq_id;
    const size_t t = 1;

    // Add frame to (f,t) slot
    data.add_frame(fv, t);

    // Check a few expected values in memory
    BOOST_CHECK(data.get_vis(f, 0, t) == N2::cfloat(1.0f, 2.0f));
    BOOST_CHECK_CLOSE_FRACTION(data.get_weight(f, 0, t), float(1000), 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_eval(f, 0, t), float(60), 1e-6f);
    BOOST_CHECK(data.get_evec(f, 0, 0, t) == N2::cfloat(0.5f, -1.5f));
    BOOST_CHECK_CLOSE_FRACTION(data.get_erms(f, t), 3.14f, 1e-6f);
    BOOST_CHECK(data.get_gain(f, 0, t) == N2::cfloat(200.0f, -200.0f));
    BOOST_CHECK_CLOSE_FRACTION(data.get_flags(f, 0, t), 300.0f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_lost(f, t), 1.0f - 80.0f / 100.0f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_rfi(f, t), 5.0f / 100.0f, 1e-6f);
    BOOST_CHECK_EQUAL(data.get_fpga_start_tick(t), uint64_t(111));
    BOOST_CHECK_EQUAL(data.get_frame_length_fpga_ticks(t), uint64_t(100));
    BOOST_CHECK_EQUAL(data.get_time_center_ut1(t), int64_t(333));
    BOOST_CHECK_EQUAL(data.get_bin_ut1(t), int64_t(444));
    BOOST_CHECK_EQUAL(data.get_added_count(), size_t(1));
    rm_tree_if_exists(base_dir);
}

// Test 2: add_frame for the same (f,t) slot twice with differing metadata values
BOOST_AUTO_TEST_CASE(test_visfiledata_era_and_fraction_guards) {
    N2Metadata force_link_marker;
    const size_t num_input = 2;
    const size_t num_prod = N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri);
    const size_t num_ev = 1;
    const size_t num_freq = 2;
    const size_t num_file_t = 2;

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_guard", "N2Metadata");
    Buffer buf(2, frame_size, pool, "n2buf_guard", "N2", 1, false, false, std::vector<int>{}, true);

    // Prepare frame view and two metadata instances for the same (f,t)
    for (int idx = 0; idx < 2; ++idx)
        buf.allocate_new_metadata_object(idx);
    auto meta1 = get_N2_metadata(&buf, 0);
    auto meta2 = get_N2_metadata(&buf, 1);

    const size_t f = 1;
    const size_t t = 1;

    // meta1
    meta1->num_elements = num_input;
    meta1->num_prod = num_prod;
    meta1->num_ev = num_ev;
    meta1->nfreq = num_freq;
    meta1->freq_id = f;
    meta1->fpga_start_tick = 1000;
    meta1->frame_start_time_ns = 2000;
    meta1->frame_length_fpga_ticks = 100;
    meta1->n_valid_fpga_ticks = 80;
    meta1->n_rfi_fpga_ticks = 30; // sum > frame_len -> should clamp to 20
    meta1->abs_time_idx = 10;
    meta1->time_center_eop.t_ut1 = 10'000;
    meta1->bin_eop.t_ut1 = 10'000;
    meta1->time_center_eop.ERA_deg = 0.0; // legitimate 0.0 value

    // meta2 (same slot), differing ERA and pathological counts
    *meta2 = *meta1;
    meta2->n_valid_fpga_ticks = 150; // > frame len -> clamp to 100
    meta2->n_rfi_fpga_ticks = 50;    // will be ignored because slot already set; kept for symmetry
    meta2->time_center_eop.t_ut1 = 11'000; // should not overwrite the first set value
    meta2->bin_eop.t_ut1 = 11'000;
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
    BOOST_REQUIRE(data.add_frame(fv1, t));
    // Verify fractions computed directly from metadata values
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_lost(f, t), 0.2f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(data.get_frac_rfi(f, t), 0.3f, 1e-6f);
    BOOST_CHECK_EQUAL(data.get_time_center_ut1(t), int64_t(10'000));
    BOOST_CHECK_EQUAL(data.get_bin_ut1(t), int64_t(10'000));

    // Second write to same (f,t) with different UT1 values should be rejected gracefully.
    bool accepted = data.add_frame(fv2, t);
    BOOST_CHECK(!accepted);
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

// Test 1: Full-block flush with transpose validation
BOOST_AUTO_TEST_CASE(test_writer_full_block_transpose) {

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer";
    const std::string in_buf_name = "n2buf";
    const std::string base_dir = "test_hdf5VisWrite_full";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    // Dims
    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    // Use 100-second frames and 200-second file windows to get file_nt=2
    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 100; // ensure fractions compute as 80/100
    const uint64_t file_seconds = 200;
    const size_t expected_file_nt = 2;

    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name,
                                   /*prefix_hostname*/ false, file_seconds,
                                   /*blocksize_f (0=all)*/ 0, /*blocksize_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ dt_ns);
    set_file_num_t(conf, unique_name, expected_file_nt);

    // Buffer + container
    const size_t num_prod = N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri);
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_full", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    // Create and start stage
    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    // Time logic: ensure base_time aligned to file window start
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t base_time_ns = 10'000'000'000ULL; // falls within file window [0,200)s

    // Send frames out of time order to exercise t-indexing
    N2::frameID fid(&buf);
    const uint64_t abs_base_idx = 0;
    // Order: all f at t=1 then all f at t=0
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, /*t*/ 1,
                               base_time_ns + 1 * frame_len_ns, frame_len_ticks, abs_base_idx + 1);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, /*t*/ 0, base_time_ns,
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

    // Find exactly one dataset in base_dir
    auto entries = list_dir_entries(base_dir);
    std::vector<std::string> datasets;
    for (auto& e : entries) {
        if (e.find(suffix) != std::string::npos)
            datasets.push_back(e);
    }
    BOOST_REQUIRE_MESSAGE(datasets.size() == 1, "Expected 1 dataset, found " << datasets.size());
    const std::string ds_path = join_path(base_dir, datasets[0]);

    {
        File f(ds_path, File::ReadOnly);
        validate_dataset_content(f, num_input, num_ev, nfreq, expected_file_nt);
    }

    // Cleanup
    rm_tree_if_exists(ds_path);
    rm_tree_if_exists(base_dir);
}

// Test 2: Partial flush triggered on exit (incomplete time block)
BOOST_AUTO_TEST_CASE(test_writer_partial_flush_on_exit) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_partial";
    const std::string in_buf_name = "n2buf_partial";
    const std::string base_dir = "test_hdf5VisWrite_partial";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    // Dims
    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    // Use a 2-second file window so file_nt=2 with 1s frames
    const uint64_t file_seconds = 2;
    const size_t file_nt = 2;
    const size_t num_prod = N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri);

    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name,
                                   /*prefix_hostname*/ false, file_seconds,
                                   /*blocksize_f (0=all)*/ 0, /*blocksize_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL);
    set_file_num_t(conf, unique_name, file_nt);

    // Buffer + container
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev, num_prod);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_partial", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
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
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, /*t*/ 0, base_time_ns,
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

    auto entries = list_dir_entries(base_dir);
    std::vector<std::string> datasets;
    for (auto& e : entries) {
        if (e.find(suffix) != std::string::npos)
            datasets.push_back(e);
    }
    BOOST_REQUIRE_MESSAGE(datasets.size() == 1, "Expected 1 dataset, found " << datasets.size());
    const std::string ds_path = join_path(base_dir, datasets[0]);

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
    rm_tree_if_exists(ds_path);
    rm_tree_if_exists(base_dir);
}

// Test 3: Multi-file rollover when time crosses a file window
BOOST_AUTO_TEST_CASE(test_writer_multi_file_rollover) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_rollover";
    const std::string in_buf_name = "n2buf_rollover";
    const std::string base_dir = "test_hdf5VisWrite_rollover";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const size_t file_nt = 2;
    const uint64_t file_seconds = 200;
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false,
                                   file_seconds, /*bs_f (0=all)*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL);
    set_file_num_t(conf, unique_name, file_nt);

    const size_t frame_size = N2FrameView::calculate_frame_size(
        num_input, num_ev, N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_roll", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 100;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t file_len_ns = file_seconds * 1'000'000'000ULL;
    const uint64_t baseA = 12'000'000'000ULL; // align to even multiple of 2s file windows
    const uint64_t baseB = baseA + file_len_ns;

    N2::frameID fid(&buf);
    const uint64_t abs_base_a = 0;
    const uint64_t abs_base_b = file_nt;
    // File window A (t=0..1)
    for (size_t t = 0; t < file_nt; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, t,
                                   baseA + t * frame_len_ns, frame_len_ticks, abs_base_a + t);
            buf.mark_frame_full("test-producer", fid);
            fid++;
        }
    // File window B (t=0..1)
    for (size_t t = 0; t < file_nt; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, t,
                                   baseB + t * frame_len_ns, frame_len_ticks, abs_base_b + t);
            buf.mark_frame_full("test-producer", fid);
            fid++;
        }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    auto entries = list_dir_entries(base_dir);
    std::vector<std::string> datasets;
    for (auto& e : entries) {
        if (e.find(suffix) != std::string::npos)
            datasets.push_back(e);
    }
    BOOST_REQUIRE_MESSAGE(datasets.size() == 2, "Expected 2 datasets, found " << datasets.size());
    // Validate each one opens and contents make sense
    for (const auto& e : datasets) {
        const std::string p = join_path(base_dir, e);
        File f(p, File::ReadOnly);
        validate_dataset_content(f, num_input, num_ev, nfreq, file_nt);
        rm_tree_if_exists(p);
    }
    rm_tree_if_exists(base_dir);
}

// Test 4: Adjacent file windows produce distinct dataset names
BOOST_AUTO_TEST_CASE(test_writer_distinct_window_names) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_names";
    const std::string in_buf_name = "n2buf_names";
    const std::string base_dir = "test_hdf5VisWrite_names";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t file_seconds = 1; // one second file window => one frame per file
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false,
                                   file_seconds, /*bs_f (0=all)*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ dt_ns);
    set_file_num_t(conf, unique_name, 1);
    const size_t frame_size = N2FrameView::calculate_frame_size(
        num_input, num_ev, N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_subsec", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
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
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, 0, baseA, frame_len_ticks,
                               abs_base_a);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // File window B
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, 0, baseB, frame_len_ticks,
                               abs_base_b);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Both datasets should exist and have different names
    const std::string d1 = get_dataset_name(base_dir, abs_base_a, baseA, suffix);
    const std::string d2 = get_dataset_name(base_dir, abs_base_b, baseB, suffix);
    bool h1 = path_exists(d1);
    bool h2 = path_exists(d2);
    BOOST_CHECK(h1 && h2);
    if (h1 && h2)
        BOOST_CHECK(d1 != d2);
    if (h1)
        rm_tree_if_exists(d1);
    if (h2)
        rm_tree_if_exists(d2);
    rm_tree_if_exists(base_dir);
}

// Test 5: Grace-based finalize of partial dataset (late_frame_grace_seconds=0)
BOOST_AUTO_TEST_CASE(test_writer_timeout_finalize_zero_threshold) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_timeout";
    const std::string in_buf_name = "n2buf_timeout";
    const std::string base_dir = "test_hdf5VisWrite_timeout";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const size_t file_nt = 2;
    const uint64_t window_seconds = 2;
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false,
                                   window_seconds, 0 /*bs_f*/, 1 /*bs_t*/,
                                   0 /*late_frame_grace_seconds*/, 1'000'000'000ULL);
    set_file_num_t(conf, unique_name, file_nt);

    const size_t frame_size = N2FrameView::calculate_frame_size(
        num_input, num_ev, N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(8, sizeof(N2Metadata), "pool_timeout", "N2Metadata");
    Buffer buf(8, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t baseA = 6'000'000'000ULL;
    const uint64_t baseB = baseA + frame_len_ns * file_nt; // next file window

    N2::frameID fid(&buf);
    const uint64_t abs_base_a = 0;
    const uint64_t abs_base_b = file_nt;
    // Produce only t=0 for file window A (all freqs)
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, 0, baseA, frame_len_ticks,
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
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, 0, baseB, frame_len_ticks,
                               abs_base_b);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    auto entries = list_dir_entries(base_dir);
    std::vector<std::string> datasets;
    for (auto& e : entries) {
        if (e.find(suffix) != std::string::npos)
            datasets.push_back(e);
    }
    BOOST_REQUIRE_MESSAGE(datasets.size() >= 1, "Expected at least 1 finalized dataset");
    for (auto& e : datasets)
        rm_tree_if_exists(join_path(base_dir, e));
    rm_tree_if_exists(base_dir);
}

// Test 6: Late-frame drop when final already exists
BOOST_AUTO_TEST_CASE(test_writer_drop_if_final_exists) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_drop";
    const std::string in_buf_name = "n2buf_drop";
    const std::string base_dir = "test_hdf5VisWrite_drop";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 2;
    const uint64_t file_seconds = 1;
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false,
                                   file_seconds, /*bs_f (0=all)*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL);
    set_file_num_t(conf, unique_name, 1);
    set_stage_log_level(conf, unique_name, "ERROR");
    const size_t frame_size = N2FrameView::calculate_frame_size(
        num_input, num_ev, N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_drop", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
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

    // Pre-create a final dataset path to force drop
    const std::string ds_final = get_dataset_name(base_dir, 0, file_start_time_ns, suffix);
    ensure_directory(base_dir);
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
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, 0, base_time_ns,
                               frame_len_ticks, abs_base_a);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // Produce next file window (should write)
    const uint64_t next_time = base_time_ns + frame_len_ns;
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, 0, next_time,
                               frame_len_ticks, abs_base_b);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Expect exactly one new dataset in addition to the pre-existing marker
    auto entries = list_dir_entries(base_dir);
    size_t datasets = 0;
    for (auto& e : entries) {
        if (e.find(suffix) != std::string::npos)
            datasets++;
    }
    BOOST_REQUIRE_MESSAGE(datasets == 2,
                          "Expected 2 dataset entries (pre-existing + new), found " << datasets);

    // Cleanup
    for (auto& e : entries) {
        rm_tree_if_exists(join_path(base_dir, e));
    }
    rm_tree_if_exists(base_dir);
}

// Test 7: Geometry mismatch within dataset (nfreq) is dropped correctly
BOOST_AUTO_TEST_CASE(test_writer_geometry_mismatch_dropped) {

    kotekan_test_logging::configure();

    const std::string suffix = ".h5";
    const std::string unique_name = "/hdf5_vis_writer_geom";
    const std::string in_buf_name = "n2buf_geom";
    const std::string base_dir = "test_hdf5VisWrite_geom";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const size_t file_nt = 2;

    const uint64_t file_seconds = file_nt; // with 1s frames, file_nt==file_seconds
    auto conf = make_writer_config(unique_name, in_buf_name, base_dir, file_name, false,
                                   file_seconds, /*bs_f (0=all)*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                   /*seq_override*/ 1'000'000'000ULL);
    set_file_num_t(conf, unique_name, file_nt);
    const size_t frame_size = N2FrameView::calculate_frame_size(
        num_input, num_ev, N2FrameView::get_num_prod(num_input, N2Layout::FullUpperTri));
    auto pool = metadataPool::create(4, sizeof(N2Metadata), "pool_geom", "N2Metadata");
    Buffer buf(4, frame_size, pool, in_buf_name, "N2", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5N2Write stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t base_time_ns = 9'000'000'000ULL;

    N2::frameID fid(&buf);
    const uint64_t abs_base = 0;
    // Produce t=0 for f=0 with normal geometry
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, f, 0, base_time_ns,
                               frame_len_ticks, abs_base);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // Send one mismatching frame for same file window with nfreq+1 (should be dropped)
    {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame_with_abs(&buf, fid, num_input, num_ev, nfreq, 0, 1,
                               base_time_ns + frame_len_ns, frame_len_ticks, abs_base + 1);
        auto meta = get_N2_metadata(&buf, fid);
        meta->nfreq = nfreq + 1; // Force mismatch
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Dataset should still exist and be readable
    auto entries = list_dir_entries(base_dir);
    std::string ds_path;
    for (auto& e : entries) {
        if (e.find(suffix) != std::string::npos) {
            ds_path = join_path(base_dir, e);
            break;
        }
    }
    BOOST_REQUIRE(!ds_path.empty());
    {
        File f(ds_path, File::ReadOnly);
        // Quick sanity: check arrays exist
        auto ds_vis = f.getDataSet("/vis");
        BOOST_REQUIRE(ds_vis.getElementCount() > 0);
    }
    rm_tree_if_exists(ds_path);
    rm_tree_if_exists(base_dir);
}
