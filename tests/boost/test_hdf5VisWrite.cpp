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
#include "hdf5VisWrite.hpp"    // for hdf5VisWrite
#include "test_utils.hpp"

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
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
#include <thread>
#include <unistd.h> // for gethostname
#include <utility>
#include <vector>

using std::string;

using HighFive::File;

// Force registration of N2Metadata in metadata factory by referencing the type
static N2Metadata _force_n2meta_registration;

// Unit-test helper for visFileData internals (moved from test_visFileData)
class TestVisFileData : public visFileData {
public:
    using visFileData::idx_feit;
    using visFileData::idx_fet;
    using visFileData::idx_fit;
    using visFileData::idx_fpt;
    using visFileData::idx_ft;
    using visFileData::visFileData;

    N2::cfloat get_vis(size_t f, size_t p, size_t t) const { return vis[idx_fpt(f, p, t)]; }
    float get_weight(size_t f, size_t p, size_t t) const { return vis_weight[idx_fpt(f, p, t)]; }
    float get_eval(size_t f, size_t e, size_t t) const { return eval[idx_fet(f, e, t)]; }
    N2::cfloat get_evec(size_t f, size_t e, size_t i, size_t t) const {
        return evec[idx_feit(f, e, i, t)];
    }
    float get_erms(size_t f, size_t t) const { return erms[idx_ft(f, t)]; }
    N2::cfloat get_gain(size_t f, size_t i, size_t t) const { return gain[idx_fit(f, i, t)]; }
    float get_flags(size_t f, size_t i, size_t t) const { return flags[idx_fit(f, i, t)]; }
    float get_frac_lost(size_t f, size_t t) const { return frac_lost[idx_ft(f, t)]; }
    float get_frac_rfi(size_t f, size_t t) const { return frac_rfi[idx_ft(f, t)]; }
    uint64_t get_n_valid(size_t f, size_t t) const { return n_valid_fpga_ticks[idx_ft(f, t)]; }
    uint64_t get_n_rfi(size_t f, size_t t) const { return n_rfi_fpga_ticks[idx_ft(f, t)]; }
    uint64_t get_fpga_start_tick(size_t t) const { return fpga_start_tick[t]; }
    uint64_t get_frame_start_time_ns(size_t t) const { return frame_start_time_ns[t]; }
    uint64_t get_frame_length_fpga_ticks(size_t) const { return frame_length_fpga_ticks; }
    double get_era_deg(size_t t) const { return era_deg[t]; }
    size_t get_added_count() const { return added_count; }
    uint8_t get_added(size_t f, size_t t) const { return added_ft[idx_ft(f, t)]; }
};

// Helper function to list items in a directory
static std::vector<std::string> list_dir_entries(const std::string& path) {
    std::vector<std::string> out;
    DIR* dir = opendir(path.c_str());
    if (!dir)
        return out;
    while (dirent* ent = readdir(dir)) {
        const char* name = ent->d_name;
        if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0)
            continue;
        out.emplace_back(name);
    }
    closedir(dir);
    return out;
}

// Join two paths with '/' if needed
static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty())
        return b;
    if (a.back() == '/')
        return a + b;
  return a + "/" + b;
}

// Check if path exists
static bool path_exists(const std::string& path) {
    struct stat st;
    return (::stat(path.c_str(), &st) == 0);
}

// Remove (recursively) directory if it exists (POSIX implementation)
static void rm_tree_if_exists(const std::string& path) {
    if (!path_exists(path))
        return;
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        (void)unlink(path.c_str());
        return;
    }
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        const char* name = ent->d_name;
        if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0)
            continue;
        const std::string sub = join_path(path, name);
        struct stat st;
        if (::lstat(sub.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode))
                rm_tree_if_exists(sub);
            else
                (void)unlink(sub.c_str());
        }
    }
    closedir(dir);
    (void)rmdir(path.c_str());
}

// ---------------- visFileData unit tests (moved and extended) ----------------

BOOST_AUTO_TEST_CASE(test_visfiledata_add_frame_single_slot) {
    N2Metadata force_link_marker;
    const size_t num_input = 3;
    const size_t num_prod = N2::get_num_prod(num_input);
    const size_t num_ev = 2;
    const size_t num_freq = 3;
    const size_t num_file_t = 2;

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(1, sizeof(N2Metadata), "test_pool", "N2Metadata");
    Buffer buf(1, frame_size, pool, "n2buf", "vis", 1, false, false, std::vector<int>{}, true);

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
    meta->eop.ERA_deg = 12.34;

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

    TestVisFileData blk(num_file_t, num_freq, num_input, num_prod, num_ev, 100, 100, 0, "", 0.0);
    const size_t f = meta->freq_id;
    const size_t t = 1;
    blk.add_frame(fv, meta, t);

    // Check a few expected values in memory
    BOOST_CHECK(blk.get_vis(f, 0, t) == N2::cfloat(1.0f, 2.0f));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_weight(f, 0, t), float(1000), 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(blk.get_eval(f, 0, t), float(60), 1e-6f);
    BOOST_CHECK(blk.get_evec(f, 0, 0, t) == N2::cfloat(0.5f, -1.5f));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_erms(f, t), 3.14f, 1e-6f);
    BOOST_CHECK(blk.get_gain(f, 0, t) == N2::cfloat(200.0f, -200.0f));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_flags(f, 0, t), 300.0f, 1e-6f);
    BOOST_CHECK_EQUAL(blk.get_n_valid(f, t), uint64_t(80));
    BOOST_CHECK_EQUAL(blk.get_n_rfi(f, t), uint64_t(5));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_frac_lost(f, t), 1.0f - 80.0f / 100.0f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(blk.get_frac_rfi(f, t), 5.0f / 100.0f, 1e-6f);
    BOOST_CHECK_EQUAL(blk.get_fpga_start_tick(t), uint64_t(111));
    BOOST_CHECK_EQUAL(blk.get_frame_start_time_ns(t), uint64_t(222));
    BOOST_CHECK_EQUAL(blk.get_frame_length_fpga_ticks(t), uint64_t(100));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_era_deg(t), 12.34, 1e-12);
    BOOST_CHECK_EQUAL(blk.get_added_count(), size_t(1));
}

BOOST_AUTO_TEST_CASE(test_visfiledata_era_and_fraction_guards) {
    N2Metadata force_link_marker;
    const size_t num_input = 2;
    const size_t num_prod = N2::get_num_prod(num_input);
    const size_t num_ev = 1;
    const size_t num_freq = 2;
    const size_t num_file_t = 2;

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_guard", "N2Metadata");
    Buffer buf(2, frame_size, pool, "n2buf_guard", "vis", 1, false, false, std::vector<int>{},
               true);

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
    meta1->eop.ERA_deg = 0.0;     // legitimate 0.0 value

    // meta2 (same slot), differing ERA and pathological counts
    *meta2 = *meta1;
    meta2->n_valid_fpga_ticks = 150; // > frame len -> clamp to 100
    meta2->n_rfi_fpga_ticks = 50;    // will be ignored because slot already set; kept for symmetry
    meta2->eop.ERA_deg = 12.34;      // should not overwrite the first set value

    N2FrameView fv1(&buf, 0);
    fv1.zero_frame();
    N2FrameView fv2(&buf, 1);
    fv2.zero_frame();

    TestVisFileData blk(num_file_t, num_freq, num_input, num_prod, num_ev, 100, 100, 0, "", 0.0);

    // First write
    blk.add_frame(fv1, meta1, t);
    // Verify clamped fractions (n_valid=80, n_rfi clamped to 20)
    BOOST_CHECK_EQUAL(blk.get_n_valid(f, t), uint64_t(80));
    BOOST_CHECK_EQUAL(blk.get_n_rfi(f, t), uint64_t(20));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_frac_lost(f, t), 0.2f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(blk.get_frac_rfi(f, t), 0.2f, 1e-6f);
    // ERA set to 0.0 and considered valid
    BOOST_CHECK_CLOSE_FRACTION(blk.get_era_deg(t), 0.0, 1e-12);

    // Second write to same (f,t) with different ERA and counts; ERA should not change
    blk.add_frame(fv2, meta2, t);
    BOOST_CHECK_CLOSE_FRACTION(blk.get_era_deg(t), 0.0, 1e-12);
}

// Capability test removed; reading validated by content checks.

// Helper to create a config just for testing
// Build a writer config using file-window-based collection and optional grace/seq override
static kotekan::Config make_base_config(const std::string& unique_name, const std::string& in_buf,
                                        const std::string& base_dir, const std::string& file_name,
                                        bool prefix_hostname, uint64_t zip_compression,
                                        uint64_t file_seconds, const std::string& format,
                                        uint64_t blocksize_f = 0, uint64_t blocksize_t = 1,
                                        uint64_t late_frame_grace_seconds = 60,
                                        uint64_t seq_length_nsec_override = 0) {
    using json = nlohmann::json;

    json j;
    // Minimal stage config; create once and assign under both '/name' and 'name' keys
    nlohmann::json s;
    s["cpu_affinity"] = std::vector<int>{0};
    s["log_level"] = "WARN";
    s["in_buf"] = in_buf;
    s["base_dir"] = base_dir;
    s["file_name"] = file_name;
    s["prefix_hostname"] = prefix_hostname;
    // Legacy key retained in tests; writer ignores ZIP storage. Keep for compatibility.
    s["zip_compression"] = zip_compression;
    // Writer format selection (HDF5)
    s["format"] = format;
    s["blocksize_f"] = blocksize_f;
    s["blocksize_p"] = 0; // unused currently
    s["blocksize_t"] = blocksize_t;
    s["file_seconds"] = file_seconds; // length of each file in seconds (must divide 86400, >0)
    s["join_timeout"] = 5;            // faster failure for tests
    s["late_frame_grace_seconds"] = late_frame_grace_seconds;
    // optionally override fpga sequence length so the Telescope singleton is not needed
    if (seq_length_nsec_override > 0)
        s["seq_length_nsec_override"] = seq_length_nsec_override;
    // Assign to both forms
    j[unique_name] = s;
    if (!unique_name.empty() && unique_name.front() == '/')
        j[unique_name.substr(1)] = s;

    kotekan::Config conf;
    conf.update_config(j);

    return conf;
}

// Build dataset filename to simulate pre-existing final files (mirrors stage logic)
static std::string compute_dataset_name(const std::string& base_dir, const std::string& file_name,
                                        bool prefix_hostname, uint64_t file_start_time_ns,
                                        const std::string& suffix) {
    std::ostringstream buf;
    buf << base_dir;
    if (!base_dir.empty() && base_dir.back() != '/')
        buf << '/';
    if (prefix_hostname) {
        char hostname[256] = {0};
        gethostname(hostname, sizeof hostname);
        buf << hostname << "_";
    }
    std::time_t tsec = file_start_time_ns / 1'000'000'000ULL;
    const uint64_t nsec = file_start_time_ns % 1'000'000'000ULL;
    buf << file_name << "." << std::put_time(std::gmtime(&tsec), "%Y%m%dT%H%M%S") << "_"
        << std::setw(9) << std::setfill('0') << nsec << suffix;
    return buf.str();
}

// Select format and expected filename suffix (HDF5 only).
static bool select_format_and_suffix(std::string& out_format, std::string& out_suffix) {
    out_format = "hdf5";
    out_suffix = ".h5";
    return true;
}

// Fill one synthetic N2 frame with deterministic data for validators
static void fill_n2_frame(Buffer* buf, int frame_id, size_t num_input, size_t num_ev, size_t nfreq,
                          size_t f_index, size_t t_index, uint64_t frame_start_time_ns,
                          uint64_t frame_length_ticks) {
    // Ensure metadata exists
    buf->allocate_new_metadata_object(frame_id);
    auto meta = get_N2_metadata(buf, frame_id);
    BOOST_REQUIRE(meta);
    const size_t num_prod = N2::get_num_prod(num_input);
    meta->num_elements = num_input;
    meta->num_prod = num_prod;
    meta->num_ev = num_ev;
    meta->nfreq = nfreq;
    meta->freq_id = f_index;
    meta->fpga_start_tick = 100 + t_index;
    meta->frame_start_time_ns = frame_start_time_ns;
    meta->frame_length_fpga_ticks = frame_length_ticks;
    meta->n_valid_fpga_ticks = 80;
    meta->n_rfi_fpga_ticks = 5;
    meta->eop.ERA_deg = 9.87 + double(t_index);

    N2FrameView fv(buf, frame_id);
    fv.zero_frame();
    // vis / weight pattern encodes (t,f,p)
    for (size_t p = 0; p < num_prod; ++p) {
        float base = 1000.0f * float(t_index) + 100.0f * float(f_index) + 10.0f * float(p);
        fv.vis[p] = N2::cfloat(base + 1.0f, base + 2.0f);
        fv.weight[p] = 1000.0f + float(p);
    }
    // eval / evec encode (e,i)
    for (size_t e = 0; e < num_ev; ++e) {
        fv.eval[e] = 60.0f + float(e);
        for (size_t i = 0; i < num_input; ++i) {
            fv.evec[num_input * e + i] = N2::cfloat(100.0f * float(e) + float(i) + 0.5f,
                                                    -(100.0f * float(e) + float(i) + 1.5f));
        }
    }
    fv.erms = 3.14f + 0.0f * float(t_index);
    for (size_t i = 0; i < num_input; ++i) {
        fv.gain[i] = N2::cfloat(200.0f + float(i), -200.0f - float(i));
        fv.flags[i] = 300.0f + float(i);
    }
}

// Wait helper: spin until a given frame ID becomes empty (consumed by the stage)
static void wait_until_frame_empty(Buffer* buf, int frame_id, double timeout_seconds = 5.0) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    double next_log = 0.25; // seconds
    while (!buf->is_frame_empty(frame_id)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        double dt = std::chrono::duration<double>(clock::now() - t0).count();
        if (dt >= next_log) {
            next_log += 0.25;
        }
        if (dt > timeout_seconds) {
            int nfull = buf->get_num_full_frames();
            BOOST_FAIL("Timed out waiting for buffer frame to become empty; num_full="
                       << nfull << ", frame_id=" << frame_id);
            break;
        }
    }
}

// Read back and validate a few arrays using the known patterns (HighFive)
static void validate_dataset_content(File& file, size_t num_input, size_t num_ev, size_t nfreq,
                                     size_t file_nt) {
    const size_t num_prod = N2::get_num_prod(num_input);

    // Check one representative frequency (e.g., f=1) across time
    size_t f = std::min<size_t>(1, nfreq - 1);

    // vis + weights
    {
        std::vector<std::vector<std::vector<cfloat>>> vis_out;
        std::vector<std::vector<std::vector<float>>> w_out;
        file.getDataSet("/vis_array").read(vis_out);
        file.getDataSet("/weights_array").read(w_out);
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
        file.getDataSet("/eval_array").read(eval_out);
        for (size_t e = 0; e < num_ev; ++e)
            for (size_t t = 0; t < file_nt; ++t)
                BOOST_CHECK_CLOSE_FRACTION(eval_out[f][e][t], 60.0f + float(e), 1e-6f);
    }

    // evec slice at i=0 spot-check
    {
        std::vector<std::vector<std::vector<std::vector<cfloat>>>> out;
        file.getDataSet("/evec_array").read(out);
        for (size_t e = 0; e < num_ev; ++e) {
            N2::cfloat expected0(100.0f * float(e) + 0.5f, -(100.0f * float(e) + 1.5f));
            for (size_t t = 0; t < file_nt; ++t)
                BOOST_CHECK(out[f][e][0][t] == expected0);
        }
    }

    // erms
    {
        std::vector<std::vector<float>> out;
        file.getDataSet("/erms_array").read(out);
        for (size_t t = 0; t < file_nt; ++t)
            BOOST_CHECK_CLOSE_FRACTION(out[f][t], 3.14f, 1e-6f);
    }

    // gain + flags spot-check i=0
    {
        std::vector<std::vector<std::vector<cfloat>>> gout;
        std::vector<std::vector<std::vector<float>>> fout;
        file.getDataSet("/gain_array").read(gout);
        file.getDataSet("/flags_array").read(fout);
        for (size_t t = 0; t < file_nt; ++t) {
            BOOST_CHECK(gout[f][0][t] == N2::cfloat(200.0f, -200.0f));
            BOOST_CHECK_CLOSE_FRACTION(fout[f][0][t], 300.0f, 1e-6f);
        }
    }

    // per-(freq,time) derived fractions and counts spot-check
    {
        std::vector<std::vector<float>> fl_out, fr_out;
        std::vector<std::vector<uint64_t>> nv_out, nr_out;
        file.getDataSet("/frac_lost_array").read(fl_out);
        file.getDataSet("/frac_rfi_array").read(fr_out);
        file.getDataSet("/n_valid_fpga_ticks").read(nv_out);
        file.getDataSet("/n_rfi_fpga_ticks").read(nr_out);
        for (size_t t = 0; t < file_nt; ++t) {
            BOOST_CHECK_EQUAL(nv_out[f][t], uint64_t(80));
            BOOST_CHECK_EQUAL(nr_out[f][t], uint64_t(5));
            BOOST_CHECK_CLOSE_FRACTION(fl_out[f][t], 1.0f - 80.0f / 100.0f, 1e-6f);
            BOOST_CHECK_CLOSE_FRACTION(fr_out[f][t], 5.0f / 100.0f, 1e-6f);
        }
    }

    // per-time arrays shape exists
    {
        std::vector<uint64_t> s0, s1, s2;
        std::vector<double> s3;
        file.getDataSet("/fpga_start_tick").read(s0);
        file.getDataSet("/frame_start_time_ns").read(s1);
        file.getDataSet("/frame_length_fpga_ticks").read(s2);
        file.getDataSet("/era_deg").read(s3);
        BOOST_CHECK(!s0.empty() && !s2.empty());
    }
}

// Read back and validate a few arrays using the known patterns


BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

// Test 1: Full-block flush with transpose validation
BOOST_AUTO_TEST_CASE(test_writer_full_block_transpose) {

    std::string format, suffix;
    select_format_and_suffix(format, suffix);
    BOOST_TEST_MESSAGE("Using format='" << format << "' suffix='" << suffix << "'");

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

    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name,
                                 /*prefix_hostname*/ false, /*zip*/ 0, file_seconds, format,
                                 /*blocksize_f (0=all)*/ 0, /*blocksize_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ dt_ns);

    // Buffer + container
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_full", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    // Create and start stage
    hdf5VisWrite stage(conf, unique_name, bc);
    stage.start();

    // Time logic: ensure base_time aligned to file window start
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t base_time_ns = 10'000'000'000ULL; // falls within file window [0,200)s

    // Send frames out of time order to exercise t-indexing
    N2::frameID fid(&buf);
    // Order: all f at t=1 then all f at t=0
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, /*t*/ 1,
                      base_time_ns + 1 * frame_len_ns, frame_len_ticks);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, /*t*/ 0, base_time_ns,
                      frame_len_ticks);
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


    std::string format, suffix;
    select_format_and_suffix(format, suffix);

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

    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name,
                                 /*prefix_hostname*/ false, /*zip*/ 0, file_seconds, format,
                                 /*blocksize_f (0=all)*/ 0, /*blocksize_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ 1'000'000'000ULL);

    // Buffer + container
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_partial", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5VisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t frame_len_ticks = 1;
    const uint64_t base_time_ns = 2'000'000'000ULL; // +2 seconds

    // Only produce t=0 for all freqs, leave t=1 missing to force partial flush on exit
    N2::frameID fid(&buf);
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, /*t*/ 0, base_time_ns,
                      frame_len_ticks);
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
        const size_t num_prod = N2::get_num_prod(num_input);
        std::vector<std::vector<std::vector<cfloat>>> vis_out;
        std::vector<std::vector<std::vector<float>>> w_out;
        f.getDataSet("/vis_array").read(vis_out);
        f.getDataSet("/weights_array").read(w_out);
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


    std::string format, suffix;
    select_format_and_suffix(format, suffix);

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
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 file_seconds, format, /*bs_f (0=all)*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ 1'000'000'000ULL);

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_roll", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5VisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 100;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t file_len_ns = file_seconds * 1'000'000'000ULL;
    const uint64_t baseA = 12'000'000'000ULL; // align to even multiple of 2s file windows
    const uint64_t baseB = baseA + file_len_ns;

    N2::frameID fid(&buf);
    // File window A (t=0..1)
    for (size_t t = 0; t < file_nt; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, t, baseA + t * frame_len_ns,
                          frame_len_ticks);
            buf.mark_frame_full("test-producer", fid);
            fid++;
        }
    // File window B (t=0..1)
    for (size_t t = 0; t < file_nt; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, t, baseB + t * frame_len_ns,
                          frame_len_ticks);
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


    std::string format, suffix;
    select_format_and_suffix(format, suffix);

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
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 file_seconds, format, /*bs_f (0=all)*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ dt_ns);
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_subsec", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5VisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t baseA = 6'000'000'000ULL;
    const uint64_t baseB = baseA + frame_len_ns; // next file window

    N2::frameID fid(&buf);
    // File window A
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, 0, baseA, frame_len_ticks);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // File window B
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, 0, baseB, frame_len_ticks);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }

    wait_until_frame_empty(&buf, fid - 1, 30.0);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Both datasets should exist and have different names
    const std::string d1 = compute_dataset_name(base_dir, file_name, false, baseA, suffix);
    const std::string d2 = compute_dataset_name(base_dir, file_name, false, baseB, suffix);
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


    std::string format, suffix;
    select_format_and_suffix(format, suffix);

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
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 window_seconds, format, 0 /*bs_f*/, 1 /*bs_t*/,
                                 0 /*late_frame_grace_seconds*/, 1'000'000'000ULL);

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(8, sizeof(N2Metadata), "pool_timeout", "N2Metadata");
    Buffer buf(8, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5VisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t baseA = 6'000'000'000ULL;
    const uint64_t baseB = baseA + frame_len_ns * file_nt; // next file window

    N2::frameID fid(&buf);
    // Produce only t=0 for file window A (all freqs)
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, 0, baseA, frame_len_ticks);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // Small pause to allow stage to open/initialize dataset and update last activity
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // Produce t=0 for file window B to trigger timeout scan and finalize A
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, 0, baseB, frame_len_ticks);
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


    std::string format, suffix;
    select_format_and_suffix(format, suffix);

    const std::string unique_name = "/hdf5_vis_writer_drop";
    const std::string in_buf_name = "n2buf_drop";
    const std::string base_dir = "test_hdf5VisWrite_drop";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 2;
    const uint64_t file_seconds = 1;
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 file_seconds, format, /*bs_f (0=all)*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ 1'000'000'000ULL);
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_drop", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5VisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t base_time_ns = 8'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t file_start_time_ns = base_time_ns; // aligned to 1-second file window

    // Pre-create a final dataset path to force drop
    const std::string ds_final =
        compute_dataset_name(base_dir, file_name, false, file_start_time_ns, suffix);
    // Create as empty directory
    ::mkdir(base_dir.c_str(), 0777);
    ::mkdir(ds_final.c_str(), 0777);

    N2::frameID fid(&buf);
    // Attempt to produce file window A (should be dropped)
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, 0, base_time_ns, frame_len_ticks);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // Produce next file window (should write)
    const uint64_t next_time = base_time_ns + frame_len_ns;
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, 0, next_time, frame_len_ticks);
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

// Test 7: Geometry mismatch within dataset (nfreq) is dropped without crash
BOOST_AUTO_TEST_CASE(test_writer_geometry_mismatch_dropped) {


    std::string format, suffix;
    select_format_and_suffix(format, suffix);

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
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 file_seconds, format, /*bs_f (0=all)*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ 1'000'000'000ULL);
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(4, sizeof(N2Metadata), "pool_geom", "N2Metadata");
    Buffer buf(4, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    hdf5VisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t base_time_ns = 9'000'000'000ULL;

    N2::frameID fid(&buf);
    // Produce t=0 for f=0 with normal geometry
    for (size_t f = 0; f < nfreq; ++f) {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, 0, base_time_ns, frame_len_ticks);
        buf.mark_frame_full("test-producer", fid);
        fid++;
    }
    // Send one mismatching frame for same file window with nfreq+1 (should be dropped)
    {
        uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
        BOOST_REQUIRE(frame != nullptr);
        fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, 0, 1, base_time_ns + frame_len_ns,
                      frame_len_ticks);
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
        auto ds_vis = f.getDataSet("/vis_array");
        BOOST_REQUIRE(ds_vis.getElementCount() > 0);
    }
    rm_tree_if_exists(ds_path);
    rm_tree_if_exists(base_dir);
}
