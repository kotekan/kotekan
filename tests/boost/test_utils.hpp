#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include "Config.hpp"
#include "N2FrameDesc.hpp"
#include "N2FrameView.hpp"
#include "N2Metadata.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "visUtil.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <locale>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Use a boost::test "Global Fixture" to set the locale...
// https://www.boost.org/doc/libs/1_75_0/libs/test/doc/html/boost_test/tests_organization/fixtures/global.html
// Enable this in your test suite by adding:
// BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);
struct GlobalFixture_Locale {
    GlobalFixture_Locale() {
        std::cout << "Setting locale (stdout)..." << std::endl;
        BOOST_TEST_MESSAGE("Setting locale...");
        try {
            std::locale::global(std::locale::classic());
        } catch (const std::exception& ex) {
            std::cerr << "Exception setting locale (stdout)..." << ex.what() << std::endl;
            BOOST_TEST_MESSAGE("Exception setting locale");
        }
    }
    ~GlobalFixture_Locale() {}
};

struct CompareCTypes {
    void check_equal(const std::vector<input_ctype>& a, const std::vector<input_ctype>& b) {
        BOOST_CHECK(!a.empty() && !b.empty());
        BOOST_CHECK(a.size() == b.size());
        auto ita = a.begin();
        auto itb = b.begin();

        while (ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->chan_id, itb->chan_id);
            BOOST_CHECK_EQUAL(ita->correlator_input, itb->correlator_input);
            if (ita != a.end())
                ++ita;
            if (itb != b.end())
                ++itb;
        }
    }

    void check_equal(const std::vector<prod_ctype>& a, const std::vector<prod_ctype>& b) {
        BOOST_CHECK(!a.empty() && !b.empty());
        BOOST_CHECK(a.size() == b.size());
        auto ita = a.begin();
        auto itb = b.begin();

        while (ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->input_a, itb->input_a);
            BOOST_CHECK_EQUAL(ita->input_b, itb->input_b);
            if (ita != a.end())
                ++ita;
            if (itb != b.end())
                ++itb;
        }
    }

    void check_equal(const std::vector<std::pair<uint32_t, freq_ctype>>& a,
                     const std::vector<std::pair<uint32_t, freq_ctype>>& b) {
        BOOST_CHECK(!a.empty() && !b.empty());
        BOOST_CHECK(a.size() == b.size());
        auto ita = a.begin();
        auto itb = b.begin();

        while (ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->first, itb->first);
            BOOST_CHECK_EQUAL(ita->second.centre, itb->second.centre);
            BOOST_CHECK_EQUAL(ita->second.width, itb->second.width);
            if (ita != a.end())
                ++ita;
            if (itb != b.end())
                ++itb;
        }
    }
};

// Ensure a directory exists (creates it if needed)
[[maybe_unused]] void ensure_directory(const std::string& path) {
    if (path.empty())
        return;
    if (::mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
        BOOST_FAIL("Failed to create directory " << path << " errno=" << errno);
    }
}

// Set the file_num_t parameter for a stage in the config
[[maybe_unused]] static void set_file_num_t(kotekan::Config& conf, const std::string& unique_name,
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

// Set the log level for a stage in the config
[[maybe_unused]] static void set_stage_log_level(kotekan::Config& conf,
                                                 const std::string& unique_name,
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

// Wait for the writer's `acq_*` subdirectory to appear under base_dir.
// hdf5N2Write generates a fresh `base_dir/acq_<timestamp>` per stage and creates it
// asynchronously from main_thread, so callers that need the path before pushing
// frames must poll. Returns the joined path or empty string on timeout.
[[maybe_unused]] static std::string wait_for_acq_dir(const std::string& base_dir,
                                                     double timeout_s = 5.0) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        DIR* dir = opendir(base_dir.c_str());
        if (dir) {
            while (dirent* ent = readdir(dir)) {
                std::string name = ent->d_name;
                if (name.rfind("acq_", 0) == 0) {
                    closedir(dir);
                    return base_dir + (base_dir.empty() || base_dir.back() == '/' ? "" : "/")
                           + name;
                }
            }
            closedir(dir);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return {};
}

// List all `.h5` file paths under the writer's `base_dir/acq_*/` subdirectories.
[[maybe_unused]] static std::vector<std::string> list_h5_datasets(const std::string& base_dir) {
    std::vector<std::string> out;
    DIR* dir = opendir(base_dir.c_str());
    if (!dir)
        return out;
    while (dirent* ent = readdir(dir)) {
        std::string name = ent->d_name;
        if (name.rfind("acq_", 0) != 0)
            continue;
        const std::string acq =
            base_dir + (base_dir.empty() || base_dir.back() == '/' ? "" : "/") + name;
        DIR* sub = opendir(acq.c_str());
        if (!sub)
            continue;
        while (dirent* sent = readdir(sub)) {
            std::string sname = sent->d_name;
            if (sname.find(".h5") != std::string::npos)
                out.push_back(acq + "/" + sname);
        }
        closedir(sub);
    }
    closedir(dir);
    return out;
}

// Helper function to list items in a directory
[[maybe_unused]] static std::vector<std::string> list_dir_entries(const std::string& path) {
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
[[maybe_unused]] static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty())
        return b;
    if (a.back() == '/')
        return a + b;
    return a + "/" + b;
}

// Check if path exists
[[maybe_unused]] static bool path_exists(const std::string& path) {
    struct stat st;
    return (::stat(path.c_str(), &st) == 0);
}

// Remove (recursively) directory if it exists (POSIX implementation)
[[maybe_unused]] static void rm_tree_if_exists(const std::string& path) {
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

// Wait helper: spin until a given frame ID becomes empty (consumed by the stage)
[[maybe_unused]] static void wait_until_frame_empty(Buffer* buf, int frame_id,
                                                    double timeout_seconds = 5.0) {
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

// Add a minimal CHORD telescope configuration (dish geometry + EOP endpoint) to a JSON config.
[[maybe_unused]] static void add_test_telescope_config(nlohmann::json& cfg) {
    nlohmann::json telescope;
    telescope["name"] = "CHORDTelescope";
    telescope["log_level"] = "WARN";
    telescope["num_dishes"] = 2;
    telescope["dish_separation_x_m"] = 6.3;
    telescope["dish_separation_y_m"] = 8.5;
    telescope["dish_grid_size_x"] = 8;
    telescope["dish_grid_size_y"] = 12;
    telescope["dish_inputs"] = nlohmann::json::array({{{"dish_idx", 0},
                                                       {"grid_x_idx", 0},
                                                       {"grid_y_idx", 0},
                                                       {"feed_pos_disp_m", {0.0, 0.0, 0.0}},
                                                       {"coelev_disp_deg", 0.0},
                                                       {"type", "ArrayDish"},
                                                       {"label", "D00"}},
                                                      {{"dish_idx", 1},
                                                       {"grid_x_idx", 1},
                                                       {"grid_y_idx", 0},
                                                       {"feed_pos_disp_m", {0.0, 0.0, 0.0}},
                                                       {"coelev_disp_deg", 0.0},
                                                       {"type", "ArrayDish"},
                                                       {"label", "D01"}}});
    telescope["eop_updatable_config"] = "/earth_rotation_data";
    telescope["require_eop"] = false;
    telescope["grid_x_axis"] = {1.0, 0.0, 0.0};
    telescope["grid_y_axis"] = {0.0, 1.0, 0.0};
    telescope["dish_elev_axis"] = {1.0, 0.0, 0.0};
    telescope["dish_vert_axis"] = {0.0, 0.0, 1.0};
    telescope["dish_grid_size_x"] = 8;
    telescope["dish_grid_size_y"] = 8;

    nlohmann::json eop_update;
    eop_update["kotekan_update_endpoint"] = "json";
    eop_update["earth_orientation_parameter_table"] = nlohmann::json::array();

    auto set_path = [&](const std::string& key, const nlohmann::json& val) {
        cfg[key] = val;
        if (!key.empty() && key.front() == '/')
            cfg[key.substr(1)] = val;
    };

    set_path("/telescope", telescope);
    set_path("/earth_rotation_data", eop_update);
}

// Helper to create a writer config for testing
[[maybe_unused]] static kotekan::Config
make_writer_config(const std::string& unique_name, const std::string& in_buf,
                   const std::string& base_dir, const std::string& file_name, bool prefix_hostname,
                   uint64_t num_file_t, uint64_t blocksize_f = 0, uint64_t blocksize_p = 0,
                   uint64_t blocksize_t = 1, uint64_t late_frame_grace_seconds = 60,
                   uint64_t seq_length_nsec_override = 0,
                   const std::string& baseband_gain_file = "") {
    using json = nlohmann::json;

    json cfg;

    add_test_telescope_config(cfg);
    nlohmann::json stage;
    stage["cpu_affinity"] = std::vector<int>{0};
    stage["log_level"] = "WARN";
    stage["in_buf"] = in_buf;
    stage["base_dir"] = base_dir;
    stage["file_name"] = file_name;
    stage["prefix_hostname"] = prefix_hostname;
    stage["blocksize_f"] = blocksize_f;
    stage["blocksize_p"] = blocksize_p;
    stage["blocksize_t"] = blocksize_t;
    stage["num_file_t"] = num_file_t;
    stage["join_timeout"] = 5;
    stage["late_frame_grace_seconds"] = late_frame_grace_seconds;
    stage["baseband_gain_file"] = baseband_gain_file;
    if (seq_length_nsec_override > 0)
        stage["seq_length_nsec_override"] = seq_length_nsec_override;

    cfg[unique_name] = stage;
    if (!unique_name.empty() && unique_name.front() == '/')
        cfg[unique_name.substr(1)] = stage;

    kotekan::Config conf;
    conf.update_config(cfg);
    return conf;
}

// Fill one synthetic N2 frame with deterministic data for validators
[[maybe_unused]] static void fill_n2_frame(Buffer* buf, int frame_id, size_t num_input,
                                           size_t num_ev, size_t f_index, size_t t_index,
                                           uint64_t frame_start_time_ns,
                                           uint64_t frame_length_ticks) {
    buf->allocate_new_metadata_object(frame_id);
    auto meta = get_N2_metadata(buf, frame_id);
    BOOST_REQUIRE(meta);

    const size_t num_prod = kotekan::N2FrameDesc::get_num_prod(num_input, N2Layout::FullUpperTri);
    meta->freq_id = f_index;
    meta->fpga_start_tick = 100 + t_index;
    meta->frame_start_time_ns = frame_start_time_ns;
    meta->frame_length_fpga_ticks = frame_length_ticks;
    meta->n_valid_fpga_ticks = 80;
    meta->n_rfi_fpga_ticks = 5;
    meta->n_rfi_only_fpga_ticks = 4;
    meta->n_pl_fpga_ticks = frame_length_ticks - (80 + 4);
    meta->time_center_eop.ERA_deg = 9.87 + double(t_index);
    meta->bin_eop.ERA_deg = 9.87 + double(t_index);
    meta->bin_start_ERA_deg = 1.23 + double(t_index);
    meta->bin_end_ERA_deg = 4.56 + double(t_index);
    meta->bin_start_ERAL_deg = 1000000 + int64_t(t_index) * 1000;
    meta->bin_end_ERAL_deg = 2000000 + int64_t(t_index) * 1000;

    N2FrameView fv(buf, frame_id);
    fv.zero_frame();

    for (size_t p = 0; p < num_prod; ++p) {
        float base = 1000.0f * float(t_index) + 100.0f * float(f_index) + 10.0f * float(p);
        fv.vis[p] = N2::cfloat(base + 1.0f, base + 2.0f);
        fv.weight[p] = 1000.0f + float(p);
    }

    for (size_t e = 0; e < num_ev; ++e) {
        fv.eval[e] = 60.0f + float(e);
        for (size_t i = 0; i < num_input; ++i) {
            fv.evec[num_input * e + i] = N2::cfloat(100.0f * float(e) + float(i) + 0.5f,
                                                    -(100.0f * float(e) + float(i) + 1.5f));
        }
    }

    fv.erms = 3.14f;
    for (size_t i = 0; i < num_input; ++i) {
        fv.gain[i] = N2::cfloat(200.0f + float(i), -200.0f - float(i));
        fv.flags[i] = 300.0f + float(i);
    }
}

// Helper to fill an N2 frame and set abs_time_idx in metadata
[[maybe_unused]] static void fill_n2_frame_with_abs(Buffer* buf, int frame_id, size_t num_input,
                                                    size_t num_ev, size_t f_index, size_t t_index,
                                                    uint64_t frame_start_time_ns,
                                                    uint64_t frame_length_ticks,
                                                    uint64_t abs_time_idx) {
    fill_n2_frame(buf, frame_id, num_input, num_ev, f_index, t_index, frame_start_time_ns,
                  frame_length_ticks);
    auto meta = get_N2_metadata(buf, frame_id);
    BOOST_REQUIRE(meta);
    meta->abs_time_idx = abs_time_idx;
    meta->time_center_eop.t_ut1_ns = get_UT1_from_time_ns(frame_start_time_ns, 0.0);
    meta->bin_eop.t_ut1_ns = get_UT1_from_time_ns(frame_start_time_ns, 0.0);
}

#endif // TEST_UTILS_HPP
