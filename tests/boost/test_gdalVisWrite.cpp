// Boost tests for the gdalVisWrite stage end-to-end, writing GDAL Zarr files

#define BOOST_TEST_MODULE "test_gdalVisWrite"

#include "test_utils.hpp"

#include "Config.hpp"              // for Config
#include "Stage.hpp"               // for Stage
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "gdalVisWrite.hpp"        // for gdalVisWrite
#include "N2FrameView.hpp"         // for N2FrameView
#include "N2Metadata.hpp"          // for N2Metadata, get_N2_metadata
#include "N2Util.hpp"              // for N2 helpers
#include "Telescope.hpp"           // for Telescope

#include <boost/test/included/unit_test.hpp>

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_vsi.h>

#include <dirent.h>     // for opendir, readdir
#include <sys/stat.h>   // for stat
#include <unistd.h>     // for gethostname

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using std::string;

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

static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty())
        return b;
    if (a.back() == '/')
        return a + b;
    return a + "/" + b;
}

static void rm_tree_if_exists(const std::string& path) {
    // Use GDAL VSI to remove trees/files; ignore errors.
    VSIUnlinkTree(path.c_str());
}

// Helper to create config and initialize telescope singleton
static kotekan::Config make_base_config(const std::string& unique_name, const std::string& in_buf,
                                        const std::string& base_dir, const std::string& file_name,
                                        bool prefix_hostname, uint64_t zip_compression,
                                        uint32_t file_nt, uint64_t blocksize_f = 0,
                                        uint64_t blocksize_t = 1) {
    using json = nlohmann::json;

    json j;
    // Minimal stage config
    j[unique_name] = nlohmann::json::object();
    j[unique_name]["cpu_affinity"] = std::vector<int>{0};
    j[unique_name]["log_level"] = "DEBUG";
    j[unique_name]["in_buf"] = in_buf;
    j[unique_name]["base_dir"] = base_dir;
    j[unique_name]["file_name"] = file_name;
    j[unique_name]["prefix_hostname"] = prefix_hostname;
    j[unique_name]["zip_compression"] = zip_compression;
    j[unique_name]["blocksize_f"] = blocksize_f;
    j[unique_name]["blocksize_p"] = 0; // unused currently
    j[unique_name]["blocksize_t"] = blocksize_t;
    j[unique_name]["file_nt"] = file_nt;
    j[unique_name]["join_timeout"] = 10; // seconds

    // Telescope config (use ICETelescope defaults, but set explicitly for determinism)
    j["/telescope"] = nlohmann::json::object();
    j["/telescope"]["name"] = "ICETelescope";
    j["/telescope"]["sampling_rate"] = 800.0; // MHz
    j["/telescope"]["fft_length"] = 2048;
    j["/telescope"]["nyquist_zone"] = 2;
    j["/telescope"]["require_gps"] = false;

    kotekan::Config conf;
    conf.update_config(j);

    // Initialize Telescope singleton before any stage uses it
    Telescope::instance(conf);

    return conf;
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
    while (!buf->is_frame_empty(frame_id)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        auto dt = std::chrono::duration<double>(clock::now() - t0).count();
        if (dt > timeout_seconds) {
            BOOST_FAIL("Timed out waiting for buffer frame to become empty");
            break;
        }
    }
}

// Open a GDAL dataset from path (Zarr dir or zip file)
static GDALDataset* open_dataset(const std::string& path) {
    char** oo = nullptr;
    GDALDataset* ds = static_cast<GDALDataset*>(
        GDALOpenEx(path.c_str(), GDAL_OF_MULTIDIM_RASTER | GDAL_OF_READONLY, nullptr,
                   const_cast<const char**>(oo), nullptr));
    return ds;
}

// Read back and validate a few arrays using the known patterns
static void validate_dataset_content(GDALDataset* ds, size_t num_input, size_t num_ev,
                                     size_t nfreq, size_t file_nt) {
    BOOST_REQUIRE(ds != nullptr);
    auto root = ds->GetRootGroup();
    BOOST_REQUIRE(root != nullptr);

    const auto c32Type = GDALExtendedDataType::Create(GDT_CFloat32);
    const auto f32Type = GDALExtendedDataType::Create(GDT_Float32);
    const auto f64Type = GDALExtendedDataType::Create(GDT_Float64);
    const auto u64Type = GDALExtendedDataType::Create(GDT_UInt64);

    // Check one representative frequency (e.g., f=1) across time
    size_t f = std::min<size_t>(1, nfreq - 1);

    // vis + weights
    {
        auto vis = root->OpenMDArray("vis_array");
        auto w = root->OpenMDArray("weights_array");
        BOOST_REQUIRE(vis && w);
        auto dims = vis->GetDimensionCount();
        BOOST_CHECK_EQUAL(dims, 3U);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0};
        std::vector<size_t> count{1, N2::get_num_prod(num_input), file_nt};
        std::vector<N2::cfloat> vis_out(count[1] * count[2]);
        std::vector<float> w_out(count[1] * count[2]);
        bool ok = vis->Read(start.data(), count.data(), nullptr, nullptr, c32Type,
                            reinterpret_cast<void*>(vis_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = w->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                     reinterpret_cast<void*>(w_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t t = 0; t < file_nt; ++t) {
            for (size_t p = 0; p < count[1]; ++p) {
                const size_t idx = file_nt * p + t;
                float base = 1000.0f * float(t) + 100.0f * float(f) + 10.0f * float(p);
                BOOST_CHECK(vis_out[idx] == N2::cfloat(base + 1.0f, base + 2.0f));
                BOOST_CHECK_CLOSE_FRACTION(w_out[idx], 1000.0f + float(p), 1e-6f);
            }
        }
    }

    // eval
    {
        auto arr = root->OpenMDArray("eval_array");
        BOOST_REQUIRE(arr);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0};
        std::vector<size_t> count{1, num_ev, file_nt};
        std::vector<float> eval_out(num_ev * file_nt);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                            reinterpret_cast<void*>(eval_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t e = 0; e < num_ev; ++e) {
            for (size_t t = 0; t < file_nt; ++t) {
                BOOST_CHECK_CLOSE_FRACTION(eval_out[file_nt * e + t], 60.0f + float(e), 1e-6f);
            }
        }
    }

    // evec slice at i=0 spot-check
    {
        auto arr = root->OpenMDArray("evec_array");
        BOOST_REQUIRE(arr);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0, 0};
        std::vector<size_t> count{1, num_ev, num_input, file_nt};
        std::vector<N2::cfloat> out(num_ev * num_input * file_nt);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, c32Type,
                            reinterpret_cast<void*>(out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t e = 0; e < num_ev; ++e) {
            N2::cfloat expected0(100.0f * float(e) + 0.5f, -(100.0f * float(e) + 1.5f));
            for (size_t t = 0; t < file_nt; ++t) {
                const size_t idx = file_nt * (num_input * e + 0) + t;
                BOOST_CHECK(out[idx] == expected0);
            }
        }
    }

    // erms
    {
        auto arr = root->OpenMDArray("erms_array");
        BOOST_REQUIRE(arr);
        std::vector<GUInt64> start{(GUInt64)f, 0};
        std::vector<size_t> count{1, file_nt};
        std::vector<float> out(file_nt);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                            reinterpret_cast<void*>(out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t t = 0; t < file_nt; ++t)
            BOOST_CHECK_CLOSE_FRACTION(out[t], 3.14f, 1e-6f);
    }

    // gain + flags spot-check i=0
    {
        auto gain = root->OpenMDArray("gain_array");
        auto flags = root->OpenMDArray("flags_array");
        BOOST_REQUIRE(gain && flags);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0};
        std::vector<size_t> count{1, num_input, file_nt};
        std::vector<N2::cfloat> gout(num_input * file_nt);
        std::vector<float> fout(num_input * file_nt);
        bool ok = gain->Read(start.data(), count.data(), nullptr, nullptr, c32Type,
                             reinterpret_cast<void*>(gout.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = flags->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                         reinterpret_cast<void*>(fout.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t t = 0; t < file_nt; ++t) {
            BOOST_CHECK(gout[file_nt * 0 + t] == N2::cfloat(200.0f, -200.0f));
            BOOST_CHECK_CLOSE_FRACTION(fout[file_nt * 0 + t], 300.0f, 1e-6f);
        }
    }

    // per-(freq,time) derived fractions and counts spot-check
    {
        auto fl = root->OpenMDArray("frac_lost_array");
        auto fr = root->OpenMDArray("frac_rfi_array");
        auto nv = root->OpenMDArray("n_valid_fpga_ticks");
        auto nr = root->OpenMDArray("n_rfi_fpga_ticks");
        BOOST_REQUIRE(fl && fr && nv && nr);
        std::vector<GUInt64> start{(GUInt64)f, 0};
        std::vector<size_t> count{1, file_nt};
        std::vector<float> fl_out(file_nt), fr_out(file_nt);
        std::vector<uint64_t> nv_out(file_nt), nr_out(file_nt);
        bool ok = fl->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                           reinterpret_cast<void*>(fl_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = fr->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                      reinterpret_cast<void*>(fr_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = nv->Read(start.data(), count.data(), nullptr, nullptr, u64Type,
                      reinterpret_cast<void*>(nv_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = nr->Read(start.data(), count.data(), nullptr, nullptr, u64Type,
                      reinterpret_cast<void*>(nr_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t t = 0; t < file_nt; ++t) {
            BOOST_CHECK_EQUAL(nv_out[t], uint64_t(80));
            BOOST_CHECK_EQUAL(nr_out[t], uint64_t(5));
            BOOST_CHECK_CLOSE_FRACTION(fl_out[t], 1.0f - 80.0f / 100.0f, 1e-6f);
            BOOST_CHECK_CLOSE_FRACTION(fr_out[t], 5.0f / 100.0f, 1e-6f);
        }
    }

    // per-time arrays shape exists
    {
        auto a0 = root->OpenMDArray("fpga_start_tick");
        auto a1 = root->OpenMDArray("frame_start_time_ns");
        auto a2 = root->OpenMDArray("frame_length_fpga_ticks");
        auto a3 = root->OpenMDArray("era_deg");
        BOOST_REQUIRE(a0 && a1 && a2 && a3);
        std::vector<GUInt64> start{0};
        std::vector<size_t> count{file_nt};
        std::vector<uint64_t> s0(file_nt), s1(file_nt), s2(file_nt);
        std::vector<double> s3(file_nt);
        bool ok = a0->Read(start.data(), count.data(), nullptr, nullptr, u64Type,
                           reinterpret_cast<void*>(s0.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = a1->Read(start.data(), count.data(), nullptr, nullptr, u64Type,
                      reinterpret_cast<void*>(s1.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = a2->Read(start.data(), count.data(), nullptr, nullptr, u64Type,
                      reinterpret_cast<void*>(s2.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = a3->Read(start.data(), count.data(), nullptr, nullptr, f64Type,
                      reinterpret_cast<void*>(s3.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        // Only check that time arrays are non-zero at least for t=0 and t=1
        BOOST_CHECK(s0[0] != 0);
        BOOST_CHECK(s2[0] != 0);
    }
}

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

// Test 1: Full-block flush with transpose validation
BOOST_AUTO_TEST_CASE(test_writer_full_block_transpose) {
    GDALAllRegister();

    const std::string unique_name = "/gdal_vis_writer";
    const std::string in_buf_name = "n2buf";
    const std::string base_dir = "test_gdalVisWrite_full";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    // Dims
    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const size_t file_nt = 2;
    const size_t num_prod = N2::get_num_prod(num_input);

    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name,
                                 /*prefix_hostname*/ false, /*zip*/ 0, file_nt);

    // Buffer + container
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_full", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*zero_new_frames*/ true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    // Create and start stage
    gdalVisWrite stage(conf, unique_name, bc);
    stage.start();

    // Time logic
    const uint64_t dt_ns = Telescope::instance().seq_length_nsec();
    const uint64_t frame_len_ticks = 1; // keep frame_len_ns == dt_ns
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t file_len_ns = frame_len_ns * file_nt;
    const uint64_t base_time_ns = 1'000'000'000ULL; // 1970-01-01 00:00:01

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
    wait_until_frame_empty(&buf, fid - 1);

    // Graceful shutdown
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    // Find exactly one dataset in base_dir
    auto entries = list_dir_entries(base_dir);
    BOOST_REQUIRE_MESSAGE(entries.size() == 1, "Expected 1 dataset, found " << entries.size());
    const std::string ds_path = join_path(base_dir, entries[0]);

    GDALDataset* ds = open_dataset(ds_path);
    BOOST_REQUIRE_MESSAGE(ds != nullptr, "Failed to open dataset: " << ds_path);
    validate_dataset_content(ds, num_input, num_ev, nfreq, file_nt);
    GDALClose(ds);

    // Cleanup
    rm_tree_if_exists(ds_path);
    VSIUnlink(base_dir.c_str());
}

// Test 2: Partial flush triggered on exit (incomplete time block)
BOOST_AUTO_TEST_CASE(test_writer_partial_flush_on_exit) {
    GDALAllRegister();

    const std::string unique_name = "/gdal_vis_writer_partial";
    const std::string in_buf_name = "n2buf_partial";
    const std::string base_dir = "test_gdalVisWrite_partial";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    // Dims
    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const size_t file_nt = 2;

    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name,
                                 /*prefix_hostname*/ false, /*zip*/ 0, file_nt);

    // Buffer + container
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_partial", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*zero_new_frames*/ true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    gdalVisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = Telescope::instance().seq_length_nsec();
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
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
    wait_until_frame_empty(&buf, fid - 1);

    // Trigger shutdown to force partial flush path
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    auto entries = list_dir_entries(base_dir);
    BOOST_REQUIRE_MESSAGE(entries.size() == 1, "Expected 1 dataset, found " << entries.size());
    const std::string ds_path = join_path(base_dir, entries[0]);

    GDALDataset* ds = open_dataset(ds_path);
    BOOST_REQUIRE(ds != nullptr);
    // Spot-check: for f=1, vis at t=0 non-zero, at t=1 default zero
    auto root = ds->GetRootGroup();
    auto arr = root->OpenMDArray("vis_array");
    auto warr = root->OpenMDArray("weights_array");
    BOOST_REQUIRE(arr && warr);
    const size_t num_prod = N2::get_num_prod(num_input);
    const auto c32Type = GDALExtendedDataType::Create(GDT_CFloat32);
    const auto f32Type = GDALExtendedDataType::Create(GDT_Float32);
    std::vector<GUInt64> start{1, 0, 0};
    std::vector<size_t> count{1, num_prod, file_nt};
    std::vector<N2::cfloat> vis_out(num_prod * file_nt);
    std::vector<float> w_out(num_prod * file_nt);
    bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, c32Type,
                        reinterpret_cast<void*>(vis_out.data()), nullptr, 0);
    BOOST_REQUIRE(ok);
    ok = warr->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                    reinterpret_cast<void*>(w_out.data()), nullptr, 0);
    BOOST_REQUIRE(ok);
    for (size_t p = 0; p < num_prod; ++p) {
        float base = 100.0f + 10.0f * float(p);
        BOOST_CHECK(vis_out[file_nt * p + 0] == N2::cfloat(base + 1.0f, base + 2.0f));
        BOOST_CHECK_CLOSE_FRACTION(w_out[file_nt * p + 0], 1000.0f + float(p), 1e-6f);
        BOOST_CHECK(vis_out[file_nt * p + 1] == N2::cfloat(0.0f, 0.0f));
        BOOST_CHECK_CLOSE_FRACTION(w_out[file_nt * p + 1], 0.0f, 1e-6f);
    }
    GDALClose(ds);

    // Cleanup
    rm_tree_if_exists(ds_path);
    VSIUnlink(base_dir.c_str());
}

// Test 3: Multi-file rollover when time crosses a file window
BOOST_AUTO_TEST_CASE(test_writer_multi_file_rollover) {
    GDALAllRegister();

    const std::string unique_name = "/gdal_vis_writer_rollover";
    const std::string in_buf_name = "n2buf_rollover";
    const std::string base_dir = "test_gdalVisWrite_rollover";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const size_t file_nt = 2;

    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0, file_nt);

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_roll", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", 0, false, false, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    gdalVisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = Telescope::instance().seq_length_nsec();
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t file_len_ns = frame_len_ns * file_nt;
    const uint64_t baseA = 3'000'000'000ULL;
    const uint64_t baseB = baseA + file_len_ns;

    N2::frameID fid(&buf);
    // Window A (t=0..1)
    for (size_t t = 0; t < file_nt; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, t, baseA + t * frame_len_ns,
                          frame_len_ticks);
            buf.mark_frame_full("test-producer", fid);
            fid++;
        }
    // Window B (t=0..1)
    for (size_t t = 0; t < file_nt; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, t, baseB + t * frame_len_ns,
                          frame_len_ticks);
            buf.mark_frame_full("test-producer", fid);
            fid++;
        }

    wait_until_frame_empty(&buf, fid - 1);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    auto entries = list_dir_entries(base_dir);
    BOOST_REQUIRE_MESSAGE(entries.size() == 2, "Expected 2 datasets, found " << entries.size());
    // Validate each one opens and contents make sense
    for (const auto& e : entries) {
        const std::string p = join_path(base_dir, e);
        GDALDataset* ds = open_dataset(p);
        BOOST_REQUIRE(ds != nullptr);
        validate_dataset_content(ds, num_input, num_ev, nfreq, file_nt);
        GDALClose(ds);
        rm_tree_if_exists(p);
    }
    VSIUnlink(base_dir.c_str());
}

// Test 4: ZIP storage and hostname prefix in filename
BOOST_AUTO_TEST_CASE(test_writer_zip_and_prefix) {
    GDALAllRegister();

    const std::string unique_name = "/gdal_vis_writer_zip";
    const std::string in_buf_name = "n2buf_zip";
    const std::string base_dir = "test_gdalVisWrite_zip";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 2; // keep it small
    const size_t file_nt = 2;

    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, /*prefix*/ true,
                                 /*zip*/ 1, file_nt, /*blocksize_f*/ 1, /*blocksize_t*/ 2);

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_zip", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", 0, false, false, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    gdalVisWrite stage(conf, unique_name, bc);
    stage.start();

    const uint64_t dt_ns = Telescope::instance().seq_length_nsec();
    const uint64_t frame_len_ticks = 1;
    const uint64_t frame_len_ns = frame_len_ticks * dt_ns;
    const uint64_t base_time_ns = 4'000'000'000ULL;

    N2::frameID fid(&buf);
    for (size_t t = 0; t < file_nt; ++t)
        for (size_t f = 0; f < nfreq; ++f) {
            uint8_t* frame = buf.wait_for_empty_frame("test-producer", fid);
            BOOST_REQUIRE(frame != nullptr);
            fill_n2_frame(&buf, fid, num_input, num_ev, nfreq, f, t, base_time_ns + t * frame_len_ns,
                          frame_len_ticks);
            buf.mark_frame_full("test-producer", fid);
            fid++;
        }

    wait_until_frame_empty(&buf, fid - 1);
    stage.stop();
    buf.send_shutdown_signal();
    stage.join();

    auto entries = list_dir_entries(base_dir);
    BOOST_REQUIRE_MESSAGE(entries.size() == 1, "Expected 1 dataset, found " << entries.size());
    const std::string ds_name = entries[0];
    const std::string ds_path = join_path(base_dir, ds_name);

    // Should be a .zarr.zip and prefixed with hostname
    BOOST_CHECK(ds_name.find(".zarr.zip") != std::string::npos);
    char hostname[256] = {0};
    gethostname(hostname, sizeof hostname);
    std::string expected_prefix = std::string(hostname) + "_" + file_name + ".";
    BOOST_CHECK_EQUAL(ds_name.rfind(expected_prefix, 0), 0U);

    GDALDataset* ds = open_dataset(ds_path);
    BOOST_REQUIRE(ds != nullptr);
    validate_dataset_content(ds, num_input, num_ev, nfreq, file_nt);
    GDALClose(ds);

    rm_tree_if_exists(ds_path);
    VSIUnlink(base_dir.c_str());
}

