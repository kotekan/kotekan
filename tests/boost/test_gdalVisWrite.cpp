// Boost tests for the gdalVisWrite stage end-to-end, writing GDAL Zarr files

#define BOOST_TEST_MODULE "test_gdalVisWrite"

#include "Config.hpp"          // for Config
#include "N2FrameView.hpp"     // for N2FrameView
#include "N2Metadata.hpp"      // for N2Metadata, get_N2_metadata
#include "N2Util.hpp"          // for N2 helpers
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "gdalVisWrite.hpp"    // for gdalVisWrite
#include "test_utils.hpp"

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cpl_conv.h>
#include <cpl_vsi.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h> // for opendir, readdir
#include <gdal.h>
#include <gdal_priv.h>
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

// Force registration of N2Metadata in metadata factory by referencing the type
static N2Metadata _force_n2meta_registration;

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

// Remove (recursively) directory if it exists
static void rm_tree_if_exists(const std::string& path) {
    if (!path_exists(path))
        return;
    // Use GDAL CPL to remove; ignore errors from unlink itself.
    CPLUnlinkTree(path.c_str());
}

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
    // Unified GDAL writer format selection ("zarr" or "hdf5").
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

// Determine a working GDAL multidimensional format and expected filename suffix.
static bool select_format_and_suffix(std::string& out_format, std::string& out_suffix) {
    GDALAllRegister();
    // Route logs to stderr for visibility during tests
    extern int __enable_syslog;
    __enable_syslog = 0;
    auto* mgr = GetGDALDriverManager();
    if (mgr && mgr->GetDriverByName("Zarr")) {
        out_format = "zarr";
        out_suffix = ".zarr";
        return true;
    }
    if (mgr && mgr->GetDriverByName("HDF5")) {
        out_format = "hdf5";
        out_suffix = ".h5";
        return true;
    }
    return false;
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
            int nfull = buf->get_num_full_frames();
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

// Open a GDAL dataset from path
static GDALDataset* open_dataset(const std::string& path) {
    char** oo = nullptr;
    GDALDataset* ds = static_cast<GDALDataset*>(
        GDALOpenEx(path.c_str(), GDAL_OF_MULTIDIM_RASTER | GDAL_OF_READONLY, nullptr,
                   const_cast<const char**>(oo), nullptr));
    return ds;
}

// Read back and validate a few arrays using the known patterns
static void validate_dataset_content(GDALDataset* ds, size_t num_input, size_t num_ev, size_t nfreq,
                                     size_t file_nt) {
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

    std::string format, suffix;
    select_format_and_suffix(format, suffix);
    BOOST_TEST_MESSAGE("Using format='" << format << "' suffix='" << suffix << "'");

    const std::string unique_name = "/gdal_vis_writer";
    const std::string in_buf_name = "n2buf";
    const std::string base_dir = "test_gdalVisWrite_full";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    // Dims
    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    // Use 100-second frames and 200-second file windows to get file_nt=2
    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 100; // ensure fractions compute as 80/100
    const size_t file_nt = 2;
    const uint64_t file_seconds = 200;

    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name,
                                 /*prefix_hostname*/ false, /*zip*/ 0, file_seconds, format,
                                 /*blocksize_f*/ 0, /*blocksize_t*/ 1, /*grace*/ 60,
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
    gdalVisWrite stage(conf, unique_name, bc);
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

    GDALDataset* ds = open_dataset(ds_path);
    BOOST_REQUIRE_MESSAGE(ds != nullptr, "Failed to open dataset: " << ds_path);
    validate_dataset_content(ds, num_input, num_ev, nfreq, file_nt);
    GDALClose(ds);

    // Cleanup
    rm_tree_if_exists(ds_path);
    rm_tree_if_exists(base_dir);
}

// Test 2: Partial flush triggered on exit (incomplete time block)
BOOST_AUTO_TEST_CASE(test_writer_partial_flush_on_exit) {
    GDALAllRegister();

    std::string format, suffix;
    select_format_and_suffix(format, suffix);

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
    // Use a 2-second file window so file_nt=2 with 1s frames
    const uint64_t file_seconds = 2;

    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name,
                                 /*prefix_hostname*/ false, /*zip*/ 0, file_seconds, format,
                                 /*blocksize_f*/ 0, /*blocksize_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ 1'000'000'000ULL);

    // Buffer + container
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_partial", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", /*numa*/ 0, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    gdalVisWrite stage(conf, unique_name, bc);
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
    rm_tree_if_exists(base_dir);
}

// Test 3: Multi-file rollover when time crosses a file window
BOOST_AUTO_TEST_CASE(test_writer_multi_file_rollover) {
    GDALAllRegister();

    std::string format, suffix;
    select_format_and_suffix(format, suffix);

    const std::string unique_name = "/gdal_vis_writer_rollover";
    const std::string in_buf_name = "n2buf_rollover";
    const std::string base_dir = "test_gdalVisWrite_rollover";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const size_t file_nt = 2;
    const uint64_t file_seconds = 200;
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 file_seconds, format, /*bs_f*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ 1'000'000'000ULL);

    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_roll", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    gdalVisWrite stage(conf, unique_name, bc);
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
        GDALDataset* ds = open_dataset(p);
        BOOST_REQUIRE(ds != nullptr);
        validate_dataset_content(ds, num_input, num_ev, nfreq, file_nt);
        GDALClose(ds);
        rm_tree_if_exists(p);
    }
    rm_tree_if_exists(base_dir);
}

// Test 4: Adjacent file windows produce distinct dataset names
BOOST_AUTO_TEST_CASE(test_writer_distinct_window_names) {
    GDALAllRegister();

    std::string format, suffix;
    select_format_and_suffix(format, suffix);

    const std::string unique_name = "/gdal_vis_writer_names";
    const std::string in_buf_name = "n2buf_names";
    const std::string base_dir = "test_gdalVisWrite_names";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const uint64_t dt_ns = 1'000'000'000ULL;
    const uint64_t frame_len_ticks = 1;
    const uint64_t file_seconds = 1; // one second file window => one frame per file
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 file_seconds, format, /*bs_f*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ dt_ns);
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_subsec", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    gdalVisWrite stage(conf, unique_name, bc);
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
    GDALAllRegister();

    std::string format, suffix;
    select_format_and_suffix(format, suffix);

    const std::string unique_name = "/gdal_vis_writer_timeout";
    const std::string in_buf_name = "n2buf_timeout";
    const std::string base_dir = "test_gdalVisWrite_timeout";
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

    gdalVisWrite stage(conf, unique_name, bc);
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
    GDALAllRegister();

    std::string format, suffix;
    select_format_and_suffix(format, suffix);

    const std::string unique_name = "/gdal_vis_writer_drop";
    const std::string in_buf_name = "n2buf_drop";
    const std::string base_dir = "test_gdalVisWrite_drop";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 2;
    const uint64_t file_seconds = 1;
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 file_seconds, format, /*bs_f*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ 1'000'000'000ULL);
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(2, sizeof(N2Metadata), "pool_drop", "N2Metadata");
    Buffer buf(2, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    gdalVisWrite stage(conf, unique_name, bc);
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
    GDALAllRegister();

    std::string format, suffix;
    select_format_and_suffix(format, suffix);

    const std::string unique_name = "/gdal_vis_writer_geom";
    const std::string in_buf_name = "n2buf_geom";
    const std::string base_dir = "test_gdalVisWrite_geom";
    const std::string file_name = "vis";
    rm_tree_if_exists(base_dir);

    const size_t num_input = 3;
    const size_t num_ev = 2;
    const size_t nfreq = 3;
    const size_t file_nt = 2;

    const uint64_t file_seconds = file_nt; // with 1s frames, file_nt==file_seconds
    auto conf = make_base_config(unique_name, in_buf_name, base_dir, file_name, false, 0,
                                 file_seconds, format, /*bs_f*/ 0, /*bs_t*/ 1, /*grace*/ 60,
                                 /*seq_override*/ 1'000'000'000ULL);
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(4, sizeof(N2Metadata), "pool_geom", "N2Metadata");
    Buffer buf(4, frame_size, pool, in_buf_name, "vis", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test-producer");
    kotekan::bufferContainer bc;
    bc.add_buffer(in_buf_name, &buf);

    gdalVisWrite stage(conf, unique_name, bc);
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
    GDALDataset* ds = open_dataset(ds_path);
    BOOST_REQUIRE(ds != nullptr);
    // Quick sanity: check arrays open
    auto root = ds->GetRootGroup();
    BOOST_REQUIRE(root != nullptr);
    BOOST_REQUIRE(root->OpenMDArray("vis_array") != nullptr);
    GDALClose(ds);
    rm_tree_if_exists(ds_path);
    rm_tree_if_exists(base_dir);
}
