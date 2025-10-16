// Boost test for gdalVisFileData add_frame() logic.

#define BOOST_TEST_MODULE "test_gdal_vis_file_data"

#include "N2FrameView.hpp"  // for N2FrameView
#include "N2Metadata.hpp"   // for N2Metadata, get_N2_metadata
#include "N2Util.hpp"       // for N2::cfloat, get_num_prod
#include "buffer.hpp"       // for Buffer
#include "gdalVisWrite.hpp" // for gdalVisFileData
#include "metadata.hpp"     // for metadataPool
#include "test_utils.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cpl_conv.h>
#include <cpl_vsi.h>
#include <cstdio>
#include <gdal.h>
#include <gdal_priv.h>
#include <string>
#include <vector>

// Helper subclass to expose protected members for inspection in tests.
class TestGdalVisFileData : public gdalVisFileData {
public:
    using gdalVisFileData::gdalVisFileData;
    using gdalVisFileData::idx_feit;
    using gdalVisFileData::idx_fet;
    using gdalVisFileData::idx_fit;
    using gdalVisFileData::idx_fpt;
    using gdalVisFileData::idx_ft;

    // Accessors for internal storage (copy out values)
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
    uint64_t get_n_valid(size_t f, size_t t) const {
        return n_valid_fpga_ticks[idx_ft(f, t)];
    }
    uint64_t get_n_rfi(size_t f, size_t t) const {
        return n_rfi_fpga_ticks[idx_ft(f, t)];
    }

    // Per-time arrays
    uint64_t get_fpga_start_tick(size_t t) const {
        return fpga_start_tick[t];
    }
    uint64_t get_frame_start_time_ns(size_t t) const {
        return frame_start_time_ns[t];
    }
    uint64_t get_frame_length_fpga_ticks(size_t) const {
        return frame_length_fpga_ticks;
    }
    double get_era_deg(size_t t) const {
        return era_deg[t];
    }

    // Added tracking (per (f,t))
    size_t get_added_count() const {
        return added_count;
    }
    uint8_t get_added(size_t f, size_t t) const {
        return added_ft[idx_ft(f, t)];
    }
};

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

BOOST_AUTO_TEST_CASE(test_add_frame_single_slot) {
    // Force-link N2Metadata registration by constructing once.
    N2Metadata force_link_marker;
    // Small synthetic dimensions
    const size_t num_input = 3; // elements
    const size_t num_prod = N2::get_num_prod(num_input);
    const size_t num_ev = 2;
    const size_t num_freq = 3;
    const size_t num_file_t = 2;

    // Build a buffer that can host one N2 frame
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(1, sizeof(N2Metadata), "test_pool", "N2Metadata");
    Buffer buf(1, frame_size, pool, "n2buf", "vis", /*numa*/ 1, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);

    // Allocate metadata and fill in structural + timing fields
    buf.allocate_new_metadata_object(0);
    auto meta = get_N2_metadata(&buf, 0);
    BOOST_REQUIRE(meta);
    meta->num_elements = num_input;
    meta->num_prod = num_prod;
    meta->num_ev = num_ev;
    meta->nfreq = num_freq;
    meta->freq_id = 1; // target frequency index

    meta->fpga_start_tick = 111;
    meta->frame_start_time_ns = 222;
    meta->frame_length_fpga_ticks = 100;
    meta->n_valid_fpga_ticks = 80;
    meta->n_rfi_fpga_ticks = 5;
    meta->eop.ERA_deg = 12.34;

    // Create a frame view and populate with synthetic data
    N2FrameView fv(&buf, 0);
    fv.zero_frame();

    for (size_t p = 0; p < num_prod; ++p) {
        fv.vis[p] = N2::cfloat(10.0f * p + 1.0f, 10.0f * p + 2.0f);
        fv.weight[p] = float(1000 + p);
    }
    for (size_t e = 0; e < num_ev; ++e) {
        fv.eval[e] = float(60 + e);
        for (size_t i = 0; i < num_input; ++i) {
            fv.evec[num_input * e + i] =
                N2::cfloat(100.0f * e + float(i) + 0.5f, -(100.0f * e + float(i) + 1.5f));
        }
    }
    fv.erms = 3.14f;
    for (size_t i = 0; i < num_input; ++i) {
        fv.gain[i] = N2::cfloat(200.0f + float(i), -200.0f - float(i));
        fv.flags[i] = float(300 + i);
    }

    // Buffer a file block and add the single (f,t) slot
    TestGdalVisFileData blk(num_file_t, num_freq, num_input, num_prod, num_ev,
                            /*frame_len_ns*/ 100, /*frame_length_fpga_ticks*/ 100,
                            /*file_start_time_ns*/ 0,
                            /*partial_path*/ std::string{}, /*open_wall_s*/ 0.0);
    const size_t f = meta->freq_id;
    const size_t t = 1; // arbitrary slot in-file
    blk.add_frame(fv, meta, t);

    // Only the targeted (f,t) should be set; others remain defaults
    // vis/weight
    BOOST_CHECK(blk.get_vis(f, 0, t) == N2::cfloat(1.0f, 2.0f));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_weight(f, 0, t), float(1000), 1e-6f);
    BOOST_CHECK(blk.get_vis(f, num_prod - 1, t)
                == N2::cfloat(10.0f * (num_prod - 1) + 1.0f, 10.0f * (num_prod - 1) + 2.0f));

    // A different time or frequency should be default (zeros)
    BOOST_CHECK(blk.get_vis(f, 0, 0) == N2::cfloat(0.0f, 0.0f));
    BOOST_CHECK(blk.get_vis((f + 1) % num_freq, 0, t) == N2::cfloat(0.0f, 0.0f));

    // eval/evec
    BOOST_CHECK_CLOSE_FRACTION(blk.get_eval(f, 0, t), float(60), 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(blk.get_eval(f, num_ev - 1, t), float(60 + (num_ev - 1)), 1e-6f);
    BOOST_CHECK(blk.get_evec(f, 0, 0, t) == N2::cfloat(0.5f, -1.5f));
    BOOST_CHECK(blk.get_evec(f, num_ev - 1, num_input - 1, t)
                == N2::cfloat(100.0f * (num_ev - 1) + float(num_input - 1) + 0.5f,
                              -(100.0f * (num_ev - 1) + float(num_input - 1) + 1.5f)));

    // erms, gain, flags
    BOOST_CHECK_CLOSE_FRACTION(blk.get_erms(f, t), 3.14f, 1e-6f);
    BOOST_CHECK(blk.get_gain(f, 0, t) == N2::cfloat(200.0f, -200.0f));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_flags(f, 0, t), 300.0f, 1e-6f);

    // Derived fractions and counts
    BOOST_CHECK_EQUAL(blk.get_n_valid(f, t), uint64_t(80));
    BOOST_CHECK_EQUAL(blk.get_n_rfi(f, t), uint64_t(5));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_frac_lost(f, t), 1.0f - 80.0f / 100.0f, 1e-6f);
    BOOST_CHECK_CLOSE_FRACTION(blk.get_frac_rfi(f, t), 5.0f / 100.0f, 1e-6f);

    // Per-time metadata
    BOOST_CHECK_EQUAL(blk.get_fpga_start_tick(t), uint64_t(111));
    BOOST_CHECK_EQUAL(blk.get_frame_start_time_ns(t), uint64_t(222));
    BOOST_CHECK_EQUAL(blk.get_frame_length_fpga_ticks(t), uint64_t(100));
    BOOST_CHECK_CLOSE_FRACTION(blk.get_era_deg(t), 12.34, 1e-12);

    // Added tracking
    BOOST_CHECK_EQUAL(blk.get_added_count(), size_t(1));
    BOOST_CHECK_EQUAL(blk.get_added(f, t), uint8_t(1));
    BOOST_CHECK_EQUAL(blk.get_added(f, 0), uint8_t(0));
    BOOST_CHECK_EQUAL(blk.get_added((f + 1) % num_freq, t), uint8_t(0));
}

// Helper to create a minimal GDAL Zarr dataset with dimensions/arrays
static GDALDataset* create_test_gdal_vis_dataset(const std::string& path, size_t num_freq,
                                                 size_t num_prod, size_t num_ev, size_t num_input,
                                                 size_t num_file_t) {
    GDALAllRegister();
    auto dm = GetGDALDriverManager();
    auto drv = dm->GetDriverByName("Zarr");
    BOOST_REQUIRE_MESSAGE(drv != nullptr, "Zarr GDAL driver not available");

    char** opts = nullptr;
    opts = CSLSetNameValue(opts, "FORMAT", "ZARR_V2");
    GDALDataset* ds =
        drv->CreateMultiDimensional(path.c_str(), nullptr, const_cast<const char**>(opts));
    CSLDestroy(opts);
    BOOST_REQUIRE(ds != nullptr);

    auto root = ds->GetRootGroup();
    BOOST_REQUIRE(root != nullptr);

    // Dimensions
    auto dim_freq = root->CreateDimension("freqs", "", "", num_freq);
    auto dim_prod_ = root->CreateDimension("products", "", "", num_prod);
    auto dim_frames = root->CreateDimension("frames", "", "", num_file_t);
    auto dim_inputs = root->CreateDimension("inputs", "", "", num_input);
    auto dim_ev_ = root->CreateDimension("ev", "", "", num_ev);

    // Arrays (names and shapes must match write_all_to_dataset)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_prod_, dim_frames};
        (void)root->CreateMDArray("vis_array", dims, GDALExtendedDataType::Create(GDT_CFloat32),
                                  nullptr);
        (void)root->CreateMDArray("weights_array", dims, GDALExtendedDataType::Create(GDT_Float32),
                                  nullptr);
    }
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_ev_, dim_frames};
        (void)root->CreateMDArray("eval_array", dims, GDALExtendedDataType::Create(GDT_Float32),
                                  nullptr);
    }
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_ev_, dim_inputs, dim_frames};
        (void)root->CreateMDArray("evec_array", dims, GDALExtendedDataType::Create(GDT_CFloat32),
                                  nullptr);
    }
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        (void)root->CreateMDArray("erms_array", dims, GDALExtendedDataType::Create(GDT_Float32),
                                  nullptr);
        (void)root->CreateMDArray("frac_lost_array", dims,
                                  GDALExtendedDataType::Create(GDT_Float32), nullptr);
        (void)root->CreateMDArray("frac_rfi_array", dims, GDALExtendedDataType::Create(GDT_Float32),
                                  nullptr);
        (void)root->CreateMDArray("n_valid_fpga_ticks", dims,
                                  GDALExtendedDataType::Create(GDT_UInt64), nullptr);
        (void)root->CreateMDArray("n_rfi_fpga_ticks", dims,
                                  GDALExtendedDataType::Create(GDT_UInt64), nullptr);
    }
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_inputs, dim_frames};
        (void)root->CreateMDArray("gain_array", dims, GDALExtendedDataType::Create(GDT_CFloat32),
                                  nullptr);
        (void)root->CreateMDArray("flags_array", dims, GDALExtendedDataType::Create(GDT_Float32),
                                  nullptr);
    }
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_frames};
        (void)root->CreateMDArray("fpga_start_tick", dims, GDALExtendedDataType::Create(GDT_UInt64),
                                  nullptr);
        (void)root->CreateMDArray("frame_start_time_ns", dims,
                                  GDALExtendedDataType::Create(GDT_UInt64), nullptr);
        (void)root->CreateMDArray("frame_length_fpga_ticks", dims,
                                  GDALExtendedDataType::Create(GDT_UInt64), nullptr);
        (void)root->CreateMDArray("era_deg", dims, GDALExtendedDataType::Create(GDT_Float64),
                                  nullptr);
    }

    return ds;
}

BOOST_AUTO_TEST_CASE(test_write_and_read_dataset) {
    // Force-link N2Metadata registration by constructing once.
    N2Metadata force_link_marker2;
    // Dimensions
    const size_t num_input = 3;
    const size_t num_prod = N2::get_num_prod(num_input);
    const size_t num_ev = 2;
    const size_t num_freq = 3;
    const size_t num_file_t = 2;

    // Build a buffer for one N2 frame
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(1, sizeof(N2Metadata), "test_pool2", "N2Metadata");
    Buffer buf(1, frame_size, pool, "n2buf2", "vis", /*numa*/ 1, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
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

    TestGdalVisFileData blk(num_file_t, num_freq, num_input, num_prod, num_ev,
                            /*frame_len_ns*/ 0, /*frame_length_fpga_ticks*/ 0,
                            /*file_start_time_ns*/ 0,
                            /*partial_path*/ std::string{}, /*open_wall_s*/ 0.0);
    const size_t f = meta->freq_id;
    const size_t t = 0;
    blk.add_frame(fv, meta, t);

    // Create dataset
    const std::string path = "test_gdal_vis_io.zarr";
    // Clean up any previous run
    CPLUnlinkTree(path.c_str());
    GDALDataset* ds =
        create_test_gdal_vis_dataset(path, num_freq, num_prod, num_ev, num_input, num_file_t);
    BOOST_REQUIRE(ds != nullptr);

    // Write buffered content via flush (writes full time range)
    blk.gdal_dataset = ds;
    blk.flush();
    GDALClose(ds);

    // Re-open read-only and validate content
    char** oo = nullptr;
    GDALDataset* ds_r = static_cast<GDALDataset*>(
        GDALOpenEx(path.c_str(), GDAL_OF_MULTIDIM_RASTER | GDAL_OF_READONLY, nullptr,
                   const_cast<const char**>(oo), nullptr));
    BOOST_REQUIRE(ds_r != nullptr);
    auto root = ds_r->GetRootGroup();
    BOOST_REQUIRE(root != nullptr);

    const auto c32Type = GDALExtendedDataType::Create(GDT_CFloat32);
    const auto f32Type = GDALExtendedDataType::Create(GDT_Float32);
    const auto f64Type = GDALExtendedDataType::Create(GDT_Float64);
    const auto u64Type = GDALExtendedDataType::Create(GDT_UInt64);

    // vis + weights for freq f
    {
        auto arr = root->OpenMDArray("vis_array");
        auto war = root->OpenMDArray("weights_array");
        BOOST_REQUIRE(arr && war);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0};
        std::vector<size_t> count{1, num_prod, num_file_t};
        std::vector<N2::cfloat> vis_out(num_prod * num_file_t);
        std::vector<float> w_out(num_prod * num_file_t);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, c32Type,
                            reinterpret_cast<void*>(vis_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = war->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                       reinterpret_cast<void*>(w_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        // Check t slice
        for (size_t p = 0; p < num_prod; ++p) {
            const size_t idx = num_file_t * p + t; // matches Write layout
            BOOST_CHECK(vis_out[idx] == N2::cfloat(10.0f * p + 1.0f, 10.0f * p + 2.0f));
            BOOST_CHECK_CLOSE_FRACTION(w_out[idx], float(1000 + p), 1e-6f);
            // Other t=0 should be default zero
            BOOST_CHECK(vis_out[num_file_t * p + (1 - t)] == N2::cfloat(0.0f, 0.0f));
            BOOST_CHECK_CLOSE_FRACTION(w_out[num_file_t * p + (1 - t)], 0.0f, 1e-6f);
        }
    }

    // eval and evec
    {
        auto arr = root->OpenMDArray("eval_array");
        BOOST_REQUIRE(arr);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0};
        std::vector<size_t> count{1, num_ev, num_file_t};
        std::vector<float> eval_out(num_ev * num_file_t);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                            reinterpret_cast<void*>(eval_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t e = 0; e < num_ev; ++e) {
            BOOST_CHECK_CLOSE_FRACTION(eval_out[num_file_t * e + t], float(60 + e), 1e-6f);
            BOOST_CHECK_CLOSE_FRACTION(eval_out[num_file_t * e + (1 - t)], 0.0f, 1e-6f);
        }
    }
    {
        auto arr = root->OpenMDArray("evec_array");
        BOOST_REQUIRE(arr);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0, 0};
        std::vector<size_t> count{1, num_ev, num_input, num_file_t};
        std::vector<N2::cfloat> evec_out(num_ev * num_input * num_file_t);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, c32Type,
                            reinterpret_cast<void*>(evec_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t e = 0; e < num_ev; ++e) {
            for (size_t i = 0; i < num_input; ++i) {
                const size_t idx = num_file_t * (num_input * e + i) + t;
                BOOST_CHECK(
                    evec_out[idx]
                    == N2::cfloat(100.0f * e + float(i) + 0.5f, -(100.0f * e + float(i) + 1.5f)));
                const size_t idx0 = num_file_t * (num_input * e + i) + (1 - t);
                BOOST_CHECK(evec_out[idx0] == N2::cfloat(0.0f, 0.0f));
            }
        }
    }

    // erms, gain, flags
    {
        auto arr = root->OpenMDArray("erms_array");
        BOOST_REQUIRE(arr);
        std::vector<GUInt64> start{(GUInt64)f, 0};
        std::vector<size_t> count{1, num_file_t};
        std::vector<float> erms_out(num_file_t);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                            reinterpret_cast<void*>(erms_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        BOOST_CHECK_CLOSE_FRACTION(erms_out[t], 3.14f, 1e-6f);
        BOOST_CHECK_CLOSE_FRACTION(erms_out[(1 - t)], 0.0f, 1e-6f);
    }
    {
        auto arr = root->OpenMDArray("gain_array");
        auto arrf = root->OpenMDArray("flags_array");
        BOOST_REQUIRE(arr && arrf);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0};
        std::vector<size_t> count{1, num_input, num_file_t};
        std::vector<N2::cfloat> gain_out(num_input * num_file_t);
        std::vector<float> flags_out(num_input * num_file_t);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, c32Type,
                            reinterpret_cast<void*>(gain_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = arrf->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                        reinterpret_cast<void*>(flags_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t i = 0; i < num_input; ++i) {
            BOOST_CHECK(gain_out[num_file_t * i + t]
                        == N2::cfloat(200.0f + float(i), -200.0f - float(i)));
            BOOST_CHECK(gain_out[num_file_t * i + (1 - t)] == N2::cfloat(0.0f, 0.0f));
            BOOST_CHECK_CLOSE_FRACTION(flags_out[num_file_t * i + t], float(300 + i), 1e-6f);
            BOOST_CHECK_CLOSE_FRACTION(flags_out[num_file_t * i + (1 - t)], 0.0f, 1e-6f);
        }
    }

    // frac_* and counts
    {
        auto arr = root->OpenMDArray("frac_lost_array");
        auto arr2 = root->OpenMDArray("frac_rfi_array");
        auto nval = root->OpenMDArray("n_valid_fpga_ticks");
        auto nrfi = root->OpenMDArray("n_rfi_fpga_ticks");
        BOOST_REQUIRE(arr && arr2 && nval && nrfi);
        std::vector<GUInt64> start{(GUInt64)f, 0};
        std::vector<size_t> count{1, num_file_t};
        std::vector<float> fl_out(num_file_t), fr_out(num_file_t);
        std::vector<uint64_t> nv_out(num_file_t), nr_out(num_file_t);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                            reinterpret_cast<void*>(fl_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = arr2->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                        reinterpret_cast<void*>(fr_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = nval->Read(start.data(), count.data(), nullptr, nullptr, u64Type,
                        reinterpret_cast<void*>(nv_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = nrfi->Read(start.data(), count.data(), nullptr, nullptr, u64Type,
                        reinterpret_cast<void*>(nr_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        BOOST_CHECK_CLOSE_FRACTION(fl_out[t], 1.0f - 80.0f / 100.0f, 1e-6f);
        BOOST_CHECK_CLOSE_FRACTION(fr_out[t], 5.0f / 100.0f, 1e-6f);
        BOOST_CHECK_CLOSE_FRACTION(fl_out[(1 - t)], 1.0f, 1e-6f);
        BOOST_CHECK_CLOSE_FRACTION(fr_out[(1 - t)], 0.0f, 1e-6f);
        BOOST_CHECK_EQUAL(nv_out[t], uint64_t(80));
        BOOST_CHECK_EQUAL(nr_out[t], uint64_t(5));
        BOOST_CHECK_EQUAL(nv_out[(1 - t)], uint64_t(0));
        BOOST_CHECK_EQUAL(nr_out[(1 - t)], uint64_t(0));
    }

    // per-time arrays
    {
        std::vector<GUInt64> start{0};
        std::vector<size_t> count{num_file_t};
        auto a0 = root->OpenMDArray("fpga_start_tick");
        auto a1 = root->OpenMDArray("frame_start_time_ns");
        auto a2 = root->OpenMDArray("frame_length_fpga_ticks");
        auto a3 = root->OpenMDArray("era_deg");
        BOOST_REQUIRE(a0 && a1 && a2 && a3);
        std::vector<uint64_t> s0(num_file_t), s1(num_file_t), s2(num_file_t);
        std::vector<double> s3(num_file_t);
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
        BOOST_CHECK_EQUAL(s0[t], uint64_t(111));
        BOOST_CHECK_EQUAL(s1[t], uint64_t(222));
        BOOST_CHECK_EQUAL(s2[t], uint64_t(100));
        BOOST_CHECK_CLOSE_FRACTION(s3[t], 12.34, 1e-12);
        // For frame_length_fpga_ticks we write the constant across all time indices
        // Other per-time arrays remain default at other indices.
        BOOST_CHECK_EQUAL(s0[(1 - t)], uint64_t(0));
        BOOST_CHECK_EQUAL(s1[(1 - t)], uint64_t(0));
        BOOST_CHECK_EQUAL(s2[(1 - t)], uint64_t(100));
        BOOST_CHECK_CLOSE_FRACTION(s3[(1 - t)], 0.0, 1e-12);
    }

    GDALClose(ds_r);

    // Cleanup output on best-effort basis
    CPLUnlinkTree(path.c_str());
}

// Ensure flush does not compact: data at t=1 remains at t=1 and t=0 stays zero.
BOOST_AUTO_TEST_CASE(test_write_no_time_compaction) {
    // Force-link N2Metadata registration by constructing once.
    N2Metadata force_link_marker3;
    // Dimensions
    const size_t num_input = 3;
    const size_t num_prod = N2::get_num_prod(num_input);
    const size_t num_ev = 2;
    const size_t num_freq = 3;
    const size_t num_file_t = 3; // choose >2 to avoid boundary confusion

    // Build a buffer for one N2 frame
    const size_t frame_size = N2FrameView::calculate_frame_size(num_input, num_ev);
    auto pool = metadataPool::create(1, sizeof(N2Metadata), "test_pool3", "N2Metadata");
    Buffer buf(1, frame_size, pool, "n2buf3", "vis", /*numa*/ 1, /*huge*/ false,
               /*mlock*/ false, /*producers*/ std::vector<int>{}, /*zero_new_frames*/ true);
    buf.allocate_new_metadata_object(0);
    auto meta = get_N2_metadata(&buf, 0);
    BOOST_REQUIRE(meta);
    meta->num_elements = num_input;
    meta->num_prod = num_prod;
    meta->num_ev = num_ev;
    meta->nfreq = num_freq;
    meta->freq_id = 1;

    meta->fpga_start_tick = 1234;
    meta->frame_start_time_ns = 5678;
    meta->frame_length_fpga_ticks = 100;
    meta->n_valid_fpga_ticks = 80;
    meta->n_rfi_fpga_ticks = 5;
    meta->eop.ERA_deg = 42.0;

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

    TestGdalVisFileData blk(num_file_t, num_freq, num_input, num_prod, num_ev,
                            /*frame_len_ns*/ 0, /*frame_length_fpga_ticks*/ 0,
                            /*file_start_time_ns*/ 0,
                            /*partial_path*/ std::string{}, /*open_wall_s*/ 0.0);
    const size_t f = meta->freq_id;
    const size_t t = 1; // place data at t=1 (middle)
    blk.add_frame(fv, meta, t);

    // Create dataset
    const std::string path = "test_gdal_vis_no_compact.zarr";
    // Clean up any previous run
    CPLUnlinkTree(path.c_str());
    GDALDataset* ds =
        create_test_gdal_vis_dataset(path, num_freq, num_prod, num_ev, num_input, num_file_t);
    BOOST_REQUIRE(ds != nullptr);

    // Write buffered content via flush (writes full time range; no compaction)
    blk.gdal_dataset = ds;
    blk.flush();
    GDALClose(ds);

    // Re-open read-only and validate positions
    char** oo = nullptr;
    GDALDataset* ds_r = static_cast<GDALDataset*>(
        GDALOpenEx(path.c_str(), GDAL_OF_MULTIDIM_RASTER | GDAL_OF_READONLY, nullptr,
                   const_cast<const char**>(oo), nullptr));
    BOOST_REQUIRE(ds_r != nullptr);
    auto root = ds_r->GetRootGroup();
    BOOST_REQUIRE(root != nullptr);

    const auto c32Type = GDALExtendedDataType::Create(GDT_CFloat32);
    const auto f32Type = GDALExtendedDataType::Create(GDT_Float32);
    const auto f64Type = GDALExtendedDataType::Create(GDT_Float64);
    const auto u64Type = GDALExtendedDataType::Create(GDT_UInt64);

    // vis + weights for freq f: expect data at t=1 only
    {
        auto arr = root->OpenMDArray("vis_array");
        auto war = root->OpenMDArray("weights_array");
        BOOST_REQUIRE(arr && war);
        std::vector<GUInt64> start{(GUInt64)f, 0, 0};
        std::vector<size_t> count{1, num_prod, num_file_t};
        std::vector<N2::cfloat> vis_out(num_prod * num_file_t);
        std::vector<float> w_out(num_prod * num_file_t);
        bool ok = arr->Read(start.data(), count.data(), nullptr, nullptr, c32Type,
                            reinterpret_cast<void*>(vis_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        ok = war->Read(start.data(), count.data(), nullptr, nullptr, f32Type,
                       reinterpret_cast<void*>(w_out.data()), nullptr, 0);
        BOOST_REQUIRE(ok);
        for (size_t p = 0; p < num_prod; ++p) {
            const size_t idx0 = num_file_t * p + 0;
            const size_t idx1 = num_file_t * p + 1;
            BOOST_CHECK(vis_out[idx0] == N2::cfloat(0.0f, 0.0f));
            BOOST_CHECK_CLOSE_FRACTION(w_out[idx0], 0.0f, 1e-6f);
            BOOST_CHECK(vis_out[idx1] == N2::cfloat(10.0f * p + 1.0f, 10.0f * p + 2.0f));
            BOOST_CHECK_CLOSE_FRACTION(w_out[idx1], float(1000 + p), 1e-6f);
        }
    }

    // per-time arrays: expect values at t=1 only; frame_length across all t
    {
        std::vector<GUInt64> start{0};
        std::vector<size_t> count{num_file_t};
        auto a0 = root->OpenMDArray("fpga_start_tick");
        auto a1 = root->OpenMDArray("frame_start_time_ns");
        auto a2 = root->OpenMDArray("frame_length_fpga_ticks");
        auto a3 = root->OpenMDArray("era_deg");
        BOOST_REQUIRE(a0 && a1 && a2 && a3);
        std::vector<uint64_t> s0(num_file_t), s1(num_file_t), s2(num_file_t);
        std::vector<double> s3(num_file_t);
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
        BOOST_CHECK_EQUAL(s0[0], uint64_t(0));
        BOOST_CHECK_EQUAL(s1[0], uint64_t(0));
        BOOST_CHECK_CLOSE_FRACTION(s3[0], 0.0, 1e-12);
        BOOST_CHECK_EQUAL(s0[1], uint64_t(1234));
        BOOST_CHECK_EQUAL(s1[1], uint64_t(5678));
        BOOST_CHECK_CLOSE_FRACTION(s3[1], 42.0, 1e-12);
        for (size_t tt = 0; tt < num_file_t; ++tt)
            BOOST_CHECK_EQUAL(s2[tt], uint64_t(100));
    }

    GDALClose(ds_r);
    CPLUnlinkTree(path.c_str());
}
