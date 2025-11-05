// Boost test for visFileData add_frame() logic (HDF5 backend, no GDAL).

#define BOOST_TEST_MODULE "test_vis_file_data"

#include "N2FrameView.hpp"
#include "N2Metadata.hpp"
#include "N2Util.hpp"
#include "buffer.hpp"
#include "hdf5VisWrite.hpp"
#include "metadata.hpp"
#include "test_utils.hpp"

#include <boost/test/included/unit_test.hpp>
#include <string>
#include <vector>

class TestVisFileData : public visFileData {
public:
    using visFileData::visFileData;
    using visFileData::idx_feit;
    using visFileData::idx_fet;
    using visFileData::idx_fit;
    using visFileData::idx_fpt;
    using visFileData::idx_ft;

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

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

BOOST_AUTO_TEST_CASE(test_add_frame_single_slot) {
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
            fv.evec[num_input * e + i] = N2::cfloat(100.0f * e + float(i) + 0.5f,
                                                    -(100.0f * e + float(i) + 1.5f));
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
