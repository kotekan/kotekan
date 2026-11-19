#define BOOST_TEST_MODULE "test_gnss_telem_chain"

#include "Config.hpp"                // for Config
#include "gnssRecord.hpp"            // for frame_floats, record_stride, REC_P_ENERGY, REC_PRN
#include "gnssTelem.hpp"             // for TelemHeader, telem_frame_bytes, telem_row_offset
#include "gnssTelemChain.hpp"        // for TelemChain
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <cstdint> // for int64_t, uint8_t
#include <cstring> // for memcpy
#include <vector>  // for vector

using json = nlohmann::json;
using kotekan::Config;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

namespace {

constexpr int N_PRN = 3, N_REC = 2, HOPS = 2, FFT = 4;
constexpr int64_t REC = HOPS * FFT, WIN = N_REC * REC;
constexpr int ROW = gnss::telem_row_floats(1);

/// A record frame whose slot p carries PRN 10 * (p + 1), despread only where @p live says so.
std::vector<float> record(const std::vector<bool>& live, double utc) {
    std::vector<float> in(gnss::frame_floats(N_PRN, 0, 1), 0.0f);
    for (int p = 0; p < N_PRN; ++p) {
        float* r = &in[(size_t)p * gnss::record_stride(0)];
        r[gnss::REC_PRN] = 10.0f * (p + 1);
        r[gnss::REC_P_ENERGY] = live[(size_t)p] ? 1.0f : 0.0f;
        in[gnss::chan_offset(p, 0, N_PRN, 1, 0)] = 10.0f * (p + 1);
    }
    std::memcpy(&in[gnss::RECORD_UTC_SLOT], &utc, sizeof(utc));
    return in;
}

struct Frame {
    gnss::TelemHeader h;
    std::vector<float> rows;
    const float* row(int slot, int wrow) const {
        return &rows[gnss::telem_row_offset(slot, wrow, N_PRN, ROW)];
    }
};

Frame pack(gnss::TelemChain& c) {
    std::vector<uint8_t> f(gnss::telem_frame_bytes(N_REC, N_PRN, 1));
    const int64_t wstart0 = c.pack(f.data());
    Frame out;
    std::memcpy(&out.h, f.data(), sizeof(out.h));
    out.rows.resize((size_t)N_REC * N_PRN * ROW);
    std::memcpy(out.rows.data(), f.data() + sizeof(out.h), out.rows.size() * sizeof(float));
    BOOST_CHECK_EQUAL(wstart0, out.h.wstart0);
    return out;
}

} // namespace

BOOST_AUTO_TEST_CASE(windows) {
    json cfg;
    cfg["log_level"] = "warn";
    cfg["telem"] = {{"chain", "gps_l1"}, {"inst", "test.0"},        {"channel_ids", {6033}},
                    {"fft_len", FFT},    {"hops_per_record", HOPS}, {"max_prn", N_PRN},
                    {"n_chan", 1},       {"n_prn", N_PRN},          {"records_per_frame", N_REC}};
    Config config;
    config.update_config(cfg);
    gnss::TelemChain c(config, "/telem", "/telem");

    BOOST_CHECK(!c.on_grid(-1));
    BOOST_CHECK(!c.on_grid(REC + 1));

    // Window 5, every slot live.
    const std::vector<float> all = record({true, true, true}, 100.0);
    const int64_t w5 = 5 * WIN;
    BOOST_CHECK(!c.window_open());
    c.add_record(all.data(), w5);
    BOOST_CHECK(!c.window_full());
    BOOST_CHECK(!c.window_changes(w5 + REC));
    c.add_record(all.data(), w5 + REC);
    BOOST_CHECK(c.window_full());
    Frame f = pack(c);
    BOOST_CHECK_EQUAL(f.h.win, 5u);
    BOOST_CHECK_EQUAL(f.h.wstart0, w5);
    BOOST_CHECK_EQUAL(f.h.seq, 0u);
    BOOST_CHECK_EQUAL(f.h.present, 3u);
    BOOST_CHECK_EQUAL(f.h.utc0, 100.0);
    for (int r = 0; r < N_REC; ++r)
        for (int p = 0; p < N_PRN; ++p)
            BOOST_CHECK_EQUAL(f.row(r, p)[gnss::REC_PRN], 10.0f * (p + 1));
    BOOST_CHECK(!c.window_open());

    // Window 6: the last slot is no longer despread, so its row from window 5 must not carry over.
    const std::vector<float> two = record({true, true, false}, 200.0);
    c.add_record(two.data(), 6 * WIN);
    c.add_record(two.data(), 6 * WIN + REC);
    f = pack(c);
    BOOST_CHECK_EQUAL(f.h.seq, 1u);
    BOOST_CHECK_EQUAL(f.h.utc0, 200.0);
    for (int r = 0; r < N_REC; ++r)
        for (int k = 0; k < ROW; ++k)
            BOOST_CHECK_EQUAL(f.row(r, 2)[k], 0.0f);

    // Window 7 without its slot 0: no utc0, whatever the window before had.
    c.add_record(two.data(), 7 * WIN + REC);
    BOOST_CHECK(!c.window_full());
    f = pack(c);
    BOOST_CHECK_EQUAL(f.h.present, 2u);
    BOOST_CHECK_EQUAL(f.h.utc0, 0.0);
    for (int k = 0; k < ROW; ++k)
        BOOST_CHECK_EQUAL(f.row(0, 0)[k], 0.0f);

    // A backwards jump (an F-engine re-base) closes the open window like a forward one.
    c.add_record(two.data(), 8 * WIN);
    BOOST_CHECK(c.window_changes(9 * WIN));
    BOOST_CHECK(c.window_changes(2 * WIN + REC));
    BOOST_CHECK(c.window_changes(0));
}
