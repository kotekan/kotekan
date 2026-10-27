#define BOOST_TEST_MODULE "test_gnss_fleet_dll"

#include "gnssFleetDll.hpp" // for FleetDll
#include "gnssRecord.hpp"   // for REC_PRN, CHAN_*
#include "gnssTelem.hpp"    // for TelemHeader, telem_frame_bytes, telem_row_offset

#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for abs
#include <cstdint> // for uint32_t, uint64_t
#include <string>  // for string
#include <vector>  // for vector

using namespace gnss;

namespace {

constexpr int N_REC = 4;
constexpr int N_PRN = 2;
constexpr int N_CHAN = 2;
constexpr int PRN = 7;

/// One sender's frame for window `win`, laid out as GnssTelemPack ships it: row 0 carries PRN,
/// row 1 is unused, and only the slots set in `present` hold rows. The others stay zero.
std::vector<char> make_frame(const std::string& inst, uint64_t win, uint32_t present,
                             const std::string& chain = "gal_e5a", int chan0 = 6000) {
    std::vector<char> buf(telem_frame_bytes(N_REC, N_PRN, N_CHAN), 0);
    auto* h = (TelemHeader*)buf.data();
    h->magic = TELEM_MAGIC;
    h->version = TELEM_VERSION;
    h->n_rec = N_REC;
    h->n_prn = N_PRN;
    h->n_row = RECORD_FLOATS;
    h->n_chan = N_CHAN;
    h->n_elem = 1;
    h->hops_per_record = 2048;
    h->fft_len = 16384;
    h->win = win;
    h->seq = win;
    h->wstart0 = (int64_t)win * N_REC * h->hops_per_record * h->fft_len;
    h->present = present;
    h->max_chan = N_CHAN;
    h->n_row_total = telem_row_floats(N_CHAN);
    telem_set_name(h->chain, chain);
    telem_set_name(h->inst, inst);
    for (int c = 0; c < N_CHAN; ++c)
        h->chan_id[c] = (uint16_t)(chan0 + c);

    float* rows = telem_rows(buf.data());
    for (int r = 0; r < N_REC; ++r) {
        if (!(present & (1u << r)))
            continue;
        float* row = rows + telem_row_offset(r, 0, N_PRN, telem_row_floats(N_CHAN));
        row[REC_PRN] = (float)PRN;
        for (int c = 0; c < N_CHAN; ++c) {
            float* cc = row + telem_chan_offset(c);
            cc[CHAN_RE] = 1.0f;
            cc[CHAN_ENERGY] = 1.0f;
            cc[CHAN_E_RE] = 1.2f;
            cc[CHAN_E_ENERGY] = 1.0f;
            cc[CHAN_L_RE] = 0.8f;
            cc[CHAN_L_ENERGY] = 1.0f;
        }
    }
    return buf;
}

/// Fold window `win` from two senders (min_instances is 2), then flush it closed.
void fold_window(FleetDll& d, uint64_t win, uint32_t present) {
    for (const char* inst : {"cx19.0", "cx27.0"}) {
        auto f = make_frame(inst, win, present);
        BOOST_REQUIRE(d.fold(f.data(), f.size()) == FoldStatus::OK);
    }
    d.flush();
}

/// The record slots of PRN that rec_series() serves for the chain.
std::vector<int> slots_of(const FleetDll& d) {
    std::vector<int> slots;
    const auto series = d.rec_series();
    const auto it = series.find("gal_e5a");
    if (it != series.end())
        for (const auto& r : it->second)
            if (r.prn == PRN)
                slots.push_back(r.slot);
    return slots;
}

} // namespace

// Every slot filled: each one is a record of PRN.
BOOST_AUTO_TEST_CASE(full_window_folds_every_slot) {
    FleetDll d(4, 2);
    fold_window(d, 5000, 0xF);
    const std::vector<int> want{0, 1, 2, 3};
    const auto got = slots_of(d);
    BOOST_CHECK_EQUAL_COLLECTIONS(got.begin(), got.end(), want.begin(), want.end());
    BOOST_CHECK_EQUAL(d.chains().at("gal_e5a").row.at(PRN).n_rec, 4);
}

// Slot 0 missing (start-up, a dropped record, the first window after a re-base): its rows are
// zero, and the PRN map comes from the first filled slot, so slots 1-3 still fold.
BOOST_AUTO_TEST_CASE(window_without_slot_0_folds_its_filled_slots) {
    FleetDll d(4, 2);
    fold_window(d, 5000, 0xE);
    const std::vector<int> want{1, 2, 3};
    const auto got = slots_of(d);
    BOOST_CHECK_EQUAL_COLLECTIONS(got.begin(), got.end(), want.begin(), want.end());
    const auto& rows = d.chains().at("gal_e5a").row;
    BOOST_REQUIRE(rows.count(PRN));
    BOOST_CHECK_EQUAL(rows.at(PRN).n_rec, 3);
    BOOST_CHECK_GT(rows.at(PRN).disc, 0.0); // E > L in every record
}

// Only the last slot filled.
BOOST_AUTO_TEST_CASE(window_with_only_the_last_slot_folds_it) {
    FleetDll d(4, 2);
    fold_window(d, 5000, 0x8);
    const std::vector<int> want{3};
    const auto got = slots_of(d);
    BOOST_CHECK_EQUAL_COLLECTIONS(got.begin(), got.end(), want.begin(), want.end());
}

// An empty armed set, as GnssFleetTrim sets for a chain whose policy expired, keeps the standing
// trim at first (no step) and then leaks it to zero and erases it.
BOOST_AUTO_TEST_CASE(disarmed_trim_leaks_to_zero_and_is_erased) {
    FleetDll d(4, 2);
    TrimPolicy p;
    p.p_floor_abs = 1e-30; // below every power here; 0 would make every window leak-only
    d.set_armed("gal_e5a", {PRN}, p);
    for (uint64_t w = 5000; w < 5020; ++w)
        fold_window(d, w, 0xF);
    const double standing = d.chains().at("gal_e5a").trim.at(PRN).trim;
    BOOST_CHECK(std::abs(standing) > 0.1);

    d.set_armed("gal_e5a", {}, p);
    BOOST_CHECK_EQUAL(d.chains().at("gal_e5a").trim.at(PRN).trim, standing);
    fold_window(d, 5020, 0xF);
    BOOST_CHECK_LT(std::abs(d.chains().at("gal_e5a").trim.at(PRN).trim), std::abs(standing));
    for (uint64_t w = 5021; w < 5400; ++w)
        fold_window(d, w, 0xF);
    BOOST_CHECK(d.chains().at("gal_e5a").trim.empty());
    BOOST_CHECK(d.trim_snapshot().empty());
}

// No slot filled: nothing to map and nothing folds.
BOOST_AUTO_TEST_CASE(empty_window_folds_nothing) {
    FleetDll d(4, 2);
    fold_window(d, 5000, 0x0);
    BOOST_CHECK(slots_of(d).empty());
    BOOST_CHECK(d.chains().at("gal_e5a").row.empty());
}

// One chain's taps without the per-channel tables (the broker's get_taps?chain=&chan=0): the
// same lobe taps as the full reduction, nothing of the other chain, no channel table.
BOOST_AUTO_TEST_CASE(taps_for_one_chain_without_channels) {
    FleetDll d(4, 2);
    // two senders on distinct channels (a channel both carry is left out of the table)
    for (uint64_t w = 5000; w < 5003; ++w) {
        for (const char* chain : {"gal_e5a", "gps_l5"}) {
            auto a = make_frame("cx19.0", w, 0xF, chain, 6000);
            auto b = make_frame("cx27.0", w, 0xF, chain, 6100);
            BOOST_REQUIRE(d.fold(a.data(), a.size()) == FoldStatus::OK);
            BOOST_REQUIRE(d.fold(b.data(), b.size()) == FoldStatus::OK);
        }
        d.flush();
    }
    const auto all = d.taps();
    const auto one = d.taps("gal_e5a", false);
    BOOST_REQUIRE_EQUAL(all.size(), 2u);
    BOOST_REQUIRE_EQUAL(one.size(), 1u);
    BOOST_REQUIRE(one.count("gal_e5a") && one.at("gal_e5a").count(PRN));
    const auto& a = all.at("gal_e5a").at(PRN);
    const auto& o = one.at("gal_e5a").at(PRN);
    BOOST_CHECK_EQUAL(o.e, a.e);
    BOOST_CHECK_EQUAL(o.p, a.p);
    BOOST_CHECK_EQUAL(o.l, a.l);
    BOOST_CHECK_EQUAL(o.n_chan, a.n_chan);
    BOOST_CHECK_EQUAL(o.n_rec, a.n_rec);
    BOOST_CHECK(!a.chan.empty());
    BOOST_CHECK(o.chan.empty());
    const auto rs = d.rec_series("gps_l5");
    BOOST_CHECK_EQUAL(rs.size(), 1u);
    BOOST_CHECK(rs.count("gps_l5"));
    BOOST_CHECK(d.taps("glo_l2of").empty());
}
