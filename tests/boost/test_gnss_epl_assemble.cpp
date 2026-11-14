#define BOOST_TEST_MODULE "test_gnss_epl_assemble"

#include "gnssEplAssemble.hpp" // for assemble_epl, CtlView, TilesView
#include "gnssGpuChain.hpp"    // for FrameHdr, PrnCtl, frame_bytes, off_*

#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for sqrt
#include <cstdint> // for int32_t, uint8_t
#include <cstring> // for memcpy
#include <vector>

using namespace gnss_gpu;

namespace {
constexpr int N_REC = 2, N_PRN = 2, N_CHAN = 2, N_ELEM = 32, N_LIVE = 32, HOPS = 4;
constexpr int N_TILE = 2; // one synth tile row (lanes 32..47) x two live tile columns
constexpr size_t TILE = 512;

// A control block with every slot running, distinct energies per (job, channel).
std::vector<uint8_t> make_ctl() {
    std::vector<uint8_t> ctl(off_corr(N_PRN) + sizeof(double) * MAX_REC * N_PRN * 4 * N_CHAN);
    auto* hdr = (FrameHdr*)ctl.data();
    hdr->n_rec = N_REC;
    hdr->n_prn = N_PRN;
    hdr->n_chan = N_CHAN;
    hdr->lanes_per_slot = LANES_FULL;
    auto* pc = (PrnCtl*)(ctl.data() + off_prnctl());
    for (int i = 0; i < N_REC * N_PRN; ++i) {
        pc[i].run = 1;
        pc[i].prn = 10 + i;
        pc[i].job0 = i * ROWS_PLAIN;
    }
    auto* en = (double*)(ctl.data() + off_corr(N_PRN));
    for (int i = 0; i < N_REC * N_PRN * ROWS_PLAIN * N_CHAN; ++i)
        en[i] = 100.0 + 7.0 * i;
    return ctl;
}
} // namespace

// A row placed at a channel offset inside a wider gather assembles exactly as the same tiles
// gathered alone.
BOOST_AUTO_TEST_CASE(wide_gather_matches_compact) {
    constexpr int FG_TOTAL = 5, FG_OFFSET = 2;
    const std::vector<uint8_t> ctl = make_ctl();

    std::vector<int32_t> wide((size_t)N_REC * FG_TOTAL * N_TILE * TILE);
    for (size_t i = 0; i < wide.size(); ++i)
        wide[i] = (int32_t)(i % 4093) - 2000;
    std::vector<int32_t> compact((size_t)N_REC * N_CHAN * N_TILE * TILE);
    for (int r = 0; r < N_REC; ++r)
        std::memcpy(&compact[(size_t)r * N_CHAN * N_TILE * TILE],
                    &wide[((size_t)r * FG_TOTAL + FG_OFFSET) * N_TILE * TILE],
                    sizeof(int32_t) * N_CHAN * N_TILE * TILE);

    const size_t nb = frame_bytes(N_PRN, N_CHAN, ROWS_PLAIN, N_LIVE);
    std::vector<uint8_t> out_w(nb, 0xAA), out_c(nb, 0xAA); // rows past n_jobs are not written
    const TilesView tw{wide.data(), (size_t)FG_TOTAL * N_TILE * TILE, FG_OFFSET, N_CHAN, N_TILE};
    const TilesView tc{compact.data(), (size_t)N_CHAN * N_TILE * TILE, 0, N_CHAN, N_TILE};
    BOOST_CHECK_EQUAL(
        assemble_epl({ctl.data()}, tw, out_w.data(), nb, N_PRN, LANES_FULL, N_ELEM, N_LIVE, HOPS),
        N_REC * N_PRN);
    assemble_epl({ctl.data()}, tc, out_c.data(), nb, N_PRN, LANES_FULL, N_ELEM, N_LIVE, HOPS);
    BOOST_CHECK(out_w == out_c);

    // Record 1, PRN slot 0, the P row, channel 1, element 21: the entry of lane P (synth input
    // N_ELEM + ROW_P) and element 21 in the channel's second tile, over the pack scale of record
    // 0's P energy.
    const int r = 1, p = 0, c = 1, e = 21, lane = ROW_P;
    const int32_t* v = &wide[((size_t)r * FG_TOTAL + FG_OFFSET + c) * N_TILE * TILE
                             + (e >> 4) * TILE + 32 * ((N_ELEM + lane) & 15) + 2 * (e & 15)];
    const double* en = (const double*)(ctl.data() + off_corr(N_PRN));
    const double s =
        7.0 / (3.0 * std::sqrt(en[(size_t)(p * ROWS_PLAIN + ROW_P) * N_CHAN + c] / HOPS));
    const auto* corr = (const double*)(out_w.data() + off_corr(N_PRN));
    const size_t orow = (size_t)(r * N_PRN + p) * ROWS_PLAIN + ROW_P;
    const size_t oi = ((orow * N_CHAN + c) * N_LIVE + e) * 2;
    BOOST_CHECK_CLOSE(corr[oi], v[0] / s, 1e-12);
    BOOST_CHECK_CLOSE(corr[oi + 1], v[1] / s, 1e-12);
}

// Slot 0 runs in every record and slot 1 joins at record 2 of five. Each record of a slot is
// dequantised by the pack scale of the slot's first running record: record 0's for slot 0, record
// 2's for slot 1, not the record's own, the previous record's or job 0's.
BOOST_AUTO_TEST_CASE(scale_by_first_running_record) {
    constexpr int NR = 5, JOIN = 2;
    std::vector<uint8_t> ctl(off_corr(N_PRN) + sizeof(double) * MAX_REC * N_PRN * 4 * N_CHAN);
    auto* hdr = (FrameHdr*)ctl.data();
    hdr->n_rec = NR;
    hdr->n_prn = N_PRN;
    hdr->n_chan = N_CHAN;
    hdr->lanes_per_slot = LANES_FULL;
    auto* pc = (PrnCtl*)(ctl.data() + off_prnctl());
    int n_jobs = 0;
    for (int r = 0; r < NR; ++r)
        for (int p = 0; p < N_PRN; ++p) {
            PrnCtl& c = pc[r * N_PRN + p];
            c.prn = 10 + p;
            c.run = (p == 0 || r >= JOIN) ? 1 : 0;
            if (c.run)
                c.job0 = ROWS_PLAIN * n_jobs++;
        }
    auto* en = (double*)(ctl.data() + off_corr(N_PRN));
    for (int i = 0; i < n_jobs * ROWS_PLAIN * N_CHAN; ++i)
        en[i] = 100.0 + 7.0 * i;

    std::vector<int32_t> tiles((size_t)NR * N_CHAN * N_TILE * TILE);
    for (size_t i = 0; i < tiles.size(); ++i)
        tiles[i] = (int32_t)(i % 4093) - 2000;
    const size_t nb = frame_bytes(N_PRN, N_CHAN, ROWS_PLAIN, N_LIVE);
    std::vector<uint8_t> out(nb, 0);
    const TilesView tv{tiles.data(), (size_t)N_CHAN * N_TILE * TILE, 0, N_CHAN, N_TILE};
    BOOST_CHECK_EQUAL(
        assemble_epl({ctl.data()}, tv, out.data(), nb, N_PRN, LANES_FULL, N_ELEM, N_LIVE, HOPS),
        n_jobs);

    const auto* opc = (const PrnCtl*)(out.data() + off_prnctl());
    const auto* corr = (const double*)(out.data() + off_corr(N_PRN));
    const int c = 1, e = 21;
    for (int p = 0; p < N_PRN; ++p) {
        const int first = (p == 0) ? 0 : JOIN;
        const double e0 = en[(size_t)(pc[first * N_PRN + p].job0 + ROW_P) * N_CHAN + c];
        const double s = 7.0 / (3.0 * std::sqrt(e0 / HOPS));
        for (int r = first; r < NR; ++r) {
            const int32_t* v =
                &tiles[((size_t)r * N_CHAN + c) * N_TILE * TILE + (e >> 4) * TILE
                       + 32 * ((N_ELEM + LANES_FULL * p + ROW_P) & 15) + 2 * (e & 15)];
            const size_t oi =
                (((size_t)(opc[r * N_PRN + p].job0 + ROW_P) * N_CHAN + c) * N_LIVE + e) * 2;
            BOOST_CHECK_CLOSE(corr[oi], v[0] / s, 1e-12);
            BOOST_CHECK_CLOSE(corr[oi + 1], v[1] / s, 1e-12);
        }
    }
}
