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
constexpr int N_REC = 2, N_PRN = 2, N_CHAN = 2, N_ELEM = 16, N_LIVE = 16, HOPS = 4;
constexpr int N_TILE = 1; // one synth tile row (lanes 16..31) x one live tile column
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

    // Record 1, PRN slot 0, the P row, channel 1, element 5: the tile entry of lane P (synth
    // input N_ELEM + ROW_P) and element 5, over the pack scale of record 0's P energy.
    const int r = 1, p = 0, c = 1, e = 5, lane = ROW_P;
    const int32_t* v = &wide[((size_t)r * FG_TOTAL + FG_OFFSET + c) * N_TILE * TILE
                             + 32 * ((N_ELEM + lane) & 15) + 2 * e];
    const double* en = (const double*)(ctl.data() + off_corr(N_PRN));
    const double s =
        7.0 / (3.0 * std::sqrt(en[(size_t)(p * ROWS_PLAIN + ROW_P) * N_CHAN + c] / HOPS));
    const auto* corr = (const double*)(out_w.data() + off_corr(N_PRN));
    const size_t orow = (size_t)(r * N_PRN + p) * ROWS_PLAIN + ROW_P;
    const size_t oi = ((orow * N_CHAN + c) * N_LIVE + e) * 2;
    BOOST_CHECK_CLOSE(corr[oi], v[0] / s, 1e-12);
    BOOST_CHECK_CLOSE(corr[oi + 1], v[1] / s, 1e-12);
}
