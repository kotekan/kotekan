/**
 * @file
 * @brief A1 REGRESSION: does the fleet fold survive an F-engine frame0 move?
 *
 * The defect (docs/CHORD_BUGLIST.md A1, rank 1 for three days because it VOIDS
 * EXPERIMENTS): `win` is absolute from the F-engine's frame0, `FleetDll` kept a strictly
 * monotone per-chain high-water mark, and an F-engine restart therefore made every
 * subsequent frame LATE **forever** -- the fold froze while `/get_dll` kept serving its
 * last aggregate at 200 OK with an identical `hop` on every row, the 60 TCP connections
 * stayed ESTABLISHED, and nothing looked down.
 *
 * WHY A DRIVER AND NOT A FIXTURE. There is no raw telemetry capture in the tree, and
 * hand-rolling the wire format in Python is exactly how the fleetdll gate spent eight days
 * green while every v5 frame came back BAD_HEADER. This builds frames with the SHIPPED
 * helpers (gnssTelem.hpp) and calls the SHIPPED fold, so a wire-format change breaks it
 * loudly instead of silently.
 *
 *   scripts/gnss/build_tool.sh epochreset && scripts/gnss/epochreset
 *
 * Prints one line per case and exits non-zero on the first failure.
 *
 * @author Keith Vanderlinde
 */

#include "gnssFleetDll.hpp"
#include "gnssTelem.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace gnss;

namespace {

constexpr int N_REC = 4;
constexpr int N_PRN = 2;
constexpr int N_CHAN = 2;
constexpr int PRN = 7;

/// One sender's frame for window `win`. `off` shifts EARLY vs LATE power so the
/// discriminator is non-zero and a frozen fold is visible as a stale value, not just a
/// missing one.
std::vector<char> make_frame(const std::string& inst, uint64_t win, uint64_t seq, double off) {
    std::vector<char> buf(telem_frame_bytes(N_REC, N_PRN), 0);
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
    h->seq = seq;
    h->wstart0 = (int64_t)win * N_REC * h->hops_per_record * h->fft_len;
    h->utc0 = 0.0;
    h->present = (1u << N_REC) - 1u;
    h->max_chan = TELEM_MAX_CHAN;
    h->n_row_total = TELEM_ROW_FLOATS;
    telem_set_name(h->chain, "gal_e5a");
    telem_set_name(h->inst, inst);
    for (int c = 0; c < N_CHAN; ++c)
        h->chan_id[c] = (uint16_t)(6000 + c);

    float* rows = telem_rows(buf.data());
    for (int r = 0; r < N_REC; ++r) {
        for (int p = 0; p < N_PRN; ++p) {
            float* row = rows + telem_row_offset(r, p, N_PRN);
            row[REC_PRN] = (p == 0) ? (float)PRN : 0.0f;
            if (p != 0)
                continue;
            for (int c = 0; c < N_CHAN; ++c) {
                float* cc = row + telem_chan_offset(c);
                cc[CHAN_RE] = 1.0f;         cc[CHAN_IM] = 0.0f; cc[CHAN_ENERGY] = 1.0f;
                cc[CHAN_E_RE] = (float)(1.0 + off); cc[CHAN_E_IM] = 0.0f; cc[CHAN_E_ENERGY] = 1.0f;
                cc[CHAN_L_RE] = (float)(1.0 - off); cc[CHAN_L_IM] = 0.0f; cc[CHAN_L_ENERGY] = 1.0f;
            }
        }
    }
    return buf;
}

int failures = 0;

void check(bool ok, const std::string& what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        failures++;
}

/// Push one window from two senders (min_instances is 2).
FoldStatus push(FleetDll& d, uint64_t win, uint64_t seq, double off) {
    FoldStatus last = FoldStatus::OK;
    for (const char* inst : {"cx19.0", "cx27.0"}) {
        auto f = make_frame(inst, win, seq, off);
        last = d.fold(f.data(), f.size());
    }
    return last;
}

double disc_of(FleetDll& d) {
    const auto it = d.chains().find("gal_e5a");
    if (it == d.chains().end())
        return 1e9;
    const auto r = it->second.row.find(PRN);
    return (r == it->second.row.end()) ? 1e9 : r->second.disc;
}

uint64_t updates_of(FleetDll& d) {
    const auto it = d.chains().find("gal_e5a");
    if (it == d.chains().end())
        return 0;
    const auto r = it->second.row.find(PRN);
    return (r == it->second.row.end()) ? 0 : r->second.n_updates;
}

} // namespace

int main() {
    printf("A1 epoch-reset regression (gnss::FleetDll, shipped wire format)\n");

    // ---- CASE 1: THE DEFECT ITSELF -- frame0 moves, the stream restarts small. --------
    {
        FleetDll d(4, 2, 8);
        for (uint64_t w = 5000; w < 5020; ++w)
            push(d, w, w, 0.20);
        const uint64_t before = updates_of(d);
        check(before > 0, "baseline: the pre-reset stream folds");
        const double d_before = disc_of(d);

        // The F-engine restarts: every sender's win restarts near zero, with the sky
        // now on the OTHER side of the peak so a frozen fold is visibly stale.
        for (uint64_t w = 0; w < 40; ++w)
            push(d, w, 1000 + w, -0.20);

        const uint64_t after = updates_of(d);
        check(after > before, "post-reset frames FOLD (the wedge is gone)");
        check(disc_of(d) * d_before < 0.0,
              "the served discriminator FOLLOWED the new epoch (sign flipped)");
        const auto& c = d.chains().at("gal_e5a");
        check(c.n_epoch_reset == 1, "exactly one epoch reset was declared");
    }

    // ---- CASE 2: A LAGGARD IS NOT AN EPOCH CHANGE. ------------------------------------
    // A sender a few windows behind must still be dropped as LATE, and must NEVER
    // re-anchor the chain -- that would make the aggregate depend on arrival order.
    {
        FleetDll d(4, 2, 8);
        for (uint64_t w = 5000; w < 5040; ++w)
            push(d, w, w, 0.20);
        const uint64_t newest_before = d.chains().at("gal_e5a").newest;
        for (int i = 0; i < 30; ++i)
            push(d, 5000, 9000 + i, 0.20);   // 40 windows back, repeatedly
        const auto& c = d.chains().at("gal_e5a");
        check(c.n_epoch_reset == 0, "a 40-window laggard does NOT re-anchor");
        check(c.newest == newest_before, "the high-water mark is unmoved by the laggard");
        check(c.n_late >= 30, "laggard frames are counted LATE");
    }

    // ---- CASE 3: ONE far-back frame cannot re-anchor (a corrupt header must not). -----
    {
        FleetDll d(4, 2, 8);
        for (uint64_t w = 5000; w < 5040; ++w)
            push(d, w, w, 0.20);
        auto f = make_frame("cx19.0", 3, 9999, 0.20);   // single wild frame
        d.fold(f.data(), f.size());
        check(d.chains().at("gal_e5a").n_epoch_reset == 0,
              "a single far-back frame does NOT re-anchor");
        for (uint64_t w = 5040; w < 5050; ++w)          // stream carries on normally
            push(d, w, w, 0.20);
        check(d.chains().at("gal_e5a").n_epoch_reset == 0,
              "and an in-order frame clears the strike run");
    }

    // ---- CASE 4: THE TRIMS SURVIVE. --------------------------------------------------
    // An F-engine restart must not turn into a fleet-wide pull-in: only the window state
    // is epoch-scoped. (This is the property d01c0c1be's trim store also protects.)
    {
        FleetDll d(4, 2, 8);
        TrimPolicy p;
        d.set_armed("gal_e5a", {PRN}, p);
        for (uint64_t w = 5000; w < 5020; ++w)
            push(d, w, w, 0.20);
        const double trim_before = d.chains().at("gal_e5a").trim.at(PRN).trim;
        check(trim_before != 0.0, "a trim accumulated before the reset");
        for (uint64_t w = 0; w < 12; ++w)
            push(d, w, 1000 + w, 0.20);
        const auto& c = d.chains().at("gal_e5a");
        check(c.n_epoch_reset == 1, "the reset fired");
        check(c.trim.count(PRN) == 1 && c.trim.at(PRN).trim != 0.0,
              "the standing trim SURVIVED the epoch reset");
        check(c.armed.count(PRN) == 1, "arming survived the epoch reset");
    }

    printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
