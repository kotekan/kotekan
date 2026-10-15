/**
 * @file
 * @brief Checks that the fleet DLL fold survives an F-engine frame0 move.
 *
 * `win` counts from the F-engine's frame0, so an F-engine restart makes every sender's window
 * index restart small. A fold that only keeps a per-chain high-water mark would then drop every
 * later frame as LATE while `/get_dll` keeps serving its last aggregate, with nothing looking
 * down. FleetDll::fold re-anchors after _epoch_strikes consecutive frames more than
 * _epoch_margin windows back. The cases check that it does, that a laggard or a single
 * far-back frame does not, and that the trims and arming survive the re-anchor.
 *
 * Frames are built with the gnssTelem.hpp helpers and fed to the shipped FleetDll::fold, so a
 * wire-format change breaks this tool instead of passing against a hand-written copy.
 *
 *   scripts/gnss/site/build_tool.sh epochreset && scripts/gnss/epochreset
 *
 * Prints one line per check and exits non-zero if any check failed.
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

/// One sender's frame for window `win`. `off` shifts early against late power so the
/// discriminator is non-zero and a frozen fold shows as a stale value, not just a missing one.
std::vector<char> make_frame(const std::string& inst, uint64_t win, uint64_t seq, double off) {
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
    h->seq = seq;
    h->wstart0 = (int64_t)win * N_REC * h->hops_per_record * h->fft_len;
    h->utc0 = 0.0;
    h->present = (1u << N_REC) - 1u;
    h->max_chan = N_CHAN;
    h->n_row_total = telem_row_floats(N_CHAN);
    telem_set_name(h->chain, "gal_e5a");
    telem_set_name(h->inst, inst);
    for (int c = 0; c < N_CHAN; ++c)
        h->chan_id[c] = (uint16_t)(6000 + c);

    float* rows = telem_rows(buf.data());
    for (int r = 0; r < N_REC; ++r) {
        for (int p = 0; p < N_PRN; ++p) {
            float* row = rows + telem_row_offset(r, p, N_PRN, telem_row_floats(N_CHAN));
            row[REC_PRN] = (p == 0) ? (float)PRN : 0.0f;
            if (p != 0)
                continue;
            for (int c = 0; c < N_CHAN; ++c) {
                float* cc = row + telem_chan_offset(c);
                cc[CHAN_RE] = 1.0f;
                cc[CHAN_IM] = 0.0f;
                cc[CHAN_ENERGY] = 1.0f;
                cc[CHAN_E_RE] = (float)(1.0 + off);
                cc[CHAN_E_IM] = 0.0f;
                cc[CHAN_E_ENERGY] = 1.0f;
                cc[CHAN_L_RE] = (float)(1.0 - off);
                cc[CHAN_L_IM] = 0.0f;
                cc[CHAN_L_ENERGY] = 1.0f;
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

    // ---- Case 1: frame0 moves and the stream restarts small. ----
    {
        FleetDll d(4, 2);
        for (uint64_t w = 5000; w < 5020; ++w)
            push(d, w, w, 0.20);
        const uint64_t before = updates_of(d);
        check(before > 0, "baseline: the pre-reset stream folds");
        const double d_before = disc_of(d);

        // The F-engine restarts: every sender's win restarts near zero, with the discriminator
        // on the other side of the peak so a frozen fold is visibly stale.
        for (uint64_t w = 0; w < 40; ++w)
            push(d, w, 1000 + w, -0.20);

        const uint64_t after = updates_of(d);
        check(after > before, "post-reset frames FOLD (the wedge is gone)");
        check(disc_of(d) * d_before < 0.0,
              "the served discriminator FOLLOWED the new epoch (sign flipped)");
        const auto& c = d.chains().at("gal_e5a");
        check(c.n_epoch_reset == 1, "exactly one epoch reset was declared");
    }

    // ---- Case 2: a laggard is not an epoch change. ----
    // A sender within _epoch_margin of newest is dropped as LATE and never re-anchors the
    // chain; re-anchoring on it would make the aggregate depend on arrival order.
    {
        FleetDll d(4, 2);
        for (uint64_t w = 5000; w < 5040; ++w)
            push(d, w, w, 0.20);
        const uint64_t newest_before = d.chains().at("gal_e5a").newest;
        for (int i = 0; i < 30; ++i)
            push(d, 5000, 9000 + i, 0.20); // 39 windows behind newest, repeatedly
        const auto& c = d.chains().at("gal_e5a");
        check(c.n_epoch_reset == 0, "a 40-window laggard does NOT re-anchor");
        check(c.newest == newest_before, "the high-water mark is unmoved by the laggard");
        check(c.n_late >= 30, "laggard frames are counted LATE");
    }

    // ---- Case 3: one far-back frame, such as a corrupt header, cannot re-anchor. ----
    {
        FleetDll d(4, 2);
        for (uint64_t w = 5000; w < 5040; ++w)
            push(d, w, w, 0.20);
        auto f = make_frame("cx19.0", 3, 9999, 0.20); // single wild frame
        d.fold(f.data(), f.size());
        check(d.chains().at("gal_e5a").n_epoch_reset == 0,
              "a single far-back frame does NOT re-anchor");
        for (uint64_t w = 5040; w < 5050; ++w) // stream carries on normally
            push(d, w, w, 0.20);
        check(d.chains().at("gal_e5a").n_epoch_reset == 0,
              "and an in-order frame clears the strike run");
    }

    // ---- Case 4: the trims survive. ----
    // An F-engine restart must not force a fleet-wide pull-in: only the window state is reset.
    {
        FleetDll d(4, 2);
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
