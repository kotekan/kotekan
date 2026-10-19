/**
 * @file
 * @brief Offline driver for gnss::FleetDll, the fast loop's discriminator, on a file.
 *
 * Reads telemetry frames in the gather's wire format (`[uint32 LE length][length bytes]`,
 * GnssTelemGather.hpp) and prints what the shipped arithmetic makes of them, as JSON on stdout.
 *
 * It includes gnssFleetDll.hpp and calls it rather than re-deriving the arithmetic (the rule of
 * scripts/gnss/e2e.cpp): a harness that re-derives it tests its author's understanding. If the
 * fold is wrong this is wrong in the same way, and scripts/gnss/fleetdll_gate.py, which
 * compares it with the Python arm, catches it.
 *
 *   scripts/gnss/site/build_tool.sh fleetdll && scripts/gnss/fleetdll frames.bin
 *
 * usage: fleetdll <file> [--n-win N] [--min-instances N] [--taps-win N] [--trim-in FILE]
 *                 [--no-flush] [--arm PRN,PRN,...] [--gain G] [--leak L] [--clamp C]
 *                 [--spacing S] [--p-floor-abs P]
 *
 * --no-flush leaves the last window open, so it is absent from the answer, as on the live path
 * (the next frame closes a window). The default flushes so a fixture's final window is not
 * missing; a comparison with the Python arm must match how the Python arm windowed the file.
 *
 * @author Keith Vanderlinde
 */

#include "gnssFleetDll.hpp"

#include "json.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// The integrator series, per chain and PRN: (last_win, disc applied, trim) rows.
using Series = std::map<std::string, std::map<int, std::vector<std::array<double, 3>>>>;

/// What the previous sample saw of one PRN's TrimState. The never-seen sentinel must not be a
/// legal window index: with 0, a fixture starting at window 0 would lose the loop's first step.
struct Seen {
    uint64_t win = UINT64_MAX;
    uint64_t n_steps = 0;
};

/// Append one row per PRN whose trim was updated since the last call, from its first step on.
/// integrate() sets last_win on every update, so no update is counted twice, as a frame-count
/// heuristic could do. A leak-only update advances last_win but neither n_steps nor last_disc,
/// so its row carries the disc it applied, 0. Updates are seen only between calls: when one
/// call follows several window closes, only the last is sampled.
void sample_series(const gnss::FleetDll& dll, std::map<std::string, std::map<int, Seen>>& seen,
                   Series& series) {
    for (const auto& cv : dll.chains())
        for (const auto& tv : cv.second.trim) {
            const gnss::FleetDll::TrimState& t = tv.second;
            Seen& prev = seen[cv.first][tv.first];
            if (t.n_steps > 0 && t.last_win != prev.win) {
                const double disc = (t.n_steps != prev.n_steps) ? t.last_disc : 0.0;
                prev.win = t.last_win;
                prev.n_steps = t.n_steps;
                series[cv.first][tv.first].push_back({(double)t.last_win, disc, t.trim});
            }
        }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <frames.bin> [--n-win N] [--min-instances N] [--taps-win N] "
                     "[--trim-in FILE]\n"
                     "        [--no-flush] [--arm PRN,PRN,...] [--gain G] [--leak L] "
                     "[--clamp C] [--spacing S] [--p-floor-abs P]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    int n_win = 4, min_inst = 2, taps_win = 0;
    std::string trim_in;
    bool do_flush = true;
    std::set<int> arm;
    gnss::TrimPolicy pol;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n-win" && i + 1 < argc)
            n_win = std::atoi(argv[++i]);
        else if (a == "--min-instances" && i + 1 < argc)
            min_inst = std::atoi(argv[++i]);
        else if (a == "--taps-win" && i + 1 < argc)
            taps_win = std::atoi(argv[++i]); // 0 = follow --n-win
        else if (a == "--trim-in" && i + 1 < argc)
            trim_in = argv[++i]; // a trim store to offer at arming, as the stage does
        else if (a == "--no-flush")
            do_flush = false;
        else if (a == "--arm" && i + 1 < argc) {
            std::string s = argv[++i], tok;
            std::stringstream ss(s);
            while (std::getline(ss, tok, ','))
                if (!tok.empty())
                    arm.insert(std::atoi(tok.c_str()));
        } else if (a == "--gain" && i + 1 < argc)
            pol.gain = std::atof(argv[++i]);
        else if (a == "--leak" && i + 1 < argc)
            pol.leak = std::atof(argv[++i]);
        else if (a == "--clamp" && i + 1 < argc)
            pol.clamp = std::atof(argv[++i]);
        else if (a == "--spacing" && i + 1 < argc)
            pol.spacing = std::atof(argv[++i]);
        else if (a == "--p-floor-abs" && i + 1 < argc)
            // The absolute window gate (TrimPolicy::p_floor_abs), the broker's probe-anchored
            // floor in p_pow units. 0 (the default) is no floor: every armed PRN's window is
            // leak-only and nothing steps.
            pol.p_floor_abs = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 2;
    }

    gnss::FleetDll dll(n_win, min_inst, taps_win);
    std::map<std::string, std::map<int, double>> restore;
    int n_adopted = 0;
    if (!trim_in.empty()) {
        std::ifstream tf(trim_in);
        if (!tf) {
            std::fprintf(stderr, "cannot read trim store %s\n", trim_in.c_str());
            return 2;
        }
        nlohmann::json tj;
        tf >> tj;
        const nlohmann::json ch = tj.value("chains", nlohmann::json::object());
        for (const auto& cv : ch.items())
            for (const auto& pv : cv.value().items())
                restore[cv.key()][std::stoi(pv.key())] = pv.value().get<double>();
    }
    std::vector<uint8_t> buf;
    uint64_t n_frames = 0, n_bad = 0, n_late = 0;

    // The integrator series: (last_win, disc, trim) per PRN at every update from its first step
    // on, so the gate can drive the Python integrator with the same disc sequence and compare
    // trims. The disc is compared per window separately, so the trims test the recurrence
    // alone. The harness arms from --arm because it has no policy cycle; live, the broker posts
    // the armed set (docs/CHORD_FAST_TRIM.md section 4).
    Series series;
    std::map<std::string, std::map<int, Seen>> seen;
    while (true) {
        uint32_t len = 0;
        if (!f.read((char*)&len, sizeof(len)))
            break; // clean EOF
        if (len == 0 || len > (1u << 24)) {
            std::fprintf(stderr,
                         "frame %llu: implausible length %u -- the stream is "
                         "desynchronised, refusing to guess\n",
                         (unsigned long long)n_frames, len);
            return 3;
        }
        buf.resize(len);
        if (!f.read((char*)buf.data(), len)) {
            std::fprintf(stderr, "frame %llu: truncated (wanted %u B)\n",
                         (unsigned long long)n_frames, len);
            return 3;
        }
        n_frames++;
        std::string chain;
        const gnss::FoldStatus st = dll.fold(buf.data(), buf.size(), &chain);
        if (st == gnss::FoldStatus::BAD_HEADER)
            n_bad++;
        else if (st == gnss::FoldStatus::LATE)
            n_late++;
        // Arm on first sight of a chain: the harness learns chain names from the wire, as the
        // stage does.
        if (!arm.empty() && st == gnss::FoldStatus::OK && !series.count(chain)) {
            series[chain];
            dll.set_armed(chain, arm, pol);
            // The stage's adoption path: a restored trim is offered only once its PRN is
            // armed, since an unarmed trim leaks away before the broker's first /set_policy.
            if (!restore.empty()) {
                auto ri = restore.find(chain);
                if (ri != restore.end())
                    for (const auto& tv : ri->second)
                        if (dll.adopt_trim(chain, tv.first, tv.second))
                            n_adopted++;
            }
        }
        sample_series(dll, seen, series);
    }
    if (do_flush) {
        dll.flush();
        sample_series(dll, seen, series);
    }

    nlohmann::json out;
    out["frames"] = n_frames;
    out["bad_frames"] = n_bad;
    out["late_frames"] = n_late;
    out["n_win"] = n_win;
    out["taps_win"] = dll.taps_win();
    out["min_instances"] = min_inst;
    out["flushed"] = do_flush;
    nlohmann::json chains = nlohmann::json::object();
    for (const auto& cv : dll.chains()) {
        nlohmann::json prns = nlohmann::json::object();
        for (const auto& pv : cv.second.row) {
            const gnss::FleetDllRow& s = pv.second;
            prns[std::to_string(pv.first)] = {{"disc", s.disc},
                                              {"q", s.q},
                                              {"e_pow", s.e_pow},
                                              {"p_pow", s.p_pow},
                                              {"l_pow", s.l_pow},
                                              {"n_src", s.n_src},
                                              {"n_chan", s.n_chan},
                                              {"n_rec", s.n_rec},
                                              {"hop", s.hop},
                                              {"win", s.win},
                                              {"n_updates", s.n_updates}};
        }
        chains[cv.first] = {{"prns", prns},
                            {"windows_closed", cv.second.n_closed},
                            {"late_frames", cv.second.n_late},
                            {"forced_closes", cv.second.n_forced},
                            {"dup_chan", cv.second.n_dup_chan},
                            {"open_windows", cv.second.open.size()},
                            {"newest_win", cv.second.newest}};
    }
    out["chains"] = chains;
    // The lobe-coherent per-PRN and per-channel taps, the object combdll.lobe_taps builds by
    // walking every (window, sender, record, PRN, channel), so the gate can compare the two
    // arms field by field on identical bytes. broker_equiv cannot cover this path: telemetry
    // arrives on a raw socket, not through gnss_broker.transport, so its replays carry none.
    nlohmann::json taps = nlohmann::json::object();
    for (const auto& cv : dll.taps()) {
        nlohmann::json pj = nlohmann::json::object();
        for (const auto& pv : cv.second) {
            const gnss::FleetDll::LobeTap& t = pv.second;
            nlohmann::json cj = nlohmann::json::object();
            // JSON has no NaN: a PRN no two senders reached says so with null, not a number
            const nlohmann::json xc =
                std::isfinite(t.xcoh) ? nlohmann::json(t.xcoh) : nlohmann::json();
            for (const auto& ch : t.chan)
                cj[std::to_string(ch.first)] = {ch.second[0], ch.second[1], ch.second[2],
                                                ch.second[3]};
            pj[std::to_string(pv.first)] = {
                {"e", t.e},         {"p", t.p},           {"l", t.l},   {"n_chan", t.n_chan},
                {"n_rec", t.n_rec}, {"n_inst", t.n_inst}, {"xcoh", xc}, {"n_xcoh", t.n_xcoh},
                {"hop", t.hop},     {"chan", cj}};
        }
        taps[cv.first] = pj;
    }
    out["taps"] = taps;
    // The per-record series (combdll.prompt_cn0's `recs`), flat and time-ordered:
    // [win, slot, prn, n_inst, n_chan, e, p, l]. Flat because it is a series: the served C/N0
    // gates individual records, so nothing may reduce it per window first.
    nlohmann::json rs = nlohmann::json::object();
    for (const auto& cv : dll.rec_series()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : cv.second)
            arr.push_back({r.win, r.slot, r.prn, r.n_inst, r.n_chan, r.e, r.p, r.l});
        rs[cv.first] = arr;
    }
    out["rec_series"] = rs;
    nlohmann::json hpr = nlohmann::json::object();
    for (const auto& cv : dll.chains())
        hpr[cv.first] = cv.second.hops_per_record;
    out["hops_per_record"] = hpr;
    // The trim store the stage would persist, and how much of an offered one was adopted.
    nlohmann::json ts = nlohmann::json::object();
    for (const auto& cv : dll.trim_snapshot()) {
        nlohmann::json pj = nlohmann::json::object();
        for (const auto& tv : cv.second)
            pj[std::to_string(tv.first)] = tv.second;
        ts[cv.first] = pj;
    }
    out["trim_snapshot"] = {{"chains", ts}};
    out["trim_adopted"] = n_adopted;
    if (!arm.empty()) {
        nlohmann::json sj = nlohmann::json::object();
        for (const auto& cv : series) {
            nlohmann::json pj = nlohmann::json::object();
            for (const auto& pv : cv.second) {
                nlohmann::json rows = nlohmann::json::array();
                for (const auto& s : pv.second)
                    rows.push_back({(uint64_t)s[0], s[1], s[2]}); // win, disc, trim
                pj[std::to_string(pv.first)] = rows;
            }
            sj[cv.first] = pj;
        }
        out["series"] = sj;
        // The integrator's step/skip census, per PRN. A skipped window is leak-only, so the
        // counts show how many windows the floor gated; fleetdll_gate.py's abs-floor leg checks
        // that a floor gates only the PRNs below it.
        nlohmann::json kj = nlohmann::json::object();
        for (const auto& cv : dll.chains()) {
            nlohmann::json pj = nlohmann::json::object();
            for (const auto& tv : cv.second.trim)
                pj[std::to_string(tv.first)] = {{"steps", tv.second.n_steps},
                                                {"skipped", tv.second.n_skipped},
                                                {"railed", tv.second.n_railed},
                                                {"trim", tv.second.trim}};
            kj[cv.first] = pj;
        }
        out["census"] = kj;
        out["policy"] = {{"gain", pol.gain},
                         {"leak", pol.leak},
                         {"clamp", pol.clamp},
                         {"spacing", pol.spacing},
                         {"p_floor_abs", pol.p_floor_abs}};
    }
    std::cout << out.dump(2) << std::endl;
    return 0;
}
