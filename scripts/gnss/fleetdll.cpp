/**
 * @file
 * @brief OFFLINE DRIVER for gnss::FleetDll -- the fast loop's discriminator, on a file.
 *
 * Task #51 milestone F1's gate. Reads telemetry frames in the gather's own wire format
 * (`[uint32 LE length][length bytes]`, GnssTelemGather.hpp) and prints what the SHIPPED
 * arithmetic makes of them, as JSON on stdout.
 *
 * WHY IT IS THE SHIPPED CODE AND NOT A MODEL OF IT. Same rule as scripts/gnss/e2e.cpp: a
 * harness that re-derives the arithmetic tests the harness author's understanding, which is
 * exactly the thing already known to be unreliable here. This includes gnssFleetDll.hpp and
 * calls it. If the fold is wrong, this is wrong in the same way, and the gate against the
 * Python arm (scripts/gnss/fleetdll_gate.py) is what catches it.
 *
 *   ./build_tool.sh fleetdll && ./fleetdll frames.bin
 *
 * usage: fleetdll <file> [--n-win N] [--min-instances N] [--max-open-win N] [--no-flush]
 *
 * ⚠️ --no-flush leaves the last window(s) OPEN, i.e. absent from the answer, which is what the
 * live path does (the next frame is the completion signal). The default flushes so a fixture's
 * final window is not silently missing -- but a gate comparing against the Python arm should
 * choose whichever matches how the Python arm windowed the same file.
 *
 * @author Keith Vanderlinde
 */

#include "gnssFleetDll.hpp"

#include "json.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <frames.bin> [--n-win N] [--min-instances N] "
                             "[--max-open-win N] [--taps-win N] [--trim-in F] [--no-flush]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    int n_win = 4, min_inst = 2, max_open = 8, taps_win = 0;
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
        else if (a == "--max-open-win" && i + 1 < argc)
            max_open = std::atoi(argv[++i]);
        else if (a == "--taps-win" && i + 1 < argc)
            taps_win = std::atoi(argv[++i]); // 0 = follow --n-win
        else if (a == "--trim-in" && i + 1 < argc)
            trim_in = argv[++i]; // a trim store to offer at ARMING, as the stage does
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
            // The broker-anchored absolute window gate (TrimPolicy::p_floor_abs). 0 (the
            // default) = the original peer-median gate, byte-identical -- which is exactly
            // what the equivalence gate asserts; a run WITH it is the arm that proves the
            // flag changes only what it claims to.
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

    gnss::FleetDll dll(n_win, min_inst, max_open, 3.0, taps_win);
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

    // THE INTEGRATOR SERIES. Every (window, disc, trim) the loop actually stepped on, so the
    // gate can drive the PYTHON integrator with the SAME disc sequence and compare trims. That
    // decomposition matters: the disc agreement is established separately, per window, against
    // Python's own fleet_dll_comb, so this leg tests the RECURRENCE in isolation rather than
    // re-testing the fold. Arming is done here at start because the offline harness has no
    // policy cycle -- on the live path the broker publishes it (docs/CHORD_FAST_TRIM.md 4).
    std::map<std::string, std::map<int, std::vector<std::array<double, 3>>>> series;
    std::map<std::string, std::map<int, uint64_t>> seen_win;
    while (true) {
        uint32_t len = 0;
        if (!f.read((char*)&len, sizeof(len)))
            break; // clean EOF
        if (len == 0 || len > (1u << 24)) {
            std::fprintf(stderr, "frame %llu: implausible length %u -- the stream is "
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
        // Arm on first sight of a chain: set_armed needs the chain to exist, and the harness
        // learns the chain names from the wire exactly as the stage does.
        if (!arm.empty() && st == gnss::FoldStatus::OK && !series.count(chain)) {
            series[chain];
            dll.set_armed(chain, arm, pol);
            // THE STAGE'S ADOPTION PATH, exercised here rather than modelled: a restored trim
            // is offered ONLY once policy has armed the PRN. Offering it at load would let it
            // leak to erasure before the first /set_policy (~5.6 s vs an ~11 s cycle).
            if (!restore.empty()) {
                auto ri = restore.find(chain);
                if (ri != restore.end())
                    for (const auto& tv : ri->second)
                        if (dll.adopt_trim(chain, tv.first, tv.second))
                            n_adopted++;
            }
        }
        // Sample whatever stepped. `last_win` changing IS the step, so this cannot double-count
        // a window nor miss one -- a frame-count heuristic could do both.
        for (const auto& cv : dll.chains())
            for (const auto& tv : cv.second.trim) {
                // ⚠️ THE SENTINEL MUST NOT BE A LEGAL WINDOW INDEX. This defaulted to 0, which
                // IS window 0 in any fixture that starts there -- so the loop's very first step
                // was silently never recorded, and every trim afterwards looked one step ahead
                // of the reference. It presented as an integrator disagreement (~2x on the
                // first sample) and was a harness bug. Caught by the gate, 2026-08-15.
                auto ins = seen_win[cv.first].emplace(tv.first, UINT64_MAX);
                uint64_t& prev = ins.first->second;
                if (tv.second.n_steps > 0 && tv.second.last_win != prev) {
                    prev = tv.second.last_win;
                    series[cv.first][tv.first].push_back(
                        {(double)tv.second.last_win, tv.second.last_disc, tv.second.trim});
                }
            }
    }
    if (do_flush)
        dll.flush();

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
            prns[std::to_string(pv.first)] = {
                {"disc", s.disc},     {"q", s.q},         {"e_pow", s.e_pow},
                {"p_pow", s.p_pow},   {"l_pow", s.l_pow}, {"n_src", s.n_src},
                {"n_chan", s.n_chan}, {"n_rec", s.n_rec}, {"hop", s.hop},
                {"win", s.win},       {"n_updates", s.n_updates}};
        }
        chains[cv.first] = {{"prns", prns},
                            {"windows_closed", cv.second.n_closed},
                            {"late_frames", cv.second.n_late},
                            {"forced_closes", cv.second.n_forced},
                            {"open_windows", cv.second.open.size()},
                            {"newest_win", cv.second.newest}};
    }
    out["chains"] = chains;
    // THE PER-INSTANCE, PER-CHANNEL TAPS -- the object combdll.instance_taps builds in Python
    // by walking every (window, instance, record, PRN, channel). Emitted so the gate can
    // compare the two arms field by field on identical bytes, which is the only thing that
    // makes moving the reduction off the broker safe: the telemetry path is INVISIBLE to
    // broker_equiv (the gather is a raw socket, not gnss_broker.transport, so a replay runs
    // with no telemetry at all and silently falls back to the polled discriminator). The
    // digest would stay green while testing none of this.
    nlohmann::json taps = nlohmann::json::object();
    for (const auto& cv : dll.taps()) {
        nlohmann::json pj = nlohmann::json::object();
        for (const auto& pv : cv.second) {
            nlohmann::json ij = nlohmann::json::object();
            for (const auto& iv : pv.second) {
                nlohmann::json cj = nlohmann::json::object();
                for (const auto& ch : iv.second.chan)
                    cj[std::to_string(ch.first)] = {ch.second[0], ch.second[1], ch.second[2],
                                                    ch.second[3]};
                ij[iv.first] = {{"e", iv.second.e},       {"p", iv.second.p},
                                {"l", iv.second.l},       {"n_chan", iv.second.n_chan},
                                {"n_rec", iv.second.n_rec}, {"hop", iv.second.hop},
                                {"chan", cj}};
            }
            pj[std::to_string(pv.first)] = ij;
        }
        taps[cv.first] = pj;
    }
    out["taps"] = taps;
    // The per-record series (combdll.prompt_cn0's `recs`), flat and time-ordered:
    // [win, slot, prn, n_inst, e, p, l]. Flat rather than nested because it is a SERIES --
    // nesting it by window would invite a consumer to reduce it, which is the one thing the
    // served C/N0 must not have done for it.
    nlohmann::json rs = nlohmann::json::object();
    for (const auto& cv : dll.rec_series()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : cv.second)
            arr.push_back({r.win, r.slot, r.prn, r.n_inst, r.e, r.p, r.l});
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
        // The integrator's step/skip census, per PRN. The skip counter is the window gate's
        // entire observable effect (a skipped window is leak-only), so an arm that changes the
        // gate MUST move exactly this and nothing else -- which is what the equivalence gate's
        // --p-floor-abs case asserts.
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
