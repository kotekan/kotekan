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
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <frames.bin> [--n-win N] [--min-instances N] "
                             "[--max-open-win N] [--no-flush]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    int n_win = 4, min_inst = 2, max_open = 8;
    bool do_flush = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n-win" && i + 1 < argc)
            n_win = std::atoi(argv[++i]);
        else if (a == "--min-instances" && i + 1 < argc)
            min_inst = std::atoi(argv[++i]);
        else if (a == "--max-open-win" && i + 1 < argc)
            max_open = std::atoi(argv[++i]);
        else if (a == "--no-flush")
            do_flush = false;
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

    gnss::FleetDll dll(n_win, min_inst, max_open);
    std::vector<uint8_t> buf;
    uint64_t n_frames = 0, n_bad = 0, n_late = 0;
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
        const gnss::FoldStatus st = dll.fold(buf.data(), buf.size());
        if (st == gnss::FoldStatus::BAD_HEADER)
            n_bad++;
        else if (st == gnss::FoldStatus::LATE)
            n_late++;
    }
    if (do_flush)
        dll.flush();

    nlohmann::json out;
    out["frames"] = n_frames;
    out["bad_frames"] = n_bad;
    out["late_frames"] = n_late;
    out["n_win"] = n_win;
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
    std::cout << out.dump(2) << std::endl;
    return 0;
}
