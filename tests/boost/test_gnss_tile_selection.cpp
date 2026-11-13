#define BOOST_TEST_MODULE "test_gnss_tile_selection"

// cudaCorrelatorDual's per-chain tile-selection lists against the list one gather over a per-half
// row table would use: rows in gnss_gathers order, fg_offset the running sum of their channel
// counts, each row's list placed at fg_offset * tiles_per_chan of a [Fg_total][Tile] list. The
// gather kernel writes out[t][k] for selection entry k (cudaGnssN2Gather.cu), so with Tc outermost
// a row's slice of the list is its slice of every sub-integration. Every row's slice must equal
// the list its own gather builds today, every entry must belong to a row, and a table with one
// row's fg_offset or lane_base off by one tile must fail.
//
// The per-chain gathers are read through Config with cudaCorrelatorDual's keys and defaults.
// GNSS_ROWS_CONFIG=<json> (scripts/gnss/gpu_rows_equiv.py extract) takes them from a node
// config's gnss<g>_n2dual block; without it that case is skipped.

#include "Config.hpp"            // for Config
#include "gnssTileSelection.hpp" // for build_tile_selection

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

/// int2's layout without the CUDA headers.
struct Sel {
    std::int32_t x, y;
};

bool operator==(const Sel& a, const Sel& b) {
    return a.x == b.x && a.y == b.y;
}

/// One gather (today) or one row (the table): its channels, lanes and blocks.
struct Row {
    std::string name;
    std::vector<std::int32_t> chans;
    int lane_base = 0, lanes = 0;
    bool aa = false, bb = false;
    int fg_offset = 0;
};

/// The correlator geometry the lists depend on.
struct Geometry {
    std::vector<std::int32_t> comb;
    int num_elements = 0;
    std::vector<int> live_cols;
};

std::vector<Sel> selection(const Geometry& g, const Row& r) {
    return build_tile_selection<Sel>(g.comb, r.chans, g.num_elements, g.live_cols, r.lane_base,
                                     r.lanes, r.aa, r.bb);
}

/// The cudaCorrelatorDual command in @p cfg and its path ("/<half>_n2dual/commands/<i>").
std::string correlator_path(const json& cfg) {
    for (const auto& [key, block] : cfg.items()) {
        if (!block.is_object() || !block.contains("commands"))
            continue;
        const json& cmds = block.at("commands");
        for (size_t i = 0; i < cmds.size(); ++i)
            if (cmds[i].value("name", "") == "cudaCorrelatorDual")
                return "/" + key + "/commands/" + std::to_string(i);
    }
    throw std::runtime_error("no cudaCorrelatorDual command in the config");
}

/// Today's gathers, read as the cudaCorrelatorDual constructor reads them, with fg_offset the
/// running sum of the channel counts in gnss_gathers order.
std::vector<Row> gathers(const kotekan::Config& config, const std::string& path, Geometry& g) {
    g.comb = config.get<std::vector<std::int32_t>>(path, "gnss_local_channels");
    g.num_elements = config.get<int>(path, "num_elements");
    g.live_cols = config.get<std::vector<int>>(path, "live_element_tiles");
    const int num_synth = config.get_default<int>(path, "num_synth", 128);
    std::vector<Row> rows;
    int fg = 0;
    for (const std::string& name : config.get<std::vector<std::string>>(path, "gnss_gathers")) {
        const std::string pre = "gather_" + name + "_";
        Row r;
        r.name = name;
        r.chans = config.get_default<std::vector<std::int32_t>>(path, pre + "channels", g.comb);
        r.lane_base = config.get_default<int>(path, pre + "lane_base", 0);
        r.lanes = config.get_default<int>(path, pre + "lanes", num_synth);
        r.aa = config.get_default<bool>(path, pre + "aa", false);
        r.bb = config.get_default<bool>(path, pre + "bb", false);
        r.fg_offset = fg;
        fg += (int)r.chans.size();
        rows.push_back(r);
    }
    return rows;
}

/// The failures of a row table against today's per-chain lists (empty: it matches). The table's
/// rows are scattered in order into a [Fg_total][Tile] list, Fg_total the sum of their channel
/// counts and Tile today's uniform tiles per channel, as one gather would write them.
std::vector<std::string> compare(const Geometry& g, const std::vector<Row>& today,
                                 const std::vector<Row>& table) {
    std::vector<std::string> fail;
    if (table.size() != today.size())
        return {"the table has " + std::to_string(table.size()) + " rows, today "
                + std::to_string(today.size()) + " gathers"};
    std::vector<std::vector<Sel>> want;
    int tiles = -1, fg_total = 0;
    for (const Row& r : today) {
        want.push_back(selection(g, r));
        const int t = (int)(want.back().size() / r.chans.size());
        if (tiles >= 0 && t != tiles)
            return {"gather " + r.name + " has " + std::to_string(t)
                    + " tiles per channel, not the uniform " + std::to_string(tiles)};
        tiles = t;
        fg_total += (int)r.chans.size();
    }
    const Sel unset{-1, -1};
    std::vector<Sel> all((size_t)fg_total * tiles, unset);
    for (const Row& r : table) {
        const std::vector<Sel> sel = selection(g, r);
        const long lo = (long)r.fg_offset * tiles;
        if (lo < 0 || lo + (long)sel.size() > (long)all.size()) {
            fail.push_back("row " + r.name + ": Fg [" + std::to_string(r.fg_offset) + ", "
                           + std::to_string(r.fg_offset + (int)r.chans.size()) + ") is outside [0, "
                           + std::to_string(fg_total) + ")");
            continue;
        }
        for (size_t k = 0; k < sel.size(); ++k)
            all[(size_t)lo + k] = sel[k];
    }
    for (size_t i = 0; i < table.size(); ++i) {
        const Row& r = table[i];
        const long lo = (long)r.fg_offset * tiles;
        if (lo < 0 || lo + (long)want[i].size() > (long)all.size())
            continue;
        for (size_t k = 0; k < want[i].size(); ++k)
            if (!(all[(size_t)lo + k] == want[i][k])) {
                fail.push_back("row " + r.name + ": tile " + std::to_string(k)
                               + " of its slice is (" + std::to_string(all[(size_t)lo + k].x) + ", "
                               + std::to_string(all[(size_t)lo + k].y) + "), gather "
                               + today[i].name + " has (" + std::to_string(want[i][k].x) + ", "
                               + std::to_string(want[i][k].y) + ")");
                break;
            }
    }
    for (size_t k = 0; k < all.size(); ++k)
        if (all[k] == unset) {
            fail.push_back("entry " + std::to_string(k) + " (Fg " + std::to_string(k / tiles)
                           + ") belongs to no row");
            break;
        }
    return fail;
}

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& x : v)
        s += (s.empty() ? "" : "; ") + x;
    return s;
}

/// The table matches; each one-tile shift of one row's fg_offset or lane_base fails.
void check_rows(const json& cfg) {
    kotekan::Config config;
    config.update_config(cfg);
    Geometry g;
    const std::vector<Row> today = gathers(config, correlator_path(cfg), g);
    BOOST_REQUIRE(!today.empty());

    const std::vector<std::string> ok = compare(g, today, today);
    BOOST_CHECK_MESSAGE(ok.empty(), join(ok));
    const Row& last = today.back();
    const size_t n_tiles = selection(g, today.front()).size() / today.front().chans.size();
    BOOST_TEST_MESSAGE(today.size() << " rows, Fg_total " << last.fg_offset + last.chans.size()
                                    << ", " << n_tiles << " tiles per channel");

    for (size_t i = 0; i < today.size(); ++i)
        for (const char* field : {"fg_offset", "lane_base"})
            for (int step : {1, -1}) {
                std::vector<Row> table = today;
                if (std::string(field) == "fg_offset")
                    table[i].fg_offset += step;
                else
                    table[i].lane_base += 16 * step;
                const std::vector<std::string> f = compare(g, today, table);
                BOOST_CHECK_MESSAGE(!f.empty(), "row " << today[i].name << " " << field << " "
                                                       << (step > 0 ? "+" : "-")
                                                       << "1 tile passed the comparison");
                if (i == 1)
                    BOOST_TEST_MESSAGE("row " << today[i].name << " " << field << " "
                                              << (step > 0 ? "+" : "-") << "1 tile: " << join(f));
            }
}

} // namespace

// Three gathers in the shape of a node's: two chains on the same channels at lanes 0 and 64, and
// one on other channels reusing lanes 0-63.
BOOST_AUTO_TEST_CASE(builtin_rows) {
    const json cfg = json::parse(R"({
        "num_elements": 128,
        "gnss0_n2dual": {"commands": [
            {"name": "cudaGnssInject"},
            {"name": "cudaCorrelatorDual",
             "num_synth": 384,
             "num_live_elements": 128,
             "live_element_tiles": [0, 1, 2, 3, 4, 5, 6, 7],
             "gnss_local_channels": [10, 11, 12, 13, 20, 21],
             "gnss_gathers": ["a", "b", "c"],
             "gather_a_channels": [10, 11, 12, 13], "gather_a_lane_base": 0, "gather_a_lanes": 64,
             "gather_b_channels": [10, 11, 12, 13], "gather_b_lane_base": 64, "gather_b_lanes": 64,
             "gather_c_channels": [20, 21], "gather_c_lane_base": 0, "gather_c_lanes": 64}]}})");
    check_rows(cfg);
}

BOOST_AUTO_TEST_CASE(production_rows) {
    const char* path = std::getenv("GNSS_ROWS_CONFIG");
    if (path == nullptr) {
        BOOST_TEST_MESSAGE("GNSS_ROWS_CONFIG not set: production rows skipped");
        return;
    }
    std::ifstream in(path);
    BOOST_REQUIRE_MESSAGE(in, "cannot read " << path);
    check_rows(json::parse(in));
}
