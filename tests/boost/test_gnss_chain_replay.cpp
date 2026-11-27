#define BOOST_TEST_MODULE "test_gnss_chain_replay"

// Replays synthetic control blocks and tiles through the four host stages of a GNSS chain
// (GnssN2RecordAssemble -> GnssGpuRecordAssemble -> GnssCoherentCombiner + GnssTelemPack), three
// chains in one process as on a node, one frame at a time, and digests everything they emit:
// buffers, REST replies at checkpoints, the ProjBoard and the REST surface. Each run is a forked
// child, which isolates the process singletons (ProjBoard, ChanPower, restServer, metrics) and the
// REST callbacks the stages never unregister.
//
// GNSS_REPLAY_CONFIG=<json> replays the production stage blocks scripts/gnss/replay_extract.py
// extracts instead of the built-in chains. GNSS_REPLAY_GOLDEN=<file> compares the baseline with a
// golden (GNSS_REPLAY_BLESS=1 writes it), and GNSS_REPLAY_DUMP=<dir> writes each run's stage
// configs and GET /endpoints reply there, <pid>.config.json and <pid>.endpoints.json, for
// node_contract. A replay takes seconds in a Test build, so by default only the baseline, which
// also runs each chain's PRN swap references (Scenario::cold_ref), and the zero-amplitude
// control run; GNSS_REPLAY_FULL=1 adds the determinism, mutation and prefill
// controls, a run whose combiners have no out_buf and one whose combiners keep slot state across
// a PRN swap. Every run also replays the chains as rows of three host stages (Scenario::rows), and
// with assemblers that read the tiles and control blocks themselves (Scenario::fold), also on
// workers (Scenario::workers), and with the combiners on workers (Scenario::cmb_workers); each must
// give the baseline's digests. workers_stall, and with GNSS_REPLAY_FULL=1 combine_workers_stall,
// hold one worker with the frames or records running free.

#include "Config.hpp"                // for Config
#include "GnssChanMetadata.hpp"      // for GnssChanMetadata, get_gnss_chan_metadata
#include "GnssCoherentCombiner.hpp"  // for GnssCoherentCombiner
#include "GnssGpuRecordAssemble.hpp" // for GnssGpuRecordAssemble
#include "GnssN2RecordAssemble.hpp"  // for GnssN2RecordAssemble
#include "GnssTelemPack.hpp"         // for GnssTelemPack
#include "Stage.hpp"                 // for Stage
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "chordMetadata.hpp"         // for chordMetadata, get_chord_metadata
#include "gnssChanPower.hpp"         // for ChanPower
#include "gnssGpuChain.hpp"          // for FrameHdr, PrnCtl, off_*
#include "gnssGpuRows.hpp"           // for row_tag
#include "gnssProjSubspace.hpp"      // for ProjBoard
#include "gnssRecRows.hpp"           // for RecDir, RecRow
#include "gnssRecord.hpp"            // for frame_floats, record_stride, cube_frame_bytes
#include "gnssTelem.hpp"             // for TelemHeader, telem_frame_bytes
#include "gnss_replay_frames.hpp"    // for ChainSpec, Sky, Scenario, write_ctl, write_tiles
#include "metadata.hpp"              // for metadataPool
#include "prometheusMetrics.hpp"     // for Metrics
#include "restClient.hpp"            // for restClient
#include "restServer.hpp"            // for restServer
#include "version.h"                 // for get_git_commit_hash

#include "fmt.hpp"  // for format
#include "json.hpp" // for json

#include <algorithm>
#include <array>
#include <atomic>
#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;
using namespace gnss_replay;

namespace {

/// What one run emitted. streams: per-frame (or one) FNV-1a digests, keyed "<pre>.epl", ".rec",
/// ".cmb", ".telem", ".cube", "<pre>.rest.<verb>@<checkpoint>", "board@<checkpoint>" and
/// "endpoints". strings: "<pre>.cube_chain" and "endpoints". live and pins: the values the
/// liveness and pin checks read. slots: per chain and combiner ("own", and "cold", "scint" and
/// "others" with Scenario::cold_ref), per-slot digests of every emit ("cmb", see cmb_slot_rows) and
/// of the GETs at the slot frames (read_slot_gets).
struct Digests {
    std::map<std::string, std::vector<uint64_t>> streams;
    std::map<std::string, std::string> strings;
    json live, pins, slots;
};

// ---- chains ----

/// Planted amplitude per element and channel, in units of the per-record noise.
constexpr double SKY_AMP = 3.0;
constexpr int64_t REC_SAMPLES = 2048LL * 16384LL;

json assemble_block(const std::vector<int>& prns, const std::vector<int>& channel_ids,
                    const std::vector<int>& xfer, double cp_mod, const Sky& sky, int n_elem) {
    std::vector<double> pos, ref;
    std::vector<int> norm_elems;
    for (int e = 0; e < n_elem; ++e) {
        const int dish = e % (n_elem / 2); // two pols per dish, pol 1 in the upper half
        pos.insert(pos.end(), {6.3 * (dish % 4), 8.5 * (dish / 4), 0.05 * dish});
        ref.insert(ref.end(), {sky.gain_e[(size_t)e].real(), sky.gain_e[(size_t)e].imag()});
        norm_elems.push_back(e);
    }
    json a = {{"n_elements", n_elem},
              {"sample_rate", 3.2e9},
              {"prns", prns},
              {"channel_ids", channel_ids},
              {"reference_element", 0},
              {"elem_sum", true},
              {"elem_sum_adapt", false},
              {"elem_sum_shared", true},
              {"elem_sum_shared_ref", ref},
              {"elem_sum_shared_ref_mode", "live"},
              {"elem_sum_tau_s", 0.5},
              {"elem_sum_pol_tau_s", 3.0},
              {"elem_sum_shared_tau_s", 300.0},
              {"elem_sum_shared_freeze_deg", 2.0},
              {"chan_noise_norm", true},
              {"chan_noise_norm_elems", norm_elems},
              {"chan_export", true},
              {"elem_dir_publish", "all"},
              {"elem_dir_hold_s", 1500.0},
              {"elem_dir_coh_min", 0.08},
              {"elem_dir_hold_coh_min", 0.08},
              {"elem_dir_hold_frac", 0.9},
              {"elem_dir_capture_cos2", 0.5},
              {"elem_dir_capture_gain", 1.5},
              {"elem_positions_enu", pos},
              {"elem_steer_sign", 1.0},
              {"elem_proj_sys", "G"},
              {"beam_cube", true},
              {"beam_cube_gpu", 0},
              {"beam_cube_max_bins", 8},
              {"beam_cube_max_prn", 32},
              {"beam_cube_ring_depth", 8},
              {"beam_cube_window_samples", (double)(48 * REC_SAMPLES)},
              {"spectrum_ring_depth", 8},
              {"spectrum_window_samples", (double)(24 * REC_SAMPLES)}};
    if (!xfer.empty())
        a["lane_transfer_freq_ids"] = xfer;
    if (cp_mod > 0.0)
        a["record_cp_mod_chips"] = cp_mod;
    return a;
}

ChainSpec spec_from_blocks(const std::string& pre, int gpu, const json& adapter,
                           const json& assemble, const json& combine, const json& telem,
                           int samples_per_data_set) {
    ChainSpec s;
    s.pre = pre;
    s.gpu = gpu;
    s.n_prn = adapter.at("n_prn").get<int>();
    s.n_chan = adapter.at("n_gnss_channels").get<int>();
    s.n_live = adapter.at("n_live_elements").get<int>();
    s.num_elements = adapter.at("num_elements").get<int>();
    s.num_synth = adapter.value("num_synth", 128);
    s.lanes_per_slot = adapter.value("lanes_per_slot", (int)gnss_gpu::LANES_FULL);
    s.hops_per_record = adapter.at("hops_per_record").get<int>();
    s.n_rec = adapter.value("samples_per_data_set", samples_per_data_set) / s.hops_per_record;
    s.fft_len = combine.at("fft_len").get<int>();
    s.sample_rate = assemble.at("sample_rate").get<double>();
    s.prns = assemble.at("prns").get<std::vector<int>>();
    s.channel_ids = assemble.at("channel_ids").get<std::vector<int>>();
    s.adapter = adapter;
    s.assemble = assemble;
    s.combine = combine;
    s.telem = telem;
    return s;
}

ChainSpec synth_chain(const std::string& pre, const std::vector<int>& prns,
                      const std::vector<int>& channel_ids, const std::vector<int>& xfer,
                      double cp_mod, int max_prn, const Sky& sky) {
    const int n_elem = 16, n_prn = (int)prns.size(), n_chan = (int)channel_ids.size();
    const json adapter = {{"hops_per_record", 2048},
                          {"lanes_per_slot", gnss_gpu::LANES_PROMPT},
                          {"n_gnss_channels", n_chan},
                          {"n_live_elements", n_elem},
                          {"n_prn", n_prn},
                          {"num_elements", n_elem},
                          {"num_synth", 64},
                          {"samples_per_data_set", 8192}};
    const json combine = {{"n_elements", n_elem},
                          {"n_prn", n_prn},
                          {"fft_len", 16384},
                          {"integration_length", 100},
                          {"integration_mode", "rolling"},
                          {"deep_coherent", true},
                          {"deep_rate_search", true},
                          {"sky_deep", true},
                          {"deep_rate_min_q", 10.0},
                          {"record_export", 128}};
    const json telem = {{"chain", pre.substr(pre.find('_') + 1)},
                        {"inst", "replay.0"},
                        {"chan_export", true},
                        {"channel_ids", channel_ids},
                        {"fft_len", 16384},
                        {"hops_per_record", 2048},
                        {"max_prn", max_prn},
                        {"n_chan", n_chan},
                        {"n_elements", n_elem},
                        {"n_prn", n_prn},
                        {"records_per_frame", 4}};
    return spec_from_blocks(pre, 0, adapter,
                            assemble_block(prns, channel_ids, xfer, cp_mod, sky, n_elem), combine,
                            telem, 8192);
}

Sky make_sky(double amp, int n_elem) {
    Sky sky;
    sky.seed = 20261113;
    sky.amp = amp;
    sky.planted = {1, 2, 4, 5, 6, 9, 11, 14, 20, 25};
    Noise u(sky.seed, "gain", 0, 0, 0);
    for (int e = 0; e < n_elem; ++e)
        sky.gain_e.push_back(std::polar(0.8 + 0.4 * u.uniform(), 2.0 * M_PI * u.uniform()));
    return sky;
}

/// A: 32 PRN x 4 channels, two lane-transfer ids, the live projection. B: 6 PRN x 1 channel.
/// C: 14 PRN x 2 channels with a record_cp_mod_chips; it shares freq_id 6033 and PRN 6 with A.
std::vector<ChainSpec> synthetic_chains(const Sky& sky) {
    std::vector<int> a_prns;
    for (int p = 1; p <= 32; ++p)
        a_prns.push_back(p);
    std::vector<ChainSpec> c = {
        synth_chain("gnss0_syn_a", a_prns, {5985, 6001, 6017, 6033}, {5953, 5969}, 0.0, 32, sky),
        synth_chain("gnss0_syn_b", {2, 5, 7, 8, 11, 13}, {6385}, {}, 0.0, 6, sky),
        synth_chain("gnss0_syn_c", {3, 4, 7, 6, 9, 10, 12, 14, 15, 17, 18, 20, 21, 24},
                    {6033, 6200}, {}, 40920.0, 14, sky)};
    for (size_t i = 0; i < c.size(); ++i)
        c[i].idx = (int)i;
    return c;
}

/// The chains of the GNSS_REPLAY_CONFIG file (scripts/gnss/replay_extract.py).
std::vector<ChainSpec> config_chains(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot read GNSS_REPLAY_CONFIG " + path);
    const json j = json::parse(in);
    const int spd = j.at("root").at("samples_per_data_set").get<int>();
    std::vector<ChainSpec> out;
    for (const auto& c : j.at("chains")) {
        out.push_back(spec_from_blocks(c.at("pre").get<std::string>(), c.at("gpu").get<int>(),
                                       c.at("adapter"), c.at("assemble"), c.at("combine"),
                                       c.at("telem"), spd));
        out.back().idx = (int)out.size() - 1;
    }
    return out;
}

int max_live(const std::vector<ChainSpec>& chains) {
    int n = 0;
    for (const auto& s : chains)
        n = std::max(n, s.n_live);
    return n;
}

const std::vector<ChainSpec>& synthetic() {
    static const std::vector<ChainSpec> c = synthetic_chains(make_sky(SKY_AMP, 16));
    return c;
}

/// The chains the baseline replays: GNSS_REPLAY_CONFIG's, else the synthetic ones.
const std::vector<ChainSpec>& chains() {
    static const std::vector<ChainSpec> c = [] {
        const char* cfg = std::getenv("GNSS_REPLAY_CONFIG");
        return cfg ? config_chains(cfg) : synthetic();
    }();
    return c;
}

/// The production blocks' element cal (elem_sum_tau_s 2 s) warms in ~150 frames, so their
/// scenario runs twice as long, is read at two later checkpoints, and B's reference element moves
/// after the cal checkpoint, as in the synthetic scenario.
Scenario base_scenario() {
    Scenario sc;
    if (std::getenv("GNSS_REPLAY_CONFIG")) {
        sc.n_frames = 240;
        sc.ref_elem_frame = 200;
        sc.checkpoints = {29, 59, 89, 119, 179, 239};
    }
    return sc;
}

Sky sky_for(const Scenario& sc) {
    return make_sky(sc.mut == Mutation::zero_amp ? 0.0 : SKY_AMP, max_live(chains()));
}

std::string assembler_name(const ChainSpec& s, int idx, const Scenario& sc) {
    if (idx == 0 && sc.mut == Mutation::rename_A)
        return "/gnss" + std::to_string(s.gpu) + "_syn_x_n2assemble";
    return "/" + s.pre + "_n2assemble";
}

std::string buf_name(const ChainSpec& s, const std::string& role) {
    return s.pre + "_" + role;
}

/// A combiner without out_buf beside a chain's own (Scenario::cold_ref), fed copies of its
/// records in frames [from, to): the swapped slot's, or for "others" every other slot's.
struct SideCombiner {
    std::string name;
    int from = 0, to = 0;
    std::unique_ptr<Buffer> in;
    GnssCoherentCombiner* cmb = nullptr;
    json rows = json::array(); // cmb_slot_rows of each emit
    int id = 0;                // the REC capture's next frame of `in`
};

/// The side combiners with Scenario::cold_ref ("cold", "scint", then "others"), up to 6 frames
/// after the swap: "cold" from the swap, "scint" from 10 frames before it and "others" from the
/// start.
std::vector<SideCombiner> side_combiners(const Scenario& sc) {
    std::vector<SideCombiner> v(3);
    v[0].name = "cold";
    v[0].from = sc.swap_frame;
    v[1].name = "scint";
    v[1].from = sc.swap_frame - 10;
    v[2].name = "others";
    v[2].from = 0;
    v[0].to = v[1].to = v[2].to = sc.swap_frame + 6;
    return v;
}

/// The host stages of Scenario::rows, for chains on GPU @p gpu.
std::string records_host(int gpu) {
    return "/gnss" + std::to_string(gpu) + "_records";
}
std::string combine_host(int gpu) {
    return "/gnss" + std::to_string(gpu) + "_combine";
}
std::string telem_host(int gpu) {
    return "/gnss" + std::to_string(gpu) + "_telem";
}

/// With Scenario::telem_shared: the GPU half's telemetry buffer.
std::string telem_shared_buf(const std::vector<ChainSpec>& cs) {
    return "gnss" + std::to_string(cs[0].gpu) + "_telem_buf";
}

/// One host block with a `chains:` row per entry of @p blocks (chain prefix -> its one-row
/// block): a key equal in every block and in neither @p row_keys nor @p row_only_keys is set once
/// on the host, every other key in each row.
json host_block(const std::map<std::string, json>& blocks, const std::vector<std::string>& row_keys,
                const std::vector<std::string>& row_only_keys) {
    auto per_row = [&](const std::string& k) {
        return std::find(row_keys.begin(), row_keys.end(), k) != row_keys.end()
               || std::find(row_only_keys.begin(), row_only_keys.end(), k) != row_only_keys.end();
    };
    json host = json::object();
    for (const auto& [k, v] : blocks.begin()->second.items()) {
        bool uniform = !per_row(k);
        for (const auto& [pre, b] : blocks)
            uniform = uniform && b.contains(k) && b.at(k) == v;
        if (uniform)
            host[k] = v;
    }
    for (const auto& [pre, b] : blocks)
        for (const auto& [k, v] : b.items())
            if (!host.contains(k))
                host["chains"][pre][k] = v;
    return host;
}

/// With Scenario::gpu_rows_shared: the GPU half's tiles and control-block buffers.
std::string rows_buf(const std::vector<ChainSpec>& cs, const std::string& role) {
    return "gnss" + std::to_string(cs[0].gpu) + "_rows_" + role;
}

/// With Scenario::gpu_rows: the chains' gnss_rows entries, in chain order, and the channel count of
/// the half's tiles.
json gpu_rows_of(const std::vector<ChainSpec>& cs) {
    json rows = json::array();
    int fg = 0;
    size_t ctl = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const ChainSpec& s = cs[i];
        std::vector<int> chans;
        for (int c = 0; c < s.n_chan; ++c)
            chans.push_back(fg + c);
        const size_t b = gnss_gpu::ctl_bytes(s.n_prn, s.n_chan);
        rows.push_back({{"name", s.pre},
                        {"chain", s.pre.substr(s.pre.find('_') + 1)},
                        {"chain_id", (int)i + 1},
                        {"row", (int)i},
                        {"channels", chans},
                        {"lane_base", 0},
                        {"lanes", s.num_synth},
                        {"fg_offset", fg},
                        {"ctl_offset", ctl},
                        {"ctl_bytes", b},
                        {"n_prn", s.n_prn}});
        fg += s.n_chan;
        ctl += b;
    }
    return rows;
}

int fg_total(const std::vector<ChainSpec>& cs) {
    int fg = 0;
    for (const auto& s : cs)
        fg += s.n_chan;
    return fg;
}

/// With Scenario::rec_shared: the GPU half's record buffer, and its depth, as
/// gen_chord_gnss_config.py's HALF_REC_DEPTH.
constexpr int REC_SHARED_DEPTH = 16;
std::string rec_shared_buf(const std::vector<ChainSpec>& cs) {
    return "gnss" + std::to_string(cs[0].gpu) + "_rec_buf";
}

/// With Scenario::rec_shared: the chains' rec_rows, in chain order, tiling a frame of the half's
/// record buffer, with gpu_rows_of()'s chain ids.
std::vector<gnss::RecRow> rec_rows_of(const std::vector<ChainSpec>& cs) {
    std::vector<gnss::RecRow> rows;
    size_t at = gnss::RecDir::bytes(cs.size());
    for (size_t i = 0; i < cs.size(); ++i) {
        const ChainSpec& s = cs[i];
        at = (at + gnss::REC_ROW_ALIGN - 1) / gnss::REC_ROW_ALIGN * gnss::REC_ROW_ALIGN;
        const int64_t b = gnss::frame_floats(s.n_prn, s.n_live, s.n_chan) * sizeof(float);
        rows.push_back({(int64_t)i + 1, s.n_prn, s.n_chan, (int64_t)at, b});
        at += b;
    }
    return rows;
}

json rec_row_json(const gnss::RecRow& r) {
    return {{"chain_id", r.chain_id},
            {"n_prn", r.n_prn},
            {"n_chan", r.n_chan},
            {"offset", r.offset},
            {"bytes", r.bytes}};
}

/// One kotekan config holding every chain's four blocks, with the harness's buffer names.
json build_config(const std::vector<ChainSpec>& cs, const Scenario& sc,
                  const std::vector<int>& cpus) {
    json cfg;
    cfg["log_level"] = "warn";
    cfg["cpu_affinity"] = cpus;
    cfg["samples_per_data_set"] = cs[0].n_rec * cs[0].hops_per_record;
    std::map<std::string, json> rows_as, rows_cb, rows_tp;
    // The assemblers' rows; the feeder lays the frames out by gpu_rows_of() itself.
    json gpu_rows = gpu_rows_of(cs);
    if (sc.mut == Mutation::ctl_offset_swap_AB)
        std::swap(gpu_rows[0]["ctl_offset"], gpu_rows[1]["ctl_offset"]);
    if (sc.mut == Mutation::fg_offset_swap_AB)
        std::swap(gpu_rows[0]["fg_offset"], gpu_rows[1]["fg_offset"]);
    if (sc.mut == Mutation::row_swap_AB)
        std::swap(gpu_rows[0], gpu_rows[1]);
    for (size_t i = 0; i < cs.size(); ++i) {
        const ChainSpec& s = cs[i];
        json ad = s.adapter, as = s.assemble, cb = s.combine, tp = s.telem;
        if (sc.learn_coh_min > 0.0)
            as["elem_sum_shared_learn_coh_min"] = sc.learn_coh_min;
        if (sc.ref_frame0 > 0)
            as["elem_sum_shared_ref_frame0"] = sc.ref_frame0;
        ad["ctl_buf"] = buf_name(s, "n2ctl_buf");
        ad["tiles_buf"] = buf_name(s, "n2tiles_buf");
        ad["out_buf"] = buf_name(s, "n2epl_buf");
        if (sc.fold) {
            as.erase("in_buf");
            as["ctl_buf"] = buf_name(s, "n2ctl_buf");
            as["tiles_buf"] = buf_name(s, "n2tiles_buf");
            for (const char* k : {"num_elements", "num_synth", "lanes_per_slot", "hops_per_record"})
                if (ad.contains(k))
                    as[k] = ad[k];
            as["fft_len"] = s.fft_len;
            if (sc.gpu_rows_shared) {
                as["ctl_buf"] = rows_buf(cs, "ctl_buf");
                as["tiles_buf"] = rows_buf(cs, "tiles_buf");
            }
            if (sc.gpu_rows) {
                as["gnss_row"] = gpu_rows[i];
                as["fg_total"] = fg_total(cs);
            }
        } else {
            as["in_buf"] = buf_name(s, "n2epl_buf");
        }
        as["out_buf"] = buf_name(s, "n2rec_buf");
        as["cube_buf"] = buf_name(s, "cube_buf");
        cb["in_bufs"] = {buf_name(s, "n2rec_buf")};
        if (sc.combiner_out_buf)
            cb["out_buf"] = buf_name(s, "n2cmb_buf");
        else
            cb.erase("out_buf");
        if (!sc.combiner_prn_reset)
            cb["reset_on_prn_change"] = false;
        tp["in_buf"] = buf_name(s, "n2rec_buf");
        tp["out_buf"] = buf_name(s, "telem_buf");
        for (json* b : {&ad, &as, &cb, &tp})
            (*b)["cpu_affinity"] = cpus;
        if (!sc.fold)
            cfg[s.pre + "_n2assemble_tiles"] = ad;
        if (sc.rows) {
            rows_as[s.pre] = as;
            rows_as[s.pre]["chain"] = tp.at("chain");
            rows_cb[s.pre] = cb;
            rows_cb[s.pre]["chain"] = tp.at("chain");
            rows_tp[s.pre] = tp;
        } else {
            cfg[assembler_name(s, (int)i, sc).substr(1)] = as;
            cfg[s.pre + "_n2combine"] = cb;
            cfg[s.pre + "_telem_pack"] = tp;
        }
        if (!sc.cold_ref)
            continue;
        for (const SideCombiner& x : side_combiners(sc)) {
            json c = cb;
            c.erase("out_buf");
            c["in_bufs"] = {buf_name(s, "n2rec_" + x.name + "_buf")};
            cfg[s.pre + "_n2combine_" + x.name] = c;
        }
    }
    if (sc.rows) {
        json& rec = cfg[records_host(cs[0].gpu).substr(1)];
        rec = host_block(rows_as, GnssGpuRecordAssemble::ROW_KEYS,
                         GnssGpuRecordAssemble::ROW_ONLY_KEYS);
        for (int w = 0; w < sc.workers; ++w) {
            json rows = json::array();
            for (size_t i = w; i < cs.size(); i += sc.workers)
                rows.push_back(cs[i].pre);
            rec["workers"].push_back({{"rows", rows}, {"cpu_affinity", cpus}});
        }
        json& cmb = cfg[combine_host(cs[0].gpu).substr(1)];
        cmb = host_block(rows_cb, GnssCoherentCombiner::ROW_KEYS,
                         GnssCoherentCombiner::ROW_ONLY_KEYS);
        for (int w = 0; w < sc.cmb_workers; ++w) {
            json rows = json::array();
            for (size_t i = w; i < cs.size(); i += sc.cmb_workers)
                rows.push_back(cs[i].pre);
            cmb["workers"].push_back({{"rows", rows}, {"cpu_affinity", cpus}});
        }
        json& tel = cfg[telem_host(cs[0].gpu).substr(1)];
        tel = host_block(rows_tp, GnssTelemPack::ROW_KEYS, {});
        if (sc.telem_shared) {
            for (json& row : tel["chains"])
                row.erase("out_buf");
            tel["out_buf"] = telem_shared_buf(cs);
        }
        for (int w = 0; w < sc.telem_workers; ++w) {
            json rows = json::array();
            for (size_t i = w; i < cs.size(); i += sc.telem_workers)
                rows.push_back(cs[i].pre);
            tel["workers"].push_back({{"rows", rows}, {"cpu_affinity", cpus}});
        }
        // Every row's records in one buffer, which each host names once on its block.
        if (sc.rec_shared) {
            const std::vector<gnss::RecRow> rr = rec_rows_of(cs);
            for (size_t i = 0; i < cs.size(); ++i) {
                const std::string& pre = cs[i].pre;
                rec["chains"][pre].erase("out_buf");
                cmb["chains"][pre].erase("in_bufs");
                tel["chains"][pre].erase("in_buf");
                for (json* h : {&rec, &cmb, &tel})
                    (*h)["chains"][pre]["rec_row"] = rec_row_json(rr[i]);
            }
            rec["out_buf"] = rec_shared_buf(cs);
            cmb["in_bufs"] = {rec_shared_buf(cs)};
            tel["in_buf"] = rec_shared_buf(cs);
            const std::string b = cs[1].pre;
            if (sc.mut == Mutation::rec_chain_id_cmb_B)
                cmb["chains"][b]["rec_row"]["chain_id"] = 99;
            if (sc.mut == Mutation::rec_n_prn_as_B)
                rec["chains"][b]["rec_row"]["n_prn"] = cs[1].n_prn + 1;
            if (sc.mut == Mutation::rec_offset_tel_B)
                tel["chains"][b]["rec_row"]["offset"] = rr[1].offset + 64;
            if (sc.mut == Mutation::rec_rows_tel_AB) {
                const std::vector<gnss::RecRow> ab =
                    rec_rows_of(std::vector<ChainSpec>(cs.begin(), cs.begin() + 2));
                tel["chains"].erase(cs[2].pre);
                for (size_t i = 0; i < 2; ++i)
                    tel["chains"][cs[i].pre]["rec_row"] = rec_row_json(ab[i]);
                for (json& w : tel["workers"])
                    w["rows"].erase(std::remove(w["rows"].begin(), w["rows"].end(), cs[2].pre),
                                    w["rows"].end());
            }
        }
    }
    return cfg;
}

// ---- the child ----

std::atomic<int> g_frame{0};
double g_frame_s = 0.0;
bool g_clock_step = false;

double replay_clock() {
    const int k = g_frame.load();
    return 1000.0 + k * g_frame_s + ((g_clock_step && k >= 50) ? 30.0 : 0.0);
}

std::string g_out_path;

/// FATAL_ERROR -> exit_kotekan -> raise(SIGTERM), in the thread that set the FATAL's message,
/// which the child leaves at its output path. Of several threads that FATAL, the first to get
/// here writes it; the others wait for its exit.
void on_sigterm(int) {
    static std::atomic_flag taken = ATOMIC_FLAG_INIT;
    if (taken.test_and_set())
        for (;;)
            pause();
    std::ofstream(g_out_path) << kotekan::kotekanLogging::get_error_message();
    _exit(70);
}

[[noreturn]] void child_fail(int code, const std::string& why) {
    std::fprintf(stderr, "replay child: %s\n", why.c_str());
    std::fflush(stderr);
    _exit(code);
}

enum Kind { EPL = 0, REC, CMB, TELEM, CUBE, N_KIND };
const char* const KIND[N_KIND] = {"epl", "rec", "cmb", "telem", "cube"};

struct ChainRun {
    const ChainSpec* s = nullptr;
    const std::vector<ChainSpec>* cs = nullptr; ///< every chain, this one included
    int idx = 0;
    const Scenario* sc = nullptr;
    /// The chain's own tiles and control-block buffers, or with Scenario::gpu_rows_shared the
    /// half's.
    Buffer *ctl = nullptr, *tiles = nullptr;
    /// The buffer of its records: its own, or with Scenario::rec_shared the half's.
    Buffer* rec = nullptr;
    std::unique_ptr<Buffer> own_ctl, own_tiles, buf[N_KIND];
    std::vector<std::unique_ptr<kotekan::Stage>> stages;
    GnssCoherentCombiner* cmb = nullptr;
    size_t cmb_row = 0; ///< the chain's row of cmb
    GnssGpuRecordAssemble* as = nullptr;
    size_t as_row = 0; ///< the chain's row of as
    std::vector<SideCombiner> side;
    std::vector<uint64_t> hashes[N_KIND];
    std::atomic<int> count[N_KIND];
    std::thread cap[N_KIND];
    int emit_every = 1;
    // Written by the capture threads before their count moves.
    std::vector<double> pin_epl;
    std::string cube_chain;
    json cube_pow;
    json cmb_rows = json::array(); // cmb_slot_rows of each emit
    int telem_bad = 0;
    int telem_bad_meta = 0;
};

/// One emit's records per slot: [digest of the record with its arc id zeroed, arc id].
json cmb_slot_rows(const ChainSpec& s, const float* f) {
    const size_t stride = gnss::record_stride(s.combine.at("n_elements").get<int>());
    std::vector<float> rec(stride);
    json rows = json::array();
    for (int p = 0; p < s.n_prn; ++p) {
        std::memcpy(rec.data(), f + p * stride, stride * sizeof(float));
        const float arc = rec[gnss::CMB_ARC];
        rec[gnss::CMB_ARC] = 0.0f;
        rows.push_back({fnv1a64(rec.data(), stride * sizeof(float)), arc});
    }
    return rows;
}

uint64_t hash_epl(const ChainRun& R, const uint8_t* f) {
    using namespace gnss_gpu;
    const ChainSpec& s = *R.s;
    const auto* h = (const FrameHdr*)f;
    const size_t rows = (size_t)std::max(0, h->n_jobs), nc = (size_t)std::max(0, h->n_chan);
    uint64_t x = fnv1a64(f, off_corr(s.n_prn));
    x = fnv1a64(f + off_corr(s.n_prn), rows * nc * s.n_live * 2 * sizeof(double), x);
    return fnv1a64(f + off_energy(s.n_prn, (int)nc, ROWS_PLAIN, s.n_live),
                   rows * nc * sizeof(double), x);
}

/// PIN_B2's reading: the prompt, channel 0, element 0, in records 1.. of the dark_r0 frame, of
/// slot dark_r0_slot and then of slot 0; 0 where the slot does not run.
void read_pin_epl(ChainRun& R, const uint8_t* f) {
    using namespace gnss_gpu;
    const ChainSpec& s = *R.s;
    const auto* h = (const FrameHdr*)f;
    const auto* pc = (const PrnCtl*)(f + off_prnctl());
    const double* corr = (const double*)(f + off_corr(s.n_prn));
    for (int slot : {R.sc->dark_r0_slot, 0})
        for (int r = 1; r < h->n_rec; ++r) {
            const PrnCtl& c = pc[(size_t)r * s.n_prn + slot];
            R.pin_epl.push_back(
                c.run ? corr[((size_t)(c.job0 + ROW_P) * h->n_chan + 0) * s.n_live * 2] : 0.0);
        }
}

void read_cube(ChainRun& R, const uint8_t* f, size_t bytes, uint64_t* h) {
    std::vector<uint8_t> c(f, f + bytes);
    char chain[gnss::CUBE_CHAIN_CHARS + 1] = {0};
    std::memcpy(chain, f + gnss::CUBE_CHAIN_OFFSET, gnss::CUBE_CHAIN_CHARS);
    R.cube_chain = chain;
    std::memset(c.data() + gnss::CUBE_CHAIN_OFFSET, 0, gnss::CUBE_CHAIN_CHARS);
    *h = fnv1a64(c.data(), c.size());
    int32_t mp, mb;
    int64_t nb, ne;
    std::memcpy(&nb, f + 48, 8);
    std::memcpy(&ne, f + 56, 8);
    std::memcpy(&mp, f + 88, 4);
    std::memcpy(&mb, f + 92, 4);
    size_t o = gnss::CUBE_HEADER_BYTES + (size_t)mp * 8 + (size_t)mb * 8;
    const int32_t* prn = (const int32_t*)(f + o);
    o += (size_t)mp * 12;
    const float* w = (const float*)(f + o);
    o += (size_t)mp * mb * 8;
    const float* cre = (const float*)(f + o);
    const float* cim = cre + (size_t)mp * mb * ne;
    json pw = json::object();
    for (int p = 0; p < mp; ++p) {
        if (prn[p] == 0)
            continue;
        double sum = 0.0;
        int nbin = 0;
        for (int b = 0; b < nb; ++b) {
            const float wb = w[(size_t)p * mb + b];
            if (!(wb > 0.0f))
                continue;
            double a = 0.0;
            for (int e = 0; e < ne; ++e) {
                const size_t k = ((size_t)p * mb + b) * ne + e;
                a += (double)cre[k] * cre[k] + (double)cim[k] * cim[k];
            }
            sum += a / ((double)wb * wb);
            ++nbin;
        }
        if (nbin > 0)
            pw[std::to_string(prn[p])] = sum / nbin;
    }
    R.cube_pow = pw;
}

/// Digests @p f, the chain's epl frame @p n.
void take_epl(ChainRun& R, const uint8_t* f, int n) {
    // The live projection rewrites chain A's epl frame in place while the capture reads it.
    if (!(R.idx == 0 && n >= R.sc->proj_live_frame))
        R.hashes[EPL].push_back(hash_epl(R, f));
    if (R.idx == 1 && n == R.sc->dark_r0_frame)
        read_pin_epl(R, f);
}

/// Counts telemetry frame @p id of @p b against @p R if its records are not all present or its
/// metadata does not carry its wstart0.
void check_telem(ChainRun& R, Buffer* b, int id) {
    gnss::TelemHeader th;
    std::memcpy(&th, b->frames[id], sizeof(th));
    if (th.present != (1u << R.s->n_rec) - 1u)
        ++R.telem_bad;
    const GnssChanMetadata* m = get_gnss_chan_metadata(b, id);
    if (m == nullptr || m->sample_seq != th.wstart0)
        ++R.telem_bad_meta;
}

/// With Scenario::telem_shared: digests each frame of the half's telemetry buffer @p b, at its
/// own length, as a frame of the chain its header names.
void capture_telem_shared(std::vector<ChainRun*> runs, Buffer* b) {
    const std::string name = "capture_telem";
    int id = 0;
    while (true) {
        const uint8_t* f = b->wait_for_full_frame(name, id);
        if (f == nullptr)
            return;
        gnss::TelemHeader th;
        std::memcpy(&th, f, sizeof(th));
        const std::string chain(th.chain, strnlen(th.chain, gnss::TELEM_NAME));
        ChainRun* R = nullptr;
        for (ChainRun* r : runs)
            if (r->s->telem.at("chain").get<std::string>() == chain)
                R = r;
        if (R == nullptr)
            child_fail(82, "a telemetry frame of chain " + chain + ", which no row sends");
        const int n = R->count[TELEM].load();
        R->hashes[TELEM].push_back(
            fnv1a64(f, std::min(gnss::telem_frame_bytes(th), b->frame_size)));
        check_telem(*R, b, id);
        b->mark_frame_empty(name, id);
        id = (id + 1) % b->num_frames;
        R->count[TELEM].store(n + 1);
    }
}

/// Digests @p f, @p bytes of record @p R->count[REC] of @p R that starts at @p seq, feeds the
/// side combiners their copies, and counts it. Returns false when shutting down.
bool take_rec(ChainRun& R, const uint8_t* f, size_t bytes, int64_t seq) {
    const std::string name = "capture_rec";
    const int n = R.count[REC].load();
    uint64_t h = fnv1a64(f, bytes);
    h = fnv1a64(&seq, sizeof(seq), h);
    for (SideCombiner& c : R.side) {
        if (n < R.s->n_rec * c.from || n >= R.s->n_rec * c.to)
            continue;
        float* x = (float*)c.in->wait_for_empty_frame(name, c.id);
        if (x == nullptr)
            return false;
        // Only the swapped slot's record, or for "others" every other slot's: the rest are not
        // compared, and empty they keep the side combiners' emits cheap.
        const size_t stride = gnss::record_stride(R.s->combine.at("n_elements").get<int>());
        float* rec = x + stride * R.sc->swap_slot;
        if (c.name == "others") {
            std::memcpy(x, f, bytes);
            std::memset(rec, 0, stride * sizeof(float));
        } else {
            std::memset(x, 0, bytes);
            std::memcpy(rec, (const float*)f + stride * R.sc->swap_slot, stride * sizeof(float));
        }
        // A is over the prompt energy: a tenth of it is ten times the amplitude.
        if (c.name == "scint" && n < R.s->n_rec * R.sc->swap_frame && n % 10 == 0)
            rec[gnss::REC_P_ENERGY] *= 0.1f;
        c.in->allocate_new_metadata_object(c.id);
        get_gnss_chan_metadata(c.in.get(), c.id)->sample_seq = seq;
        c.in->mark_frame_full(name, c.id);
        c.id = (c.id + 1) % c.in->num_frames;
    }
    R.hashes[REC].push_back(h);
    R.count[REC].store(n + 1);
    return true;
}

/// With Scenario::rec_shared: takes each frame of the half's record buffer @p b, whose directory
/// @p dir must carry, as take_rec takes each chain's record from it.
void capture_rec_shared(std::vector<ChainRun*> runs, Buffer* b, const gnss::RecDir* dir) {
    const std::string name = "capture_rec";
    for (int id = 0;; id = (id + 1) % b->num_frames) {
        const uint8_t* f = b->wait_for_full_frame(name, id);
        if (f == nullptr)
            return;
        dir->check(name, id, f);
        for (ChainRun* R : runs) {
            const gnss::RecRow& row = dir->row((size_t)R->idx);
            if (!take_rec(*R, f + row.offset, (size_t)row.bytes, dir->wstart(f, (size_t)R->idx)))
                return;
        }
        b->mark_frame_empty(name, id);
    }
}

void capture(ChainRun* R, int kind) {
    Buffer* b = R->buf[kind].get();
    const std::string name = std::string("capture_") + KIND[kind];
    int id = 0;
    while (true) {
        uint8_t* f = b->wait_for_full_frame(name, id);
        if (f == nullptr)
            return;
        const int n = R->count[kind].load();
        uint64_t h = 0;
        if (kind == EPL) {
            take_epl(*R, f, n);
        } else if (kind == REC) {
            const GnssChanMetadata* m = get_gnss_chan_metadata(b, id);
            if (!take_rec(*R, f, b->frame_size, m ? m->sample_seq : -1))
                return;
            b->mark_frame_empty(name, id);
            id = (id + 1) % b->num_frames;
            continue;
        } else if (kind == CUBE) {
            read_cube(*R, f, b->frame_size, &h);
        } else {
            h = fnv1a64(f, b->frame_size);
            if (kind == CMB)
                R->cmb_rows.push_back(cmb_slot_rows(*R->s, (const float*)f));
            if (kind == TELEM)
                check_telem(*R, b, id);
        }
        if (kind != EPL)
            R->hashes[kind].push_back(h);
        b->mark_frame_empty(name, id);
        id = (id + 1) % b->num_frames;
        R->count[kind].store(n + 1);
    }
}

/// Records side combiner @p c has been fed by the end of frame @p k.
int side_records(const ChainRun& R, const SideCombiner& c, int k) {
    return R.s->n_rec * std::max(0, std::min(k + 1, c.to) - c.from);
}

bool sides_quiescent(const ChainRun& R, int k) {
    for (const SideCombiner& c : R.side)
        if (c.in->get_num_full_frames() != 0
            || c.cmb->records_done() != (uint64_t)side_records(R, c, k))
            return false;
    return true;
}

/// Whether @p R's combiner has taken every record of frames 0..k, and its out_buf every emit.
bool combined(ChainRun& R, int k) {
    const int n_rec = R.s->n_rec;
    return R.rec->get_num_full_frames() == 0
           && R.cmb->records_done(R.cmb_row) == (uint64_t)n_rec * (k + 1)
           && (!R.sc->combiner_out_buf || R.count[CMB].load() == n_rec * (k + 1) / R.emit_every);
}

/// With Scenario::cmb_workers a worker takes a record of each of its rows in turn, so a chain's
/// records are combined only once every chain of the frame is fed: then its combiner is waited
/// for, and read, after the frame (combined_after). Scenario::telem_workers' telemetry likewise,
/// and the combiners with it, as a record frame is released only once its telemetry row took it.
/// With Scenario::rec_shared a chain's records reach their readers only once every chain has
/// written its own, so only its assembler is waited for here, and its readers after the frame
/// (records_read).
bool quiescent(ChainRun& R, int k) {
    const int n_rec = R.s->n_rec;
    const bool epl = R.sc->fold || (R.buf[EPL]->is_frame_empty(0) && R.count[EPL].load() == k + 1);
    if (R.sc->rec_shared)
        return R.ctl->is_frame_empty(0) && R.tiles->is_frame_empty(0) && epl
               && R.buf[CUBE]->get_num_full_frames() == 0
               && R.as->records_done(R.as_row) == (uint64_t)n_rec * (k + 1);
    return R.ctl->is_frame_empty(0) && R.tiles->is_frame_empty(0) && epl
           && R.buf[CUBE]->get_num_full_frames() == 0 && R.count[REC].load() == n_rec * (k + 1)
           && (R.sc->telem_workers > 0 || R.count[TELEM].load() == k + 1)
           && (R.sc->cmb_workers > 0 || R.sc->telem_workers > 0 || combined(R, k))
           && sides_quiescent(R, k);
}

/// Whether a combiner without out_buf, fed @p records by the end of frame @p k and of whose
/// emits @p done were read, made one in frame @p k. Its scratch holds only the last emit.
bool scratch_emit(const ChainRun& R, int records, int done, int k) {
    const int want = records / R.emit_every;
    if (want > done + 1)
        child_fail(78, fmt::format("{:s}: {:d} combiner emits in frame {:d}; the scratch holds one",
                                   R.s->pre, want - done, k));
    return want > done;
}

/// Without out_buf, digests the combiner's emit of frame @p k, if it made one, from its scratch.
void capture_scratch(ChainRun& R, int k) {
    const int n = R.count[CMB].load();
    if (!scratch_emit(R, R.s->n_rec * (k + 1), n, k))
        return;
    const std::vector<float>& f = R.cmb->scratch(R.cmb_row);
    R.hashes[CMB].push_back(fnv1a64(f.data(), f.size() * sizeof(float)));
    R.cmb_rows.push_back(cmb_slot_rows(*R.s, f.data()));
    R.count[CMB].store(n + 1);
}

/// With Scenario::fold: a frame released is the GPU's to overwrite, so the feeder fills each with
/// 0xA5 as soon as the assembler releases it, whichever comes first. An assembler that reads one
/// after releasing it reads this instead.
void refill_released(ChainRun& R, int k) {
    Buffer* const bufs[2] = {R.ctl, R.tiles};
    bool done[2] = {false, false};
    const auto t0 = std::chrono::steady_clock::now();
    while (!done[0] || !done[1]) {
        for (int i = 0; i < 2; ++i)
            if (!done[i] && bufs[i]->is_frame_empty(0)) {
                std::memset(bufs[i]->frames[0], 0xA5, bufs[i]->frame_size);
                done[i] = true;
            }
        std::this_thread::yield();
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30))
            child_fail(
                71, fmt::format("{:s}: tiles or ctl frame {:d} not released in 30 s", R.s->pre, k));
    }
}

/// With Scenario::fold, fails unless the released tiles and ctl frames still hold refill_released's
/// 0xA5: in production cudaOutputData's next frame lands there.
void check_released(const ChainRun& R, int k) {
    for (const Buffer* b : {R.ctl, R.tiles})
        for (size_t j = 0; j < b->frame_size; ++j)
            if (b->frames[0][j] != 0xA5)
                child_fail(81, fmt::format("{:s}: {:s} byte {:d} written after frame {:d} was "
                                           "released",
                                           R.s->pre, b->buffer_name, j, k));
}

/// With Scenario::fold, digests the assembler's own epl frame of frame @p k.
void capture_fold(ChainRun& R, int k) {
    take_epl(R, R.as->epl_frame(R.as_row), k);
    R.count[EPL].store(k + 1);
}

void capture_side(ChainRun& R, int k) {
    for (SideCombiner& c : R.side)
        if (scratch_emit(R, side_records(R, c, k), (int)c.rows.size(), k))
            c.rows.push_back(cmb_slot_rows(*R.s, c.cmb->scratch().data()));
}

void wait_quiescent(ChainRun& R, int k) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!quiescent(R, k)) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30))
            child_fail(71, fmt::format("{:s} not quiescent 30 s after frame {:d}: epl {:d} rec "
                                       "{:d} combined {:d} cmb {:d} telem {:d}",
                                       R.s->pre, k, R.count[EPL].load(), R.count[REC].load(),
                                       R.cmb->records_done(R.cmb_row), R.count[CMB].load(),
                                       R.count[TELEM].load()));
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

void wait_combined(ChainRun& R, int k) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!combined(R, k)) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30))
            child_fail(71, fmt::format("{:s} not combined 30 s after frame {:d}: rec {:d} combined "
                                       "{:d} cmb {:d}",
                                       R.s->pre, k, R.count[REC].load(),
                                       R.cmb->records_done(R.cmb_row), R.count[CMB].load()));
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

/// With Scenario::rec_shared, waits until @p R's records of frames 0..k are digested, in its side
/// combiners and, unless wait_telem waits for them, packed.
void wait_records_read(ChainRun& R, int k) {
    const auto t0 = std::chrono::steady_clock::now();
    while (R.count[REC].load() != R.s->n_rec * (k + 1) || !sides_quiescent(R, k)
           || (R.sc->telem_workers == 0 && R.count[TELEM].load() != k + 1)) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30))
            child_fail(71, fmt::format("{:s}: records of frame {:d} not read in 30 s: rec {:d} "
                                       "telem {:d}",
                                       R.s->pre, k, R.count[REC].load(), R.count[TELEM].load()));
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

/// With Scenario::telem_workers, waits for @p R's telemetry frame of frame @p k.
void wait_telem(ChainRun& R, int k) {
    const auto t0 = std::chrono::steady_clock::now();
    while (R.count[TELEM].load() != k + 1) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30))
            child_fail(71, fmt::format("{:s}: telemetry frame {:d} not in 30 s ({:d} so far)",
                                       R.s->pre, k, R.count[TELEM].load()));
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

void strip_timing(json& j) {
    if (j.is_object()) {
        j.erase("us_per_record");
        for (auto& v : j)
            strip_timing(v);
    } else if (j.is_array()) {
        for (auto& v : j)
            strip_timing(v);
    }
}

json rest(unsigned short port, const std::string& path, const json& body) {
    const auto r =
        restClient::instance().make_request_blocking(path, body, "127.0.0.1", port, 0, 10);
    if (!r.first)
        return json{{"_error", r.second}};
    if (r.second.empty()) // send_empty_reply
        return json();
    json j = json::parse(r.second, nullptr, false);
    if (j.is_discarded())
        return json{{"_error", "unparsable reply"}};
    strip_timing(j);
    return j;
}

void post(unsigned short port, const std::string& path, const json& body) {
    const json r = rest(port, path, body);
    if (r.is_object() && r.contains("_error"))
        child_fail(72, "POST " + path + ": " + r["_error"].dump());
}

/// The chain's broker chain, which its rows name in the hosts.
std::string chain_of(const ChainSpec& s) {
    return s.telem.at("chain").get<std::string>();
}

/// POST @p body to @p verb of chain @p R's assembler @p u, at the record host by chain with
/// Scenario::by_chain.
void post_assembler(const ChainRun& R, unsigned short port, const std::string& u,
                    const std::string& verb, const json& body) {
    if (R.sc->by_chain)
        post(port, records_host(R.s->gpu) + "/" + verb,
             {{"chain", chain_of(*R.s)}, {"body", body}});
    else
        post(port, u + "/" + verb, body);
}

/// The path of GET @p verb of chain @p R's assembler @p u, or with @p combiner of its combiner:
/// at the host by chain with Scenario::by_chain.
std::string get_path(const ChainRun& R, const std::string& u, const std::string& verb,
                     bool combiner) {
    if (R.sc->by_chain)
        return (combiner ? combine_host(R.s->gpu) : records_host(R.s->gpu)) + "/" + verb
               + "?chain=" + chain_of(*R.s);
    return (combiner ? "/" + R.s->pre + "_n2combine" : u) + "/" + verb;
}

/// The scenario's POSTs to chain @p R after frame @p k.
void fire_posts(ChainRun& R, const Sky& sky, unsigned short port, const std::string& u, int k) {
    const ChainSpec& s = *R.s;
    const Scenario& sc = *R.sc;
    const double utc = record_utc(s, k, 0);
    if (k % sc.geom_every == 0) {
        json g = {{"_bore", {10.0, utc}}};
        for (int p = 0; p < s.n_prn; ++p) {
            double az, el;
            const int prn = prn_at(s, sc, k, p);
            azel(prn, &az, &el);
            g[std::to_string(prn)] = {az, el, 0.0, 0.0, utc};
        }
        post_assembler(R, port, u, "set_sat_geometry", g);
    }
    if (k == sc.gain_frame && s.assemble.value("elem_sum", false)) {
        json gain = json::array();
        for (int e = 0; e < s.assemble.at("n_elements").get<int>(); ++e) {
            const std::complex<double> g =
                sky.gain_e[(size_t)e % sky.gain_e.size()] * std::polar(1.0 + 0.02 * e, -0.2);
            gain.push_back({g.real(), g.imag()});
        }
        post_assembler(R, port, u, "set_elem_gain", {{"gain", gain}});
    }
    if (k == sc.shared_ref_frame) {
        json ref = json::array();
        for (int e = 0; e < s.n_live; ++e) {
            const std::complex<double> g = sky.gain_e[(size_t)e] * std::polar(1.0, 0.3);
            ref.push_back({g.real(), g.imag()});
        }
        json body = {{"ref", ref}, {"mode", "live"}, {"slew_deg_s", 5.0}};
        if (sc.ref_frame0 > 0)
            body["frame0"] = sc.ref_frame0;
        post_assembler(R, port, u, "set_elem_sum_shared_ref", body);
    }
    if (R.idx == 0 && k == sc.proj_live_frame)
        post_assembler(R, port, u, "set_elem_proj", {{"mode", "live"}});
    if (R.idx == 1 && k == sc.ref_elem_frame)
        post_assembler(R, port, u, "set_reference_element", {{"element", 1}});
}

uint64_t digest(const json& j) {
    const std::string b = j.dump();
    return fnv1a64(b.data(), b.size());
}

/// Per-slot digests of combiner GET @p verb's reply @p rows after frame @p k, into @p out:
/// "status@k" as [digest of the row without adr_arc, prn, adr_arc, s4_raw], "elements@k" as
/// [digest, prn, keff] and "records@k" as {prn: digest}.
void slot_digests(const std::string& verb, const json& rows, int k, json& out) {
    const std::string cp = "@" + std::to_string(k);
    if (!rows.is_array())
        child_fail(79, verb + cp + ": " + rows.dump());
    if (verb == "get_status") {
        json st = json::array();
        for (json r : rows) {
            const json arc = r.at("adr_arc");
            r.erase("adr_arc");
            st.push_back({digest(r), r.at("prn"), arc, r.at("s4_raw")});
        }
        out["status" + cp] = st;
    } else if (verb == "get_elements") {
        json el = json::array();
        for (const json& r : rows)
            el.push_back({digest(r), r.at("prn"), r.at("keff")});
        out["elements" + cp] = el;
    } else if (verb == "get_records") {
        json rc = json::object();
        for (const json& r : rows)
            rc[std::to_string(r.at("prn").get<int>())] = digest(r);
        out["records" + cp] = rc;
    }
}

/// GET @p verb of chain @p R's assembler @p u, or with @p combiner of its combiner, after frame
/// @p k, digested, and the parts the checks read kept in @p live.
void take_get(ChainRun& R, unsigned short port, const std::string& u, const std::string& verb,
              bool combiner, int k, Digests& d) {
    const std::string cp = "@" + std::to_string(k);
    const json j = rest(port, get_path(R, u, verb, combiner), json{});
    const std::string body = j.dump();
    d.streams[R.s->pre + ".rest." + verb + cp] = {fnv1a64(body.data(), body.size())};
    // The combiner's replies also give the per-slot digests at the checkpoint.
    if (R.sc->mut == Mutation::none && combiner)
        slot_digests(verb, j, k, d.slots[R.s->pre]["own"]);
    json& L = d.live[R.s->pre];
    if (verb == "get_elem_cal" && j.contains("prns")) {
        json rows = json::array();
        for (const auto& r : j["prns"])
            rows.push_back({r["prn"], r["warm"], r["shadow_warm"]});
        L["elem_cal" + cp] = rows;
        json held = json::array();
        for (const auto& r : j["prns"])
            held.push_back({r["prn"], r["shadow_warm"], r.value("held", true)});
        L["held" + cp] = held;
        L["shared" + cp] = json::array({j["shared_warm"], j["shared_n"]});
        if (j.contains("fleet_ref"))
            L["frame0_current" + cp] = j["fleet_ref"].value("frame0_current", false);
        if (j.contains("proj"))
            L["proj" + cp] = json::array({j["proj"]["mode"], j["proj"]["k"]});
    } else if (verb == "get_status" && j.is_array()) {
        json rows = json::array();
        for (const auto& r : j)
            rows.push_back({r["prn"], r["amplitude"], r["adr_arc"], r["coh_amplitude"],
                            r["deep_records"], r["adr_records"]});
        L["status" + cp] = rows;
    } else if (verb == "get_records" && j.is_array()) {
        json rows = json::object();
        for (const auto& r : j) {
            double lo = 1e300;
            for (const auto& x : r["records"])
                lo = std::min(lo, x[0].get<double>());
            rows[std::to_string(r["prn"].get<int>())] = lo;
        }
        L["records_min_hop" + cp] = rows;
    }
}

/// Every GET of the chain's combiner, as take_get.
void fire_combiner_gets(ChainRun& R, unsigned short port, int k, Digests& d) {
    for (const char* v : {"get_status", "get_records", "get_elements"})
        take_get(R, port, "", v, true, k, d);
}

/// Every GET of the chain's assembler @p u, and with @p combiner its combiner's, as take_get.
void fire_gets(ChainRun& R, unsigned short port, const std::string& u, int k, Digests& d,
               bool combiner) {
    for (const char* v : {"get_elem_cal", "get_spectrum", "get_lane_inst"})
        take_get(R, port, u, v, false, k, d);
    if (combiner)
        fire_combiner_gets(R, port, k, d);
}

/// Frames after which the combiners' GETs are read per slot: the first four after the start and
/// after the swap, which hold the first emit of each, the last before the swap, and the
/// checkpoints.
bool slot_frame(const Scenario& sc, int k) {
    return k < 4 || (k >= sc.swap_frame - 1 && k < sc.swap_frame + 4)
           || std::find(sc.checkpoints.begin(), sc.checkpoints.end(), k) != sc.checkpoints.end();
}

/// The GETs of combiner @p name, or with @p chain of that row of combine host @p name, after
/// frame @p k, digested per slot into @p out, get_records only with @p records.
void read_slot_gets(unsigned short port, const std::string& name, int k, bool records, json& out,
                    const std::string& chain = "") {
    for (const std::string verb : {"get_status", "get_elements", "get_records"})
        if (records || verb != "get_records")
            slot_digests(
                verb,
                rest(port, name + "/" + verb + (chain.empty() ? "" : "?chain=" + chain), json{}), k,
                out);
}

/// The combiner GETs of chain @p R after frame @p k, digested per slot, as read_slot_gets: at its
/// combine host by chain with Scenario::by_chain.
void read_own_slot_gets(const ChainRun& R, unsigned short port, int k, json& out) {
    const bool records = k >= R.sc->swap_frame;
    if (R.sc->by_chain)
        read_slot_gets(port, combine_host(R.s->gpu), k, records, out, chain_of(*R.s));
    else
        read_slot_gets(port, "/" + R.s->pre + "_n2combine", k, records, out);
}

/// A GET's reply as it came: whether it was a 200 and its bytes.
std::pair<bool, std::string> get_raw(unsigned short port, const std::string& path) {
    return restClient::instance().make_request_blocking(path, json{}, "127.0.0.1", port, 0, 10);
}

/// With Scenario::rows, after frame @p k: every GET of every chain's assembler and combiner at
/// its host by chain is byte for byte the GET at the chain's own path (a refused one refused
/// there too), and without chain the host's reply holds each chain's under its chain.
void check_by_chain(const std::vector<std::unique_ptr<ChainRun>>& runs, unsigned short port,
                    int k) {
    const int gpu = runs[0]->s->gpu;
    for (const bool combiner : {false, true}) {
        const std::vector<std::string> verbs =
            combiner ? std::vector<std::string>{"get_status", "get_records", "get_elements"}
                     : std::vector<std::string>{"get_elem_cal", "get_spectrum", "get_lane_inst"};
        const std::string host = combiner ? combine_host(gpu) : records_host(gpu);
        for (const auto& v : verbs) {
            json want = json::object();
            for (const auto& R : runs) {
                const std::string own = (combiner ? "/" + R->s->pre + "_n2combine"
                                                  : assembler_name(*R->s, R->idx, *R->sc))
                                        + "/" + v;
                const auto a = get_raw(port, own);
                const auto b = get_raw(port, host + "/" + v + "?chain=" + chain_of(*R->s));
                if (a != b)
                    child_fail(82, fmt::format("after frame {:d}: {:s} {:s} {:.200s}, by chain "
                                               "{:s} {:.200s}",
                                               k, own, a.first ? "200" : "refused", a.second,
                                               b.first ? "200" : "refused", b.second));
                if (a.first)
                    want[chain_of(*R->s)] = json::parse(a.second);
            }
            const auto all = get_raw(port, host + "/" + v);
            if (!all.first || json::parse(all.second) != want)
                child_fail(82, fmt::format("after frame {:d}: {:s}/{:s} without chain {:.200s}", k,
                                           host, v, all.second));
        }
    }
}

/// The per-slot GETs of @p R's side combiners after frame @p k, which only the unmutated runs'
/// checks read: from the frame of the chain's first emit since the swap, and "scint" also at the
/// frame before it (its raw S4); "others" from the frame before the swap. get_records is the
/// costly one, so it is read only on "cold", where the swap checks compare it.
void read_side_slots(const ChainRun& R, unsigned short port, int k, Digests& d) {
    const ChainSpec& s = *R.s;
    const Scenario& sc = *R.sc;
    const int first = (s.n_rec * sc.swap_frame / R.emit_every + 1) * R.emit_every;
    for (const SideCombiner& x : R.side)
        if (k < x.to
            && (x.name == "others"
                    ? k >= sc.swap_frame - 1
                    : k >= (first - 1) / s.n_rec || (x.name == "scint" && k == sc.swap_frame - 1)))
            read_slot_gets(port, "/" + s.pre + "_n2combine_" + x.name, k, x.name == "cold",
                           d.slots[s.pre][x.name]);
}

void snapshot_board(int k, Digests& d) {
    json rows = json::array(), summary = json::array();
    for (const auto& [key, e] : gnss::ProjBoard::instance().snapshot()) {
        json q = json::array();
        for (const auto& v : e.q)
            q.push_back({v.real(), v.imag()});
        rows.push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key), e.owner, e.src, e.n,
                        e.k, e.t_pub, e.power, e.coh, e.held_s, e.t_utc, e.xfer, e.los, e.los_rate,
                        q});
        summary.push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key), e.src, e.coh});
    }
    const std::string body = rows.dump();
    d.streams["board@" + std::to_string(k)] = {fnv1a64(body.data(), body.size())};
    d.live["board@" + std::to_string(k)] = summary;
}

void apply_mutation(const Scenario& sc, int idx, int frame, const ChainSpec& s, uint8_t* ctl,
                    int32_t* tiles) {
    using namespace gnss_gpu;
    auto* pc = (PrnCtl*)(ctl + off_prnctl());
    if (sc.mut == Mutation::rot_sign_A10 && idx == 0 && frame == 10)
        for (int r = 0; r < s.n_rec; ++r)
            for (float& x : pc[(size_t)r * s.n_prn + 1].rot)
                x = -x;
    if (sc.mut == Mutation::job0_plus4_C12 && idx == 2 && frame == 12)
        for (int r = 0; r < s.n_rec; ++r)
            if (pc[(size_t)r * s.n_prn + 2].run)
                pc[(size_t)r * s.n_prn + 2].job0 += ROWS_PLAIN;
    if (sc.mut == Mutation::tile_word_B10 && idx == 1 && frame == 10)
        tiles[tile_index(s, 0, 0, lane_of_row(s.lanes_per_slot, ROW_P), 0)] ^= (1 << 20);
    auto* hdr = (FrameHdr*)ctl;
    if (idx == 1 && frame == 3) {
        hdr->n_prn += sc.mut == Mutation::n_prn_B3;
        hdr->n_chan += sc.mut == Mutation::n_chan_B3;
        if (sc.mut == Mutation::lanes_B3)
            hdr->lanes_per_slot = s.lanes_per_slot == LANES_FULL ? LANES_PROMPT : LANES_FULL;
    }
}

std::vector<int> allowed_cpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    sched_getaffinity(0, sizeof(set), &set);
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set))
            cpus.push_back(c);
    return cpus;
}

std::string short_hostname() {
    char hn[64] = {0};
    gethostname(hn, sizeof(hn) - 1);
    char* dot = std::strchr(hn, '.');
    if (dot)
        *dot = '\0';
    return hn;
}

/// Writes frame @p k into @p R's tiles and control-block buffers and marks them full: the chain's
/// own frames, or with Scenario::gpu_rows the whole GPU half's, every chain at its row of
/// @p layout (with gpu_rows_shared, the half's one pair of buffers that every chain reads).
void feed(const std::vector<std::unique_ptr<ChainRun>>& runs, ChainRun& R, const json& layout,
          const Sky& sky, const Scenario& sc, int k, Digests& d) {
    uint8_t* ctl_f = R.ctl->wait_for_empty_frame("feeder", 0);
    int32_t* tiles_f = (int32_t*)R.tiles->wait_for_empty_frame("feeder", 0);
    if (ctl_f == nullptr || tiles_f == nullptr)
        child_fail(74, "feeder: buffer shut down");
    int64_t hop0 = 0;
    for (auto& W : runs) {
        if (!sc.gpu_rows && W.get() != &R)
            continue;
        const ChainSpec& s = *W->s;
        const size_t ctl_b = gnss_gpu::ctl_bytes(s.n_prn, s.n_chan);
        const size_t tiles_n = (size_t)s.n_rec * s.n_chan * n_tile(s) * 512;
        // A row's frames are written as the chain's own, then copied to its place in the half's.
        std::vector<uint8_t> own_ctl(sc.gpu_rows ? ctl_b : 0);
        std::vector<int32_t> own_tiles(sc.gpu_rows ? tiles_n : 0);
        uint8_t* ctl = sc.gpu_rows ? own_ctl.data() : ctl_f;
        int32_t* tiles = sc.gpu_rows ? own_tiles.data() : tiles_f;
        write_ctl(ctl, ctl_b, s, sky, sc, k);
        write_tiles(tiles, tiles_n, s, sky, sc, k);
        apply_mutation(sc, W->idx, k, s, ctl, tiles);
        if (W->idx == 1 && k == sc.dark_r0_frame) {
            json t = json::array();
            for (int slot : {sc.dark_r0_slot, 0}) {
                const int lane = s.lanes_per_slot * slot
                                 + gnss_gpu::lane_of_row(s.lanes_per_slot, gnss_gpu::ROW_P);
                for (int r = 1; r < s.n_rec; ++r)
                    t.push_back(tiles[tile_index(s, r, 0, lane, 0)]);
            }
            d.pins["B2_tiles"] = t;
        }
        hop0 = ((const gnss_gpu::FrameHdr*)ctl)->seq0 / s.fft_len
               + (sc.mut == Mutation::seq0_skew_B3 && W->idx == 1 && k >= 3);
        if (!sc.gpu_rows)
            continue;
        const json& row = layout.at(W->idx);
        const int fg_offset = row.at("fg_offset").get<int>();
        const int bad = sc.mut == Mutation::row_tag_B3 && W->idx == 1 && k >= 3;
        ((gnss_gpu::FrameHdr*)ctl)->_pad1 = gnss_gpu::row_tag(
            row.at("chain_id").get<int>(), row.at("row").get<int>() + bad, fg_offset);
        std::memcpy(ctl_f + row.at("ctl_offset").get<size_t>(), ctl, ctl_b);
        const size_t slice = (size_t)s.n_chan * n_tile(s) * 512;
        const int fg = fg_total(*W->cs);
        for (int r = 0; r < s.n_rec; ++r)
            std::memcpy(tiles_f + ((size_t)r * fg + fg_offset) * n_tile(s) * 512, tiles + r * slice,
                        slice * sizeof(int32_t));
    }
    R.tiles->allocate_new_metadata_object(0);
    get_chord_metadata(R.tiles, 0)->set_fpga_seq_num(hop0);
    R.ctl->mark_frame_full("feeder", 0);
    R.tiles->mark_frame_full("feeder", 0);
}

/// One band-power pass before any frame: chan_noise_norm weights from the first record.
void publish_band_power(const std::vector<ChainSpec>& cs) {
    std::vector<int> fids;
    for (const auto& s : cs)
        for (int f : s.channel_ids)
            if (std::find(fids.begin(), fids.end(), f) == fids.end())
                fids.push_back(f);
    const int ne = max_live(cs);
    std::vector<double> power;
    for (int f : fids)
        for (int e = 0; e < ne; ++e)
            power.push_back(1.0 + 0.001 * (f % 97) + 0.01 * e);
    gnss::ChanPower::instance().publish(fids, power, ne);
}

int child_main(const std::vector<ChainSpec>& cs, const Scenario& sc, const std::string& out_path) {
    // Boost.Test's handlers would report a crash from inside the child and carry on; the parent
    // reports the signal instead.
    for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGALRM})
        std::signal(sig, SIG_DFL);
    g_out_path = out_path;
    std::signal(SIGTERM, on_sigterm);
    alarm(300);
    if (sc.cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(sc.cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            child_fail(73, "sched_setaffinity failed");
    }
    const std::vector<int> cpus = allowed_cpus();
    kotekan::restServer::instance().start("127.0.0.1", 0);
    const unsigned short port = kotekan::restServer::instance().port();
    g_frame_s = (double)cs[0].n_rec * cs[0].hops_per_record * cs[0].fft_len / cs[0].sample_rate;
    g_clock_step = sc.mut == Mutation::clock_step_50;
    const Sky sky = sky_for(sc);

    auto pool = metadataPool::create(64 * (int)cs.size(), 64, "pool", "GnssChanMetadata");
    // The tiles carry the correlator's metadata, as cudaOutputData forwards it.
    auto tiles_pool = metadataPool::create(2 * (int)cs.size(), sizeof(chordMetadata), "tiles_pool",
                                           "chordMetadata");
    kotekan::bufferContainer bc;
    // With Scenario::gpu_rows every tiles and control-block frame is a whole GPU half's: every
    // chain's tiles on one channel axis and every control block, in chain order (gpu_rows_of).
    // With gpu_rows_shared one pair of buffers holds them for every chain.
    const json layout = gpu_rows_of(cs);
    const size_t rows_ctl_b =
        layout.back().at("ctl_offset").get<size_t>() + layout.back().at("ctl_bytes").get<size_t>();
    const size_t rows_tiles_b =
        (size_t)cs[0].n_rec * fg_total(cs) * n_tile(cs[0]) * 512 * sizeof(int32_t);
    std::unique_ptr<Buffer> rows_ctl, rows_tiles;
    if (sc.gpu_rows_shared) {
        const size_t ctl_b = rows_ctl_b, tiles_b = rows_tiles_b;
        rows_ctl = std::make_unique<Buffer>(1, ctl_b, pool, rows_buf(cs, "ctl_buf"), "standard", 0,
                                            false, false, std::vector<int>{}, true, sc.prefill);
        rows_tiles =
            std::make_unique<Buffer>(1, tiles_b, tiles_pool, rows_buf(cs, "tiles_buf"), "standard",
                                     0, false, false, std::vector<int>{}, true, sc.prefill);
        bc.add_buffer(rows_buf(cs, "ctl_buf"), rows_ctl.get());
        bc.add_buffer(rows_buf(cs, "tiles_buf"), rows_tiles.get());
        rows_ctl->register_producer("feeder");
        rows_tiles->register_producer("feeder");
    }
    std::vector<std::unique_ptr<ChainRun>> runs;
    for (size_t i = 0; i < cs.size(); ++i) {
        const ChainSpec& s = cs[i];
        auto R = std::make_unique<ChainRun>();
        R->s = &s;
        R->cs = &cs;
        R->idx = (int)i;
        R->sc = &sc;
        const size_t ctl_b = sc.gpu_rows ? rows_ctl_b : gnss_gpu::ctl_bytes(s.n_prn, s.n_chan);
        const size_t tiles_b = sc.gpu_rows
                                   ? rows_tiles_b
                                   : (size_t)s.n_rec * s.n_chan * n_tile(s) * 512 * sizeof(int32_t);
        const int max_prn = s.telem.at("max_prn").get<int>();
        const size_t sizes[N_KIND] = {
            gnss_gpu::frame_bytes(s.n_prn, s.n_chan, gnss_gpu::ROWS_PLAIN, s.n_live),
            gnss::frame_floats(s.n_prn, s.n_live, s.n_chan) * sizeof(float),
            (size_t)s.n_prn * gnss::record_stride(s.n_live) * sizeof(float),
            gnss::telem_frame_bytes(s.n_rec, max_prn, s.n_chan),
            gnss::cube_frame_bytes(s.assemble.value("beam_cube_max_prn", s.n_prn),
                                   s.assemble.value("beam_cube_max_bins", s.n_chan), s.n_live)};
        const char* role[N_KIND] = {"n2epl_buf", "n2rec_buf", "n2cmb_buf", "telem_buf", "cube_buf"};
        R->own_ctl =
            std::make_unique<Buffer>(1, ctl_b, pool, buf_name(s, "n2ctl_buf"), "standard", 0, false,
                                     false, std::vector<int>{}, true, sc.prefill);
        R->own_tiles =
            std::make_unique<Buffer>(1, tiles_b, tiles_pool, buf_name(s, "n2tiles_buf"), "standard",
                                     0, false, false, std::vector<int>{}, true, sc.prefill);
        bc.add_buffer(buf_name(s, "n2ctl_buf"), R->own_ctl.get());
        bc.add_buffer(buf_name(s, "n2tiles_buf"), R->own_tiles.get());
        R->own_ctl->register_producer("feeder");
        R->own_tiles->register_producer("feeder");
        R->ctl = sc.gpu_rows_shared ? rows_ctl.get() : R->own_ctl.get();
        R->tiles = sc.gpu_rows_shared ? rows_tiles.get() : R->own_tiles.get();
        for (int k = 0; k < N_KIND; ++k) {
            R->buf[k] = std::make_unique<Buffer>(k == EPL ? 1 : 4, sizes[k], pool,
                                                 buf_name(s, role[k]), "standard", 0, false, false,
                                                 std::vector<int>{}, true, sc.prefill);
            bc.add_buffer(buf_name(s, role[k]), R->buf[k].get());
            R->buf[k]->register_consumer(std::string("capture_") + KIND[k]);
            R->count[k].store(0);
        }
        if (sc.cold_ref)
            R->side = side_combiners(sc);
        for (SideCombiner& c : R->side) {
            const std::string bn = buf_name(s, "n2rec_" + c.name + "_buf");
            c.in = std::make_unique<Buffer>(4, sizes[REC], pool, bn, "standard", 0, false, false,
                                            std::vector<int>{}, true, sc.prefill);
            bc.add_buffer(bn, c.in.get());
            c.in->register_producer("capture_rec");
        }
        R->emit_every = std::max(1, s.combine.value("integration_length", 1) / 10);
        runs.push_back(std::move(R));
    }

    // With Scenario::telem_shared: the half's telemetry buffer, sized for the widest row.
    std::unique_ptr<Buffer> telem_shared;
    if (sc.telem_shared) {
        size_t bytes = 0;
        for (const ChainSpec& s : cs)
            bytes = std::max(bytes, gnss::telem_frame_bytes(
                                        s.n_rec, s.telem.at("max_prn").get<int>(), s.n_chan));
        telem_shared = std::make_unique<Buffer>(4 * (int)cs.size(), bytes, pool,
                                                telem_shared_buf(cs), "standard", 0, false, false,
                                                std::vector<int>{}, true, sc.prefill);
        bc.add_buffer(telem_shared_buf(cs), telem_shared.get());
        telem_shared->register_consumer("capture_telem");
    }
    // With Scenario::rec_shared: the half's record buffer, as deep as a node's, and the directory
    // its frames must carry.
    std::unique_ptr<Buffer> rec_half;
    std::unique_ptr<gnss::RecDir> rec_dir;
    if (sc.rec_shared) {
        const std::vector<gnss::RecRow> rr = rec_rows_of(cs);
        rec_half = std::make_unique<Buffer>(
            REC_SHARED_DEPTH, (size_t)(rr.back().offset + rr.back().bytes), nullptr,
            rec_shared_buf(cs), "standard", 0, false, false, std::vector<int>{}, true, sc.prefill);
        bc.add_buffer(rec_shared_buf(cs), rec_half.get());
        rec_half->register_consumer("capture_rec");
        rec_dir = std::make_unique<gnss::RecDir>("capture_rec", rr, rec_half.get());
    }
    for (auto& R : runs)
        R->rec = sc.rec_shared ? rec_half.get() : R->buf[REC].get();

    kotekan::Config config;
    config.update_config(build_config(cs, sc, cpus));
    std::vector<std::unique_ptr<kotekan::Stage>> hosts;
    GnssCoherentCombiner* cmb_host = nullptr;
    GnssGpuRecordAssemble* as_host = nullptr;
    if (sc.rows) {
        as_host = new GnssGpuRecordAssemble(config, records_host(cs[0].gpu), bc, &replay_clock);
        hosts.emplace_back(as_host);
        cmb_host = new GnssCoherentCombiner(config, combine_host(cs[0].gpu), bc);
        hosts.emplace_back(cmb_host);
        hosts.emplace_back(new GnssTelemPack(config, telem_host(cs[0].gpu), bc));
    }
    for (auto& R : runs) {
        const ChainSpec& s = *R->s;
        if (!sc.fold)
            R->stages.emplace_back(
                new GnssN2RecordAssemble(config, "/" + s.pre + "_n2assemble_tiles", bc));
        if (sc.rows) {
            // The host's rows are in key order.
            R->as = as_host;
            R->cmb = cmb_host;
            for (const auto& c : cs)
                R->cmb_row += c.pre < s.pre;
            R->as_row = R->cmb_row;
        } else {
            R->as =
                new GnssGpuRecordAssemble(config, assembler_name(s, R->idx, sc), bc, &replay_clock);
            R->stages.emplace_back(R->as);
            R->cmb = new GnssCoherentCombiner(config, "/" + s.pre + "_n2combine", bc);
            R->stages.emplace_back(R->cmb);
            R->stages.emplace_back(new GnssTelemPack(config, "/" + s.pre + "_telem_pack", bc));
        }
        for (SideCombiner& c : R->side) {
            c.cmb = new GnssCoherentCombiner(config, "/" + s.pre + "_n2combine_" + c.name, bc);
            R->stages.emplace_back(c.cmb);
        }
    }
    publish_band_power(cs);
    for (auto& h : hosts)
        h->start();
    for (auto& R : runs) {
        for (auto& st : R->stages)
            st->start();
        // An assembler's own epl frame is prefilled as the buffers are.
        if (sc.fold)
            std::memset(R->as->epl_frame(R->as_row), sc.prefill, R->buf[EPL]->frame_size);
        for (int k = 0; k < N_KIND; ++k)
            if ((k != CMB || sc.combiner_out_buf) && (k != EPL || !sc.fold)
                && (k != TELEM || !sc.telem_shared) && (k != REC || !sc.rec_shared))
                R->cap[k] = std::thread(capture, R.get(), k);
    }
    std::vector<ChainRun*> rs;
    for (auto& R : runs)
        rs.push_back(R.get());
    std::thread telem_cap, rec_cap;
    if (sc.telem_shared)
        telem_cap = std::thread(capture_telem_shared, rs, telem_shared.get());
    if (sc.rec_shared)
        rec_cap = std::thread(capture_rec_shared, rs, rec_half.get(), rec_dir.get());

    // See quiescent().
    const bool combined_after = sc.cmb_workers > 0 || sc.telem_workers > 0 || sc.rec_shared;
    Digests d;
    for (int k = 0; k < sc.n_frames; ++k) {
        g_frame.store(k);
        // With gpu_rows_shared the half's frame k is written whole and released whole before
        // any chain's frame k is read back.
        if (sc.gpu_rows_shared) {
            if (k > 0)
                check_released(*runs[0], k - 1);
            feed(runs, *runs[0], layout, sky, sc, k, d);
            refill_released(*runs[0], k);
        }
        for (auto& R : runs) {
            const ChainSpec& s = *R->s;
            if (!sc.gpu_rows_shared) {
                if (sc.fold && k > 0)
                    check_released(*R, k - 1);
                feed(runs, *R, layout, sky, sc, k, d);
                if (sc.fold)
                    refill_released(*R, k);
            }
            wait_quiescent(*R, k);
            if (sc.fold)
                capture_fold(*R, k);
            if (!sc.combiner_out_buf && !combined_after)
                capture_scratch(*R, k);
            if (!sc.rec_shared)
                capture_side(*R, k);
            const std::string u = assembler_name(s, R->idx, sc);
            fire_posts(*R, sky, port, u, k);
            const bool checkpoint =
                std::find(sc.checkpoints.begin(), sc.checkpoints.end(), k) != sc.checkpoints.end();
            if (checkpoint)
                fire_gets(*R, port, u, k, d, !combined_after);
            // The per-slot GETs, which only the unmutated runs' checks read. get_records is the
            // costly one, so the chain's own is read only from the swap on, where the swap checks
            // compare it.
            if (sc.mut == Mutation::none && slot_frame(sc, k)) {
                if (!checkpoint && !combined_after) // fire_gets read them
                    read_own_slot_gets(*R, port, k, d.slots[s.pre]["own"]);
                if (!sc.rec_shared)
                    read_side_slots(*R, port, k, d);
            }
            if (sc.refill_a5)
                std::memset(sc.fold ? R->as->epl_frame(R->as_row) : R->buf[EPL]->frames[0], 0xA5,
                            R->buf[EPL]->frame_size);
        }
        // With rec_shared, the records' digests and side combiners of the frame, as above.
        if (sc.rec_shared)
            for (auto& R : runs) {
                wait_records_read(*R, k);
                capture_side(*R, k);
                if (sc.mut == Mutation::none && slot_frame(sc, k))
                    read_side_slots(*R, port, k, d);
            }
        if (sc.telem_workers > 0)
            for (auto& R : runs)
                wait_telem(*R, k);
        // With combined_after, the combiners' captures and GETs of the frame, as above.
        if (combined_after) {
            const bool checkpoint =
                std::find(sc.checkpoints.begin(), sc.checkpoints.end(), k) != sc.checkpoints.end();
            for (auto& R : runs) {
                wait_combined(*R, k);
                if (!sc.combiner_out_buf)
                    capture_scratch(*R, k);
                if (checkpoint)
                    fire_combiner_gets(*R, port, k, d);
                else if (sc.mut == Mutation::none && slot_frame(sc, k))
                    read_own_slot_gets(*R, port, k, d.slots[R->s->pre]["own"]);
            }
        }
        if (std::find(sc.checkpoints.begin(), sc.checkpoints.end(), k) != sc.checkpoints.end()) {
            snapshot_board(k, d);
            if (sc.rows)
                check_by_chain(runs, port, k);
        }
    }

    for (auto& R : runs) {
        const std::string& pre = R->s->pre;
        for (int k = 0; k < N_KIND; ++k)
            d.streams[pre + "." + KIND[k]] = R->hashes[k];
        d.strings[pre + ".cube_chain"] = R->cube_chain;
        d.live[pre]["cube_frames"] = R->count[CUBE].load();
        d.live[pre]["cube_pow"] = R->cube_pow;
        d.live[pre]["telem_frames"] = R->count[TELEM].load();
        d.live[pre]["telem_bad_present"] = R->telem_bad;
        d.live[pre]["telem_bad_meta"] = R->telem_bad_meta;
        if (R->idx == 1)
            d.pins["B2_epl"] = R->pin_epl;
        d.slots[pre]["own"]["cmb"] = R->cmb_rows;
        for (const SideCombiner& x : R->side)
            d.slots[pre][x.name]["cmb"] = x.rows;
    }
    // Each host lists its rows, in key order, with the one-row stage each stands for and its
    // chain.
    if (sc.rows) {
        std::map<std::string, std::string> pres;
        for (const auto& s : cs)
            pres[s.pre] = chain_of(s);
        for (const auto& [host, suffix] : std::vector<std::pair<std::string, std::string>>{
                 {records_host(cs[0].gpu), "_n2assemble"},
                 {combine_host(cs[0].gpu), "_n2combine"},
                 {telem_host(cs[0].gpu), "_telem_pack"}}) {
            json want = json::array();
            for (const auto& [p, chain] : pres)
                want.push_back({{"name", p}, {"stage", "/" + p + suffix}, {"chain", chain}});
            const json got = rest(port, host + "/chains", json{});
            if (got != want)
                child_fail(80, host + "/chains: " + got.dump() + ", not " + want.dump());
        }
    }
    // The REST surface under the chains' stage names.
    {
        const json ep = rest(port, "/endpoints", json{});
        if (const char* dir = std::getenv("GNSS_REPLAY_DUMP")) {
            const std::string stem = std::string(dir) + "/" + std::to_string(getpid());
            std::ofstream(stem + ".config.json") << build_config(cs, sc, cpus).dump(1);
            std::ofstream(stem + ".endpoints.json") << ep.dump(1);
        }
        std::vector<std::string> lines;
        for (const char* verb : {"GET", "POST"})
            for (const auto& p : ep.value(verb, std::vector<std::string>{}))
                for (size_t i = 0; i < cs.size(); ++i) {
                    const std::string& pre = cs[i].pre;
                    const std::vector<std::string> names = {
                        "/" + pre + "_n2assemble_tiles/", assembler_name(cs[i], (int)i, sc) + "/",
                        "/" + pre + "_n2combine/", "/" + pre + "_telem_pack/"};
                    for (const auto& n : names)
                        if (p.compare(0, n.size(), n) == 0)
                            lines.push_back(std::string(verb) + " " + p);
                }
        std::sort(lines.begin(), lines.end());
        lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
        std::string all;
        for (const auto& l : lines)
            all += l + "\n";
        d.strings["endpoints"] = all;
        d.streams["endpoints"] = {fnv1a64(all.data(), all.size())};
    }

    if (sc.fold)
        for (auto& R : runs)
            check_released(*R, sc.n_frames - 1);

    json out = {{"streams", d.streams},
                {"strings", d.strings},
                {"live", d.live},
                {"pins", d.pins},
                {"slots", d.slots}};
    std::ofstream f(out_path);
    f << out.dump();
    f.close();
    if (!f)
        child_fail(75, "cannot write " + out_path);
    std::fflush(nullptr);
    _exit(0);
}

// ---- the stall child ----

/// Frames the stall test feeds, the depth of its tiles and control-block buffers (as
/// gen_chord_gnss_config.py's FOLD_EPL_DEPTH), and how long it holds a worker.
constexpr int STALL_FRAMES = 40, STALL_DEPTH = 8;
constexpr double STALL_HOLD_S = 2.0;
/// The record of chain 0 at which its record consumer, or worker 0, is held.
constexpr int STALL_AT = 40;

/// What the stall test holds: chain 0's record consumer, or worker 0 itself between its reads of
/// the tiles and control blocks (from the assemblers' clock, on that worker's thread).
enum class Hold { rec, worker };

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::atomic<int> g_taken0{0};
std::atomic<bool> g_hold_worker{false}, g_held{false};
std::atomic<double> g_hold_t0{0.0}, g_hold_t1{0.0};

/// Holds the calling thread for STALL_HOLD_S, once, and notes when.
void hold_here() {
    if (g_held.exchange(true))
        return;
    g_hold_t0.store(now_s());
    std::this_thread::sleep_for(std::chrono::duration<double>(STALL_HOLD_S));
    g_hold_t1.store(now_s());
}

/// The assemblers' clock in the stall test: with Hold::worker it holds worker 0 once chain 0's
/// record consumer has taken STALL_AT records.
double stall_clock() {
    if (g_hold_worker.load() && !g_held.load() && g_taken0.load() >= STALL_AT) {
        char name[16] = {0};
        pthread_getname_np(pthread_self(), name, sizeof(name));
        const std::string n(name);
        if (n.size() >= 3 && n.compare(n.size() - 3, 3, "/w0") == 0)
            hold_here();
    }
    return gnss::steady_now_s();
}

/// Writes frame @p k of every chain into frame k % depth of one GPU half's control-block and tiles
/// buffers @p ctl and @p tiles, at the rows of @p layout, and the tiles' metadata. The caller marks
/// them full. Returns false when the buffers shut down.
bool feed_half(const std::vector<ChainSpec>& cs, const json& layout, const Sky& sky,
               const Scenario& sc, int k, Buffer* ctl, Buffer* tiles) {
    const int id = k % ctl->num_frames;
    uint8_t* ctl_f = ctl->wait_for_empty_frame("feeder", id);
    int32_t* tiles_f = (int32_t*)tiles->wait_for_empty_frame("feeder", id);
    if (ctl_f == nullptr || tiles_f == nullptr)
        return false;
    // As the GPU would: a worker still reading a frame it released reads this.
    std::memset(ctl_f, 0xA5, ctl->frame_size);
    std::memset(tiles_f, 0xA5, tiles->frame_size);
    int64_t hop0 = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const ChainSpec& s = cs[i];
        const json& row = layout.at(i);
        std::vector<uint8_t> own_ctl(gnss_gpu::ctl_bytes(s.n_prn, s.n_chan), 0);
        std::vector<int32_t> own_tiles((size_t)s.n_rec * s.n_chan * n_tile(s) * 512, 0);
        write_ctl(own_ctl.data(), own_ctl.size(), s, sky, sc, k);
        write_tiles(own_tiles.data(), own_tiles.size(), s, sky, sc, k);
        auto* hdr = (gnss_gpu::FrameHdr*)own_ctl.data();
        hdr->_pad1 = gnss_gpu::row_tag(row.at("chain_id").get<int>(), row.at("row").get<int>(),
                                       row.at("fg_offset").get<int>());
        hop0 = hdr->seq0 / s.fft_len;
        std::memcpy(ctl_f + row.at("ctl_offset").get<size_t>(), own_ctl.data(), own_ctl.size());
        const size_t slice = (size_t)s.n_chan * n_tile(s) * 512;
        for (int r = 0; r < s.n_rec; ++r)
            std::memcpy(tiles_f
                            + ((size_t)r * fg_total(cs) + row.at("fg_offset").get<int>())
                                  * n_tile(s) * 512,
                        own_tiles.data() + r * slice, slice * sizeof(int32_t));
    }
    tiles->allocate_new_metadata_object(id);
    get_chord_metadata(tiles, id)->set_fpga_seq_num(hop0);
    return true;
}

/// The stall test's child: the chains as rows of one GPU half on two workers (chain i on worker
/// i % 2) reading one pair of tiles and control-block buffers STALL_DEPTH frames deep, fed as
/// fast as the workers release them, every row's records taken by a consumer of its own. One of
/// @p hold holds for STALL_HOLD_S. Writes {"fed": [time of frame k], "taken": [[[time, first
/// sample]...] per chain], "hold": [start, end]} to @p out_path.
void stall_child(const std::vector<ChainSpec>& cs, Hold hold, const std::string& out_path) {
    for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGALRM})
        std::signal(sig, SIG_DFL);
    g_out_path = out_path;
    std::signal(SIGTERM, on_sigterm);
    alarm(120);
    g_hold_worker.store(hold == Hold::worker);
    const std::vector<int> cpus = allowed_cpus();
    kotekan::restServer::instance().start("127.0.0.1", 0);
    Scenario sc = base_scenario();
    sc.fold = sc.gpu_rows = sc.gpu_rows_shared = sc.rows = true;
    sc.workers = 2;
    const Sky sky = sky_for(sc);
    const json layout = gpu_rows_of(cs);
    const size_t ctl_b =
        layout.back().at("ctl_offset").get<size_t>() + layout.back().at("ctl_bytes").get<size_t>();
    const size_t tiles_b =
        (size_t)cs[0].n_rec * fg_total(cs) * n_tile(cs[0]) * 512 * sizeof(int32_t);
    auto pool = metadataPool::create(64 * (int)cs.size(), 64, "pool", "GnssChanMetadata");
    auto tiles_pool =
        metadataPool::create(STALL_DEPTH, sizeof(chordMetadata), "tiles_pool", "chordMetadata");
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> bufs;
    auto add = [&](int frames, size_t bytes, std::shared_ptr<metadataPool> p,
                   const std::string& name) {
        bufs.push_back(std::make_unique<Buffer>(frames, bytes, p, name, "standard", 0, false, false,
                                                std::vector<int>{}, true, 0xA5));
        bc.add_buffer(name, bufs.back().get());
        return bufs.back().get();
    };
    Buffer* ctl = add(STALL_DEPTH, ctl_b, pool, rows_buf(cs, "ctl_buf"));
    Buffer* tiles = add(STALL_DEPTH, tiles_b, tiles_pool, rows_buf(cs, "tiles_buf"));
    ctl->register_producer("feeder");
    tiles->register_producer("feeder");
    std::vector<Buffer*> rec, cube;
    for (const ChainSpec& s : cs) {
        rec.push_back(add(4, gnss::frame_floats(s.n_prn, s.n_live, s.n_chan) * sizeof(float), pool,
                          buf_name(s, "n2rec_buf")));
        cube.push_back(
            add(4,
                gnss::cube_frame_bytes(s.assemble.value("beam_cube_max_prn", s.n_prn),
                                       s.assemble.value("beam_cube_max_bins", s.n_chan), s.n_live),
                pool, buf_name(s, "cube_buf")));
        rec.back()->register_consumer("taker");
        cube.back()->register_consumer("drain");
    }
    kotekan::Config config;
    config.update_config(build_config(cs, sc, cpus));
    publish_band_power(cs);
    GnssGpuRecordAssemble host(config, records_host(cs[0].gpu), bc, &stall_clock);
    host.start();

    std::vector<double> fed(STALL_FRAMES, 0.0);
    std::thread feeder([&] {
        for (int k = 0; k < STALL_FRAMES; ++k) {
            if (!feed_half(cs, layout, sky, sc, k, ctl, tiles))
                return;
            fed[k] = now_s();
            ctl->mark_frame_full("feeder", k % ctl->num_frames);
            tiles->mark_frame_full("feeder", k % tiles->num_frames);
        }
    });
    std::vector<std::vector<std::pair<double, int64_t>>> taken(cs.size());
    std::vector<std::thread> takers;
    for (size_t i = 0; i < cs.size(); ++i)
        takers.emplace_back([&, i] {
            Buffer* b = rec[i];
            for (int j = 0; j < STALL_FRAMES * cs[i].n_rec; ++j) {
                const int id = j % b->num_frames;
                if (b->wait_for_full_frame("taker", id) == nullptr)
                    return;
                const GnssChanMetadata* m = get_gnss_chan_metadata(b, id);
                taken[i].push_back({now_s(), m ? m->sample_seq : -1});
                if (i == 0) {
                    g_taken0.store(j + 1);
                    if (hold == Hold::rec && j == STALL_AT)
                        hold_here();
                }
                b->mark_frame_empty("taker", id);
            }
        });
    std::vector<std::thread> drains;
    for (Buffer* b : cube)
        drains.emplace_back([b] {
            for (int id = 0; b->wait_for_full_frame("drain", id) != nullptr;
                 id = (id + 1) % b->num_frames)
                b->mark_frame_empty("drain", id);
        });
    feeder.join();
    for (std::thread& t : takers)
        t.join();
    host.stop();
    for (auto& b : bufs)
        b->send_shutdown_signal();
    host.join();
    for (std::thread& t : drains)
        t.join();

    json out = {{"fed", fed},
                {"taken", taken},
                {"hold", {g_hold_t0.load(), g_hold_t1.load()}},
                {"n_rec", cs[0].n_rec}};
    std::ofstream f(out_path);
    f << out.dump();
    f.close();
    if (!f)
        child_fail(75, "cannot write " + out_path);
    std::fflush(nullptr);
    _exit(0);
}

// ---- the combiner stall child ----

/// Records each row of the combiner stall test is fed, one every CMB_STALL_DT_S, and chain 0's
/// record at which its feeder or its taker is held.
constexpr int CMB_STALL_RECORDS = 1000, CMB_STALL_AT = 200;
/// The PRN of every record the combiner stall test feeds.
constexpr float CMB_STALL_PRN = 7.0f;
constexpr double CMB_STALL_DT_S = 0.004;

/// What the combiner stall test holds: chain 0's feeder, so that its worker waits for chain 0's
/// next record, or chain 0's taker, so that its worker blocks in an emit on chain 0's full
/// out_buf.
enum class CmbHold { input, output };

/// The combiner stall test's child: the chains' combiners as rows of one host on two workers
/// (chain i on worker i % 2). A feeder per chain writes CMB_STALL_RECORDS empty records of PRN
/// CMB_STALL_PRN, each with its window's sample_seq, one every CMB_STALL_DT_S, into a 4-deep
/// in_buf, and a thread per chain sets a record's PRNs to 0 as soon as its frame is released;
/// integration_length 10 emits every record, labelled with its hop, into a 4-deep out_buf that a
/// taker per chain reads.
/// @p hold holds for STALL_HOLD_S at chain 0's record CMB_STALL_AT. Writes {"t0": start, "fed":
/// [[time of record j] per chain], "taken": [[[time, hop]...] per chain], "hold": [start, end]} to
/// @p out_path.
void cmb_stall_child(const std::vector<ChainSpec>& cs, CmbHold hold, const std::string& out_path) {
    for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGALRM})
        std::signal(sig, SIG_DFL);
    g_out_path = out_path;
    std::signal(SIGTERM, on_sigterm);
    alarm(120);
    const std::vector<int> cpus = allowed_cpus();
    kotekan::restServer::instance().start("127.0.0.1", 0);
    auto pool = metadataPool::create(16 * (int)cs.size(), 64, "pool", "GnssChanMetadata");
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> bufs;
    std::vector<Buffer*> in, out;
    std::map<std::string, json> rows;
    for (const ChainSpec& s : cs) {
        const size_t bytes = (size_t)s.n_prn
                             * gnss::record_stride(s.combine.at("n_elements").get<int>())
                             * sizeof(float);
        for (const char* role : {"n2rec_buf", "n2cmb_buf"}) {
            bufs.push_back(std::make_unique<Buffer>(4, bytes, pool, buf_name(s, role), "standard",
                                                    0, false, false, std::vector<int>{}, true, 0));
            bc.add_buffer(buf_name(s, role), bufs.back().get());
        }
        in.push_back(bufs[bufs.size() - 2].get());
        out.push_back(bufs.back().get());
        in.back()->register_producer("feeder");
        out.back()->register_consumer("taker");
        json cb = s.combine;
        cb["integration_length"] = 10;
        cb["in_bufs"] = {buf_name(s, "n2rec_buf")};
        cb["out_buf"] = buf_name(s, "n2cmb_buf");
        cb["chain"] = s.telem.at("chain");
        cb["cpu_affinity"] = cpus;
        rows[s.pre] = cb;
    }
    json cfg = {{"log_level", "warn"}, {"cpu_affinity", cpus}};
    json& h = cfg[combine_host(cs[0].gpu).substr(1)];
    h = host_block(rows, GnssCoherentCombiner::ROW_KEYS, GnssCoherentCombiner::ROW_ONLY_KEYS);
    for (int w = 0; w < 2; ++w) {
        json r = json::array();
        for (size_t i = w; i < cs.size(); i += 2)
            r.push_back(cs[i].pre);
        h["workers"].push_back({{"rows", r}, {"cpu_affinity", cpus}});
    }
    kotekan::Config config;
    config.update_config(cfg);
    GnssCoherentCombiner host(config, combine_host(cs[0].gpu), bc);
    host.start();

    const double t0 = now_s() + 0.2;
    std::vector<std::vector<double>> fed(cs.size(), std::vector<double>(CMB_STALL_RECORDS, 0.0));
    std::vector<std::vector<std::pair<double, int64_t>>> taken(cs.size());
    // Per chain: the in_buf's frames, and the records fed and the releases poisoned so far.
    std::vector<std::vector<float*>> frames(cs.size(), std::vector<float*>(in[0]->num_frames));
    std::vector<std::atomic<int>> n_fed(cs.size()), n_poisoned(cs.size());
    std::vector<std::thread> threads;
    for (size_t i = 0; i < cs.size(); ++i) {
        const size_t stride = gnss::record_stride(cs[i].combine.at("n_elements").get<int>());
        threads.emplace_back([&, i, stride] {
            Buffer* b = in[i];
            const int64_t rec_samples = (int64_t)cs[i].hops_per_record * cs[i].fft_len;
            for (int j = 0; j < CMB_STALL_RECORDS; ++j) {
                if (i == 0 && j == CMB_STALL_AT && hold == CmbHold::input)
                    hold_here();
                const int id = j % b->num_frames;
                float* f = (float*)b->wait_for_empty_frame("feeder", id);
                if (f == nullptr)
                    return;
                if (j < b->num_frames)
                    frames[i][id] = f;
                while (n_poisoned[i].load() < j + 1 - b->num_frames)
                    std::this_thread::yield();
                while (now_s() < t0 + j * CMB_STALL_DT_S)
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                for (int p = 0; p < cs[i].n_prn; ++p)
                    f[p * stride + gnss::REC_PRN] = CMB_STALL_PRN;
                b->allocate_new_metadata_object(id);
                get_gnss_chan_metadata(b, id)->sample_seq = (REC_BASE + j) * rec_samples;
                fed[i][j] = now_s();
                b->mark_frame_full("feeder", id);
                n_fed[i].store(j + 1);
            }
        });
        // Sets each record's PRNs to 0 as soon as the combiner releases its frame, so that a read
        // after the release takes PRN 0.
        threads.emplace_back([&, i, stride] {
            Buffer* b = in[i];
            for (int k = 0; k < CMB_STALL_RECORDS; ++k) {
                const int id = k % b->num_frames;
                while (n_fed[i].load() <= k || !b->is_frame_empty(id))
                    std::this_thread::yield();
                for (int p = 0; p < cs[i].n_prn; ++p)
                    frames[i][id][p * stride + gnss::REC_PRN] = 0.0f;
                n_poisoned[i].store(k + 1);
            }
        });
        threads.emplace_back([&, i, stride] {
            Buffer* b = out[i];
            for (int j = 0; j < CMB_STALL_RECORDS; ++j) {
                const int id = j % b->num_frames;
                const uint8_t* f = b->wait_for_full_frame("taker", id);
                if (f == nullptr)
                    return;
                if (i == 0 && j == CMB_STALL_AT && hold == CmbHold::output)
                    hold_here();
                int64_t hop;
                std::memcpy(&hop, f + gnss::CMB_HOP_SLOT * sizeof(float), sizeof(hop));
                // A poisoned PRN read shows as the record's hop negated.
                bool poisoned = false;
                for (int p = 0; p < cs[i].n_prn; ++p)
                    poisoned |= ((const float*)f)[p * stride + gnss::REC_PRN] != CMB_STALL_PRN;
                taken[i].push_back({now_s(), poisoned ? -hop : hop});
                b->mark_frame_empty("taker", id);
            }
        });
    }
    for (std::thread& t : threads)
        t.join();
    host.stop();
    for (auto& b : bufs)
        b->send_shutdown_signal();
    host.join();

    json out_j = {
        {"t0", t0}, {"fed", fed}, {"taken", taken}, {"hold", {g_hold_t0.load(), g_hold_t1.load()}}};
    std::ofstream f(out_path);
    f << out_j.dump();
    f.close();
    if (!f)
        child_fail(75, "cannot write " + out_path);
    std::fflush(nullptr);
    _exit(0);
}

// ---- the record buffer stall child ----

/// GPU frames the record buffer stall test feeds, one per frame period, and the record of chain 0
/// at which it holds.
constexpr int REC_STALL_FRAMES = 60, REC_STALL_AT = 40;

/// What the record buffer stall test holds, each a reader or writer of one row among others: the
/// reader of combine row 0's out_buf, so that combine worker 0 blocks in an emit; the reader of
/// telemetry row 0's own out_buf, so that telemetry worker 0 blocks in a flush; or record worker 0
/// itself, in its clock.
enum class RecHold { combine, telem, record };

/// The record buffer stall test's child: the chains as rows of one GPU half, fed one frame per
/// frame period into one pair of tiles and control-block buffers STALL_DEPTH frames deep. Their
/// record host writes the half's record buffer, REC_SHARED_DEPTH frames with peek_hold as on a
/// node, which their combine and telemetry hosts read; every host runs two workers, chain i on
/// worker i % 2. Every combine row emits each record into an out_buf of its own, every telemetry
/// row writes one of its own, and a thread reads each; another thread reads every frame of the
/// record buffer as one more consumer. @p hold holds for STALL_HOLD_S. Writes {"t0": first
/// frame's time, "dt": frame period, "fed": [time of frame k], "rec": [[[time, wstart]...] per
/// chain], "torn": [records whose PRN rows are not all of their wstart, per chain], "cmb":
/// [[[time, hop]...] per chain], "telem": [[[time, seq, win, present]...] per chain],
/// "progress": [[time, records written per chain..., records combined per chain..., full
/// frames]...], "hold": [start, end]} to @p out_path.
void rec_stall_child(const std::vector<ChainSpec>& cs, RecHold hold, const std::string& out_path) {
    for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGALRM})
        std::signal(sig, SIG_DFL);
    g_out_path = out_path;
    std::signal(SIGTERM, on_sigterm);
    alarm(120);
    g_hold_worker.store(hold == RecHold::record);
    const std::vector<int> cpus = allowed_cpus();
    kotekan::restServer::instance().start("127.0.0.1", 0);
    Scenario sc = base_scenario();
    sc.fold = sc.gpu_rows = sc.gpu_rows_shared = sc.rows = sc.rec_shared = true;
    sc.workers = sc.cmb_workers = sc.telem_workers = 2;
    const Sky sky = sky_for(sc);
    const json layout = gpu_rows_of(cs);
    const std::vector<gnss::RecRow> rr = rec_rows_of(cs);
    const size_t ctl_b =
        layout.back().at("ctl_offset").get<size_t>() + layout.back().at("ctl_bytes").get<size_t>();
    const size_t tiles_b =
        (size_t)cs[0].n_rec * fg_total(cs) * n_tile(cs[0]) * 512 * sizeof(int32_t);
    auto pool = metadataPool::create(64 * (int)cs.size(), 64, "pool", "GnssChanMetadata");
    auto tiles_pool =
        metadataPool::create(STALL_DEPTH, sizeof(chordMetadata), "tiles_pool", "chordMetadata");
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> bufs;
    auto add = [&](int frames, size_t bytes, std::shared_ptr<metadataPool> p,
                   const std::string& name) {
        bufs.push_back(std::make_unique<Buffer>(frames, bytes, p, name, "standard", 0, false, false,
                                                std::vector<int>{}, true, 0xA5));
        bc.add_buffer(name, bufs.back().get());
        return bufs.back().get();
    };
    Buffer* ctl = add(STALL_DEPTH, ctl_b, pool, rows_buf(cs, "ctl_buf"));
    Buffer* tiles = add(STALL_DEPTH, tiles_b, tiles_pool, rows_buf(cs, "tiles_buf"));
    ctl->register_producer("feeder");
    tiles->register_producer("feeder");
    Buffer* rec = add(REC_SHARED_DEPTH, (size_t)(rr.back().offset + rr.back().bytes), nullptr,
                      rec_shared_buf(cs));
    rec->enable_peek_hold();
    rec->register_consumer("taker");
    const gnss::RecDir dir("taker", rr, rec);
    std::vector<Buffer*> cube, cmb, telem;
    for (const ChainSpec& s : cs) {
        cube.push_back(
            add(4,
                gnss::cube_frame_bytes(s.assemble.value("beam_cube_max_prn", s.n_prn),
                                       s.assemble.value("beam_cube_max_bins", s.n_chan), s.n_live),
                pool, buf_name(s, "cube_buf")));
        cmb.push_back(add(4, (size_t)s.n_prn * gnss::record_stride(s.n_live) * sizeof(float), pool,
                          buf_name(s, "n2cmb_buf")));
        telem.push_back(
            add(4, gnss::telem_frame_bytes(s.n_rec, s.telem.at("max_prn").get<int>(), s.n_chan),
                pool, buf_name(s, "telem_buf")));
        cube.back()->register_consumer("drain");
        cmb.back()->register_consumer("taker");
        telem.back()->register_consumer("taker");
    }
    json cfg = build_config(cs, sc, cpus);
    // Every record emits.
    cfg[combine_host(cs[0].gpu).substr(1)]["integration_length"] = 10;
    kotekan::Config config;
    config.update_config(cfg);
    publish_band_power(cs);
    GnssGpuRecordAssemble as_host(config, records_host(cs[0].gpu), bc, &stall_clock);
    GnssCoherentCombiner cmb_host(config, combine_host(cs[0].gpu), bc);
    GnssTelemPack tel_host(config, telem_host(cs[0].gpu), bc);
    as_host.start();
    cmb_host.start();
    tel_host.start();

    const double dt =
        (double)cs[0].n_rec * cs[0].hops_per_record * cs[0].fft_len / cs[0].sample_rate;
    const double t0 = now_s() + 0.2;
    std::vector<double> fed(REC_STALL_FRAMES, 0.0);
    std::thread feeder([&] {
        for (int k = 0; k < REC_STALL_FRAMES; ++k) {
            while (now_s() < t0 + k * dt)
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            if (!feed_half(cs, layout, sky, sc, k, ctl, tiles))
                return;
            fed[k] = now_s();
            ctl->mark_frame_full("feeder", k % ctl->num_frames);
            tiles->mark_frame_full("feeder", k % tiles->num_frames);
        }
    });
    const int n_rec_total = REC_STALL_FRAMES * cs[0].n_rec;
    std::vector<std::vector<std::pair<double, int64_t>>> taken(cs.size()), emits(cs.size());
    std::vector<std::vector<std::array<double, 4>>> windows(cs.size());
    std::vector<int> torn(cs.size(), 0);
    // How many of each the readers have, for the wait below.
    std::vector<std::atomic<int>> n_taken(cs.size()), n_emits(cs.size()), n_windows(cs.size());
    std::vector<std::thread> threads;
    // The record buffer's frames: each row's wstart, and whether every PRN row of its block
    // carries that wstart's UTC.
    threads.emplace_back([&] {
        for (int j = 0; j < n_rec_total; ++j) {
            const int id = j % rec->num_frames;
            const uint8_t* f = rec->wait_for_full_frame("taker", id);
            if (f == nullptr)
                return;
            dir.check("taker", id, f);
            for (size_t i = 0; i < cs.size(); ++i) {
                const ChainSpec& s = cs[i];
                const int64_t w = dir.wstart(f, i);
                const double utc = UTC0 + (double)w / s.sample_rate;
                const float* block = (const float*)(f + rr[i].offset);
                for (int p = 0; p < s.n_prn; ++p) {
                    double u;
                    std::memcpy(&u,
                                block + (size_t)p * gnss::record_stride(s.n_live)
                                    + gnss::RECORD_UTC_SLOT,
                                sizeof(u));
                    if (u != utc) {
                        ++torn[i];
                        break;
                    }
                }
                taken[i].push_back({now_s(), w});
                n_taken[i].store(j + 1);
            }
            g_taken0.store(j + 1);
            rec->mark_frame_empty("taker", id);
        }
    });
    for (size_t i = 0; i < cs.size(); ++i) {
        threads.emplace_back([&, i] {
            Buffer* b = cmb[i];
            for (int j = 0; j < n_rec_total; ++j) {
                const int id = j % b->num_frames;
                const uint8_t* f = b->wait_for_full_frame("taker", id);
                if (f == nullptr)
                    return;
                if (i == 0 && j == REC_STALL_AT && hold == RecHold::combine)
                    hold_here();
                int64_t hop;
                std::memcpy(&hop, f + gnss::CMB_HOP_SLOT * sizeof(float), sizeof(hop));
                emits[i].push_back({now_s(), hop});
                n_emits[i].store(j + 1);
                b->mark_frame_empty("taker", id);
            }
        });
        threads.emplace_back([&, i] {
            Buffer* b = telem[i];
            for (int j = 0; j < n_rec_total / cs[i].n_rec; ++j) {
                const int id = j % b->num_frames;
                const uint8_t* f = b->wait_for_full_frame("taker", id);
                if (f == nullptr)
                    return;
                if (i == 0 && j == REC_STALL_AT / cs[i].n_rec && hold == RecHold::telem)
                    hold_here();
                gnss::TelemHeader th;
                std::memcpy(&th, f, sizeof(th));
                windows[i].push_back({now_s(), (double)th.seq, (double)th.win, (double)th.present});
                n_windows[i].store(j + 1);
                b->mark_frame_empty("taker", id);
            }
        });
    }
    for (Buffer* b : cube)
        threads.emplace_back([b] {
            for (int id = 0; b->wait_for_full_frame("drain", id) != nullptr;
                 id = (id + 1) % b->num_frames)
                b->mark_frame_empty("drain", id);
        });
    // What every host has done, every 5 ms, until every reader has every record.
    json progress = json::array();
    auto done = [&] {
        bool all = true;
        for (size_t i = 0; i < cs.size(); ++i)
            all = all && n_taken[i].load() == n_rec_total && n_emits[i].load() == n_rec_total
                  && n_windows[i].load() == n_rec_total / cs[i].n_rec;
        return all;
    };
    const double t_end = t0 + REC_STALL_FRAMES * dt + 30.0;
    while (!done() && now_s() < t_end) {
        json p = json::array({now_s()});
        for (size_t i = 0; i < cs.size(); ++i)
            p.push_back(as_host.records_done(i));
        for (size_t i = 0; i < cs.size(); ++i)
            p.push_back(cmb_host.records_done(i));
        p.push_back(rec->get_num_full_frames());
        progress.push_back(p);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    feeder.join();
    as_host.stop();
    cmb_host.stop();
    tel_host.stop();
    for (auto& b : bufs)
        b->send_shutdown_signal();
    for (std::thread& t : threads)
        t.join();
    as_host.join();
    cmb_host.join();
    tel_host.join();

    json out_j = {{"t0", t0},
                  {"dt", dt},
                  {"fed", fed},
                  {"rec", taken},
                  {"torn", torn},
                  {"cmb", emits},
                  {"telem", windows},
                  {"progress", progress},
                  {"n_rec", cs[0].n_rec},
                  {"hold", {g_hold_t0.load(), g_hold_t1.load()}}};
    std::ofstream f(out_path);
    f << out_j.dump();
    f.close();
    if (!f)
        child_fail(75, "cannot write " + out_path);
    std::fflush(nullptr);
    _exit(0);
}

// ---- the parent ----

/// Runs @p child in a forked child, which must _exit. Returns its exit code.
int run_forked(const std::function<void()>& child) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    BOOST_REQUIRE(pid >= 0);
    if (pid == 0) {
        try {
            child();
        } catch (const std::exception& e) {
            child_fail(76, std::string("exception: ") + e.what());
        }
        _exit(77);
    }
    int status = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (waitpid(pid, &status, WNOHANG) == 0) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(310)) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            BOOST_FAIL("replay child timed out");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (WIFSIGNALED(status))
        BOOST_FAIL("replay child killed by signal " << WTERMSIG(status));
    return WEXITSTATUS(status);
}

/// Runs @p sc in a forked child, which writes its digests to @p path. Returns its exit code.
int run_child(const std::vector<ChainSpec>& cs, const Scenario& sc, const char* path) {
    return run_forked([&] { child_main(cs, sc, path); });
}

Digests run_scenario(const std::vector<ChainSpec>& cs, const Scenario& sc) {
    char path[] = "/tmp/gnss_replay_XXXXXX";
    const int fd = mkstemp(path);
    BOOST_REQUIRE(fd >= 0);
    close(fd);
    int rc = -1;
    try {
        rc = run_child(cs, sc, path);
    } catch (...) {
        unlink(path);
        throw;
    }
    Digests d;
    if (rc != 0) {
        std::ifstream f(path);
        const std::string msg((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        unlink(path);
        BOOST_FAIL("replay child exit " << rc
                                        << (rc == 70 ? " (FATAL in child: " + msg + ")" : ""));
    }
    std::ifstream in(path);
    const json j = json::parse(in);
    unlink(path);
    d.streams = j.at("streams").get<std::map<std::string, std::vector<uint64_t>>>();
    d.strings = j.at("strings").get<std::map<std::string, std::string>>();
    d.live = j.at("live");
    d.pins = j.at("pins");
    d.slots = j.at("slots");
    return d;
}

/// The base scenario with the PRN swap references beside every chain's combiner.
const Digests& baseline() {
    static const Digests d = [] {
        Scenario sc = base_scenario();
        sc.cold_ref = true;
        return run_scenario(chains(), sc);
    }();
    return d;
}

Digests run_with(Mutation m) {
    Scenario sc = base_scenario();
    sc.mut = m;
    return run_scenario(chains(), sc);
}

/// The mutation controls name the synthetic chains' roles, so they always replay those.
const Digests& synthetic_baseline() {
    static const Digests d =
        std::getenv("GNSS_REPLAY_CONFIG") ? run_scenario(synthetic(), Scenario{}) : baseline();
    return d;
}

/// Each mutation must change the keys in `differ` (a key ending in '*' matches any key with that
/// prefix, and one changed match is enough) and no key matching `same`.
struct MutationCase {
    Mutation m;
    const char* name;
    std::vector<std::string> differ, same;
};

bool matches(const std::string& key, const std::string& pat) {
    if (!pat.empty() && pat.back() == '*')
        return key.compare(0, pat.size() - 1, pat, 0, pat.size() - 1) == 0;
    return key == pat;
}

bool full_run() {
    const char* f = std::getenv("GNSS_REPLAY_FULL");
    if (f != nullptr && std::string(f) == "1")
        return true;
    BOOST_TEST_MESSAGE("GNSS_REPLAY_FULL=1 not set: skipped");
    return false;
}

/// 1/s as GnssN2RecordAssemble computes it from a replica energy.
double adapter_inv_scale(const ChainSpec& s, double e) {
    const double rms = std::sqrt(e / (double)s.hops_per_record);
    return 1.0 / (7.0 / (3.0 * rms));
}

// ---- checks ----

std::set<std::string> changed(const Digests& a, const Digests& b) {
    std::set<std::string> out;
    for (const auto& [k, v] : a.streams) {
        auto it = b.streams.find(k);
        if (it == b.streams.end() || it->second != v)
            out.insert(k);
    }
    for (const auto& [k, v] : b.streams)
        if (!a.streams.count(k))
            out.insert(k);
    for (const auto& [k, v] : a.strings) {
        auto it = b.strings.find(k);
        if (it == b.strings.end() || it->second != v)
            out.insert("strings:" + k);
    }
    for (const auto& [k, v] : b.strings)
        if (!a.strings.count(k))
            out.insert("strings:" + k);
    return out;
}

std::string join(const std::set<std::string>& s) {
    std::string o;
    for (const auto& x : s)
        o += (o.empty() ? "" : " ") + x;
    return o;
}

/// The indices (arrays) or keys (objects) at which @p a and @p b differ.
std::string differing(const json& a, const json& b) {
    std::set<std::string> out;
    if (a.is_array() && b.is_array()) {
        for (size_t i = 0; i < std::max(a.size(), b.size()); ++i)
            if (i >= a.size() || i >= b.size() || a[i] != b[i])
                out.insert(std::to_string(i));
    } else {
        for (const auto& [k, v] : a.items())
            if (!b.contains(k) || b.at(k) != v)
                out.insert(k);
        for (const auto& [k, v] : b.items())
            if (!a.contains(k))
                out.insert(k);
    }
    return join(out);
}

/// Notes in @p bad where @p got differs from @p want, under @p what.
void note_diff(const std::string& what, const json& got, const json& want,
               std::vector<std::string>& bad) {
    if (got != want)
        bad.push_back(what + " differs at " + differing(got, want));
}

bool planted_running(const ChainSpec& s, int prn) {
    return make_sky(SKY_AMP, 1).planted.count(prn) > 0
           && std::find(s.prns.begin(), s.prns.end(), prn) != s.prns.end();
}

/// The PRNs of running slots without a signal, at the end of the scenario.
std::set<int> unplanted_running(const ChainSpec& s) {
    const Scenario sc = base_scenario();
    const Sky sky = make_sky(SKY_AMP, 1);
    std::set<int> out;
    for (int p = 0; p < s.n_prn; ++p) {
        const int prn = prn_at(s, sc, sc.n_frames - 1, p);
        if (runs(s, sc, sky, sc.n_frames - 1, 1, p) && !sky.planted.count(prn))
            out.insert(prn);
    }
    return out;
}

double median(std::vector<double> v) {
    if (v.empty())
        return NAN;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/// At the second-last checkpoint (89) every planted slot is warm with its own shadow warm, and no
/// unplanted slot's shadow is warm. Once the shared model forms, every running slot's live cal
/// holds it, so only the shadow, which learns from the slot's own data, tells a signal from noise.
/// A noise-only shadow passes its 3-sigma element gate now and then, and the live cal keeps what
/// it held, so one warm planted slot is not enough.
bool cal_live(const Digests& d, const ChainSpec& s, std::string* why) {
    const std::vector<int>& cp = base_scenario().checkpoints;
    const json rows =
        d.live.at(s.pre).value("elem_cal@" + std::to_string(cp[cp.size() - 2]), json::array());
    const std::set<int> un = unplanted_running(s);
    int planted = 0, planted_warm = 0, unplanted_warm = 0;
    for (const auto& r : rows) {
        const int prn = r[0].get<int>();
        if (planted_running(s, prn)) {
            ++planted;
            planted_warm += r[1].get<bool>() && r[2].get<bool>();
        }
        if (un.count(prn) && r[2].get<bool>())
            ++unplanted_warm;
    }
    *why = fmt::format("{:s}: {:d}/{:d} planted warm, {:d}/{:d} unplanted shadow-warm", s.pre,
                       planted_warm, planted, unplanted_warm, un.size());
    return planted > 0 && planted_warm == planted && unplanted_warm == 0;
}

/// The combiner's coherent amplitude |<A>| (get_status coh_amplitude) on planted slots against
/// unplanted ones. Its incoherent amplitude reads the per-record SNR, about 2-10 here.
bool combiner_live(const Digests& d, const ChainSpec& s, std::string* why) {
    const json rows = d.live.at(s.pre).value(
        "status@" + std::to_string(base_scenario().checkpoints.back()), json::array());
    const std::set<int> un = unplanted_running(s);
    std::vector<double> pl, up;
    for (const auto& r : rows) {
        const int prn = r[0].get<int>();
        if (planted_running(s, prn))
            pl.push_back(r[3].get<double>());
        else if (un.count(prn))
            up.push_back(r[3].get<double>());
    }
    const double a = median(pl), b = median(up);
    *why =
        fmt::format("{:s}: combiner coh_amplitude median planted {:g} unplanted {:g}", s.pre, a, b);
    return a > 10.0 * b;
}

bool cube_live(const Digests& d, const ChainSpec& s, std::string* why) {
    const json& L = d.live.at(s.pre);
    const std::set<int> un = unplanted_running(s);
    std::vector<double> pl, up;
    const json pw = L.value("cube_pow", json::object());
    for (const auto& [k, v] : pw.items()) {
        const int prn = std::stoi(k);
        if (planted_running(s, prn))
            pl.push_back(v.get<double>());
        else if (un.count(prn))
            up.push_back(v.get<double>());
    }
    const int n = L.value("cube_frames", 0);
    const double a = median(pl), b = median(up);
    *why = fmt::format("{:s}: {:d} cube frames, coherent power median planted {:g} unplanted {:g}",
                       s.pre, n, a, b);
    return n >= 3 && a > 10.0 * b;
}

/// Every board entry belongs to group /gnss<g> and to one of the chains' assemblers by the
/// owner form its source kind publishes: "<u>/lane/<prn>" (lane), "<u>" (own rows), "<u>/probe".
bool board_ok(const Digests& d, const std::vector<ChainSpec>& cs, const Scenario& sc, int cp,
              std::string* why) {
    const json rows = d.live.value("board@" + std::to_string(cp), json::array());
    if (rows.empty()) {
        *why = "board empty";
        return false;
    }
    for (const auto& r : rows) {
        const std::string group = r[0], owner = r[2];
        const int src = r[3].get<int>();
        bool ok = false;
        for (size_t i = 0; i < cs.size() && !ok; ++i) {
            const std::string u = assembler_name(cs[i], (int)i, sc);
            if (group != "/gnss" + std::to_string(cs[i].gpu))
                continue;
            if (src == 2) {
                for (int p = 0; p < cs[i].n_prn && !ok; ++p)
                    ok = owner == u + "/lane/" + std::to_string(prn_at(cs[i], sc, cp, p));
            } else {
                ok = owner == (src == 1 ? u + "/probe" : u);
            }
        }
        if (!ok) {
            *why = "board entry " + r.dump() + " has no matching chain owner";
            return false;
        }
    }
    *why = fmt::format("{:d} board entries", rows.size());
    return true;
}

/// The (freq_id, PRN) lanes two chains both publish, as two entries, one per owner.
std::set<std::pair<int, int>> shared_lanes(const Digests& d, const std::vector<ChainSpec>& cs,
                                           int cp) {
    const json rows = d.live.value("board@" + std::to_string(cp), json::array());
    const Scenario sc = base_scenario();
    std::set<std::pair<int, int>> out;
    for (size_t i = 0; i < cs.size(); ++i)
        for (size_t j = i + 1; j < cs.size(); ++j)
            for (int f : cs[i].channel_ids) {
                if (std::find(cs[j].channel_ids.begin(), cs[j].channel_ids.end(), f)
                    == cs[j].channel_ids.end())
                    continue;
                for (int prn : make_sky(SKY_AMP, 1).planted) {
                    if (!planted_running(cs[i], prn) || !planted_running(cs[j], prn))
                        continue;
                    const std::string oi =
                        assembler_name(cs[i], (int)i, sc) + "/lane/" + std::to_string(prn);
                    const std::string oj =
                        assembler_name(cs[j], (int)j, sc) + "/lane/" + std::to_string(prn);
                    bool hi = false, hj = false;
                    for (const auto& r : rows)
                        if (r[1].get<int>() == f && r[3].get<int>() == 2) {
                            hi = hi || r[2] == oi;
                            hj = hj || r[2] == oj;
                        }
                    if (hi && hj)
                        out.insert({f, prn});
                }
            }
    return out;
}

/// The REST paths the stages register under this config (GnssGpuRecordAssemble's constructor and
/// GnssCoherentCombiner::main_thread).
std::string expected_endpoints(const std::vector<ChainSpec>& cs, const Scenario& sc) {
    std::vector<std::string> l;
    for (size_t i = 0; i < cs.size(); ++i) {
        const ChainSpec& s = cs[i];
        const json& a = s.assemble;
        const std::string u = assembler_name(s, (int)i, sc), c = "/" + s.pre + "_n2combine";
        const bool elem_sum = a.value("elem_sum", false);
        const std::vector<int> ch = a.value("channel_ids", std::vector<int>{});
        const bool lane_on =
            a.value("elem_dir_publish", std::string("off")) == "all" && !ch.empty();
        const bool inst_on =
            lane_on && !a.value("lane_transfer_freq_ids", std::vector<int>{}).empty()
            && ch.size() >= 2 && ch[1] != ch[0] && a.value("lane_transfer_inst_slope", true);
        l.push_back("GET " + u + "/get_spectrum");
        l.push_back("POST " + u + "/set_sat_geometry");
        l.push_back("POST " + u + "/set_reference_element");
        if (elem_sum) {
            for (const char* v : {"set_elem_gain", "set_elem_sum_adapt", "set_elem_sum_shared",
                                  "set_elem_sum_shared_ref"})
                l.push_back("POST " + u + "/" + v);
            l.push_back("GET " + u + "/get_elem_cal");
            if (!ch.empty())
                l.push_back("POST " + u + "/set_elem_proj");
        }
        if (inst_on)
            l.push_back("GET " + u + "/get_lane_inst");
        l.push_back("GET " + c + "/get_status");
        l.push_back("GET " + c + "/get_elements");
        if (s.combine.value("record_export", 0) > 0)
            l.push_back("GET " + c + "/get_records");
    }
    std::sort(l.begin(), l.end());
    std::string all;
    for (const auto& x : l)
        all += x + "\n";
    return all;
}

std::string hex(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof(b), "%016llx", (unsigned long long)v);
    return b;
}

/// The production config by file name and a digest of its bytes, or "synthetic".
std::string config_id() {
    const char* cfg = std::getenv("GNSS_REPLAY_CONFIG");
    if (cfg == nullptr)
        return "synthetic";
    std::ifstream in(cfg, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string path(cfg);
    return path.substr(path.rfind('/') + 1) + " " + hex(fnv1a64(bytes.data(), bytes.size()));
}

json golden_of(const Digests& d) {
    json st = json::object();
    for (const auto& [k, v] : d.streams) {
        json a = json::array();
        for (uint64_t x : v)
            a.push_back(hex(x));
        st[k] = a;
    }
    return {{"host", short_hostname()},
            {"compiler", std::string(__VERSION__)},
            {"build_type", REPLAY_BUILD_TYPE},
            {"git_sha", get_git_commit_hash()},
            {"config", config_id()},
            {"streams", st},
            {"strings", d.strings}};
}

/// With Scenario::swap_chain, checks that in run @p d the swapped row restarted its slot cold on
/// the new PRN (PIN_B1), and that every other row kept its slot's PRN and ADR arc.
void check_one_row_swapped(const Digests& d, const Scenario& sc) {
    for (const ChainSpec& s : chains()) {
        const json& L = d.live.at(s.pre);
        const int was = s.prns[(size_t)sc.swap_slot];
        const json before = L.at("status@29").at((size_t)sc.swap_slot);
        const json after = L.at("status@59").at((size_t)sc.swap_slot);
        BOOST_CHECK_EQUAL(before[0].get<int>(), was);
        if (s.idx == sc.swap_chain) {
            const double swap_hop =
                (double)(REC_BASE + (int64_t)s.n_rec * sc.swap_frame) * s.hops_per_record;
            BOOST_CHECK_EQUAL(after[0].get<int>(), was + 100);
            BOOST_CHECK_MESSAGE(before[2] != after[2], s.pre << ": the ADR arc did not break");
            BOOST_CHECK_MESSAGE(
                L.at("records_min_hop@59").at(std::to_string(was + 100)).get<double>() >= swap_hop,
                s.pre << ": get_records holds a record from before the swap");
        } else {
            BOOST_CHECK_EQUAL(after[0].get<int>(), was);
            BOOST_CHECK_MESSAGE(before[2] == after[2], s.pre << ": the ADR arc broke: " << before[2]
                                                             << " -> " << after[2]);
        }
    }
}

/// A node as gen_chord_gnss_config.py builds it with --record-host --combine-host
/// --telem-pack-host on --gpu-rows: the chains as rows of a GPU half, on a worker each on every
/// host, with the swap references; with @p rec_shared also --half-rec-buf.
Scenario half_rec_node(bool rec_shared) {
    Scenario sc = base_scenario();
    const int n = (int)chains().size();
    sc.cold_ref = true;
    sc.rows = sc.fold = sc.gpu_rows = sc.telem_shared = true;
    sc.workers = sc.cmb_workers = sc.telem_workers = n;
    sc.rec_shared = rec_shared;
    return sc;
}

/// The child output of @p child, run forked: its exit code and what it wrote.
std::pair<int, std::string> child_output(const std::function<void(const std::string&)>& child) {
    char path[] = "/tmp/gnss_replay_XXXXXX";
    const int fd = mkstemp(path);
    BOOST_REQUIRE(fd >= 0);
    close(fd);
    const int rc = run_forked([&] { child(path); });
    std::ifstream f(path);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    unlink(path);
    return {rc, text};
}

} // namespace

BOOST_AUTO_TEST_CASE(deterministic) {
    if (!full_run())
        return;
    const Digests& a = baseline();
    const Digests b = run_scenario(chains(), base_scenario());
    Scenario pinned = base_scenario();
    pinned.cpu = 0;
    const Digests c = run_scenario(chains(), pinned);
    BOOST_CHECK_MESSAGE(changed(a, b).empty(), "baseline twice: " << join(changed(a, b)));
    BOOST_CHECK_MESSAGE(changed(a, c).empty(), "pinned to cpu 0: " << join(changed(a, c)));
    BOOST_CHECK(a.live == b.live && a.live == c.live);
    BOOST_CHECK(a.pins == b.pins && a.pins == c.pins);
    for (const auto& s : chains())
        BOOST_CHECK(a.slots.at(s.pre).at("own") == b.slots.at(s.pre).at("own")
                    && a.slots.at(s.pre).at("own") == c.slots.at(s.pre).at("own"));
    BOOST_TEST_MESSAGE(fmt::format("{:d} digest streams", a.streams.size()));
}

BOOST_AUTO_TEST_CASE(live) {
    const Digests& d = baseline();
    const Digests z = run_with(Mutation::zero_amp);
    const Scenario sc = base_scenario();
    std::string why;
    for (const auto& s : chains()) {
        BOOST_CHECK_MESSAGE(cal_live(d, s, &why), why);
        BOOST_TEST_MESSAGE(why);
        BOOST_CHECK_MESSAGE(combiner_live(d, s, &why), why);
        BOOST_TEST_MESSAGE(why);
        BOOST_CHECK_MESSAGE(cube_live(d, s, &why), why);
        BOOST_TEST_MESSAGE(why);
        const json& L = d.live.at(s.pre);
        BOOST_CHECK_EQUAL(L.value("telem_frames", 0), sc.n_frames);
        BOOST_CHECK_EQUAL(L.value("telem_bad_present", -1), 0);
        BOOST_CHECK_EQUAL(L.value("telem_bad_meta", -1), 0);
        // The zero-amplitude sky must fail every one of these.
        BOOST_CHECK_MESSAGE(!cal_live(z, s, &why), "zero_amp passed: " << why);
        BOOST_CHECK_MESSAGE(!combiner_live(z, s, &why), "zero_amp passed: " << why);
        BOOST_CHECK_MESSAGE(!cube_live(z, s, &why), "zero_amp passed: " << why);
    }
    // The projection posted live on A at proj_live_frame is off before it and in force after.
    {
        const json& L = d.live.at(chains()[0].pre);
        int after = sc.checkpoints.back();
        for (int cp : sc.checkpoints)
            if (cp > sc.proj_live_frame && cp < after)
                after = cp;
        const json pb = L.value("proj@" + std::to_string(sc.checkpoints[0]), json());
        const json pa = L.value("proj@" + std::to_string(after), json());
        BOOST_CHECK_MESSAGE(pb == json::array({"off", 0}), "projection before the POST: " << pb);
        BOOST_CHECK_MESSAGE(pa.is_array() && pa[0] == "live" && pa[1].get<int>() > 0,
                            "projection after the POST: " << pa);
        BOOST_TEST_MESSAGE("projection [mode, k] at " << sc.checkpoints[0] << ": " << pb << ", at "
                                                      << after << ": " << pa);
    }
    for (int cp : {sc.checkpoints[sc.checkpoints.size() - 2], sc.checkpoints.back()}) {
        BOOST_CHECK_MESSAGE(board_ok(d, chains(), sc, cp, &why), why);
        BOOST_TEST_MESSAGE(why);
        const std::set<std::pair<int, int>> sh = shared_lanes(d, chains(), cp);
        BOOST_CHECK_MESSAGE(!sh.empty(), "board@" << cp << ": no lane published by two chains");
        BOOST_TEST_MESSAGE(fmt::format(
            "board@{:d}: {:d} (freq_id, PRN) lanes published by two chains as two entries", cp,
            sh.size()));
        // A and C share freq_id 6033 and PRN 6 (synthetic_chains).
        if (!std::getenv("GNSS_REPLAY_CONFIG"))
            BOOST_CHECK_MESSAGE(sh.count({6033, 6}) == 1,
                                "board@" << cp
                                         << ": (6033, PRN 6) is not one entry each from A "
                                            "and C");
    }
}

/// Without out_buf each combiner emits to its scratch, and every digest, its bytes included, is
/// the baseline's, whose combiners write out_buf.
BOOST_AUTO_TEST_CASE(combiner_without_out_buf) {
    if (!full_run())
        return;
    const Digests& base = baseline();
    Scenario sc = base_scenario();
    sc.combiner_out_buf = false;
    const Digests d = run_scenario(chains(), sc);
    BOOST_CHECK_MESSAGE(changed(base, d).empty(), "without out_buf: " << join(changed(base, d)));
    BOOST_CHECK(base.live == d.live);
    BOOST_CHECK(base.pins == d.pins);
    for (const auto& s : chains()) {
        BOOST_CHECK_MESSAGE(!d.streams.at(s.pre + ".cmb").empty(), s.pre << ": no combiner emit");
        BOOST_CHECK(base.slots.at(s.pre).at("own") == d.slots.at(s.pre).at("own"));
    }
}

/// The assemblers, combiners and telemetry packs hosted as rows of one stage each give the
/// baseline's digests, endpoints, checks and per-slot reads.
BOOST_AUTO_TEST_CASE(rows) {
    const Digests& base = baseline();
    Scenario sc = base_scenario();
    sc.cold_ref = true;
    sc.rows = true;
    const Digests d = run_scenario(chains(), sc);
    BOOST_CHECK_MESSAGE(changed(base, d).empty(), "as rows: " << join(changed(base, d)));
    BOOST_CHECK(base.live == d.live);
    BOOST_CHECK(base.pins == d.pins);
    BOOST_CHECK(base.slots == d.slots);
}

/// The assemblers reading the tiles and control blocks themselves give the baseline's digests,
/// their epl frames included; with GNSS_REPLAY_FULL=1 also as rows of one host, and with their
/// epl frames left stale between frames.
BOOST_AUTO_TEST_CASE(fold) {
    const Digests& base = baseline();
    Scenario sc = base_scenario();
    sc.cold_ref = true;
    sc.fold = true;
    std::vector<Scenario> runs = {sc};
    if (full_run()) {
        runs.push_back(sc);
        runs.back().rows = true;
        runs.push_back(sc);
        runs.back().refill_a5 = false;
    }
    for (const Scenario& x : runs) {
        const Digests d = run_scenario(chains(), x);
        BOOST_CHECK_MESSAGE(changed(base, d).empty(), "fold, rows " << x.rows << ", refill "
                                                                    << x.refill_a5 << ": "
                                                                    << join(changed(base, d)));
        BOOST_CHECK(base.live == d.live);
        BOOST_CHECK(base.pins == d.pins);
        BOOST_CHECK(base.slots == d.slots);
    }
}

/// The chains as rows of one GPU half, each assembler reading its row of whole-half tiles and
/// control-block frames, give the baseline's digests; with GNSS_REPLAY_FULL=1 also as rows of
/// one host. With one pair of buffers that every assembler reads, as on a node, the chains
/// process each frame concurrently, so only what a frame alone determines can be compared: the
/// epl frames. The rest depends on the order in which chains read each other's ProjBoard entries.
BOOST_AUTO_TEST_CASE(gpu_rows) {
    const Digests& base = baseline();
    Scenario sc = base_scenario();
    sc.cold_ref = true;
    sc.fold = true;
    sc.gpu_rows = true;
    std::vector<Scenario> runs = {sc};
    if (full_run()) {
        runs.push_back(sc);
        runs.back().rows = true;
    }
    for (const Scenario& x : runs) {
        const Digests d = run_scenario(chains(), x);
        BOOST_CHECK_MESSAGE(changed(base, d).empty(),
                            "gpu rows, rows " << x.rows << ": " << join(changed(base, d)));
        BOOST_CHECK(base.live == d.live);
        BOOST_CHECK(base.pins == d.pins);
        BOOST_CHECK(base.slots == d.slots);
    }
    sc.gpu_rows_shared = true;
    const Digests d = run_scenario(chains(), sc);
    for (const auto& s : chains())
        BOOST_CHECK_MESSAGE(d.streams.at(s.pre + ".epl") == base.streams.at(s.pre + ".epl"),
                            "gpu rows shared: " << s.pre << ".epl");
    BOOST_CHECK(base.pins == d.pins);
}

/// The chains as rows of one GPU half on one host whose rows run on workers, each reading its
/// rows' tiles and control blocks as one consumer, give the baseline's digests with one worker
/// per row, with more rows than workers and with one worker; with GNSS_REPLAY_FULL=1 also as
/// rows reading the adapters' epl frames. With one pair of buffers that every worker reads, as on
/// a node, the workers process each frame concurrently, so only the epl frames can be compared,
/// except with one worker, which runs every row of the frame in turn.
BOOST_AUTO_TEST_CASE(workers) {
    const Digests& base = baseline();
    const int n = (int)chains().size();
    for (int w : {n, 2, 1}) {
        Scenario sc = base_scenario();
        sc.cold_ref = true;
        sc.fold = true;
        sc.gpu_rows = true;
        sc.rows = true;
        sc.workers = w;
        Digests d = run_scenario(chains(), sc);
        BOOST_CHECK_MESSAGE(changed(base, d).empty(), w << " workers: " << join(changed(base, d)));
        BOOST_CHECK(base.live == d.live);
        BOOST_CHECK(base.pins == d.pins);
        BOOST_CHECK(base.slots == d.slots);
        sc.gpu_rows_shared = true;
        d = run_scenario(chains(), sc);
        for (const auto& s : chains())
            BOOST_CHECK_MESSAGE(d.streams.at(s.pre + ".epl") == base.streams.at(s.pre + ".epl"),
                                w << " workers, shared: " << s.pre << ".epl");
        BOOST_CHECK(base.pins == d.pins);
        if (w == 1)
            BOOST_CHECK_MESSAGE(changed(base, d).empty(),
                                "1 worker, shared: " << join(changed(base, d)));
    }
    if (!full_run())
        return;
    Scenario sc = base_scenario();
    sc.cold_ref = true;
    sc.rows = true;
    sc.workers = 2;
    const Digests d = run_scenario(chains(), sc);
    BOOST_CHECK_MESSAGE(changed(base, d).empty(),
                        "unfolded, 2 workers: " << join(changed(base, d)));
    BOOST_CHECK(base.live == d.live);
    BOOST_CHECK(base.slots == d.slots);
}

/// Two workers, more rows than workers (chain i on worker i % 2), holding first chain 0's record
/// consumer for STALL_HOLD_S at its record STALL_AT, and then worker 0 itself between its reads of
/// the tiles and control blocks. While worker 0 is held, worker 1 runs on until the feeder has
/// used the buffers' depth and is blocked; then every row takes every record of every frame once,
/// in order, the held rows included.
BOOST_AUTO_TEST_CASE(workers_stall) {
    const std::vector<ChainSpec>& cs = synthetic();
    for (const Hold hold : {Hold::rec, Hold::worker}) {
        const char* what = hold == Hold::rec ? "record consumer held" : "worker held";
        char path[] = "/tmp/gnss_replay_XXXXXX";
        const int fd = mkstemp(path);
        BOOST_REQUIRE(fd >= 0);
        close(fd);
        const int rc = run_forked([&] { stall_child(cs, hold, path); });
        std::ifstream in(path);
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        unlink(path);
        BOOST_REQUIRE_MESSAGE(rc == 0, what << ": child exit " << rc << " " << text);
        const json j = json::parse(text);
        const double h0 = j.at("hold")[0].get<double>(), h1 = j.at("hold")[1].get<double>();
        BOOST_REQUIRE_MESSAGE(h1 - h0 >= STALL_HOLD_S, what << ": no hold");
        const std::vector<double> fed = j.at("fed").get<std::vector<double>>();
        int fed_max = -1, fed_late = 0;
        for (int k = 0; k < STALL_FRAMES; ++k) {
            fed_max = fed[k] < h1 ? k : fed_max;
            fed_late += fed[k] > h0 + 0.5 && fed[k] < h1;
        }
        // The last frame of which each worker's rows took a record before the hold ended.
        int last[2] = {-1, -1};
        for (size_t i = 0; i < cs.size(); ++i) {
            const ChainSpec& s = cs[i];
            const int64_t rec_samples = (int64_t)s.hops_per_record * s.fft_len;
            std::vector<int64_t> want, got;
            int after = 0;
            for (int k = 0; k < STALL_FRAMES * s.n_rec; ++k)
                want.push_back((REC_BASE + k) * rec_samples);
            for (const json& t : j.at("taken").at(i)) {
                got.push_back(t[1].get<int64_t>());
                const int frame = (int)((got.back() / rec_samples - REC_BASE) / s.n_rec);
                if (t[0].get<double>() < h1)
                    last[i % 2] = std::max(last[i % 2], frame);
                else
                    ++after;
            }
            BOOST_CHECK_MESSAGE(got == want, what << ": " << s.pre << " took " << got.size()
                                                  << " records, not each of " << want.size()
                                                  << " once in order");
            BOOST_CHECK_MESSAGE(after > 0, what << ": " << s.pre << " took nothing after the hold");
        }
        BOOST_TEST_MESSAGE(fmt::format("{:s}: {:.2f} s; fed frames 0-{:d} before its end, {:d} "
                                       "in its last {:.1f} s; worker 0 at frame {:d}, worker 1 "
                                       "at {:d}",
                                       what, h1 - h0, fed_max, fed_late, h1 - h0 - 0.5, last[0],
                                       last[1]));
        BOOST_CHECK_MESSAGE(fed_late == 0, what << ": the feeder was not blocked");
        BOOST_CHECK_MESSAGE(last[1] == fed_max, what << ": worker 1 stopped at frame " << last[1]
                                                     << ", the feeder at " << fed_max);
        BOOST_CHECK_MESSAGE(fed_max - last[0] >= STALL_DEPTH
                                && fed_max - last[0] <= STALL_DEPTH + 1,
                            what << ": the feeder ran " << fed_max - last[0]
                                 << " frames ahead of worker 0, not the depth " << STALL_DEPTH);
    }
}

/// The combiners as rows of one host whose rows run on workers, each taking a record of each of
/// its rows in turn as one consumer of their in_bufs, give the baseline's digests with more rows
/// than workers; with GNSS_REPLAY_FULL=1 also with one worker per row and with one worker, without
/// out_buf, and beside assemblers that read their rows of the half's tiles and control blocks, on
/// one worker per row each, as gen_chord_gnss_config.py --record-host --combine-host builds a
/// node, and on two workers each, so that one record worker feeds each combine worker's rows.
BOOST_AUTO_TEST_CASE(combine_workers) {
    const Digests& base = baseline();
    const int n = (int)chains().size();
    std::vector<Scenario> runs(1, base_scenario());
    runs[0].cold_ref = true;
    runs[0].rows = true;
    runs[0].cmb_workers = 2;
    if (full_run()) {
        for (int w : {n, 1}) {
            runs.push_back(runs[0]);
            runs.back().cmb_workers = w;
        }
        runs.push_back(runs[0]);
        runs.back().combiner_out_buf = false;
        for (int w : {n, 2}) {
            runs.push_back(runs[0]);
            runs.back().cmb_workers = runs.back().workers = w;
            runs.back().fold = runs.back().gpu_rows = true;
        }
    }
    for (const Scenario& x : runs) {
        const Digests d = run_scenario(chains(), x);
        BOOST_CHECK_MESSAGE(changed(base, d).empty(),
                            x.cmb_workers << " combine workers, out_buf " << x.combiner_out_buf
                                          << ", record workers " << x.workers << ": "
                                          << join(changed(base, d)));
        BOOST_CHECK(base.live == d.live);
        BOOST_CHECK(base.pins == d.pins);
        BOOST_CHECK(base.slots == d.slots);
    }
}

/// The telemetry packs as rows of one host that write one out_buf, which one capture reads and
/// sorts by chain, give the baseline's digests, each frame read at its own length, on one worker
/// per row as gen_chord_gnss_config.py --telem-pack-host builds a node; with GNSS_REPLAY_FULL=1
/// also with more rows than workers, with one worker and without a workers block.
BOOST_AUTO_TEST_CASE(telem_rows) {
    const Digests& base = baseline();
    const int n = (int)chains().size();
    std::vector<Scenario> runs(1, base_scenario());
    runs[0].cold_ref = true;
    runs[0].rows = true;
    runs[0].telem_shared = true;
    runs[0].telem_workers = n;
    if (full_run())
        for (int w : {2, 1, 0}) {
            runs.push_back(runs[0]);
            runs.back().telem_workers = w;
        }
    for (const Scenario& x : runs) {
        const Digests d = run_scenario(chains(), x);
        BOOST_CHECK_MESSAGE(changed(base, d).empty(),
                            x.telem_workers << " telemetry workers: " << join(changed(base, d)));
        BOOST_CHECK(base.live == d.live);
        BOOST_CHECK(base.pins == d.pins);
        BOOST_CHECK(base.slots == d.slots);
    }
}

/// A PRN swap on one row of a combine host on workers, with more rows than workers: the run
/// equals the same swap through one stage per chain, the swapped row restarts the slot cold
/// (PIN_B1), and the other rows, on its worker and on the other, keep their slot's PRN and ADR
/// arc.
BOOST_AUTO_TEST_CASE(combine_workers_swap_one_row) {
    if (!full_run())
        return;
    Scenario sc = base_scenario();
    sc.swap_chain = 1;
    const Digests stages = run_scenario(chains(), sc);
    sc.rows = true;
    sc.cmb_workers = 2;
    const Digests d = run_scenario(chains(), sc);
    BOOST_CHECK_MESSAGE(changed(stages, d).empty(),
                        "one row swapped, 2 combine workers: " << join(changed(stages, d)));
    BOOST_CHECK(stages.live == d.live);
    BOOST_CHECK(stages.slots == d.slots);
    check_one_row_swapped(d, sc);
}

/// The combiners as rows of one host on two workers, more rows than workers (chain i on worker
/// i % 2), holding for STALL_HOLD_S first chain 0's input, so that worker 0 waits for its next
/// record, and then chain 0's out_buf consumer, so that worker 0 blocks in an emit. Chain 1, on
/// worker 1, keeps up through either hold; chain 2 shares worker 0 and stalls with chain 0; then
/// every row takes every record once, in order, the held rows included, and reads none after
/// releasing it.
BOOST_AUTO_TEST_CASE(combine_workers_stall) {
    if (!full_run())
        return;
    const std::vector<ChainSpec>& cs = synthetic();
    BOOST_REQUIRE(cs.size() == 3);
    for (const CmbHold hold : {CmbHold::input, CmbHold::output}) {
        const char* what = hold == CmbHold::input ? "input held" : "out_buf consumer held";
        char path[] = "/tmp/gnss_replay_XXXXXX";
        const int fd = mkstemp(path);
        BOOST_REQUIRE(fd >= 0);
        close(fd);
        const int rc = run_forked([&] { cmb_stall_child(cs, hold, path); });
        std::ifstream in(path);
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        unlink(path);
        BOOST_REQUIRE_MESSAGE(rc == 0, what << ": child exit " << rc << " " << text);
        const json j = json::parse(text);
        const double h0 = j.at("hold")[0].get<double>(), h1 = j.at("hold")[1].get<double>();
        const double t0 = j.at("t0").get<double>();
        BOOST_REQUIRE_MESSAGE(h1 - h0 >= STALL_HOLD_S, what << ": no hold");
        int during[3] = {0, 0, 0}, after[3] = {0, 0, 0};
        double late[3] = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < cs.size(); ++i) {
            const ChainSpec& s = cs[i];
            std::vector<int64_t> want, got;
            for (int r = 0; r < CMB_STALL_RECORDS; ++r)
                want.push_back((REC_BASE + r) * s.hops_per_record);
            for (const json& t : j.at("taken").at(i)) {
                got.push_back(t[1].get<int64_t>());
                const double at = t[0].get<double>();
                during[i] += at > h0 + 0.25 && at < h1 - 0.05;
                after[i] += at >= h1;
            }
            const std::vector<double> fed = j.at("fed").at(i).get<std::vector<double>>();
            for (int r = 0; r < CMB_STALL_RECORDS; ++r)
                late[i] = std::max(late[i], fed[r] - (t0 + r * CMB_STALL_DT_S));
            BOOST_CHECK_MESSAGE(got == want, what << ": " << s.pre << " took " << got.size()
                                                  << " records, not each of " << want.size()
                                                  << " once in order, none read after release");
            BOOST_CHECK_MESSAGE(after[i] > 0, what << ": " << s.pre
                                                   << " took nothing after the "
                                                      "hold");
        }
        const double held_s = h1 - h0 - 0.3;
        BOOST_TEST_MESSAGE(
            fmt::format("{:s}: {:.2f} s; records taken in it by chain 0 (held) {:d}, "
                        "chain 1 (other worker) {:d}, chain 2 (held worker) {:d}; "
                        "feeders at most {:.3f} / {:.3f} / {:.3f} s late",
                        what, h1 - h0, during[0], during[1], during[2], late[0], late[1], late[2]));
        BOOST_CHECK_MESSAGE(during[1] >= 0.8 * held_s / CMB_STALL_DT_S,
                            what << ": chain 1 took " << during[1] << " records in the hold");
        BOOST_CHECK_MESSAGE(late[1] < 0.1,
                            what << ": chain 1's feeder was held " << late[1] << " s");
        BOOST_CHECK_MESSAGE(during[0] == 0 && during[2] <= 8,
                            what << ": chains 0 and 2 took " << during[0] << " and " << during[2]
                                 << " records in the hold");
    }
}

/// The chains' assemblers writing one record buffer of the GPU half, which their combiners and
/// telemetry packs read, each chain at its rec_row, give the baseline's digests, each chain's
/// records digested from its row of each frame: with one worker per row on every host, as a node
/// with --half-rec-buf, and with more rows than workers and one worker on the record and combine
/// hosts; with GNSS_REPLAY_FULL=1 also the same node without --half-rec-buf, and with one pair of
/// tiles and control-block buffers, where the record workers run each frame concurrently, so that
/// with one worker per row only the epl frames can be compared and with one worker everything.
BOOST_AUTO_TEST_CASE(half_rec_buf) {
    const Digests& base = baseline();
    const int n = (int)chains().size();
    std::vector<Scenario> runs;
    for (int w : {n, 2, 1}) {
        runs.push_back(half_rec_node(true));
        runs.back().workers = runs.back().cmb_workers = w;
    }
    if (full_run()) {
        runs.push_back(half_rec_node(false));
        runs.push_back(half_rec_node(true));
        runs.back().gpu_rows_shared = true;
        runs.back().workers = runs.back().cmb_workers = runs.back().telem_workers = 1;
    }
    for (const Scenario& x : runs) {
        const Digests d = run_scenario(chains(), x);
        BOOST_CHECK_MESSAGE(changed(base, d).empty(),
                            "record buffer " << x.rec_shared << ", " << x.workers << " record, "
                                             << x.cmb_workers << " combine and " << x.telem_workers
                                             << " telemetry workers, shared tiles "
                                             << x.gpu_rows_shared << ": "
                                             << join(changed(base, d)));
        BOOST_CHECK(base.live == d.live);
        BOOST_CHECK(base.pins == d.pins);
        BOOST_CHECK(base.slots == d.slots);
    }
    if (!full_run())
        return;
    Scenario x = half_rec_node(true);
    x.gpu_rows_shared = true;
    const Digests d = run_scenario(chains(), x);
    for (const auto& s : chains())
        BOOST_CHECK_MESSAGE(d.streams.at(s.pre + ".epl") == base.streams.at(s.pre + ".epl"),
                            "record buffer, shared tiles: " << s.pre << ".epl");
    BOOST_CHECK(base.pins == d.pins);
}

/// The POSTs through the hosts by chain, and the GETs there, ?chain=<c>, give the baseline's
/// digests, as a node with --half-rec-buf; with GNSS_REPLAY_FULL=1 also with the rows reading the
/// adapters' epl frames, each naming its chain without a gnss_row.
BOOST_AUTO_TEST_CASE(by_chain) {
    const Digests& base = baseline();
    std::vector<Scenario> runs(1, half_rec_node(true));
    if (full_run()) {
        runs.push_back(base_scenario());
        runs.back().cold_ref = true;
        runs.back().rows = true;
    }
    for (Scenario& x : runs) {
        x.by_chain = true;
        const Digests d = run_scenario(chains(), x);
        BOOST_CHECK_MESSAGE(changed(base, d).empty(),
                            "by chain, gpu rows " << x.gpu_rows << ": " << join(changed(base, d)));
        BOOST_CHECK(base.live == d.live);
        BOOST_CHECK(base.pins == d.pins);
        BOOST_CHECK(base.slots == d.slots);
    }
}

/// A PRN swap on one row with the half's one record buffer and more rows than workers: the run
/// equals the same swap through one stage per chain, and only the swapped row restarts its slot.
BOOST_AUTO_TEST_CASE(half_rec_buf_swap_one_row) {
    if (!full_run())
        return;
    Scenario sc = base_scenario();
    sc.swap_chain = 1;
    const Digests stages = run_scenario(chains(), sc);
    Scenario x = half_rec_node(true);
    x.cold_ref = false;
    x.swap_chain = 1;
    x.workers = x.cmb_workers = x.telem_workers = 2;
    const Digests d = run_scenario(chains(), x);
    BOOST_CHECK_MESSAGE(changed(stages, d).empty(),
                        "one row swapped, record buffer: " << join(changed(stages, d)));
    BOOST_CHECK(stages.live == d.live);
    BOOST_CHECK(stages.slots == d.slots);
    check_one_row_swapped(d, x);
}

/// With the half's one record buffer a host FATALs on a row whose rec_row disagrees with the
/// frames the real assemblers write: on the first frame, a combine row with another chain_id and a
/// telemetry host that reads a row less; at the start, an assembler row whose rec_row does not
/// hold its record and a telemetry row off the frame's tiling.
BOOST_AUTO_TEST_CASE(half_rec_buf_refuses_a_mismatch) {
    const std::vector<ChainSpec>& cs = chains();
    const ChainSpec& b = cs[1];
    const std::vector<gnss::RecRow> rr = rec_rows_of(cs);
    const std::string buf = rec_shared_buf(cs);
    const std::string g = "/gnss" + std::to_string(cs[0].gpu);
    const std::vector<std::pair<Mutation, std::string>> cases = {
        {Mutation::rec_chain_id_cmb_B,
         g + "_combine/w0: frame 0 of " + buf
             + ": directory row 1 has chain_id 2, this host's row 1 (chain_id 99) has 99"},
        {Mutation::rec_rows_tel_AB, g + "_telem/w0: frame 0 of " + buf
                                        + " lists 3 rows (chain_id 1, 2, 3), this host reads 2 "
                                          "(chain_id 1, 2)"},
        {Mutation::rec_n_prn_as_B,
         fmt::format("/{:s}_n2assemble: rec_row holds {:d} PRN x {:d} comb columns in {:d} B, "
                     "this row's record is {:d} PRN of {:d} elements x {:d} comb columns in {:d} B",
                     b.pre, b.n_prn + 1, b.n_chan, rr[1].bytes, b.n_prn, b.n_live, b.n_chan,
                     rr[1].bytes)},
        {Mutation::rec_offset_tel_B,
         fmt::format("{:s}_telem: the rec_row of chain_id 2 holds bytes [{:d}, {:d}) of {:s}; the "
                     "directory and the rows before it put it at byte {:d}",
                     g, rr[1].offset + 64, rr[1].offset + 64 + rr[1].bytes, buf, rr[1].offset)}};
    for (const auto& [m, want] : cases) {
        Scenario sc = half_rec_node(true);
        sc.cold_ref = false;
        sc.cmb_workers = sc.telem_workers = 1;
        sc.mut = m;
        const auto [rc, msg] =
            child_output([&](const std::string& path) { child_main(cs, sc, path); });
        BOOST_CHECK_MESSAGE(rc == 70 && msg == want,
                            "mutation " << (int)m << ": child exit " << rc << ", " << msg);
        BOOST_TEST_MESSAGE("mutation " << (int)m << ": " << msg);
    }
}

/// The half's one record buffer, its rows on two workers on every host (more rows than workers),
/// fed at the frame rate, held for STALL_HOLD_S by the reader of combine row 0's out_buf, then of
/// telemetry row 0's, then by record worker 0 itself. A held reader holds every row's frames: the
/// writers run on until they have filled the record buffer, the feeder until it has filled the
/// tiles and control blocks too, and it is on time until then. A held writer holds every row's
/// frames from the readers, and the other writer runs a buffer's depth ahead of it. Then every
/// reader takes every record of every row once, in order, each row's block whole.
BOOST_AUTO_TEST_CASE(half_rec_buf_stall) {
    const std::vector<ChainSpec>& cs = synthetic();
    BOOST_REQUIRE(cs.size() == 3);
    for (const RecHold hold : {RecHold::combine, RecHold::telem, RecHold::record}) {
        const char* what = hold == RecHold::combine ? "combine row 0's reader held"
                           : hold == RecHold::telem ? "telemetry row 0's reader held"
                                                    : "record worker 0 held";
        const auto [rc, text] =
            child_output([&](const std::string& path) { rec_stall_child(cs, hold, path); });
        BOOST_REQUIRE_MESSAGE(rc == 0, what << ": child exit " << rc << " " << text);
        const json j = json::parse(text);
        const double h0 = j.at("hold")[0].get<double>(), h1 = j.at("hold")[1].get<double>();
        BOOST_REQUIRE_MESSAGE(h1 - h0 >= STALL_HOLD_S, what << ": no hold");
        const int n_rec = j.at("n_rec").get<int>();
        const int n = REC_STALL_FRAMES * n_rec;
        // Every reader took every record of every row once, in order, each block whole.
        for (size_t i = 0; i < cs.size(); ++i) {
            const ChainSpec& s = cs[i];
            const int64_t rec_samples = (int64_t)s.hops_per_record * s.fft_len;
            std::vector<int64_t> want_w, got_w, want_hop, got_hop;
            for (int r = 0; r < n; ++r) {
                want_w.push_back((REC_BASE + r) * rec_samples);
                want_hop.push_back((REC_BASE + r) * s.hops_per_record);
            }
            for (const json& t : j.at("rec").at(i))
                got_w.push_back(t[1].get<int64_t>());
            for (const json& t : j.at("cmb").at(i))
                got_hop.push_back(t[1].get<int64_t>());
            BOOST_CHECK_MESSAGE(got_w == want_w, what << ": " << s.pre << ": the record buffer "
                                                      << "gave " << got_w.size()
                                                      << " records, not each once in order");
            BOOST_CHECK_MESSAGE(j.at("torn").at(i).get<int>() == 0,
                                what << ": " << s.pre << ": " << j.at("torn").at(i)
                                     << " records with PRN rows of another record");
            BOOST_CHECK_MESSAGE(got_hop == want_hop, what << ": " << s.pre << ": the combiner took "
                                                          << got_hop.size()
                                                          << " records, not each once in order");
            const json& win = j.at("telem").at(i);
            bool in_order = (int)win.size() == n / n_rec;
            for (size_t k = 0; in_order && k < win.size(); ++k)
                in_order = win[k][1].get<double>() == (double)k
                           && win[k][2].get<double>() == (double)(REC_BASE / n_rec + (int64_t)k)
                           && win[k][3].get<double>() == (double)((1 << n_rec) - 1);
            BOOST_CHECK_MESSAGE(in_order, what << ": " << s.pre << ": " << win.size()
                                               << " telemetry windows, not each once, whole, in "
                                                  "order");
        }
        // The feeder is on time until the hold has used the slack before it: the tiles' and
        // control blocks' depth, and with a reader held the record buffer's too. Then it waits.
        const double t0 = j.at("t0").get<double>(), dt = j.at("dt").get<double>();
        const std::vector<double> fed = j.at("fed").get<std::vector<double>>();
        const int slack =
            STALL_DEPTH + (hold == RecHold::record ? 0 : (REC_SHARED_DEPTH - 1) / n_rec);
        int on_time = 0, first_late = -1;
        double late_max = 0.0;
        for (int k = 0; k < REC_STALL_FRAMES; ++k) {
            const double late = fed[k] - (t0 + k * dt);
            late_max = std::max(late_max, late);
            if (t0 + k * dt < h0)
                BOOST_CHECK_MESSAGE(late < dt / 2, what << ": frame " << k << " fed " << late
                                                        << " s late before the hold");
            else if (first_late < 0 && late >= dt / 2)
                first_late = k;
            else if (first_late < 0)
                ++on_time;
        }
        BOOST_TEST_MESSAGE(fmt::format("{:s}: {:.2f} s; the feeder fed {:d} frames on time in it, "
                                       "then waited up to {:.2f} s",
                                       what, h1 - h0, on_time, late_max));
        BOOST_CHECK_MESSAGE(on_time >= slack - 1, what << ": the feeder was late after " << on_time
                                                       << " frames of the hold, not " << slack - 1);
        BOOST_CHECK_MESSAGE(late_max > 1.0, what << ": the hold never reached the feeder");
        // At the end of the hold: what each host had done (records written per row, records
        // combined per row) and the record buffer's full frames.
        json last;
        for (const json& p : j.at("progress"))
            if (p[0].get<double>() < h1 - 0.05)
                last = p;
        BOOST_REQUIRE(!last.is_null());
        const int64_t w0 = last[1], w1 = last[2], w2 = last[3], c0 = last[4], c1 = last[5],
                      c2 = last[6], full = last[7];
        BOOST_TEST_MESSAGE(fmt::format("{:s}: at its end, records written {:d} {:d} {:d}, "
                                       "combined {:d} {:d} {:d}, {:d} frames full",
                                       what, w0, w1, w2, c0, c1, c2, full));
        const int64_t w_min = std::min({w0, w1, w2});
        if (hold == RecHold::record) {
            // Worker 1 ran the record buffer's depth, less the frame peek_hold keeps, ahead of
            // worker 0's last row, which marks worker 0's frames, and the readers went no further
            // than that row.
            BOOST_CHECK_MESSAGE(w1 - w2 >= REC_SHARED_DEPTH - 2 && w1 - w2 <= REC_SHARED_DEPTH + 1,
                                what << ": worker 1 ran " << w1 - w2 << " records ahead, not "
                                     << REC_SHARED_DEPTH - 1);
            BOOST_CHECK_MESSAGE(full <= 1 && c1 <= w2 && c1 >= w2 - 1,
                                what << ": combine row 1 took " << c1 << " records of worker 0's "
                                     << w2 << ", " << full << " frames full");
        } else {
            // Every frame is held, every row's: full, or written by the rows of one record
            // worker but not yet by those of the other; the writers stopped within a correlator
            // frame of each other, a buffer's depth on, and the readers not held took all they
            // wrote.
            const int64_t w_max = std::max({w0, w1, w2});
            BOOST_CHECK_MESSAGE(full + w_max - w_min == REC_SHARED_DEPTH,
                                what << ": " << full << " frames full and " << w_max - w_min
                                     << " written in part, not the " << REC_SHARED_DEPTH);
            BOOST_CHECK_MESSAGE(w_max - w_min <= n_rec,
                                what << ": the rows' writers stopped apart");
            if (hold == RecHold::combine)
                BOOST_CHECK_MESSAGE(c1 >= w_min - 1 && w_min - c0 >= REC_SHARED_DEPTH - 2,
                                    what << ": combine row 1 took " << c1 << " of " << w_min
                                         << " records, row 0 " << c0);
            else
                BOOST_CHECK_MESSAGE(std::min({c0, c1, c2}) >= w_min - 1,
                                    what << ": the combiners took " << c0 << ", " << c1 << ", "
                                         << c2 << " of " << w_min << " records");
        }
    }
}

/// A row FATALs on another row's control block: chain A's and B's rows swapping their ctl_offset
/// or their fg_offset, which shape checks cannot see between rows of one shape, B's block
/// carrying another row's tag, or A and B swapping their whole rows, tag included.
BOOST_AUTO_TEST_CASE(gpu_rows_refuse_a_mismatch) {
    const std::string A = "/" + chains()[0].pre + "_n2assemble: ";
    const std::string B = "/" + chains()[1].pre + "_n2assemble: ";
    for (const Mutation m : {Mutation::ctl_offset_swap_AB, Mutation::fg_offset_swap_AB,
                             Mutation::row_tag_B3, Mutation::row_swap_AB}) {
        Scenario sc = base_scenario();
        sc.fold = true;
        sc.gpu_rows = true;
        // Both rows of a swap FATAL, so they are fed one after the other: two at once can leave
        // the child's SIGTERM handler without either message.
        sc.gpu_rows_shared = m == Mutation::row_tag_B3;
        sc.mut = m;
        char path[] = "/tmp/gnss_replay_XXXXXX";
        const int fd = mkstemp(path);
        BOOST_REQUIRE(fd >= 0);
        close(fd);
        const int rc = run_child(chains(), sc, path);
        std::ifstream f(path);
        const std::string msg((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        unlink(path);
        // Rows of one shape read each other's blocks and refuse them by their tag; a swap
        // between rows of different shapes may already put a row outside the frames.
        const bool who = msg.compare(0, B.size(), B) == 0
                         || (m != Mutation::row_tag_B3 && msg.compare(0, A.size(), A) == 0);
        const bool same_shape =
            chains()[0].n_prn == chains()[1].n_prn && chains()[0].n_chan == chains()[1].n_chan;
        const bool why =
            m == Mutation::row_swap_AB
                ? msg.find("is another chain's row") != std::string::npos
                : msg.find("carries row tag") != std::string::npos
                      || (!same_shape && msg.find("are not inside") != std::string::npos);
        BOOST_CHECK_MESSAGE(rc == 70 && who && why,
                            "mutation " << (int)m << ": child exit " << rc << ", " << msg);
        BOOST_TEST_MESSAGE("mutation " << (int)m << ": " << msg);
    }
}

/// An assembler reading the tiles and control blocks itself FATALs when the tiles frame is not
/// the control block's or the control block is not the configured shape.
BOOST_AUTO_TEST_CASE(fold_refuses_a_mismatch) {
    const std::string B = "/" + chains()[1].pre + "_n2assemble: ";
    const std::vector<std::pair<Mutation, std::string>> cases = {
        {Mutation::seq0_skew_B3, "the two buffers are out of step"},
        {Mutation::n_prn_B3, "the control block is"},
        {Mutation::n_chan_B3, "the control block is"},
        {Mutation::lanes_B3, "lanes per slot"}};
    for (const auto& [m, want] : cases) {
        Scenario sc = base_scenario();
        sc.fold = true;
        sc.mut = m;
        char path[] = "/tmp/gnss_replay_XXXXXX";
        const int fd = mkstemp(path);
        BOOST_REQUIRE(fd >= 0);
        close(fd);
        const int rc = run_child(chains(), sc, path);
        std::ifstream f(path);
        const std::string msg((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        unlink(path);
        BOOST_CHECK_MESSAGE(rc == 70 && msg.compare(0, B.size(), B) == 0
                                && msg.find(want) != std::string::npos,
                            "mutation " << (int)m << ": child exit " << rc << ", " << msg);
        BOOST_TEST_MESSAGE("mutation " << (int)m << ": " << msg);
    }
}

BOOST_AUTO_TEST_CASE(identity) {
    const Digests& d = baseline();
    const Scenario sc = base_scenario();
    const std::string host = short_hostname();
    for (size_t i = 0; i < chains().size(); ++i) {
        const ChainSpec& s = chains()[i];
        const std::string want =
            (host + "/" + assembler_name(s, (int)i, sc)).substr(0, gnss::CUBE_CHAIN_CHARS - 1);
        BOOST_CHECK_EQUAL(d.strings.at(s.pre + ".cube_chain"), want);
    }
    BOOST_CHECK_EQUAL(d.strings.at("endpoints"), expected_endpoints(chains(), sc));
}

BOOST_AUTO_TEST_CASE(mutations) {
    if (!full_run())
        return;
    const Digests& base = synthetic_baseline();
    const std::string A = synthetic()[0].pre, B = synthetic()[1].pre, C = synthetic()[2].pre;
    // E and L are the prompt rotated per channel, so a flipped rotation moves only the DLL
    // powers: the cal, lanes, cube and records come from the prompt.
    const std::vector<MutationCase> cases = {
        {Mutation::rot_sign_A10,
         "rot_sign_A10",
         {A + ".epl", A + ".rec", A + ".telem", A + ".cmb", A + ".rest.get_status@*"},
         {B + ".*", C + ".*", "board@*", A + ".cube", A + ".rest.get_elem_cal@*",
          A + ".rest.get_records@*", A + ".rest.get_elements@*"}},
        {Mutation::job0_plus4_C12,
         "job0_plus4_C12",
         {C + ".epl", C + ".rec", C + ".telem", C + ".cmb", C + ".cube"},
         {A + ".*", B + ".*"}},
        {Mutation::rename_A,
         "rename_A",
         {"board@*", "strings:" + A + ".cube_chain", "endpoints"},
         {A + ".epl", A + ".rec", A + ".telem", A + ".cmb", A + ".cube", A + ".rest.*", B + ".*",
          C + ".*"}},
        {Mutation::tile_word_B10, "tile_word_B10", {B + ".epl", B + ".rec"}, {A + ".*", C + ".*"}},
        {Mutation::clock_step_50,
         "clock_step_50",
         {A + ".rest.get_elem_cal@59", B + ".rest.get_elem_cal@59", C + ".rest.get_elem_cal@59"},
         {"board@29", A + ".rest.get_elem_cal@29", B + ".rest.get_elem_cal@29",
          C + ".rest.get_elem_cal@29"}},
    };
    for (const auto& mc : cases) {
        Scenario sc;
        sc.mut = mc.m;
        const std::set<std::string> ch = changed(base, run_scenario(synthetic(), sc));
        BOOST_TEST_MESSAGE(mc.name << " changes: " << join(ch));
        for (const auto& pat : mc.differ) {
            bool hit = false;
            for (const auto& k : ch)
                hit = hit || matches(k, pat);
            BOOST_CHECK_MESSAGE(hit, mc.name << ": " << pat << " did not change");
        }
        for (const auto& pat : mc.same)
            for (const auto& k : ch)
                BOOST_CHECK_MESSAGE(!matches(k, pat), mc.name << ": " << k << " changed");
    }
}

/// Bytes a stage leaves unwritten would carry the buffers' fill into a digest. The prefill is
/// written only when a frame is allocated, so it shows bytes left unwritten in each frame's first
/// use; the epl refill covers every frame.
BOOST_AUTO_TEST_CASE(prefill) {
    if (!full_run())
        return;
    const Digests& base = baseline();
    Scenario stale = base_scenario();
    stale.refill_a5 = false;
    Scenario zero = base_scenario();
    zero.prefill = 0x00;
    const std::set<std::string> a = changed(base, run_scenario(chains(), stale));
    const std::set<std::string> b = changed(base, run_scenario(chains(), zero));
    BOOST_CHECK_MESSAGE(a.empty(), "no 0xA5 epl refill changes: " << join(a));
    BOOST_CHECK_MESSAGE(b.empty(), "a 0x00 prefill changes: " << join(b));
}

/// Behaviour that a fix changed (PIN_B1) or must change (PIN_B2).
BOOST_AUTO_TEST_CASE(shared_model_learns_from_held_prns_only) {
    // A shadow learns whatever its PRN's rows hold. Every running slot without a planted signal
    // carries a weak source its code does not match (Scenario::probe_amp): the shadows of those
    // PRNs warm on it while their lanes stay incoherent. Without elem_sum_shared_learn_coh_min the
    // shared model averages them in, and on a sky with no satellite it forms from them alone;
    // with it, only the PRNs with a coherent lane feed the model, and with none no model forms.
    // A reference of the records' own start reads as current, from the config and from the POST.
    Scenario lit = base_scenario(), lit_off = base_scenario(), dark = base_scenario(),
             dark_off = base_scenario();
    for (Scenario* sc : {&lit, &lit_off, &dark, &dark_off})
        sc->probe_amp = 0.5;
    lit.learn_coh_min = dark.learn_coh_min = 0.3;
    lit.ref_frame0 = (int64_t)std::floor(UTC0);
    dark.mut = dark_off.mut = Mutation::zero_amp;
    const Digests dl = run_scenario(chains(), lit), dlo = run_scenario(chains(), lit_off),
                  dd = run_scenario(chains(), dark), ddo = run_scenario(chains(), dark_off);
    const std::set<int> planted = sky_for(lit).planted;
    const std::string cp0 = "@" + std::to_string(lit.checkpoints.front()),
                      cp = "@" + std::to_string(lit.checkpoints.back());
    int n_learning = 0;
    for (const ChainSpec& A : chains()) {
        const json &L = dl.live.at(A.pre), &LO = dlo.live.at(A.pre), &D = dd.live.at(A.pre),
                   &DO = ddo.live.at(A.pre);
        BOOST_CHECK_MESSAGE(L.at("frame0_current" + cp0).get<bool>(),
                            A.pre << ": the configured reference's start does not read as current");
        BOOST_CHECK_MESSAGE(L.at("frame0_current" + cp).get<bool>(),
                            A.pre << ": the posted reference's start does not read as current");
        if (!LO.at("shared" + cp)[0].get<bool>())
            continue; // this chain's shadows hold no model in the scenario
        ++n_learning;
        int n_held = 0, n_unheld_warm = 0;
        for (const auto& h : L.at("held" + cp)) {
            const int prn = h[0].get<int>();
            const bool warm = h[1].get<bool>(), held = h[2].get<bool>();
            BOOST_CHECK_MESSAGE(!held || planted.count(prn),
                                A.pre << ": PRN " << prn << " has no signal and passed the gate");
            n_held += warm && held;
            n_unheld_warm += warm && !held;
        }
        BOOST_CHECK_MESSAGE(n_unheld_warm > 0, A.pre << ": no shadow warmed on the probe source");
        BOOST_CHECK_MESSAGE(L.at("shared" + cp)[0].get<bool>() && L.at("shared" + cp)[1] == n_held,
                            A.pre << ": the shared model is not of the " << n_held
                                  << " held PRNs: " << L.at("shared" + cp));
        BOOST_CHECK_MESSAGE(LO.at("shared" + cp)[1].get<int>() > n_held,
                            A.pre << ": without the gate the model is of no more than the "
                                  << n_held << " held PRNs: " << LO.at("shared" + cp));
        for (const auto& h : D.at("held" + cp))
            BOOST_CHECK_MESSAGE(!h[2].get<bool>(),
                                A.pre << ": PRN " << h[0] << " of an empty sky passed the gate");
        BOOST_CHECK_MESSAGE(
            !D.at("shared" + cp)[0].get<bool>(),
            A.pre << ": a shared model formed from the probe source behind the gate");
        BOOST_CHECK_MESSAGE(DO.at("shared" + cp)[0].get<bool>(),
                            A.pre << ": without the gate no model formed from the probe source");
    }
    BOOST_CHECK_MESSAGE(n_learning > 0, "no chain's shadows hold a shared model");
}

BOOST_AUTO_TEST_CASE(current_behaviour_pins) {
    const Digests& d = baseline();
    const Scenario sc = base_scenario();
    // PIN_B1: GnssCoherentCombiner restarts a slot cold on a PRN swap. At the next checkpoint
    // the swapped slot's get_records holds only the new PRN's records, its ADR arc broke at the
    // swap, and its deep window and ADR arc hold no record from before the swap.
    for (const ChainSpec& A : chains()) {
        const json& L = d.live.at(A.pre);
        const int was = A.prns[(size_t)sc.swap_slot], now = was + 100;
        const double swap_hop =
            (double)(REC_BASE + (int64_t)A.n_rec * sc.swap_frame) * A.hops_per_record;
        const int since = (59 - sc.swap_frame + 1) * A.n_rec;
        const json before = L.at("status@29").at((size_t)sc.swap_slot);
        const json after = L.at("status@59").at((size_t)sc.swap_slot);
        BOOST_CHECK_EQUAL(before[0].get<int>(), was);
        BOOST_CHECK_EQUAL(after[0].get<int>(), now);
        BOOST_CHECK_MESSAGE(L.at("records_min_hop@59").at(std::to_string(now)).get<double>()
                                >= swap_hop,
                            "PIN_B1: get_records of PRN " << now
                                                          << " holds a record from before the "
                                                             "swap");
        BOOST_CHECK_MESSAGE(before[2] != after[2], "PIN_B1: the ADR arc did not break at the "
                                                   "swap: "
                                                       << before[2] << " -> " << after[2]);
        BOOST_CHECK_MESSAGE(after[4].get<int>() <= since, "PIN_B1: the deep window holds "
                                                              << after[4] << " records, " << since
                                                              << " since the swap");
        // The first record after the swap has no predecessor on the new PRN and the second
        // opens the arc, which then counts every record.
        BOOST_CHECK_MESSAGE(after[5].get<int>() == since - 2, "PIN_B1: the ADR arc holds "
                                                                  << after[5] << " records, not "
                                                                  << since - 2 << " (" << since
                                                                  << " since the swap)");
    }
    // PIN_B2: in the dark_r0 frame, where the energies differ by record, GnssN2RecordAssemble
    // dequantises each slot in the later records with the energy of its first running record, as
    // the pack scaled it: record dark_r0_join's for the slot dark before it, not record 0's job-0
    // energy (another slot's), and record 0's for slot 0, not the record's own.
    {
        const ChainSpec& B = chains()[1];
        const json& t = d.pins.at("B2_tiles");
        const json& v = d.pins.at("B2_epl");
        const int f = sc.dark_r0_frame, n = B.n_rec - 1;
        BOOST_REQUIRE_EQUAL(v.size(), (size_t)(2 * n));
        BOOST_REQUIRE(sc.dark_r0_join >= 2 && sc.dark_r0_join < B.n_rec);
        for (int i = 0; i < 2 * n; ++i) {
            const int r = 1 + i % n, slot = (i < n) ? sc.dark_r0_slot : 0;
            if (slot != 0 && r < sc.dark_r0_join)
                continue;
            const double tile = t[i].get<double>(), got = v[i].get<double>();
            const double e_rule = energy_at(B, sc, f, slot == 0 ? 0 : sc.dark_r0_join, slot, 0);
            const double e_not =
                (slot == 0) ? energy_at(B, sc, f, r, 0, 0) : energy_at(B, sc, f, 0, 0, 0);
            const double want = tile * adapter_inv_scale(B, e_rule);
            BOOST_REQUIRE(tile != 0.0);
            BOOST_CHECK_MESSAGE(std::fabs(got - want) <= 1e-12 * std::fabs(got),
                                "PIN_B2: slot " << slot << " record " << r << " reads " << got
                                                << ", not tile/s(first running record) " << want);
            BOOST_CHECK_MESSAGE(
                std::fabs(got - tile * adapter_inv_scale(B, e_not)) > 1e-6 * std::fabs(got),
                "PIN_B2: slot " << slot << " record " << r << " is scaled by "
                                << (slot == 0 ? "its own record's" : "job 0's") << " energy");
        }
    }
}

/// get_elements' keff for slot @p p after frame @p k: the slot's records at the last emit since
/// the stage's start, or with @p reset since the swap, at most integration_length.
int keff_at(const ChainSpec& s, const Scenario& sc, int p, int k, bool reset) {
    const int K = s.combine.value("integration_length", 1);
    const int E = std::max(1, K / 10);
    const int last = s.n_rec * (k + 1) / E * E;
    const int swap_rec = s.n_rec * sc.swap_frame;
    return std::min(K, (reset && p == sc.swap_slot && last > swap_rec) ? last - swap_rec : last);
}

/// Notes in @p bad every slot frame where combiner @p got's get_elements keff is not keff_at.
void note_keff(const ChainSpec& s, const Scenario& sc, const json& got, bool reset,
               std::vector<std::string>& bad) {
    for (int k = 0; k < sc.n_frames; ++k) {
        if (!slot_frame(sc, k))
            continue;
        const json& el = got.at("elements@" + std::to_string(k));
        if (keff_at(s, sc, 0, k, false) == 0) {
            if (!el.empty())
                bad.push_back(fmt::format("elements@{:d}: rows before the first emit", k));
            continue;
        }
        if (el.size() != (size_t)s.n_prn) {
            bad.push_back(fmt::format("elements@{:d}: {:d} rows", k, el.size()));
            continue;
        }
        for (int p = 0; p < s.n_prn; ++p)
            if (el[p][2].get<double>() != keff_at(s, sc, p, k, reset))
                bad.push_back(fmt::format("elements@{:d} slot {:d}: keff {:s}, not {:d}", k, p,
                                          el[p][2].dump(), keff_at(s, sc, p, k, reset)));
    }
}

/// get_elements' keff counts each slot's records since the stage's start or its last reset, at
/// most integration_length, so the broker's min_keff holds back a slot only just reset.
BOOST_AUTO_TEST_CASE(elements_keff) {
    const Digests& d = baseline();
    const Scenario sc = base_scenario();
    for (const ChainSpec& s : chains()) {
        std::vector<std::string> bad;
        note_keff(s, sc, d.slots.at(s.pre).at("own"), true, bad);
        BOOST_CHECK_MESSAGE(bad.empty(), s.pre << ": " << bad.size() << " keff differ, first: "
                                               << (bad.empty() ? "" : bad[0]));
    }
}

/// A cold reset leaves the swapped slot as a combiner that first saw it on the new PRN (the
/// baseline's "cold" side combiner, fed the same records from the swap on), in every emit and in
/// every get_status, get_elements and (for the chain's own) get_records read since, except that
/// its arc id counts on from the departed PRN's arcs. It holds for the chain's own combiner and
/// for "scint", whose history before the swap has a raw S4 above 1: carried across the swap, that
/// would survive the thermal floor (at most 1) into the first emit's S4. prn_swap_spares_others
/// and prn_swap_without_reset check the other slots.
BOOST_AUTO_TEST_CASE(prn_swap_cold_start) {
    const Digests& d = baseline();
    const Scenario sc = base_scenario();
    const int q = sc.swap_slot;
    const std::vector<SideCombiner> side = side_combiners(sc);
    for (const ChainSpec& s : chains()) {
        const json& cold = d.slots.at(s.pre).at("cold");
        const json& cc = cold.at("cmb");
        const int E = std::max(1, s.combine.value("integration_length", 1) / 10);
        // The first emit since the swap must be read at a slot frame.
        BOOST_REQUIRE(slot_frame(sc, (s.n_rec * sc.swap_frame + E - 1) / s.n_rec));
        BOOST_REQUIRE(!cc.empty());
        const int now = s.prns[(size_t)q] + 100;
        const std::string key = std::to_string(now);
        const std::string last = "status@" + std::to_string(sc.swap_frame - 1);
        const double s4_last = d.slots.at(s.pre).at("scint").at(last)[q][3].get<double>();
        BOOST_CHECK_MESSAGE(s4_last > 1.0, s.pre << " scint: raw S4 " << s4_last
                                                 << " before the swap, not above 1");
        for (const auto& [name, from] : {std::make_pair("own", 0), {"scint", side[1].from}}) {
            const json& got = d.slots.at(s.pre).at(name);
            const json& oc = got.at("cmb");
            // It emits with the cold reference only if the swap falls on one of its emits.
            const int before = s.n_rec * (sc.swap_frame - from);
            BOOST_REQUIRE(before > 0 && before % E == 0);
            const size_t n_before = (size_t)(before / E);
            BOOST_REQUIRE(oc.size() >= n_before + cc.size());
            // The arc id at the last emit before the swap.
            const double arc0 = oc[n_before - 1][q][1].get<double>();
            std::vector<std::string> bad;
            for (size_t i = 0; i < cc.size(); ++i) {
                const json& c = cc[i][q];
                note_diff("emit " + std::to_string(n_before + i), oc[n_before + i][q],
                          {c[0], arc0 + c[1].get<double>()}, bad);
            }
            int n_swapped = 0;
            for (int k = sc.swap_frame; k < side[0].to; ++k) {
                const std::string cp = "@" + std::to_string(k);
                if (!cold.contains("status" + cp))
                    continue; // not read: no emit since the swap yet
                const json& c = cold.at("status" + cp)[q];
                note_diff("status" + cp, got.at("status" + cp)[q],
                          {c[0], c[1], arc0 + c[2].get<double>(), c[3]}, bad);
                note_diff("elements" + cp, got.at("elements" + cp)[q], cold.at("elements" + cp)[q],
                          bad);
                if (std::string(name) == "own")
                    note_diff("records" + cp, got.at("records" + cp).at(key),
                              cold.at("records" + cp).at(key), bad);
                ++n_swapped;
            }
            BOOST_CHECK_MESSAGE(n_swapped > 0, s.pre << " " << name
                                                     << ": no slot frame after the swap's first "
                                                        "emit");
            BOOST_CHECK_MESSAGE(bad.empty(), s.pre << " " << name << " slot " << q << ": "
                                                   << bad.size()
                                                   << " emits or GETs differ from the cold start, "
                                                      "first: "
                                                   << (bad.empty() ? "" : bad[0]));
            BOOST_TEST_MESSAGE(s.pre << " " << name << ": " << cc.size() << " emits and "
                                     << n_swapped << " slot frames since the swap; arc id " << arc0
                                     << " before it");
        }
        BOOST_TEST_MESSAGE(s.pre << " scint: raw S4 " << s4_last << " before the swap");
    }
}

/// The reset touches only the swapped slot: every other slot of the chain's own combiner equals
/// "others", a combiner fed only those slots from the start, in every emit and in every
/// get_status and get_elements read from the frame before the swap until "others" stops.
BOOST_AUTO_TEST_CASE(prn_swap_spares_others) {
    const Digests& d = baseline();
    const Scenario sc = base_scenario();
    const int q = sc.swap_slot;
    for (const ChainSpec& s : chains()) {
        const json& own = d.slots.at(s.pre).at("own");
        const json& oth = d.slots.at(s.pre).at("others");
        const json& xc = oth.at("cmb");
        BOOST_REQUIRE(!xc.empty() && own.at("cmb").size() >= xc.size());
        std::vector<std::string> bad;
        auto others = [&](json r) {
            BOOST_REQUIRE_EQUAL(r.size(), (size_t)s.n_prn);
            r[q] = nullptr;
            return r;
        };
        for (size_t i = 0; i < xc.size(); ++i)
            note_diff("emit " + std::to_string(i), others(own.at("cmb")[i]), others(xc[i]), bad);
        int n_read = 0;
        for (int k = sc.swap_frame - 1; k < sc.n_frames; ++k) {
            const std::string cp = "@" + std::to_string(k);
            if (!oth.contains("status" + cp))
                continue;
            for (const char* v : {"status", "elements"})
                note_diff(v + cp, others(own.at(v + cp)), others(oth.at(v + cp)), bad);
            ++n_read;
        }
        BOOST_CHECK_MESSAGE(n_read > 1, s.pre << ": " << n_read << " slot frames read");
        BOOST_CHECK_MESSAGE(bad.empty(), s.pre << ": " << bad.size()
                                               << " emits or GETs of the other slots differ, "
                                                  "first: "
                                               << (bad.empty() ? "" : bad[0]));
        BOOST_TEST_MESSAGE(s.pre << " others: " << xc.size() << " emits and " << n_read
                                 << " slot frames");
    }
}

/// reset_on_prn_change false keeps a slot's state across a PRN swap: the run differs from the
/// baseline only in the combiners from the swap on, and there only in the swapped slot, whose
/// ADR arc then runs on through the swap.
BOOST_AUTO_TEST_CASE(prn_swap_without_reset) {
    if (!full_run())
        return;
    const Digests& base = baseline();
    Scenario sc = base_scenario();
    sc.combiner_prn_reset = false;
    const Digests d = run_scenario(chains(), sc);
    const std::set<std::string> ch = changed(base, d);
    BOOST_TEST_MESSAGE("without the reset: " << join(ch));
    const int q = sc.swap_slot;
    std::set<std::string> allowed;
    for (const auto& s : chains()) {
        allowed.insert(s.pre + ".cmb");
        for (int cp : sc.checkpoints)
            for (const char* v : {"get_status", "get_records", "get_elements"})
                if (cp >= sc.swap_frame)
                    allowed.insert(s.pre + ".rest." + v + "@" + std::to_string(cp));
        BOOST_CHECK_MESSAGE(ch.count(s.pre + ".cmb") == 1, s.pre << ".cmb did not change");
        // The emits before the swap are the baseline's.
        const std::vector<uint64_t>& x = base.streams.at(s.pre + ".cmb");
        const std::vector<uint64_t>& y = d.streams.at(s.pre + ".cmb");
        const size_t n_before =
            (size_t)(s.n_rec * sc.swap_frame
                     / std::max(1, s.combine.value("integration_length", 1) / 10));
        BOOST_REQUIRE(x.size() == y.size() && x.size() > n_before);
        BOOST_CHECK(std::equal(x.begin(), x.begin() + n_before, y.begin()));
        // Per slot, every emit and every GET at the slot frames is the baseline's, except the
        // swapped slot's from the swap's first emit on.
        const json& a = base.slots.at(s.pre).at("own");
        const json& b = d.slots.at(s.pre).at("own");
        BOOST_REQUIRE_EQUAL(a.at("cmb").size(), b.at("cmb").size());
        std::vector<std::string> bad;
        for (size_t i = 0; i < a.at("cmb").size(); ++i) {
            json ra = a.at("cmb")[i], rb = b.at("cmb")[i];
            if (i >= n_before)
                ra[q] = rb[q] = nullptr;
            note_diff("emit " + std::to_string(i), rb, ra, bad);
        }
        const int now = s.prns[(size_t)q] + 100;
        for (int k = 0; k < sc.n_frames; ++k) {
            if (!slot_frame(sc, k))
                continue;
            const std::string cp = "@" + std::to_string(k);
            json sa = a.at("status" + cp), sb = b.at("status" + cp);
            json ea = a.at("elements" + cp), eb = b.at("elements" + cp);
            json ra = a.value("records" + cp, json::object());
            json rb = b.value("records" + cp, json::object());
            if (sa[q][1].get<int>() == now) {
                sa[q] = sb[q] = ea[q] = eb[q] = nullptr;
                ra.erase(std::to_string(now));
                rb.erase(std::to_string(now));
            }
            note_diff("status" + cp, sb, sa, bad);
            note_diff("elements" + cp, eb, ea, bad);
            note_diff("records" + cp, rb, ra, bad);
        }
        // Without the reset the swapped slot's keff runs on through the swap.
        note_keff(s, sc, b, false, bad);
        BOOST_CHECK_MESSAGE(bad.empty(), s.pre << ": " << bad.size()
                                               << " emits or GETs differ from the baseline, "
                                                  "first: "
                                               << (bad.empty() ? "" : bad[0]));
        const json& L = d.live.at(s.pre);
        BOOST_CHECK_EQUAL(L.at("status@29").at((size_t)q)[2], L.at("status@59").at((size_t)q)[2]);
    }
    for (const auto& k : ch)
        BOOST_CHECK_MESSAGE(allowed.count(k) == 1, k << " changed");
}

BOOST_AUTO_TEST_CASE(golden) {
    const char* path = std::getenv("GNSS_REPLAY_GOLDEN");
    const Digests& d = baseline();
    if (path == nullptr) {
        BOOST_TEST_MESSAGE("GNSS_REPLAY_GOLDEN not set: no golden compare");
        return;
    }
    const json now = golden_of(d);
    const char* bless = std::getenv("GNSS_REPLAY_BLESS");
    if (bless && std::string(bless) == "1") {
        std::ofstream(path) << now.dump(1);
        BOOST_TEST_MESSAGE("blessed " << path);
        return;
    }
    std::ifstream in(path);
    BOOST_REQUIRE_MESSAGE(in, "cannot read golden " << path);
    const json g = json::parse(in);
    for (const char* k : {"host", "compiler", "build_type", "config"})
        BOOST_REQUIRE_MESSAGE(g.at(k) == now.at(k), "golden " << k << " " << g.at(k).dump()
                                                              << " is not this run's "
                                                              << now.at(k).dump());
    std::set<std::string> diff;
    for (const char* part : {"streams", "strings"}) {
        for (const auto& [k, v] : g.at(part).items())
            if (!now.at(part).contains(k) || now.at(part).at(k) != v)
                diff.insert(std::string(part) + ":" + k);
        for (const auto& [k, v] : now.at(part).items())
            if (!g.at(part).contains(k))
                diff.insert(std::string(part) + ":" + k);
    }
    BOOST_CHECK_MESSAGE(diff.empty(), "differs from golden " << path << " (" << g.at("git_sha")
                                                             << "): " << join(diff));
}
