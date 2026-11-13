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
// golden (GNSS_REPLAY_BLESS=1 writes it), and GNSS_REPLAY_DUMP=<dir> writes the endpoints and the
// stage configs. A replay takes seconds in a Test build, so by default only the baseline and the
// zero-amplitude control run; GNSS_REPLAY_FULL=1 adds the determinism, mutation and prefill
// controls.

#include "Config.hpp"                // for Config
#include "GnssChanMetadata.hpp"      // for GnssChanMetadata, get_gnss_chan_metadata
#include "GnssCoherentCombiner.hpp"  // for GnssCoherentCombiner
#include "GnssGpuRecordAssemble.hpp" // for GnssGpuRecordAssemble
#include "GnssN2RecordAssemble.hpp"  // for GnssN2RecordAssemble
#include "GnssTelemPack.hpp"         // for GnssTelemPack
#include "Stage.hpp"                 // for Stage
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "gnssChanPower.hpp"         // for ChanPower
#include "gnssClock.hpp"             // for steady_now_hook
#include "gnssGpuChain.hpp"          // for FrameHdr, PrnCtl, off_*
#include "gnssProjSubspace.hpp"      // for ProjBoard
#include "gnssRecord.hpp"            // for frame_floats, record_stride, cube_frame_bytes
#include "gnssTelem.hpp"             // for TelemHeader, telem_frame_bytes
#include "gnss_replay_frames.hpp"    // for ChainSpec, Sky, Scenario, write_ctl, write_tiles
#include "metadata.hpp"              // for metadataPool
#include "restClient.hpp"            // for restClient
#include "restServer.hpp"            // for restServer
#include "version.h"                 // for get_git_commit_hash

#include "fmt.hpp"  // for format
#include "json.hpp" // for json

#include <algorithm>
#include <atomic>
#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sched.h>
#include <set>
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
/// liveness and pin checks read.
struct Digests {
    std::map<std::string, std::vector<uint64_t>> streams;
    std::map<std::string, std::string> strings;
    json live, pins;
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

/// One kotekan config holding every chain's four blocks, with the harness's buffer names.
json build_config(const std::vector<ChainSpec>& cs, const Scenario& sc,
                  const std::vector<int>& cpus) {
    json cfg;
    cfg["log_level"] = "warn";
    cfg["cpu_affinity"] = cpus;
    cfg["samples_per_data_set"] = cs[0].n_rec * cs[0].hops_per_record;
    for (size_t i = 0; i < cs.size(); ++i) {
        const ChainSpec& s = cs[i];
        json ad = s.adapter, as = s.assemble, cb = s.combine, tp = s.telem;
        ad["ctl_buf"] = buf_name(s, "n2ctl_buf");
        ad["tiles_buf"] = buf_name(s, "n2tiles_buf");
        ad["out_buf"] = buf_name(s, "n2epl_buf");
        as["in_buf"] = buf_name(s, "n2epl_buf");
        as["out_buf"] = buf_name(s, "n2rec_buf");
        as["cube_buf"] = buf_name(s, "cube_buf");
        cb["in_bufs"] = {buf_name(s, "n2rec_buf")};
        cb["out_buf"] = buf_name(s, "n2cmb_buf");
        tp["in_buf"] = buf_name(s, "n2rec_buf");
        tp["out_buf"] = buf_name(s, "telem_buf");
        for (json* b : {&ad, &as, &cb, &tp})
            (*b)["cpu_affinity"] = cpus;
        cfg[s.pre + "_n2assemble_tiles"] = ad;
        cfg[assembler_name(s, (int)i, sc).substr(1)] = as;
        cfg[s.pre + "_n2combine"] = cb;
        cfg[s.pre + "_telem_pack"] = tp;
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

void on_sigterm(int) {
    _exit(70); // FATAL_ERROR -> exit_kotekan -> raise(SIGTERM)
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
    int idx = 0;
    const Scenario* sc = nullptr;
    std::unique_ptr<Buffer> ctl, tiles, buf[N_KIND];
    std::vector<std::unique_ptr<kotekan::Stage>> stages;
    std::vector<uint64_t> hashes[N_KIND];
    std::atomic<int> count[N_KIND];
    std::thread cap[N_KIND];
    int emit_every = 1;
    // Written by the capture threads before their count moves.
    std::vector<double> pin_epl;
    std::string cube_chain;
    json cube_pow;
    int telem_bad = 0;
};

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

/// PIN_B2's reading: slot dark_r0_slot's prompt, channel 0, element 0, in records 1.. of the
/// dark_r0 frame.
void read_pin_epl(ChainRun& R, const uint8_t* f) {
    using namespace gnss_gpu;
    const ChainSpec& s = *R.s;
    const auto* h = (const FrameHdr*)f;
    const auto* pc = (const PrnCtl*)(f + off_prnctl());
    const double* corr = (const double*)(f + off_corr(s.n_prn));
    for (int r = 1; r < h->n_rec; ++r) {
        const PrnCtl& c = pc[(size_t)r * s.n_prn + R.sc->dark_r0_slot];
        R.pin_epl.push_back(c.run ? corr[((size_t)(c.job0 + ROW_P) * h->n_chan + 0) * s.n_live * 2]
                                  : NAN);
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
            // The live projection rewrites chain A's epl frame in place while this reads it.
            if (!(R->idx == 0 && n >= R->sc->proj_live_frame))
                h = hash_epl(*R, f);
            if (R->idx == 1 && n == R->sc->dark_r0_frame)
                read_pin_epl(*R, f);
        } else if (kind == REC) {
            h = fnv1a64(f, b->frame_size);
            const GnssChanMetadata* m = get_gnss_chan_metadata(b, id);
            const int64_t seq = m ? m->sample_seq : -1;
            h = fnv1a64(&seq, sizeof(seq), h);
        } else if (kind == CUBE) {
            read_cube(*R, f, b->frame_size, &h);
        } else {
            h = fnv1a64(f, b->frame_size);
            if (kind == TELEM) {
                gnss::TelemHeader th;
                std::memcpy(&th, f, sizeof(th));
                if (th.present != (1u << R->s->n_rec) - 1u)
                    ++R->telem_bad;
            }
        }
        if (!(kind == EPL && R->idx == 0 && n >= R->sc->proj_live_frame))
            R->hashes[kind].push_back(h);
        b->mark_frame_empty(name, id);
        id = (id + 1) % b->num_frames;
        R->count[kind].store(n + 1);
    }
}

bool quiescent(ChainRun& R, int k) {
    const int n_rec = R.s->n_rec;
    return R.ctl->is_frame_empty(0) && R.tiles->is_frame_empty(0) && R.buf[EPL]->is_frame_empty(0)
           && R.buf[REC]->get_num_full_frames() == 0 && R.buf[CUBE]->get_num_full_frames() == 0
           && R.count[EPL].load() == k + 1 && R.count[REC].load() == n_rec * (k + 1)
           && R.count[TELEM].load() == k + 1
           && R.count[CMB].load() == n_rec * (k + 1) / R.emit_every;
}

void wait_quiescent(ChainRun& R, int k) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!quiescent(R, k)) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30))
            child_fail(71, fmt::format("{:s} not quiescent 30 s after frame {:d}: epl {:d} rec "
                                       "{:d} cmb {:d} telem {:d}",
                                       R.s->pre, k, R.count[EPL].load(), R.count[REC].load(),
                                       R.count[CMB].load(), R.count[TELEM].load()));
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
        post(port, u + "/set_sat_geometry", g);
    }
    if (k == sc.shared_ref_frame) {
        json ref = json::array();
        for (int e = 0; e < s.n_live; ++e) {
            const std::complex<double> g = sky.gain_e[(size_t)e] * std::polar(1.0, 0.3);
            ref.push_back({g.real(), g.imag()});
        }
        post(port, u + "/set_elem_sum_shared_ref",
             {{"ref", ref}, {"mode", "live"}, {"slew_deg_s", 5.0}});
    }
    if (R.idx == 0 && k == sc.proj_live_frame)
        post(port, u + "/set_elem_proj", {{"mode", "live"}});
    if (R.idx == 1 && k == sc.ref_elem_frame)
        post(port, u + "/set_reference_element", {{"element", 1}});
}

/// Every GET the chain serves, digested, and the parts the checks read kept in @p live.
void fire_gets(ChainRun& R, unsigned short port, const std::string& u, int k, Digests& d) {
    const std::string c = "/" + R.s->pre + "_n2combine";
    const std::string cp = "@" + std::to_string(k);
    for (const auto& path : {u + "/get_elem_cal", u + "/get_spectrum", u + "/get_lane_inst",
                             c + "/get_status", c + "/get_records", c + "/get_elements"}) {
        const json j = rest(port, path, json{});
        const std::string verb = path.substr(path.rfind('/') + 1);
        const std::string body = j.dump();
        d.streams[R.s->pre + ".rest." + verb + cp] = {fnv1a64(body.data(), body.size())};
        json& L = d.live[R.s->pre];
        if (verb == "get_elem_cal" && j.contains("prns")) {
            json rows = json::array();
            for (const auto& r : j["prns"])
                rows.push_back({r["prn"], r["warm"], r["shadow_warm"]});
            L["elem_cal" + cp] = rows;
            if (j.contains("proj"))
                L["proj" + cp] = json::array({j["proj"]["mode"], j["proj"]["k"]});
        } else if (verb == "get_status" && j.is_array()) {
            json rows = json::array();
            for (const auto& r : j)
                rows.push_back({r["prn"], r["amplitude"], r["adr_arc"], r["coh_amplitude"]});
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

int child_main(const std::vector<ChainSpec>& cs, const Scenario& sc, const std::string& out_path) {
    // Boost.Test's handlers would report a crash from inside the child and carry on; the parent
    // reports the signal instead.
    for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGALRM})
        std::signal(sig, SIG_DFL);
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
    gnss::steady_now_hook = &replay_clock;
    const Sky sky = sky_for(sc);

    auto pool = metadataPool::create(64 * (int)cs.size(), 64, "pool", "GnssChanMetadata");
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<ChainRun>> runs;
    for (size_t i = 0; i < cs.size(); ++i) {
        const ChainSpec& s = cs[i];
        auto R = std::make_unique<ChainRun>();
        R->s = &s;
        R->idx = (int)i;
        R->sc = &sc;
        const size_t ctl_b =
            gnss_gpu::off_prnctl() + sizeof(gnss_gpu::PrnCtl) * gnss_gpu::MAX_REC * s.n_prn
            + sizeof(double) * gnss_gpu::max_jobs(s.n_prn, gnss_gpu::ROWS_PLAIN) * s.n_chan;
        const size_t tiles_b = (size_t)s.n_rec * s.n_chan * n_tile(s) * 512 * sizeof(int32_t);
        const int max_prn = s.telem.at("max_prn").get<int>();
        const size_t sizes[N_KIND] = {
            gnss_gpu::frame_bytes(s.n_prn, s.n_chan, gnss_gpu::ROWS_PLAIN, s.n_live),
            gnss::frame_floats(s.n_prn, s.n_live, s.n_chan) * sizeof(float),
            (size_t)s.n_prn * gnss::record_stride(s.n_live) * sizeof(float),
            gnss::telem_frame_bytes(s.n_rec, max_prn, s.n_chan),
            gnss::cube_frame_bytes(s.assemble.value("beam_cube_max_prn", s.n_prn),
                                   s.assemble.value("beam_cube_max_bins", s.n_chan), s.n_live)};
        const char* role[N_KIND] = {"n2epl_buf", "n2rec_buf", "n2cmb_buf", "telem_buf", "cube_buf"};
        R->ctl = std::make_unique<Buffer>(1, ctl_b, pool, buf_name(s, "n2ctl_buf"), "standard", 0,
                                          false, false, std::vector<int>{}, true, sc.prefill);
        R->tiles =
            std::make_unique<Buffer>(1, tiles_b, pool, buf_name(s, "n2tiles_buf"), "standard", 0,
                                     false, false, std::vector<int>{}, true, sc.prefill);
        bc.add_buffer(buf_name(s, "n2ctl_buf"), R->ctl.get());
        bc.add_buffer(buf_name(s, "n2tiles_buf"), R->tiles.get());
        R->ctl->register_producer("feeder");
        R->tiles->register_producer("feeder");
        for (int k = 0; k < N_KIND; ++k) {
            R->buf[k] = std::make_unique<Buffer>(k == EPL ? 1 : 4, sizes[k], pool,
                                                 buf_name(s, role[k]), "standard", 0, false, false,
                                                 std::vector<int>{}, true, sc.prefill);
            bc.add_buffer(buf_name(s, role[k]), R->buf[k].get());
            R->buf[k]->register_consumer(std::string("capture_") + KIND[k]);
            R->count[k].store(0);
        }
        R->emit_every = std::max(1, s.combine.value("integration_length", 1) / 10);
        runs.push_back(std::move(R));
    }

    kotekan::Config config;
    config.update_config(build_config(cs, sc, cpus));
    for (auto& R : runs) {
        const ChainSpec& s = *R->s;
        R->stages.emplace_back(
            new GnssN2RecordAssemble(config, "/" + s.pre + "_n2assemble_tiles", bc));
        R->stages.emplace_back(
            new GnssGpuRecordAssemble(config, assembler_name(s, R->idx, sc), bc));
        R->stages.emplace_back(new GnssCoherentCombiner(config, "/" + s.pre + "_n2combine", bc));
        R->stages.emplace_back(new GnssTelemPack(config, "/" + s.pre + "_telem_pack", bc));
    }
    // One band-power pass before any frame: chan_noise_norm weights from the first record.
    {
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
    for (auto& R : runs) {
        for (auto& st : R->stages)
            st->start();
        for (int k = 0; k < N_KIND; ++k)
            R->cap[k] = std::thread(capture, R.get(), k);
    }

    Digests d;
    for (int k = 0; k < sc.n_frames; ++k) {
        g_frame.store(k);
        for (auto& R : runs) {
            const ChainSpec& s = *R->s;
            uint8_t* ctl = R->ctl->wait_for_empty_frame("feeder", 0);
            int32_t* tiles = (int32_t*)R->tiles->wait_for_empty_frame("feeder", 0);
            if (ctl == nullptr || tiles == nullptr)
                child_fail(74, "feeder: buffer shut down");
            write_ctl(ctl, R->ctl->frame_size, s, sky, sc, k);
            write_tiles(tiles, R->tiles->frame_size / sizeof(int32_t), s, sky, sc, k);
            apply_mutation(sc, R->idx, k, s, ctl, tiles);
            if (R->idx == 1 && k == sc.dark_r0_frame) {
                json t = json::array();
                const int lane = s.lanes_per_slot * sc.dark_r0_slot
                                 + gnss_gpu::lane_of_row(s.lanes_per_slot, gnss_gpu::ROW_P);
                for (int r = 1; r < s.n_rec; ++r)
                    t.push_back(tiles[tile_index(s, r, 0, lane, 0)]);
                d.pins["B2_tiles"] = t;
            }
            R->ctl->mark_frame_full("feeder", 0);
            R->tiles->mark_frame_full("feeder", 0);
            wait_quiescent(*R, k);
            const std::string u = assembler_name(s, R->idx, sc);
            fire_posts(*R, sky, port, u, k);
            if (std::find(sc.checkpoints.begin(), sc.checkpoints.end(), k) != sc.checkpoints.end())
                fire_gets(*R, port, u, k, d);
            if (sc.refill_a5)
                std::memset(R->buf[EPL]->frames[0], 0xA5, R->buf[EPL]->frame_size);
        }
        if (std::find(sc.checkpoints.begin(), sc.checkpoints.end(), k) != sc.checkpoints.end())
            snapshot_board(k, d);
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
        if (R->idx == 1)
            d.pins["B2_epl"] = R->pin_epl;
    }
    // The REST surface under the chains' stage names.
    {
        const json ep = rest(port, "/endpoints", json{});
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

    json out = {{"streams", d.streams}, {"strings", d.strings}, {"live", d.live}, {"pins", d.pins}};
    std::ofstream f(out_path);
    f << out.dump();
    f.close();
    if (!f)
        child_fail(75, "cannot write " + out_path);
    std::fflush(nullptr);
    _exit(0);
}

// ---- the parent ----

Digests run_scenario(const std::vector<ChainSpec>& cs, const Scenario& sc) {
    char path[] = "/tmp/gnss_replay_XXXXXX";
    const int fd = mkstemp(path);
    BOOST_REQUIRE(fd >= 0);
    close(fd);
    std::fflush(nullptr);
    const pid_t pid = fork();
    BOOST_REQUIRE(pid >= 0);
    if (pid == 0) {
        try {
            child_main(cs, sc, path);
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
            unlink(path);
            BOOST_FAIL("replay child timed out");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    Digests d;
    if (WIFSIGNALED(status)) {
        unlink(path);
        BOOST_FAIL("replay child killed by signal " << WTERMSIG(status));
    }
    const int rc = WEXITSTATUS(status);
    if (rc != 0) {
        unlink(path);
        BOOST_FAIL("replay child exit " << rc << (rc == 70 ? " (FATAL in child)" : ""));
    }
    std::ifstream in(path);
    const json j = json::parse(in);
    unlink(path);
    d.streams = j.at("streams").get<std::map<std::string, std::vector<uint64_t>>>();
    d.strings = j.at("strings").get<std::map<std::string, std::string>>();
    d.live = j.at("live");
    d.pins = j.at("pins");
    return d;
}

const Digests& baseline() {
    static const Digests d = run_scenario(chains(), base_scenario());
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

/// Current behaviour that a fix must flip.
BOOST_AUTO_TEST_CASE(current_behaviour_pins) {
    const Digests& d = baseline();
    const Scenario sc = base_scenario();
    // PIN_B1: GnssCoherentCombiner keeps slot state across a PRN swap. At the next checkpoint the
    // swapped slot's get_records still holds the departed PRN's records, and its ADR arc did not
    // break at the swap.
    {
        const ChainSpec& A = chains()[0];
        const json& L = d.live.at(A.pre);
        const int was = A.prns[(size_t)sc.swap_slot], now = was + 100;
        const double swap_hop =
            (double)(REC_BASE + (int64_t)A.n_rec * sc.swap_frame) * A.hops_per_record;
        const json before = L.at("status@29").at((size_t)sc.swap_slot);
        const json after = L.at("status@59").at((size_t)sc.swap_slot);
        BOOST_CHECK_EQUAL(before[0].get<int>(), was);
        BOOST_CHECK_EQUAL(after[0].get<int>(), now);
        BOOST_CHECK_MESSAGE(L.at("records_min_hop@59").at(std::to_string(now)).get<double>()
                                < swap_hop,
                            "PIN_B1: get_records of PRN " << now
                                                          << " holds no record from before the "
                                                             "swap");
        BOOST_CHECK_MESSAGE(before[2] == after[2], "PIN_B1: the ADR arc broke at the swap: "
                                                       << before[2] << " -> " << after[2]);
    }
    // PIN_B2: GnssN2RecordAssemble dequantises a slot dark at record 0 with record 0's job-0
    // energy (another slot's) in the later records, not with its own.
    {
        const ChainSpec& B = chains()[1];
        const json& t = d.pins.at("B2_tiles");
        const json& v = d.pins.at("B2_epl");
        const double e_job0 = energy(B, 0, 0), e_own = energy(B, sc.dark_r0_slot, 0);
        BOOST_REQUIRE_EQUAL(v.size(), (size_t)(B.n_rec - 1));
        for (size_t r = 0; r < v.size(); ++r) {
            const double tile = t[r].get<double>(), got = v[r].get<double>();
            const double want = tile * adapter_inv_scale(B, e_job0);
            BOOST_REQUIRE(tile != 0.0);
            BOOST_CHECK_MESSAGE(std::fabs(got - want) <= 1e-12 * std::fabs(got),
                                "PIN_B2: record " << r + 1 << " reads " << got
                                                  << ", not tile/s(job 0) " << want);
            BOOST_CHECK_MESSAGE(std::fabs(got - tile * adapter_inv_scale(B, e_own))
                                    > 1e-6 * std::fabs(got),
                                "PIN_B2: record " << r + 1 << " is scaled by its own energy");
        }
    }
}

BOOST_AUTO_TEST_CASE(golden) {
    const char* path = std::getenv("GNSS_REPLAY_GOLDEN");
    const Digests& d = baseline();
    if (const char* dir = std::getenv("GNSS_REPLAY_DUMP")) {
        std::ofstream(std::string(dir) + "/endpoints.txt") << d.strings.at("endpoints");
        std::ofstream(std::string(dir) + "/config.json")
            << build_config(chains(), base_scenario(), {}).dump(1);
    }
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
