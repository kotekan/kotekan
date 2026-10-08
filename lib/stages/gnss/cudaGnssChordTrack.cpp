#include "cudaGnssChordTrack.hpp"

#include "GnssChanMetadata.hpp"
#include "Telescope.hpp" // for Telescope (the LIVE record epoch, not a config copy)
#include "cudaGnssChordDespread.hpp"
#include "cudaUtils.hpp"
#include "gnssBandPlan.hpp"
#include "gnssGpuChain.hpp"
#include "gnssSeedTransport.hpp"
#include "gnssSignal.hpp"
#include "kotekanLogging.hpp"
#include "pfbPrototype.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------
cudaGnssChordTrackState::cudaGnssChordTrackState(Config& config, const std::string& unique_name,
                                                 bufferContainer& host_buffers,
                                                 cudaDeviceInterface& device) :
    cudaCommandState(config, unique_name, host_buffers, device) {
    using namespace std::placeholders;

    prns = config.get<std::vector<int>>(unique_name, "prns");
    n_prn = (int)prns.size();
    n_chan = config.get<int>(unique_name, "n_channels");
    n_elem = config.get<int>(unique_name, "n_elements");
    n_hops_frame = config.get<int>(unique_name, "samples_per_data_set");
    hops_per_record = config.get_default<int>(unique_name, "hops_per_record", 2048);
    fft_len = config.get_default<int>(unique_name, "fft_length", 16384);
    sample_rate = config.get_default<double>(unique_name, "sample_rate", 3.2e9);
    f_offset_hz = config.get_default<double>(unique_name, "f_offset_hz", 0.0);
    // F-engine conjugation, measured on sky 2026-07-30 -- see DespreadParams::conj_data.
    // Without it the tracker despreads the conjugate of the sky and never locks.
    _conjugate = config.get_default<bool>(unique_name, "conjugate", false);
    dll_spacing = config.get_default<double>(unique_name, "dll_spacing", 0.5);
    trim_ttl_s = config.get_default<double>(unique_name, "trim_ttl_s", 0.0);
    trim_clamp = config.get_default<double>(unique_name, "trim_clamp", 3.0);
    // ── THE RECORD EPOCH COMES FROM THE TELESCOPE, NOT FROM THE CONFIG ──────────────────
    // utc0 is the UTC of ABSOLUTE SAMPLE 0, and this process already knows it exactly:
    // CHORDTelescope::to_time_ns(seq) = time0_ns + seq*dt_ns, where time0_ns is fetched
    // live from the FPGA controller at startup (query_gps/require_gps) WITH the GPS
    // week-rollover correction applied. to_time_ns(0) is therefore the same quantity the
    // config's frame0_utc has been hand-carrying -- from a better source.
    //
    // ⚠️ WHY THIS CHANGED (2026-08-18). The config copy went stale twice: 13.78 days from
    // 2026-07-24, and 9.165 days tonight when the F-engine restarted at 18:30 UTC during
    // site work. Both times /telescope served the truth while every GNSS record would have
    // been stamped days in the past, and nothing downstream could tell -- every
    // cross-record estimator works in differences, so a uniformly-wrong epoch stays
    // uniform. The generator grew a check against the running fleet to catch it; that
    // check was a workaround for a duplicated source of truth, and this removes the
    // duplication instead. A value the process can look up should never be pasted into a
    // config.
    //
    // The config key SURVIVES as an explicit override, because --frame0-nano exists to
    // start without chive's timing service and that path must keep working. Precedence:
    // an explicitly-configured value wins (it is a deliberate operator act), otherwise the
    // telescope, otherwise 0 -> the assembler's host-clock fallback and its loud warning.
    const double cfg_frame0_utc = config.get_default<double>(unique_name, "frame0_utc", 0.0);
    double tel_frame0_utc = 0.0;
    {
        const auto& tel = Telescope::instance();
        if (tel.gps_time_enabled()) {
            const int64_t t0 = tel.to_time_ns(0);
            if (t0 > 0)
                tel_frame0_utc = (double)t0 * 1e-9;
        }
    }
    frame0_utc = cfg_frame0_utc > 0.0 ? cfg_frame0_utc : tel_frame0_utc;
    if (tel_frame0_utc > 0.0 && cfg_frame0_utc > 0.0
        && std::abs(cfg_frame0_utc - tel_frame0_utc) > 1e-3) {
        // The generator refuses to EMIT a disagreeing config, but it can only check at
        // generation time -- an F-engine restart after that is invisible to it and not to
        // us. Do not silently prefer one: an operator who set frame0_utc meant it, and an
        // operator who did not needs to know their file is stale.
        WARN("cudaGnssChordTrack: frame0_utc CONFLICT -- config {:.9f}, telescope {:.9f} "
             "({:.3f} days apart). USING THE CONFIG because an explicit value is an "
             "operator decision, but this almost always means the F-engine restarted and "
             "the config was not regenerated. Every record from this node is stamped "
             "{:.3f} days off, and nothing downstream can see it. Clear frame0_utc from "
             "the node file to follow the telescope.",
             cfg_frame0_utc, tel_frame0_utc, std::abs(cfg_frame0_utc - tel_frame0_utc) / 86400.0,
             std::abs(cfg_frame0_utc - tel_frame0_utc) / 86400.0);
    } else if (tel_frame0_utc > 0.0 && cfg_frame0_utc <= 0.0) {
        INFO("cudaGnssChordTrack: record epoch from the TELESCOPE, frame0_utc {:.9f} "
             "(live GPS time0, week-rollover corrected). No config value to go stale.",
             frame0_utc);
    } else if (tel_frame0_utc <= 0.0) {
        WARN("cudaGnssChordTrack: the telescope has no GPS time0; record epoch falls back "
             "to the config's frame0_utc {:.9f}. That value cannot self-correct across an "
             "F-engine restart -- verify it against telescope/time0_ns before trusting any "
             "absolute record time.",
             frame0_utc);
    }
    // 0 = never expire (the old latching behaviour). The broker re-seeds every --interval (2 s
    // live), so 60 s is ~30 refreshes of margin while still retiring a set satellite promptly.
    seed_ttl_s = config.get_default<double>(unique_name, "seed_ttl_s", 60.0);

    if (hops_per_record <= 0 || n_hops_frame % hops_per_record != 0)
        FATAL_ERROR("cudaGnssChordTrack: hops_per_record {:d} must divide samples_per_data_set "
                    "{:d} exactly -- a partial trailing record would silently drop data.",
                    hops_per_record, n_hops_frame);
    const int n_rec = n_hops_frame / hops_per_record;
    if (n_rec > gnss_gpu::MAX_REC)
        FATAL_ERROR("cudaGnssChordTrack: {:d} records/frame exceeds MAX_REC {:d}; raise "
                    "hops_per_record.",
                    n_rec, gnss_gpu::MAX_REC);

    const std::string signame = config.get<std::string>(unique_name, "signal");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signame);
    if (!sig)
        FATAL_ERROR("cudaGnssChordTrack: unknown signal '{:s}'", signame);

    // The tap already selected the covering channels, so in the DATA they are dense
    // 0..n_chan-1 and every one is covered. Their SKY identity is a different matter: the
    // replica for local channel c must be synthesized at the centre frequency of the global
    // bin it came from, and CHORD's comb is stride-16 (5972, 5988, ...), not contiguous.
    // `channel_ids` carries those global bins, in local order.
    covering.resize((size_t)n_chan);
    for (int c = 0; c < n_chan; ++c)
        covering[(size_t)c] = c;
    channel_ids = config.get<std::vector<int>>(unique_name, "channel_ids");
    if ((int)channel_ids.size() != n_chan)
        FATAL_ERROR("cudaGnssChordTrack: channel_ids has {:d} entries but n_channels is {:d} -- "
                    "these are the GLOBAL bins of this GPU's covering comb, in local order.",
                    (int)channel_ids.size(), n_chan);

    // The replica bank must be built with the F-ENGINE's exact PFB, or the channelized replica
    // does not match the data: CHORD is a 4-tap Hamming critically-sampled bank over 8192
    // positive-frequency bins (arXiv:2607.01625 s2.2.2, recorded in chord_gnss_node.yaml).
    const int N = fft_len / 2;
    replica = std::make_unique<gnss::ChannelizedReplicaBank>(
        *sig, sample_rate, f_offset_hz, N, /*num_taps=*/4, dsp::window_from_string("hamming"),
        prns);
    // SPARSE comb -> the chan_ids overload. Passing chan_offset 0 here built every replica at
    // global bins 0..6 (DC) while the data sat at 5972..6076: noise at every code phase, which
    // is why nothing ever locked (found 2026-07-31). See GnssCudaDespread.hpp.
    despread = std::make_unique<GnssCudaDespread>(*replica, n_prn, channel_ids, hops_per_record,
                                                  sample_rate, f_offset_hz);
    // Truncate the per-hop PFB chip gather. Synthesis is 89% of this kernel and LINEAR in the
    // depth, so this is the biggest single lever on tracker GPU load.
    //
    // VALIDATED FLOOR IS 120 (scripts/gnss/e2e --max-chips, CHORD geometry, 2026-08-05): every
    // value from 120 to the full 210 reproduces the full-depth answer EXACTLY (+0.123 chips),
    // and 105 is 13 chips out with the wrong NH alignment -- a discontinuity, not a gradual
    // degradation. 140 is the recommended operating point: 1.50x, with 17% margin over 120.
    // Below 120 the truncated replica stops making the true lobe win and the search settles on a
    // grating lobe, which is the old refine_span: 4096 failure wearing a different hat.
    const int max_chips = config.get_default<int>(unique_name, "despread_max_chips", 0);
    // Item 6: CENTERED window placement. Changes where the max_chips window sits, so the
    // validated floors differ: one-sided 120 (9.5), centered 60 (80 recommended -- the harsh
    // 2-node comb flips at 52 and margin is cheap). Default OFF = the shipped one-sided cap.
    const bool centered = config.get_default<bool>(unique_name, "despread_chips_centered", false);
    if (centered)
        despread->set_chips_centered(true);
    if (max_chips > 0) {
        despread->set_max_chips(max_chips);
        if (centered && max_chips < 60)
            WARN_NON_OO("cudaGnssChordTrack: despread_max_chips={:d} CENTERED is BELOW THE "
                        "VALIDATED FLOOR of 60 (the 2-node comb flips to a grating lobe at "
                        "52). Timing experiments only; detections and locks are INVALID.",
                        max_chips);
        else if (centered)
            INFO_NON_OO("cudaGnssChordTrack: despread_max_chips={:d} CENTERED (item 6) -- "
                        "central window of the ~210-chip span; {:.2f}x less synthesis work",
                        max_chips, 210.0 / (double)max_chips);
        else if (max_chips < 120)
            WARN_NON_OO("cudaGnssChordTrack: despread_max_chips={:d} is BELOW THE VALIDATED "
                        "FLOOR of 120 -- the replica is truncated past the point where the "
                        "search resolves the true lobe. Timing experiments only; detections "
                        "and locks from this run are INVALID.",
                        max_chips);
        else
            INFO_NON_OO("cudaGnssChordTrack: despread_max_chips={:d} (full span ~210) -- gather "
                        "truncated at a validated depth; {:.2f}x less synthesis work",
                        max_chips, 210.0 / (double)max_chips);
    }

    // fp16 Phi tables (31896a862:docs/CHORD_GPU_TODO.md item 3): halve the RESIDENT table -- the
    // one lever the DRAM-footprint verdict (§10.6c) says pays; 1.27-1.37x measured on synthesis,
    // storage error 3.3e-4 (~0.14 dB class against the 4-bit voltage floor). Default OFF.
    // READ THE RETURN: "armed" and "in effect" are different states (#96/#97) -- the engine
    // refuses fp16 while shared tables are armed.
    if (config.get_default<bool>(unique_name, "phi_fp16", false)) {
        if (despread->set_phi_fp16(true))
            INFO_NON_OO("cudaGnssChordTrack: fp16 Phi tables ARMED (item 3) -- resident Phi "
                        "halved; despread_batch/device/peel paths will refuse while set");
        else
            WARN_NON_OO("cudaGnssChordTrack: phi_fp16 requested but the despread REFUSED it -- "
                        "running fp32; check the engine's own warning for why");
    }

    seeds.assign((size_t)n_prn, Seed{});
    trim.assign((size_t)n_prn, 0.0);
    trim_n.assign((size_t)n_prn, 0);
    trim_t_recv.assign((size_t)n_prn, 0.0);

    slot_gen.assign((size_t)n_prn, 0);
    // The fold histories are per PRN slot too; an unsized history is an out-of-bounds write on
    // the first frame (the slot count never changes, so this is the one sizing point).
    fold.init(n_prn);

    const std::string ep =
        config.get_default<std::string>(unique_name, "seed_endpoint", "/chord_track/set_seeds");
    kotekan::restServer::instance().register_post_callback(
        ep, std::bind(&cudaGnssChordTrackState::set_seeds_callback, this, _1, _2));
    const std::string tep =
        config.get_default<std::string>(unique_name, "trim_endpoint", "/chord_track/get_trim");
    kotekan::restServer::instance().register_get_callback(
        tep, std::bind(&cudaGnssChordTrackState::get_trim_callback, this, _1));
    // TASK #51 F2: the fleet controller's actuator.
    const std::string sep =
        config.get_default<std::string>(unique_name, "set_trim_endpoint", "/chord_track/set_trim");
    kotekan::restServer::instance().register_post_callback(
        sep, std::bind(&cudaGnssChordTrackState::set_trim_callback, this, _1, _2));
    // LIVE PRN MEMBERSHIP. Registered ALWAYS and carrying its own authority, exactly as
    // /set_trim does: there is no "enable live reconfiguration" flag on the node, because a
    // node that will not answer "which satellite is slot 7?" is strictly worse than one that
    // will, and the arming decision belongs to the BROKER (which is the only side that knows
    // whether it is in report-only or apply mode). GET is a diagnostic; POST is the actuator.
    const std::string gpe =
        config.get_default<std::string>(unique_name, "get_prns_endpoint", "/chord_track/get_prns");
    kotekan::restServer::instance().register_get_callback(
        gpe, std::bind(&cudaGnssChordTrackState::get_prns_callback, this, _1));
    const std::string spe =
        config.get_default<std::string>(unique_name, "set_prns_endpoint", "/chord_track/set_prns");
    kotekan::restServer::instance().register_post_callback(
        spe, std::bind(&cudaGnssChordTrackState::set_prns_callback, this, _1, _2));
    INFO_NON_OO("cudaGnssChordTrack: {:d} PRN x {:d} chan x {:d} elem, {:d} hops/record "
                "({:d} rec/frame), seeds on {:s}",
                n_prn, n_chan, n_elem, hops_per_record, n_rec, ep);
}

void cudaGnssChordTrackState::note_frame_hop(long long hop) {
    if (hop < 0)
        return;
    std::lock_guard<std::mutex> lk(prn_mtx);
    last_hop = (int64_t)hop;
}

std::vector<int> cudaGnssChordTrackState::prn_map() {
    std::lock_guard<std::mutex> lk(prn_mtx);
    return prns;
}

void cudaGnssChordTrackState::get_prns_callback(kotekan::connectionInstance& conn) {
    nlohmann::json out = nlohmann::json::array();
    nlohmann::json gen = nlohmann::json::array();
    bool pend = false;
    std::string err;
    uint64_t swaps = 0;
    int64_t at_h = -1, last_h = -1;
    {
        std::lock_guard<std::mutex> lk(prn_mtx);
        for (int p = 0; p < n_prn; ++p) {
            out.push_back(prns[(size_t)p]);
            gen.push_back(slot_gen[(size_t)p]);
        }
        pend = prn_pending;
        err = prn_last_err;
        swaps = prn_swaps;
        at_h = prn_at_hop;
        last_h = last_hop;
    }
    nlohmann::json reply = {{"prns", out},
                            {"n_prn", n_prn},
                            {"slot_gen", gen},
                            {"swaps", swaps},
                            {"pending", pend},
                            {"last_error", err},
                            // So the broker can see WHEN a staged swap is due and whether
                            // this node's clock has reached it -- a swap that is merely
                            // waiting must not read like a swap that was refused. And
                            // last_hop < 0 on a LIVE node is the alarm: it means no producer
                            // is feeding the deadline clock, so every swap silently degrades
                            // to apply-immediately (which is exactly how this shipped inert).
                            {"pending_at_hop", at_h},
                            {"last_hop", last_h}};
    conn.send_json_reply(reply);
}

void cudaGnssChordTrackState::set_prns_callback(kotekan::connectionInstance& conn,
                                                nlohmann::json& request) {
    // Parse and validate OUTSIDE every lock. The GPU thread takes prn_mtx once per frame, so
    // a parse under it would stall the tracker for the duration of an HTTP body -- the same
    // lesson set_seeds_callback records above.
    std::vector<int> want;
    std::string err;
    int64_t at_hop = -1;
    bool obsolete_at_seq = false;
    try {
        // OPTIONAL, and absent means "as soon as possible" -- the pre-scheduling behaviour,
        // kept so a hand-driven curl still works and so an older broker is not broken by a
        // newer node.
        if (request.is_object() && request.contains("at_hop") && !request.at("at_hop").is_null())
            at_hop = request.at("at_hop").get<int64_t>();
        // ⚠️ THE OBSOLETE FIELD IS REFUSED, NOT REINTERPRETED. `at_seq` carried the same
        // deadline in a different currency (see the header). Reading it as hops would be a
        // guess about which side of the 2026-08-27 fix the sender is on, and guessing wrong
        // is a swap 16384 frames early or late; ignoring it costs only the fleet-wide
        // simultaneity, which is what an unscheduled post already costs. Loud, never silent.
        else if (request.is_object() && request.contains("at_seq")
                 && !request.at("at_seq").is_null())
            obsolete_at_seq = true;
        const nlohmann::json& arr = request.is_object() ? request.at("prns") : request;
        if (!arr.is_array())
            throw std::runtime_error("expected an array of PRNs, or {\"prns\": [...]}");
        want.reserve(arr.size());
        for (const auto& v : arr)
            want.push_back(v.get<int>());
    } catch (const std::exception& e) {
        err = std::string("bad prns payload: ") + e.what();
    }
    if (err.empty() && (int)want.size() != n_prn)
        err = "prns has " + std::to_string(want.size()) + " entries but this chain has "
              + std::to_string(n_prn)
              + " slots. THE SLOT COUNT IS NOT LIVE: it sizes every buffer, GPU allocation and "
                "wire frame in the pipeline, so a resize is a fleet-wide re-plumb and a node "
                "restart, not a swap. Send exactly one PRN per slot.";
    if (err.empty())
        for (size_t i = 0; i < want.size() && err.empty(); ++i) {
            if (want[i] < 1 || want[i] > 63)
                err = "PRN " + std::to_string(want[i]) + " at slot " + std::to_string(i)
                      + " is outside 1..63";
            for (size_t j = 0; j < i && err.empty(); ++j)
                if (want[j] == want[i])
                    // A duplicate would put one satellite in two slots: both would be seeded,
                    // both would report, and the broker would fold two copies of one ray into
                    // its per-satellite state as though they were independent measurements.
                    err = "PRN " + std::to_string(want[i]) + " appears in slots "
                          + std::to_string(j) + " and " + std::to_string(i);
        }
    if (!err.empty()) {
        {
            std::lock_guard<std::mutex> lk(prn_mtx);
            prn_last_err = err;
        }
        conn.send_error(err, kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    int n_diff = 0;
    {
        std::lock_guard<std::mutex> lk(prn_mtx);
        for (int p = 0; p < n_prn; ++p)
            if (want[(size_t)p] != prns[(size_t)p])
                ++n_diff;
        prn_last_err.clear();
        if (n_diff > 0) {
            pending_prns = want;
            prn_pending = true;
            prn_at_hop = at_hop;
            prn_stage_hop = last_hop; // for the re-base guard in apply_prn_swaps
        }
    }
    if (n_diff > 0) {
        if (obsolete_at_seq)
            WARN_NON_OO("set_prns: the payload carries the OBSOLETE `at_seq` field (samples) "
                        "and no `at_hop` (hops). Staging UNSCHEDULED: this node will swap at "
                        "its own next frame boundary, so the fleet will NOT cross together. "
                        "Update the broker -- see cudaGnssChordTrack.hpp on the currency.");
        if (at_hop >= 0)
            INFO_NON_OO("set_prns: {:d} slot(s) staged for frame hop >= {:d} "
                        "(now {:d}, {:d} frame(s) ahead)",
                        n_diff, (long long)at_hop, (long long)last_hop,
                        (long long)((at_hop - last_hop) / (n_hops_frame > 0 ? n_hops_frame : 1)));
        else
            INFO_NON_OO("set_prns: {:d} slot(s) staged for the next frame boundary", n_diff);
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

std::vector<int> cudaGnssChordTrackState::apply_prn_swaps(void* stream) {
    std::vector<int> changed;
    std::vector<std::pair<int, int>> log; // (slot, old prn) for logging outside the lock
    {
        std::lock_guard<std::mutex> lk(prn_mtx);
        if (!prn_pending)
            return changed;
        // NOT YET DUE -- leave it staged and come back next frame.
        if (prn_at_hop >= 0 && last_hop >= 0 && last_hop < prn_at_hop) {
            // ⚠️ UNLESS THE COUNTER WENT BACKWARDS, in which case the deadline is unreachable
            // and waiting for it would wedge the swap FOREVER. An F-engine restart moves seq
            // back to ~0 ([[chord-gather-wedges-on-frame0-reset]]), so a schedule written
            // against the old epoch can never mature. Apply immediately and say so: a swap
            // that arrives early is a re-acquisition, a swap that never arrives is a slot
            // stuck on a satellite that has set.
            if (prn_stage_hop >= 0 && last_hop < prn_stage_hop) {
                WARN_NON_OO("set_prns: frame hop went BACKWARDS ({:d} -> {:d}) -- the F-engine "
                            "re-based, so the scheduled swap at {:d} can never arrive. "
                            "Applying NOW; the fleet-wide alignment for this swap is lost.",
                            (long long)prn_stage_hop, (long long)last_hop, (long long)prn_at_hop);
            } else {
                return changed;
            }
        }
        prn_pending = false;
        prn_at_hop = -1;
        prn_stage_hop = -1;
        for (int p = 0; p < n_prn; ++p) {
            const int newp = pending_prns[(size_t)p];
            if (newp == prns[(size_t)p])
                continue;
            // THE BANK AND THE DEVICE FIRST. If the code table cannot be built (a PRN with no
            // code for this signal) nothing else must move: a slot whose seed was cleared but
            // whose code is unchanged would despread the OLD satellite against the NEW one's
            // model, which is a lock loss dressed as a swap.
            if (!despread || !despread->set_prn(p, newp, stream)) {
                prn_last_err = "slot " + std::to_string(p) + ": no code for PRN "
                               + std::to_string(newp) + " on this signal (or an FDMA bank)";
                continue;
            }
            log.emplace_back(p, prns[(size_t)p]);
            prns[(size_t)p] = newp;
            ++slot_gen[(size_t)p];
            ++prn_swaps;
            changed.push_back(p);
        }
        pending_prns.clear();
    }
    if (changed.empty())
        return changed;
    // ⚠️ SEQUENTIAL, NEVER NESTED. prn_mtx is released above before either of these is taken:
    // expire_trims_locked reads `prns` with trim_mtx already held, so a prn_mtx -> trim_mtx
    // nesting here would close a deadlock cycle against every /set_trim POST.
    {
        std::lock_guard<std::mutex> lk(seed_mtx);
        for (int p : changed)
            seeds[(size_t)p] = Seed{}; // have=false: the slot goes dark until the broker seeds it
    }
    {
        std::lock_guard<std::mutex> lk(trim_mtx);
        for (int p : changed) {
            // A standing trim is a correction to the OLD satellite's code phase, in chips of
            // its geometry. Carried across it is a fixed offset applied to a satellite that
            // never asked for it -- the disease of chord-gather-restart-wipes-trims, inverted.
            trim[(size_t)p] = 0.0;
            trim_n[(size_t)p] = 0;
            trim_t_recv[(size_t)p] = 0.0;
        }
    }
    for (const auto& l : log)
        WARN_NON_OO("PRN SWAP: slot {:d} PRN {:d} -> {:d}. Code table, Phi cache, carrier NCO, "
                    "seed, trim and power averages all reset COLD for this slot; it stays dark "
                    "until the broker seeds the new satellite.",
                    l.first, l.second, prns[(size_t)l.first]);
    return changed;
}

void cudaGnssChordTrackState::set_seeds_callback(kotekan::connectionInstance& conn,
                                                 nlohmann::json& request) {
    // Parse first, lock last -- the execute path takes seed_mtx every GPU frame, so holding it
    // across a parse would stall the tracker directly.
    std::vector<std::pair<int, Seed>> upd;
    // prn_mtx is the OUTERMOST lock (see the header): the GPU thread can be rewriting `prns`
    // in apply_prn_swaps while this runs, and resolving a PRN against a half-written map would
    // seed the wrong slot.
    std::lock_guard<std::mutex> pk(prn_mtx);
    try {
        upd.reserve(request.size());
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            for (int i = 0; i < n_prn; ++i)
                if (prns[i] == prn) {
                    Seed sd;
                    sd.have = true;
                    sd.doppler_hz = s.at("doppler_hz").get<double>();
                    sd.cp_chips = s.at("code_phase_chips").get<double>();
                    sd.cp_rate = s.value("code_phase_rate", 0.0);
                    sd.dop_rate = s.value("doppler_rate_hz_s", 0.0);
                    sd.ctrim_hz = s.value("carrier_trim_hz", 0.0);
                    sd.ref_hop = s.value("ref_hop", (long long)0);
                    // PHYSICAL phase at ref_hop, when the producer can supply it. Preferred
                    // over cp_chips: an argument back-references to sample 0 through a
                    // Doppler-scaled rate, so the producer's Doppler error arrives multiplied
                    // by ~5900 chips/Hz. A phase carries no such lever.
                    sd.phase_ref_chips = s.value("code_phase_at_ref_chips", -1.0);
                    upd.emplace_back(i, sd);
                    break;
                }
        }
    } catch (const std::exception& e) {
        conn.send_error(std::string("bad seed payload: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    {
        const double now =
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        std::lock_guard<std::mutex> lk(seed_mtx);
        for (auto& u : upd) {
            u.second.t_recv = now;
            seeds[(size_t)u.first] = u.second;
        }
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void cudaGnssChordTrackState::expire_trims_locked(std::vector<int>& expired) {
    // EXPIRE A TRIM WHOSE CONTROLLER STOPPED TALKING. A frozen trim is a permanent silent code
    // offset that the broker's own slow DLL would then fight, and "latched forever" is the #13
    // failure. Only STAMPED trims expire: the in-tracker loop leaves trim_t_recv at 0, and its
    // silence means the SIGNAL went away, not the controller.
    if (trim_ttl_s <= 0.0)
        return;
    const double now =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    for (int p = 0; p < n_prn; ++p)
        if (trim_t_recv[(size_t)p] > 0.0 && (now - trim_t_recv[(size_t)p]) > trim_ttl_s) {
            if (trim[(size_t)p] != 0.0)
                expired.push_back(prns[(size_t)p]);
            trim[(size_t)p] = 0.0;
            trim_t_recv[(size_t)p] = 0.0;
            ++trim_expired;
        }
}

std::vector<double> cudaGnssChordTrackState::snapshot_trims(std::vector<int>& expired) {
    std::lock_guard<std::mutex> lk(trim_mtx);
    expire_trims_locked(expired);
    return trim;
}

std::vector<cudaGnssChordTrackState::Seed>
cudaGnssChordTrackState::snapshot_seeds(std::vector<int>& expired) {
    // EXPIRE STALE SEEDS, under the same lock that copies them (see the long note at the call
    // site in cudaGnssChordTrack::execute -- the latch-forever failure this replaced).
    const double now =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lk(seed_mtx);
    if (seed_ttl_s > 0.0)
        for (int p = 0; p < n_prn; ++p) {
            auto& sd = seeds[(size_t)p];
            if (sd.have && sd.t_recv > 0.0 && (now - sd.t_recv) > seed_ttl_s) {
                sd.have = false;
                expired.push_back(prns[(size_t)p]);
            }
        }
    return seeds;
}

void cudaGnssChordTrackState::set_trim_callback(kotekan::connectionInstance& conn,
                                                nlohmann::json& request) {
    // Parse first, lock last -- the same discipline as set_seeds: the execute path takes
    // trim_mtx every GPU frame, so holding it across a parse would stall the tracker directly.
    std::vector<std::pair<int, double>> upd;
    std::lock_guard<std::mutex> pk(prn_mtx); // OUTERMOST -- see set_seeds_callback
    try {
        upd.reserve(request.size());
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            const double t = s.at("trim_chips").get<double>();
            // A NaN here would propagate into the commanded code phase and despread garbage
            // for as long as it stood. Reject the whole request rather than a field: a partial
            // application would leave the controller and the tracker disagreeing about what is
            // commanded, silently.
            if (!std::isfinite(t))
                throw std::runtime_error("trim_chips is not finite");
            for (int i = 0; i < n_prn; ++i)
                if (prns[i] == prn) {
                    upd.emplace_back(i, std::max(-trim_clamp, std::min(trim_clamp, t)));
                    break;
                }
        }
    } catch (const std::exception& e) {
        conn.send_error(std::string("bad trim payload: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    const double now =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    {
        std::lock_guard<std::mutex> lk(trim_mtx);
        // Sweep here too: this callback still runs when execute() has stopped (a wedged GPU
        // chain answers REST perfectly well), and that is precisely when a stale trim would
        // otherwise stand indefinitely. See expire_trims_locked.
        std::vector<int> gone;
        expire_trims_locked(gone);
        for (const auto& u : upd) {
            trim[(size_t)u.first] = u.second;
            trim_t_recv[(size_t)u.first] = now;
            ++trim_n[(size_t)u.first];
        }
        ++trim_posts;
    }
    // ⚠️ NOTHING ELSE IS TOUCHED. Not seeds, not t_recv. See the header note: an
    // actuator that quietly resets another loop's state is the failure this endpoint exists
    // to avoid, and it is why this is not a field on /set_seeds.
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void cudaGnssChordTrackState::get_trim_callback(kotekan::connectionInstance& conn) {
    nlohmann::json out = nlohmann::json::array();
    {
        std::lock_guard<std::mutex> pk(prn_mtx); // OUTERMOST -- see set_seeds_callback
        std::lock_guard<std::mutex> lk(trim_mtx);
        for (int p = 0; p < n_prn; ++p)
            out.push_back({{"prn", prns[(size_t)p]},
                           {"trim_chips", trim[(size_t)p]},
                           {"age_s", trim_t_recv[(size_t)p] > 0.0
                                         ? (std::chrono::duration<double>(
                                                std::chrono::steady_clock::now().time_since_epoch())
                                                .count()
                                            - trim_t_recv[(size_t)p])
                                         : -1.0},
                           {"updates", trim_n[(size_t)p]}});
    }
    nlohmann::json reply = {{"posts", trim_posts},
                            {"expired", trim_expired},
                            {"ttl_s", trim_ttl_s},
                            {"clamp", trim_clamp},
                            {"trims", out}};
    conn.send_json_reply(reply);
}
