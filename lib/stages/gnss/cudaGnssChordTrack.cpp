#include "cudaGnssChordTrack.hpp"

#include "Telescope.hpp" // for Telescope (the record epoch)
#include "gnssGpuChain.hpp"
#include "gnssSignal.hpp"
#include "kotekanLogging.hpp"
#include "pfbPrototype.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
    hops_per_record = config.get<int>(unique_name, "hops_per_record");
    fft_len = config.get<int>(unique_name, "fft_length");
    sample_rate = config.get<double>(unique_name, "sample_rate");
    f_offset_hz = config.get<double>(unique_name, "f_offset_hz");
    // CHORD's F-engine output is conjugated relative to the decode, and cudaGnssInject always
    // absorbs that by conjugating the replica (launch_pack44).
    if (!config.get_default<bool>(unique_name, "conjugate", true))
        FATAL_ERROR("cudaGnssChordTrack: conjugate: false is not supported; the replica is "
                    "always conjugated to absorb the F-engine conjugation");
    dll_spacing = config.get<double>(unique_name, "dll_spacing");
    trim_ttl_s = config.get_default<double>(unique_name, "trim_ttl_s", 4.0);
    if (!(trim_ttl_s > 0.0))
        FATAL_ERROR("cudaGnssChordTrack: trim_ttl_s {:g} must be > 0", trim_ttl_s);
    // Record epoch: frame0_utc is the UTC of absolute sample 0, which is the telescope's
    // to_time_ns(0), from the GPS time0 it reads at startup. A config copy cannot follow an
    // F-engine restart, and an epoch wrong by a constant is invisible to every estimator that
    // works in differences. The config key stays as an explicit override (the generator's
    // --frame0-nano sets it). Precedence: config if set, else the telescope, else 0, which the
    // record assembler refuses.
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
        // The generator checks the config against the running fleet, but only when it runs;
        // an F-engine restart after that shows up only here. The config still wins, since
        // setting it is deliberate, but the conflict is logged.
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

    // The despread's channels are dense, 0..n_chan-1, and all covered. The replica for local
    // channel c is synthesized at the centre frequency of global bin channel_ids[c]; CHORD's
    // comb is not contiguous (stride 16).
    covering.resize((size_t)n_chan);
    for (int c = 0; c < n_chan; ++c)
        covering[(size_t)c] = c;
    const std::vector<int> channel_ids = config.get<std::vector<int>>(unique_name, "channel_ids");
    if ((int)channel_ids.size() != n_chan)
        FATAL_ERROR("cudaGnssChordTrack: channel_ids has {:d} entries but n_channels is {:d} -- "
                    "these are the GLOBAL bins of this GPU's covering comb, in local order.",
                    (int)channel_ids.size(), n_chan);

    // The replica bank must use the F-engine's PFB, or the channelized replica does not match
    // the data: CHORD's is a 4-tap Hamming critically-sampled bank over 8192 positive-frequency
    // bins (arXiv:2607.01625 s2.2.2, recorded in chord_gnss_node.yaml).
    const int N = fft_len / 2;
    replica = std::make_unique<gnss::ChannelizedReplicaBank>(
        *sig, sample_rate, f_offset_hz, N, /*num_taps=*/4, dsp::window_from_string("hamming"),
        prns);
    // The despread takes each channel's global bin, since the comb is sparse.
    despread = std::make_unique<GnssCudaDespread>(*replica, n_prn, channel_ids, hops_per_record,
                                                  sample_rate);
    // The per-hop PFB chip gather is an 80-chip window centred on the ~210-chip span, 2.6x
    // less synthesis work than the full span. A capped gather is a different replica, but at
    // CHORD geometry (scripts/gnss/e2e --max-chips --chips-centered) a centred window
    // reproduces the full-depth code phase down to 60 chips. The Phi tables are fp16, half the
    // device footprint at a storage error of ~3.3e-4. The config keys are read only to refuse
    // any other setting.
    constexpr int max_chips = 80;
    if (config.get_default<int>(unique_name, "despread_max_chips", max_chips) != max_chips
        || !config.get_default<bool>(unique_name, "despread_chips_centered", true)
        || !config.get_default<bool>(unique_name, "phi_fp16", true))
        FATAL_ERROR("cudaGnssChordTrack: the despread runs only a centred {:d}-chip gather with "
                    "fp16 Phi; despread_max_chips must be {:d}, despread_chips_centered and "
                    "phi_fp16 true",
                    max_chips, max_chips);
    despread->set_max_chips(max_chips);
    despread->set_phi_fp16();

    seeds.assign((size_t)n_prn, Seed{});
    trim.assign((size_t)n_prn, 0.0);
    trim_n.assign((size_t)n_prn, 0);
    trim_t_recv.assign((size_t)n_prn, 0.0);

    slot_gen.assign((size_t)n_prn, 0);
    // Sized once here: the slot count never changes, and the producer indexes it from the
    // first frame.
    fold.init(n_prn);

    const std::string ep = config.get<std::string>(unique_name, "seed_endpoint");
    kotekan::restServer::instance().register_post_callback(
        ep, std::bind(&cudaGnssChordTrackState::set_seeds_callback, this, _1, _2));
    const std::string tep = config.get<std::string>(unique_name, "trim_endpoint");
    kotekan::restServer::instance().register_get_callback(
        tep, std::bind(&cudaGnssChordTrackState::get_trim_callback, this, _1));
    // GnssFleetTrim's actuator.
    const std::string sep = config.get<std::string>(unique_name, "set_trim_endpoint");
    kotekan::restServer::instance().register_post_callback(
        sep, std::bind(&cudaGnssChordTrackState::set_trim_callback, this, _1, _2));
    // Slot membership. Always registered, with no enable flag on the node: whether to apply
    // swaps is the broker's decision (it also has a report-only mode).
    const std::string gpe = config.get<std::string>(unique_name, "get_prns_endpoint");
    kotekan::restServer::instance().register_get_callback(
        gpe, std::bind(&cudaGnssChordTrackState::get_prns_callback, this, _1));
    const std::string spe = config.get<std::string>(unique_name, "set_prns_endpoint");
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

void cudaGnssChordTrackState::get_prns_callback(kotekan::connectionInstance& conn) {
    nlohmann::json out = nlohmann::json::array();
    bool pend = false;
    int64_t last_h = -1;
    {
        std::lock_guard<std::mutex> lk(prn_mtx);
        for (int p = 0; p < n_prn; ++p)
            out.push_back(prns[(size_t)p]);
        pend = !pending_prns.empty();
        last_h = last_hop;
    }
    nlohmann::json reply = {{"prns", out},
                            {"pending", pend},
                            // last_hop < 0 on a running node means nothing calls
                            // note_frame_hop, so every scheduled swap applies immediately.
                            {"last_hop", last_h}};
    conn.send_json_reply(reply);
}

void cudaGnssChordTrackState::set_prns_callback(kotekan::connectionInstance& conn,
                                                nlohmann::json& request) {
    // Parse and validate before taking prn_mtx, which the GPU thread takes every frame.
    std::vector<int> want;
    std::string err;
    int64_t at_hop = -1;
    try {
        if (!request.is_object())
            throw std::runtime_error("expected {\"prns\": [...]}");
        // Optional; absent or null means the next frame boundary.
        if (request.contains("at_hop") && !request.at("at_hop").is_null())
            at_hop = request.at("at_hop").get<int64_t>();
        const nlohmann::json& arr = request.at("prns");
        if (!arr.is_array())
            throw std::runtime_error("expected {\"prns\": [...]}");
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
                    // A duplicate would put one satellite in two slots, and the broker would
                    // fold both copies into its per-satellite state as independent measurements.
                    err = "PRN " + std::to_string(want[i]) + " appears in slots "
                          + std::to_string(j) + " and " + std::to_string(i);
        }
    if (!err.empty()) {
        conn.send_error(err, kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    int n_diff = 0;
    {
        std::lock_guard<std::mutex> lk(prn_mtx);
        for (int p = 0; p < n_prn; ++p)
            if (want[(size_t)p] != prns[(size_t)p])
                ++n_diff;
        if (n_diff > 0) {
            pending_prns = want;
            prn_at_hop = at_hop;
            prn_stage_hop = last_hop; // for the re-base guard in apply_prn_swaps
        }
    }
    if (n_diff > 0) {
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

void cudaGnssChordTrackState::apply_prn_swaps(void* stream) {
    std::vector<int> changed;
    std::vector<std::pair<int, int>> log;     // (slot, old prn) for logging outside the lock
    std::vector<std::pair<int, int>> refused; // (slot, refused prn)
    {
        std::lock_guard<std::mutex> lk(prn_mtx);
        if (pending_prns.empty())
            return;
        // Not yet due: leave it staged.
        if (prn_at_hop >= 0 && last_hop >= 0 && last_hop < prn_at_hop) {
            // Unless the hop counter went backwards since staging: an F-engine restart moves it
            // back to ~0, so the deadline may never arrive. Applying early costs a
            // re-acquisition; never applying leaves the slot on a satellite that has set.
            if (prn_stage_hop >= 0 && last_hop < prn_stage_hop) {
                WARN_NON_OO("set_prns: frame hop went BACKWARDS ({:d} -> {:d}) -- the F-engine "
                            "re-based, so the scheduled swap at {:d} can never arrive. "
                            "Applying NOW; the fleet-wide alignment for this swap is lost.",
                            (long long)prn_stage_hop, (long long)last_hop, (long long)prn_at_hop);
            } else {
                return;
            }
        }
        prn_at_hop = -1;
        prn_stage_hop = -1;
        for (int p = 0; p < n_prn; ++p) {
            const int newp = pending_prns[(size_t)p];
            if (newp == prns[(size_t)p])
                continue;
            // Code table first. If it cannot be built (no code for this PRN on this signal, or
            // an FDMA bank), nothing else about the slot changes, so its seed and trim stay
            // consistent with the code it still despreads.
            if (!despread->set_prn(p, newp, stream)) {
                refused.emplace_back(p, newp);
                continue;
            }
            log.emplace_back(p, prns[(size_t)p]);
            prns[(size_t)p] = newp;
            ++slot_gen[(size_t)p];
            changed.push_back(p);
        }
        pending_prns.clear();
    }
    for (const auto& r : refused)
        WARN_NON_OO("set_prns: slot {:d} REFUSED PRN {:d} (no code for it on this signal, or an "
                    "FDMA bank) and keeps PRN {:d}; this node's map now differs from the one "
                    "posted.",
                    r.first, r.second, prns[(size_t)r.first]);
    if (changed.empty())
        return;
    // prn_mtx is already released, so a seed or trim posted for the new PRN in between is
    // cleared here as well; the next post restores it.
    {
        std::lock_guard<std::mutex> lk(seed_mtx);
        for (int p : changed)
            seeds[(size_t)p] = Seed{}; // have=false: the slot goes dark until the broker seeds it
    }
    {
        std::lock_guard<std::mutex> lk(trim_mtx);
        for (int p : changed) {
            // A trim corrects the old satellite's code phase; kept, it would offset the new one.
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
}

void cudaGnssChordTrackState::set_seeds_callback(kotekan::connectionInstance& conn,
                                                 nlohmann::json& request) {
    // Parse before taking seed_mtx, which the GPU thread takes every frame.
    std::vector<std::pair<int, Seed>> upd;
    // prn_mtx first (see the header): apply_prn_swaps on the GPU thread rewrites `prns`, and
    // resolving a PRN against a half-written map would seed the wrong slot.
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
                    // Physical phase at ref_hop, when the sender supplies it. Preferred over
                    // cp_chips: an argument is referred to sample 0 through a Doppler-scaled
                    // rate, so the sender's Doppler error arrives multiplied by thousands of
                    // chips per Hz (~5095 at 6.8 days of uptime). A phase has no such lever.
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
    // Zero a trim whose controller stopped posting (see trim_ttl_s). Slots with no post since
    // start or since their last reset have trim_t_recv 0 and are skipped.
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
    // Expire stale seeds under the same lock that copies them.
    const double now =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lk(seed_mtx);
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
    // Parse before taking trim_mtx, which the GPU thread takes every frame.
    std::vector<std::pair<int, double>> upd;
    std::lock_guard<std::mutex> pk(prn_mtx); // first; see set_seeds_callback
    try {
        upd.reserve(request.size());
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            const double t = s.at("trim_chips").get<double>();
            // A NaN would propagate into the commanded code phase. Reject the whole request,
            // not just the entry, so the controller and the node never disagree about what is
            // commanded.
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
        // Sweep here too: this callback still runs when execute() has stopped. See
        // expire_trims_locked.
        std::vector<int> gone;
        expire_trims_locked(gone);
        for (const auto& u : upd) {
            trim[(size_t)u.first] = u.second;
            trim_t_recv[(size_t)u.first] = now;
            ++trim_n[(size_t)u.first];
        }
        ++trim_posts;
    }
    // Nothing else is touched, not the seeds nor their t_recv (see trim_ttl_s in the header).
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void cudaGnssChordTrackState::get_trim_callback(kotekan::connectionInstance& conn) {
    nlohmann::json out = nlohmann::json::array();
    {
        std::lock_guard<std::mutex> pk(prn_mtx); // first; see set_seeds_callback
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
