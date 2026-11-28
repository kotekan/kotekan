#include "gnssTrackChain.hpp"

#include "Telescope.hpp" // for Telescope (the record epoch)
#include "dspPfbPrototype.hpp"
#include "gnssGpuChain.hpp"
#include "gnssSeedTransport.hpp" // for propagate_seed
#include "gnssSignal.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

using kotekan::Config;

namespace gnss {

// The Doppler margin of an FDMA slot's own lobe: GLONASS's L2 Doppler stays under ~3.7 kHz here.
static constexpr double FDMA_DOPPLER_MARGIN_HZ = 5e3;

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------
TrackChain::TrackChain(Config& config, const std::string& cfg_path) {
    prns = config.get<std::vector<int>>(cfg_path, "prns");
    n_prn = (int)prns.size();
    n_chan = config.get<int>(cfg_path, "n_channels");
    n_elem = config.get<int>(cfg_path, "n_elements");
    n_hops_frame = config.get<int>(cfg_path, "samples_per_data_set");
    hops_per_record = config.get<int>(cfg_path, "hops_per_record");
    fft_len = config.get<int>(cfg_path, "fft_length");
    sample_rate = config.get<double>(cfg_path, "sample_rate");
    f_offset_hz = config.get<double>(cfg_path, "f_offset_hz");
    // CHORD's F-engine output is conjugated relative to the decode, and cudaGnssInject always
    // absorbs that by conjugating the replica (launch_pack44).
    if (!config.get_default<bool>(cfg_path, "conjugate", true))
        FATAL_ERROR("cudaGnssChordTrack: conjugate: false is not supported; the replica is "
                    "always conjugated to absorb the F-engine conjugation");
    dll_spacing = config.get<double>(cfg_path, "dll_spacing");
    epl_from_prompt = config.get_default<bool>(cfg_path, "epl_from_prompt", false);
    psi_synth = config.get_default<bool>(cfg_path, "psi_synth", false);
    trim_ttl_s = config.get_default<double>(cfg_path, "trim_ttl_s", 4.0);
    if (!(trim_ttl_s > 0.0))
        FATAL_ERROR("cudaGnssChordTrack: trim_ttl_s {:g} must be > 0", trim_ttl_s);
    // Record epoch: frame0_utc is the UTC of absolute sample 0, which is the telescope's
    // to_time_ns(0), from the GPS time0 it reads at startup. A config copy cannot follow an
    // F-engine restart, and an epoch wrong by a constant is invisible to every estimator that
    // works in differences. The config key stays as an explicit override (the generator's
    // --frame0-nano sets it). Precedence: config if set, else the telescope, else 0, which the
    // record assembler refuses.
    const double cfg_frame0_utc = config.get_default<double>(cfg_path, "frame0_utc", 0.0);
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
             "({:.3f} days apart). Using the config because an explicit value is an "
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

    const std::string signame = config.get<std::string>(cfg_path, "signal");
    sig = gnss::signal_by_name(signame);
    if (!sig)
        FATAL_ERROR("cudaGnssChordTrack: unknown signal '{:s}'", signame);

    // The despread's channels are dense, 0..n_chan-1, and all covered. The replica for local
    // channel c is synthesized at the centre frequency of global bin channel_ids[c]; CHORD's
    // comb is not contiguous (stride 16).
    covering.resize((size_t)n_chan);
    for (int c = 0; c < n_chan; ++c)
        covering[(size_t)c] = c;
    const std::vector<int> channel_ids = config.get<std::vector<int>>(cfg_path, "channel_ids");
    if ((int)channel_ids.size() != n_chan)
        FATAL_ERROR("cudaGnssChordTrack: channel_ids has {:d} entries but n_channels is {:d} -- "
                    "these are the GLOBAL bins of this GPU's covering comb, in local order.",
                    (int)channel_ids.size(), n_chan);

    // The replica bank must use the F-engine's PFB, or the channelized replica does not match
    // the data. The defaults are CHORD's: a 4-tap Hamming critically-sampled bank over 8192
    // positive-frequency bins (arXiv:2607.01625 s2.2.2, recorded in chord_gnss_node.yaml).
    gnss::Channelization chz;
    chz.sample_rate = sample_rate;
    chz.complex_input = config.get_default<std::string>(cfg_path, "input_type", "real") != "real";
    chz.spectrum_length = chz.complex_input ? fft_len : fft_len / 2;
    chz.num_taps = config.get_default<int>(cfg_path, "num_taps", 4);
    chz.window =
        dsp::window_from_string(config.get_default<std::string>(cfg_path, "pfb_window", "hamming"));
    chz.time_order_fold = config.get_default<bool>(cfg_path, "pfb_time_order", false);
    // The Psi tables assume real input's near and far carrier images and leave the fold's
    // phase out.
    if (psi_synth && (chz.complex_input || chz.time_order_fold))
        FATAL_ERROR("cudaGnssChordTrack: psi_synth supports neither I/Q input nor "
                    "pfb_time_order");
    replica = std::make_unique<gnss::ChannelizedReplicaBank>(
        *sig, chz, f_offset_hz, prns,
        config.get_default<std::vector<int>>(cfg_path, "bds3s_prns", {}));
    // FDMA (GLONASS L2OF): each slot's carrier offset from the descriptor's, before the despread
    // reads the bank. Empty for a CDMA signal.
    replica->set_prn_freq_offsets(
        config.get_default<std::vector<double>>(cfg_path, "prn_freq_offset_hz", {}));
    // Each FDMA slot owns the channels of its own lobe (main lobe and the Doppler margin),
    // not the comb: the record and lane sums over the others would add noise and nothing else.
    if (replica->prn_freq_offsets_set() > 0) {
        slot_covering.assign((size_t)n_prn, {});
        slot_chan_mask.assign((size_t)n_prn, 0);
        std::string owned;
        for (int p = 0; p < n_prn; ++p) {
            const auto bins = replica->covering_bins(0.0, FDMA_DOPPLER_MARGIN_HZ, p);
            for (int c = 0; c < n_chan; ++c)
                if (std::find(bins.begin(), bins.end(), channel_ids[(size_t)c]) != bins.end()) {
                    slot_covering[(size_t)p].push_back(c);
                    slot_chan_mask[(size_t)p] |= 1ULL << c;
                }
            owned += fmt::format("{:s}{:d}", p ? "," : "", (int)slot_covering[(size_t)p].size());
        }
        INFO("cudaGnssChordTrack: FDMA, {:d} slot offsets; channels each slot owns here: {:s}",
             replica->prn_freq_offsets_set(), owned);
    }
    // The despread takes each channel's global bin, since the comb is sparse.
    despread = std::make_unique<GnssCudaDespread>(*replica, n_prn, channel_ids, hops_per_record,
                                                  sample_rate);
    if (psi_synth) {
        // Code-phase tables over the full PFB span: no Phi table, no chip cap (the Psi
        // interpolation needs the whole span; cudaGnssPsiSynth.hpp). The Phi keys are not
        // read, so a config carrying them is not refused and not honoured.
        const double refresh = config.get_default<double>(cfg_path, "psi_refresh_hz", 30.0);
        if (!(refresh > 0.0))
            FATAL_ERROR("cudaGnssChordTrack: psi_refresh_hz {:g} must be > 0", refresh);
        if (config.get_default<bool>(cfg_path, "psi_fp16", true))
            despread->set_psi_fp16();
        despread->set_psi_synth(refresh);
        if (config.get_default<bool>(cfg_path, "psi_sync_build", false)) {
            despread->set_psi_sync_build();
            WARN("cudaGnssChordTrack: psi_sync_build: every Psi build is waited for on the GPU "
                 "thread (for tests only)");
        }
        INFO("cudaGnssChordTrack: psi_synth: replicas from code-phase tables over the full PFB "
             "span (+1.1% replica energy against the 80-chip gather: absolute SNR gates step)");
    } else {
        // The per-hop PFB chip gather is an 80-chip window centred on the ~210-chip span, 2.6x
        // less synthesis work than the full span. A capped gather is a different replica, but
        // at CHORD geometry (scripts/gnss/e2e --max-chips --chips-centered) a centred window
        // reproduces the full-depth code phase down to 60 chips. The Phi tables are fp16, half
        // the device footprint at a storage error of ~3.3e-4. The config keys are read only to
        // refuse any other setting.
        constexpr int max_chips = 80;
        if (config.get_default<int>(cfg_path, "despread_max_chips", max_chips) != max_chips
            || !config.get_default<bool>(cfg_path, "despread_chips_centered", true)
            || !config.get_default<bool>(cfg_path, "phi_fp16", true))
            FATAL_ERROR("cudaGnssChordTrack: the despread runs only a centred {:d}-chip gather "
                        "with fp16 Phi; despread_max_chips must be {:d}, "
                        "despread_chips_centered and phi_fp16 true",
                        max_chips, max_chips);
        despread->set_max_chips(max_chips);
        despread->set_phi_fp16();
    }
    if (epl_from_prompt) {
        despread->set_epl_from_prompt();
        INFO("cudaGnssChordTrack: epl_from_prompt: the E and L lanes are the rotated prompt");
    }

    seeds.assign((size_t)n_prn, Seed{});
    trim.assign((size_t)n_prn, 0.0);
    trim_t_recv.assign((size_t)n_prn, 0.0);

    slot_gen.assign((size_t)n_prn, 0);
    // Sized once here: the slot count never changes, and the producer indexes it from the
    // first frame.
    fold.init(n_prn);

    _seed_ep = config.get<std::string>(cfg_path, "seed_endpoint");
    // trim_endpoint has no effect. The live configs still carry it, so it is read only so a
    // usage report does not list it unused.
    config.get_default<std::string>(cfg_path, "trim_endpoint", "");
    _set_trim_ep = config.get<std::string>(cfg_path, "set_trim_endpoint");
    _get_prns_ep = config.get<std::string>(cfg_path, "get_prns_endpoint");
    _set_prns_ep = config.get<std::string>(cfg_path, "set_prns_endpoint");
    _set_nav_ep = config.get_default<std::string>(cfg_path, "set_nav_symbols_endpoint", "");
    INFO_NON_OO("cudaGnssChordTrack: {:d} PRN x {:d} chan x {:d} elem, {:d} hops/record "
                "({:d} rec/frame), seeds on {:s}",
                n_prn, n_chan, n_elem, hops_per_record, n_rec, _seed_ep);
}

std::vector<RestVerb> TrackChain::rest_verbs() {
    using namespace std::placeholders;
    // set_trim is GnssFleetTrim's actuator. Slot membership is always served, with no enable
    // flag on the node: whether to apply swaps is the broker's decision (it also has a
    // report-only mode).
    std::vector<RestVerb> verbs = {
        {"set_seeds", _seed_ep, nullptr, std::bind(&TrackChain::set_seeds_callback, this, _1, _2)},
        {"set_trim", _set_trim_ep, nullptr,
         std::bind(&TrackChain::set_trim_callback, this, _1, _2)},
        {"get_prns", _get_prns_ep, std::bind(&TrackChain::get_prns, this), nullptr},
        {"set_prns", _set_prns_ep, nullptr,
         std::bind(&TrackChain::set_prns_callback, this, _1, _2)}};
    if (!_set_nav_ep.empty())
        verbs.push_back({"set_nav_symbols", _set_nav_ep, nullptr,
                         std::bind(&TrackChain::set_nav_symbols_callback, this, _1, _2)});
    return verbs;
}

std::vector<GnssCudaDespread::Spec>
TrackChain::record_specs(long long hop0, const std::vector<Seed>& seeds,
                         const std::vector<double>& trim_now, int jobs_before,
                         gnss_gpu::PrnCtl* pctl, int* slot2spec, void* stream, int& n_psi_dark) {
    const long long wstart = hop0 * (long long)fft_len;
    std::vector<GnssCudaDespread::Spec> specs;
    specs.reserve((size_t)n_prn);
    for (int p = 0; p < n_prn; ++p) {
        if (slot2spec)
            slot2spec[p] = -1;
        // Identity first, before any `continue`: a slot with no seed still holds a
        // satellite, and the assembler must see a PRN change even while the slot is dark,
        // so it drops the old satellite's element cal rather than carry it to the new one.
        pctl[p].prn = (uint16_t)prns[(size_t)p];
        const auto& sd = seeds[(size_t)p];
        if (!sd.have) {
            // Seed gone (set, expired, never acquired): the replica phase history is
            // meaningless across the gap, so drop it. The next record this PRN produces
            // emits reanchored = 1 and the assembler breaks the arc.
            fold.ok[(size_t)p] = 0;
            continue;
        }
        // FDMA: a slot whose lobe has no channel on this GPU is dark here, as unseeded.
        const bool fdma = !slot_covering.empty();
        if (fdma && slot_covering[(size_t)p].empty()) {
            fold.ok[(size_t)p] = 0;
            continue;
        }

        // Propagate the seed to this record's window, with the fleet trim.
        gnss::SeedState ss;
        ss.cp_chips = sd.cp_chips;
        ss.phase_ref_chips = sd.phase_ref_chips;
        ss.doppler_hz = sd.doppler_hz;
        ss.dop_rate = sd.dop_rate;
        ss.cp_rate = sd.cp_rate;
        ss.ref_hop = sd.ref_hop;
        const gnss::SeedPropagation pr = gnss::propagate_seed(*replica, ss, hop0, sample_rate,
                                                              sig->carrier_hz, trim_now[(size_t)p]);

        // Psi tables are built on the despread's builder thread and uploaded one per
        // record: a slot whose tables are not on the device yet (first seed, a swap, a
        // bucket past its grace) is dark this record exactly as a slot without a seed,
        // and joins when they land. Asked here, before anything is written for the slot:
        // no run, no job row, no fold history, slot2spec stays -1.
        if (psi_synth && !despread->psi_ready(p, pr.doppler_hz, stream)) {
            fold.ok[(size_t)p] = 0;
            ++n_psi_dark;
            continue;
        }

        GnssCudaDespread::Spec sp;
        sp.p = p;
        sp.doppler_hz = pr.doppler_hz;
        sp.cp_seed = pr.cp;
        sp.spacing_chips = dll_spacing;
        sp.ctrim_hz = sd.ctrim_hz;
        sp.covering = fdma ? slot_covering[(size_t)p] : covering;
        // GuoWang data-sign wipe: the frame at this record's reference sample is the
        // seed's frame at ref_hop plus the whole periods the phase advanced since (the
        // phase at ref_hop is in [0, L)).
        if (sd.nav_pages && sd.nav_frame_at_ref >= 0) {
            sp.nav_pages = sd.nav_pages.get();
            sp.nav_frame =
                sd.nav_frame_at_ref
                + (long long)std::floor(pr.phase_now / (double)replica->eff_code_length());
        } else if (sd.nav_frame_at_ref >= 0 && psi_synth && !_set_nav_ep.empty()) {
            // Streamed data symbols: the periods of this record's first and last hops (the
            // kernel's wrap count, at each hop's newest sample) and those between, all held, or
            // no wipe. The table repeats its last entry.
            const double L = (double)replica->eff_code_length();
            const gnss::SeedPropagation pe =
                gnss::propagate_seed(*replica, ss, hop0 + hops_per_record - 1, sample_rate,
                                     sig->carrier_hz, trim_now[(size_t)p]);
            const long long q0 = (long long)std::floor(pr.phase_now / L);
            const long long n_span = (long long)std::floor(pe.phase_now / L) - q0 + 1;
            if (n_span >= 1 && n_span <= GnssCudaDespread::Spec::NAV_N) {
                std::lock_guard<std::mutex> lk(nav_mtx);
                const auto it = nav_rings.find(prns[(size_t)p]);
                if (it != nav_rings.end()
                    && it->second.get(sd.nav_frame_at_ref + q0, (int)n_span, sp.nav.data())) {
                    std::fill(sp.nav.begin() + n_span, sp.nav.end(), sp.nav[n_span - 1]);
                    sp.nav_on = true;
                }
            }
        }

        // PrnCtl, the gnssGpuChain.hpp contract the assembler reads. job0 indexes the 4 rows
        // (E/P/L/P_HEAD) this spec occupies in the consumer's corr/energy arrays.
        //
        // Re-pin phase step: the replica's carrier phase is anchored absolutely,
        // 2*pi*(f_offset + applied)*t_abs, so a change in the applied carrier since the
        // previous record steps it by (applied - dop_prev)*t_abs cycles, many cycles at days
        // of uptime. It is taken here, in the Doppler domain, where the difference keeps full
        // precision (see PrnCtl::dcyc). Both terms use this record's t_abs: the step is the
        // phase difference at this instant between the old and the new replica.
        const double t_abs = (double)wstart / sample_rate;
        const bool have_hist = fold.ok[(size_t)p] != 0;
        // The applied carrier includes the broker's carrier trim: ctrim is part of the
        // despread's reference frequency (ang0 and wc both take it), so a ctrim change steps
        // the replica phase by dctrim*t_abs just as a Doppler change does.
        const double applied = pr.doppler_hz + sd.ctrim_hz;
        const double dcyc = have_hist ? (applied - fold.dop_prev[(size_t)p]) * t_abs : 0.0;
        fold.dop_prev[(size_t)p] = applied;
        fold.ok[(size_t)p] = 1;

        gnss_gpu::PrnCtl& c = pctl[p];
        c.run = 1;
        c.reanchored = have_hist ? 3 : 1;
        c.dcyc = dcyc;
        c.job0 = (jobs_before + (int)specs.size()) * gnss_gpu::ROWS_PLAIN;
        c.fcar_report = (float)pr.doppler_hz;
        c.n_owned = fdma ? (float)slot_covering[(size_t)p].size() : (float)n_chan;
        c.cp_seed = pr.cp;
        c.f_nco = sd.ctrim_hz;
        c.chan_mask = fdma             ? slot_chan_mask[(size_t)p]
                      : (n_chan >= 64) ? ~0ULL
                                       : ((1ULL << n_chan) - 1ULL);
        c.ctrim_hz = sd.ctrim_hz;
        // carrier_offset: f_offset plus the slot's FDMA offset (GLONASS L2OF), the carrier
        // the despread rotates to.
        c.fcar = replica->carrier_offset(p) + pr.doppler_hz;

        if (slot2spec)
            slot2spec[p] = (int)specs.size();
        specs.push_back(sp);
    }
    return specs;
}

void TrackChain::record_built(const std::vector<GnssCudaDespread::Spec>& specs,
                              gnss_gpu::PrnCtl* pctl) {
    // ang0 exists only once the despread has built the jobs, so this is a second pass over
    // pctl. It is read from the despread's record of what it handed the kernel, not re-derived
    // here.
    for (const auto& sp : specs) {
        gnss_gpu::PrnCtl& c = pctl[sp.p];
        c.ang0 = despread->last_ang0(sp.p);
        c.phi_ddop = despread->last_phi_ddop(sp.p);
        // The rotation the kernel applied to its E trial this record, for the assembler to
        // form the E and L rows under LANES_PROMPT (PrnCtl::rot). The control block is zeroed
        // per frame, so a chain without the rotated path leaves 0.
        if (const float* rot = despread->last_rot(sp.p))
            for (int ch = 0; ch < n_chan && ch < gnss_gpu::MAX_ROT_CHAN; ++ch)
                c.rot[ch] = rot[ch];
    }
}

void TrackChain::begin_frame(long long hop0_frame, void* stream) {
    note_frame_hop(hop0_frame);
    apply_prn_swaps(stream);
    std::lock_guard<std::mutex> lk(prn_mtx);
    for (int p = 0; p < n_prn; ++p)
        if (fold.slot_gen_seen[(size_t)p] != slot_gen[(size_t)p]) {
            fold.slot_gen_seen[(size_t)p] = slot_gen[(size_t)p];
            fold.dop_prev[(size_t)p] = 0.0;
            fold.ok[(size_t)p] = 0;
        }
}

void TrackChain::note_frame_hop(long long hop) {
    if (hop < 0)
        return;
    std::lock_guard<std::mutex> lk(prn_mtx);
    last_hop = (int64_t)hop;
}

nlohmann::json TrackChain::get_prns() {
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
    return reply;
}

void TrackChain::set_prns_callback(kotekan::connectionInstance& conn, nlohmann::json& request) {
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
              + " slots. The slot count is not live: it sizes every buffer, GPU allocation and "
                "wire frame in the pipeline, so a resize is a fleet-wide re-plumb and a node "
                "restart, not a swap. Send exactly one PRN per slot.";
    if (err.empty())
        err = gnss::prn_map_error(*sig, want);
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

void TrackChain::apply_prn_swaps(void* stream) {
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
            trim_t_recv[(size_t)p] = 0.0;
        }
    }
    for (const auto& l : log)
        WARN_NON_OO("PRN SWAP: slot {:d} PRN {:d} -> {:d}. Code table, Phi cache, carrier NCO, "
                    "seed, trim and power averages all reset COLD for this slot; it stays dark "
                    "until the broker seeds the new satellite.",
                    l.first, l.second, prns[(size_t)l.first]);
}

void TrackChain::set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request) {
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
                    // GuoWang data-sign wipe: both or neither.
                    sd.nav_frame_at_ref = s.value("nav_frame_at_ref", (long long)-1);
                    if (s.contains("nav_pages")) {
                        const auto& arr = s.at("nav_pages");
                        std::array<std::string, guowang::GW1202_NAV_PAGES> hex;
                        if (!arr.is_array() || arr.size() != hex.size())
                            throw std::runtime_error("nav_pages: expected 4 hex strings");
                        for (size_t k = 0; k < hex.size(); k++)
                            hex[k] = arr.at(k).get<std::string>();
                        auto pages = std::make_shared<guowang::Gw1202NavPages>();
                        if (!guowang::parse_gw1202_nav_pages(hex, *pages))
                            throw std::runtime_error("nav_pages: 79 hex digits per page");
                        sd.nav_pages = pages;
                    }
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

void TrackChain::expire_trims_locked(std::vector<int>& expired) {
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
        }
}

std::vector<double> TrackChain::snapshot_trims(std::vector<int>& expired) {
    std::lock_guard<std::mutex> lk(trim_mtx);
    expire_trims_locked(expired);
    return trim;
}

std::vector<TrackChain::Seed> TrackChain::snapshot_seeds(std::vector<int>& expired) {
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

void TrackChain::set_nav_symbols_callback(kotekan::connectionInstance& conn,
                                          nlohmann::json& request) {
    struct Row {
        int prn;
        long long sym0;
        std::vector<int8_t> signs;
        uint32_t stream_id;
    };
    std::vector<Row> rows;
    try {
        if (!request.is_array())
            throw std::runtime_error("expected an array");
        rows.reserve(request.size());
        for (const auto& r : request) {
            Row row{r.at("prn").get<int>(),
                    r.at("sym0").get<long long>(),
                    {},
                    r.value("stream_id", (uint32_t)0)};
            const std::string s = r.at("signs").get<std::string>();
            if (s.size() > (size_t)gnss::NavSymbolRing::N)
                throw std::runtime_error("more signs than a ring holds");
            row.signs.reserve(s.size());
            for (char c : s) {
                if (c != '+' && c != '-')
                    throw std::runtime_error("signs: only '+' and '-'");
                row.signs.push_back(c == '+' ? 1 : -1);
            }
            rows.push_back(std::move(row));
        }
    } catch (const std::exception& e) {
        conn.send_error(std::string("bad nav symbols payload: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(nav_mtx);
        for (const auto& r : rows)
            nav_rings[r.prn].put(r.sym0, (int)r.signs.size(), r.signs.data(), r.stream_id);
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void TrackChain::set_trim_callback(kotekan::connectionInstance& conn, nlohmann::json& request) {
    std::vector<std::pair<int, double>> upd;
    std::lock_guard<std::mutex> pk(prn_mtx); // first; see set_seeds_callback
    const std::string err = parse_trims(request, upd);
    if (!err.empty()) {
        conn.send_error("bad trim payload: " + err, kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    apply_trims(upd);
    // Nothing else is touched, not the seeds nor their t_recv (see trim_ttl_s in the header).
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

std::string TrackChain::parse_trims(const nlohmann::json& request,
                                    std::vector<std::pair<int, double>>& upd) {
    // Parse before taking trim_mtx, which the GPU thread takes every frame.
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
        return e.what();
    }
    return "";
}

void TrackChain::apply_trims(const std::vector<std::pair<int, double>>& upd) {
    const double now =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    {
        std::lock_guard<std::mutex> lk(trim_mtx);
        // Sweep here too: posts still arrive when execute() has stopped. See
        // expire_trims_locked.
        std::vector<int> gone;
        expire_trims_locked(gone);
        for (const auto& u : upd) {
            trim[(size_t)u.first] = u.second;
            trim_t_recv[(size_t)u.first] = now;
        }
    }
}

} // namespace gnss
