#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "gnssGpuChain.hpp"
#include "gnssRecord.hpp"
#include "gnssSharedPin.hpp" // for ref_offset, pin_rotation (#154)
#include "visUtil.hpp"       // for frameID

#include "json.hpp" // for the /get_spectrum reply

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h> // for gethostname

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssGpuRecordAssemble);

GnssGpuRecordAssemble::GnssGpuRecordAssemble(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssGpuRecordAssemble::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    _prns = config.get<std::vector<int>>(unique_name, "prns");
    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    // Per-channel prompt-phase dump (see the hpp note; diagnostic, default off).
    _chan_dump_prn = config.get_default<int>(unique_name, "chan_dump_prn", -1);
    // --phase-dump-prn: what the NCO fold is HANDED and what it does, per record, one PRN.
    _phi_dump_prn = config.get_default<int>(unique_name, "phi_dump_prn", -1);
    if (_phi_dump_prn >= 0) {
        _phi_dump_left = config.get_default<int>(unique_name, "phi_dump_records", 6000);
        const std::string path =
            config.get_default<std::string>(unique_name, "phi_dump_path", "/tmp/gnss_phi_dump.txt");
        _phi_dump = std::fopen(path.c_str(), "w");
        if (_phi_dump)
            std::fprintf(_phi_dump, "# r wstart reanchored c_dcyc phi_before phi_after f_nco dt "
                                    "ang0 fcar arg_raw arg_out\n");
    }
    _chan_dump_decim = std::max(1, config.get_default<int>(unique_name, "chan_dump_decim", 10));
    if (_chan_dump_prn >= 0) {
        const std::string path = config.get_default<std::string>(unique_name, "chan_dump_path",
                                                                 "/tmp/gnss_chan_phase_dump.txt");
        _chan_dump = std::fopen(path.c_str(), "a");
    }
    // ELEMENT AXIS (CHORD). 0 = the single-antenna airspy layout, byte-for-byte -- the default,
    // so nothing existing changes. >0 appends n_elements per-antenna blocks to every PRN record
    // (gnssRecord.hpp). The GPU correlation array grows the same axis: [rows][n_chan][n_elem].
    _n_elements = config.get_default<int>(unique_name, "n_elements", 0);
    // Which element the record HEADER's correlation slots carry. The broker closes the DLL and
    // carrier loops off those slots, so this must be a phase-coherent single-antenna view (see
    // gnssRecord.hpp) -- pick a healthy, high-gain feed. Ignored when n_elements == 0.
    _reference_element = config.get_default<int>(unique_name, "reference_element", 0);
    if (_n_elements > 0 && (_reference_element < 0 || _reference_element >= _n_elements)) {
        FATAL_ERROR("GnssGpuRecordAssemble: reference_element {:d} outside [0, n_elements={:d})",
                    _reference_element, _n_elements);
        return;
    }
    // SELF-CALIBRATED ELEMENT SUM (hpp note / gnssElemCal.hpp). Off by default; no-op unless
    // the element axis exists.
    _elem_sum = config.get_default<bool>(unique_name, "elem_sum", false) && _n_elements > 0;
    _elem_sum_tau_s = config.get_default<double>(unique_name, "elem_sum_tau_s", 5.0);
    _elem_hold_on_reanchor =
        config.get_default<bool>(unique_name, "elem_sum_hold_on_reanchor", true);
    _elem_sum_min_w = config.get_default<double>(unique_name, "elem_sum_min_w", 0.02);
    // HOLD vs ADAPT. The per-PRN cal learns its element phases FROM THAT PRN'S OWN DATA, so a
    // satellite near boresight, whose cross-correlation dominates every weak satellite's
    // per-element despread, teaches every cal ITS phases and the array gain is gone (measured
    // on a BeiDou transit: element vectors 80-90% identical to the bright satellite's inside
    // 2 deg, p down 100x, spared only the bright one). With geometry steered and one shared
    // instrumental prior seeded, adapt=false freezes the weights at the prior: nothing a
    // single satellite can move. A SHADOW cal keeps learning per PRN so the capture stays
    // visible (/get_elem_cal) without being acted on.
    _elem_adapt = config.get_default<bool>(unique_name, "elem_sum_adapt", true);
    // SHARED MODEL (hpp note): one instrument per pol + one inter-pol coefficient per PRN.
    // Shared implies held -- a learner that also combined would defeat the point.
    _elem_shared = config.get_default<bool>(unique_name, "elem_sum_shared", false);
    _elem_shared_tau_s = config.get_default<double>(unique_name, "elem_sum_shared_tau_s", 300.0);
    _elem_pol_tau_s = config.get_default<double>(unique_name, "elem_sum_pol_tau_s", 3.0);
    _elem_shared_freeze_deg =
        config.get_default<double>(unique_name, "elem_sum_shared_freeze_deg", 6.0);
    _elem_shared_freeze_hold_s =
        config.get_default<double>(unique_name, "elem_sum_shared_freeze_hold_s", 60.0);
    if (_elem_shared) {
        _elem_adapt = false;
        INFO("GnssGpuRecordAssemble[{:s}]: SHARED element model ON (consensus tau {:.0f} s, "
             "inter-pol tau {:.1f} s); per-PRN weights held to it",
             unique_name, _elem_shared_tau_s, _elem_pol_tau_s);
    }
    // #154 THE FLEET REFERENCE (hpp note): flat [re0, im0, re1, im1, ...] over n_elements, the
    // generator's copy of one band's entry in the committed snapshot (--elem-shared-ref).
    {
        const auto flat =
            config.get_default<std::vector<double>>(unique_name, "elem_sum_shared_ref", {});
        if (!flat.empty()) {
            if ((int)flat.size() != 2 * _n_elements) {
                FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: elem_sum_shared_ref has {:d} values, "
                            "need 2 x n_elements = {:d} (re, im per element) -- regenerate the "
                            "config against this array",
                            unique_name, flat.size(), 2 * _n_elements);
                return;
            }
            _g_fleet_ref.resize((size_t)_n_elements);
            for (int e = 0; e < _n_elements; ++e)
                _g_fleet_ref[(size_t)e] =
                    std::complex<double>(flat[(size_t)2 * e], flat[(size_t)2 * e + 1]);
            _fleet_ref_present = true;
        }
        const std::string mode =
            config.get_default<std::string>(unique_name, "elem_sum_shared_ref_mode", "off");
        if (mode == "off")
            _fleet_mode = 0;
        else if (mode == "log")
            _fleet_mode = 1;
        else if (mode == "live")
            _fleet_mode = 2;
        else {
            FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: elem_sum_shared_ref_mode '{:s}' is not "
                        "off | log | live",
                        unique_name, mode);
            return;
        }
        _fleet_slew_rad_s =
            config.get_default<double>(unique_name, "elem_sum_shared_ref_slew_deg_s", 1.0) * M_PI
            / 180.0;
        _fleet_min_sim =
            config.get_default<double>(unique_name, "elem_sum_shared_ref_min_sim", 0.5);
        if (_fleet_mode != 0)
            INFO("GnssGpuRecordAssemble[{:s}]: fleet reference {:s} ({:s}), slew {:.2f} deg/s, "
                 "min sim {:.2f}",
                 unique_name, mode, _g_fleet_ref.empty() ? "NONE configured: own pin" : "loaded",
                 _fleet_slew_rad_s * 180.0 / M_PI, _fleet_min_sim);
    }
    // ── #102 ELEMENT STEERING: geometric per-(sat, channel, element) phasors ─────────
    // OFF unless elem_positions_enu is provided (flat [n_elements][3], metres ENU of any
    // fixed array point -- re-referenced to the reference element below so the header's
    // slots and the ADR observable are UNTOUCHED by steering: the ref phasor is unity by
    // construction). The SIGN is a measured config parameter (gnssElemSteer.hpp note).
    {
        auto pos = config.get_default<std::vector<double>>(unique_name, "elem_positions_enu", {});
        const double st_sign = config.get_default<double>(unique_name, "elem_steer_sign", 1.0);
        const double st_hold = config.get_default<double>(unique_name, "elem_steer_hold_s", 120.0);
        if (!pos.empty() && _n_elements > 0) {
            if ((int)pos.size() != 3 * _n_elements)
                FATAL_ERROR("GnssGpuRecordAssemble: elem_positions_enu has {:d} values, "
                            "want 3*n_elements = {:d}",
                            (int)pos.size(), 3 * _n_elements);
            // channel_ids is read into _spec_freq_ids LATER in this constructor (the
            // chan-export block) -- read it locally here; init order bit the fleet once
            // (crash-loop, 2026-08-29 22:5x).
            auto _st_fids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});
            if (_st_fids.empty())
                FATAL_ERROR("GnssGpuRecordAssemble: elem steering needs channel_ids (the "
                            "per-channel RF frequencies)");
            for (int i = 0; i < 3; ++i) {
                const double r0 = pos[(size_t)_reference_element * 3 + i];
                for (int e = 0; e < _n_elements; ++e)
                    pos[(size_t)e * 3 + i] -= r0;
            }
            std::vector<double> f_mhz;
            for (int id : _st_fids)
                f_mhz.push_back(id * 0.1953125);
            _steer = gnss::ElemSteer(std::move(pos), std::move(f_mhz), (int)_prns.size(), st_sign,
                                     st_hold);
            INFO("GnssGpuRecordAssemble[{:s}]: #102 element steering READY ({:d} elements, "
                 "{:d} channels, sign {:+.0f}) -- awaiting /set_sat_geometry posts",
                 unique_name, _n_elements, (int)_st_fids.size(), st_sign);
        }
    }
    const int n = (int)_prns.size();
    // Record width is a C++ constant; the frame is sized in yaml (n_prn * record_floats *
    // sizeof_float) with nothing linking the two. A stale config under-sizes the frame and this
    // stage writes past its end -- die at construction rather than corrupt memory.
    const int rec_stride = gnss::record_stride(_n_elements);
    const size_t rec_bytes_need = (size_t)n * rec_stride * sizeof(float);
    if ((size_t)out_buf->frame_size < rec_bytes_need) {
        FATAL_ERROR("GnssGpuRecordAssemble: out_buf frame {:d} B < {:d} PRN x {:d} floats "
                    "({:d} B). Bump 'record_floats' in the config to {:d}.",
                    (size_t)out_buf->frame_size, n, rec_stride, rec_bytes_need, rec_stride);
        return;
    }
    if (_elem_sum) {
        _cal.assign((size_t)n, gnss::ElemCal(_n_elements, _reference_element, _elem_sum_tau_s,
                                             _elem_sum_min_w));
        _cal_shadow = _cal;
        _cal_sim.assign((size_t)n, -1.0);
        _anchor_warned.assign((size_t)n, 0);
        _g_shared.assign((size_t)_n_elements, std::complex<double>(0.0, 0.0));
        _pol_num.assign((size_t)n, std::complex<double>(0.0, 0.0));
        _pol_den.assign((size_t)n, 0.0);
        _pol_warmth.assign((size_t)n, 0.0);
        _pol_c.assign((size_t)n, std::complex<double>(0.0, 0.0));
        _w_scratch.assign((size_t)_n_elements, std::complex<double>(0.0, 0.0));
    }
    // ── BRIGHT-SATELLITE PROJECTION (hpp note; gnssProjSubspace.hpp) ──────────────────────
    // Needs the element axis, the channel labels (the board is keyed on freq_id) and the
    // element cal (the shadow learners). Ships OFF; the mode is a live switch (/set_elem_proj)
    // so a canary can be armed without a config regeneration.
    {
        auto fids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});
        const std::string mode_s =
            config.get_default<std::string>(unique_name, "elem_proj_mode", "off");
        _proj_mode = (mode_s == "live") ? 2 : (mode_s == "shadow") ? 1 : 0;
        _proj_deg = config.get_default<double>(unique_name, "elem_proj_deg", 4.0);
        _proj_kmax_alloc =
            std::max(1, std::min(4, config.get_default<int>(unique_name, "elem_proj_rank_max", 2)));
        _proj_rank_max = _proj_kmax_alloc;
        _proj_max_age_s = config.get_default<double>(unique_name, "elem_proj_max_age_s", 2.0);
        _proj_probe_frac_min =
            config.get_default<double>(unique_name, "elem_proj_probe_frac_min", 0.5);
        _proj_tau_s = config.get_default<double>(unique_name, "elem_proj_tau_s", 4.0);
        _proj_probe_tau_s = config.get_default<double>(unique_name, "elem_proj_probe_tau_s", 1.0);
        _proj_probe_since_s =
            config.get_default<double>(unique_name, "elem_proj_probe_since_s", 90.0);
        _bore_az_deg = config.get_default<double>(unique_name, "boresight_az_deg", 180.0);
        _bore_el_deg = config.get_default<double>(unique_name, "boresight_el_deg", 81.41);
        _proj_group = config.get_default<std::string>(unique_name, "elem_proj_group",
                                                      unique_name.substr(0, unique_name.find('_')));
        std::string sys = config.get_default<std::string>(unique_name, "gnss_system", "");
        if (sys.empty()) {
            // Fallback from the stage name (the generator's chain tokens); the key wins when set.
            auto has = [&](const char* t) { return unique_name.find(t) != std::string::npos; };
            sys = (has("_e5a") || has("_e5b") || has("_e6") || has("_e1"))    ? "E"
                  : (has("_b2a") || has("_b2b") || has("_b3i") || has("_b1")) ? "C"
                  : (has("_l1of") || has("_l2of") || has("_l3oc"))            ? "R"
                                                                              : "G";
        }
        _proj_sys = sys[0];
        _proj_ready = _elem_sum && _n_elements > 0 && !fids.empty();
        if (_proj_ready) {
            _proj_fids = fids;
            _proj_own.resize((size_t)n);
            _proj_probe = gnss::ProjSubspace(_n_elements, (int)fids.size(), _proj_probe_tau_s,
                                             _proj_kmax_alloc);
            _proj_steer.assign((size_t)_proj_kmax_alloc * fids.size() * _n_elements,
                               gnss::ElemSteer::cf(1.0f, 0.0f));
            _proj_Q.resize(fids.size());
            for (auto& Q : _proj_Q)
                Q.init(_n_elements, _proj_kmax_alloc);
            _proj_isB.assign((size_t)n, 0);
            _proj_isProbe.assign((size_t)n, 0);
            _proj_dropped.assign((size_t)n, 0);
            _slot_was_run.assign((size_t)n, 0);
            _slot_run_since.assign((size_t)n, 0.0);
            _slot_probe_broker.assign((size_t)n, 0);
            _cal_proj = _cal;
            _cap_plain.assign((size_t)n, -1.0);
            _cap_proj.assign((size_t)n, -1.0);
            _b_cos2.assign((size_t)n, -1.0);
            _sim_pp.assign((size_t)n, -1.0);
            _g_proj.assign((size_t)_n_elements, {0.0, 0.0});
            _v_scratch.assign((size_t)_n_elements, {0.0, 0.0});
            _proj_probe_frac.assign(fids.size(), 0.0);
            _proj_probe_on.assign(fids.size(), 0);
            _proj_probe_used.assign(fids.size(), 0);
            _proj_steer_all.assign((size_t)n * fids.size() * _n_elements,
                                   gnss::ElemSteer::cf(1.0f, 0.0f));
            _proj_steered.assign((size_t)n, 0);
            _proj_owner_probe = unique_name + "/probe";
            _proj_k_ch.assign(fids.size(), 0);
            _proj_src_ch.assign(fids.size(), 0);
            INFO("GnssGpuRecordAssemble[{:s}]: bright-satellite projection READY, mode {:s} "
                 "(system {:c}, group {:s}, {:d} channels, deg {:.1f}, rank <= {:d}, tau own "
                 "{:.1f} s / probe {:.1f} s, probe trigger frac >= {:.2f}); POST /set_elem_proj "
                 "to change",
                 unique_name, mode_s, _proj_sys, _proj_group, (int)fids.size(), _proj_deg.load(),
                 _proj_kmax_alloc, _proj_tau_s, _proj_probe_tau_s, _proj_probe_frac_min.load());
        } else if (_proj_mode.load() != 0) {
            WARN("GnssGpuRecordAssemble[{:s}]: elem_proj_mode {:s} requested but the projection "
                 "needs elem_sum, an element axis and channel_ids -- OFF",
                 unique_name, mode_s);
            _proj_mode = 0;
        }
    }
    _phi.assign(n, 0.0);
    _phi_cyc.assign(n, 0.0);
    _phi_cmd_prev.assign(n, 0.0);
    _phi_cmd_ok.assign(n, 0);
    _fcar_prev.assign(n, 0.0);
    _fnco_prev.assign(n, 0.0);
    _fcar_prev_ok.assign(n, 0);
    _a_prev.assign(n, {0.0, 0.0});
    _a_prev_ok.assign(n, 0);
    _elem_prev_ok.assign(n, 0);
    _wstart_prev.assign(n, 0);
    // PER-CHANNEL PROMPT SPECTRUM (hpp note; task #32). Keyed on `channel_ids` being present:
    // the generator wires the same per-GPU freq_id list the despread runs, so the export knows
    // which sky frequency each channel index is. No channel_ids (airspy, legacy configs) ->
    // fully inert: no accumulation, and the endpoint answers with an explanatory error rather
    // than not existing, so a mis-wired poller sees WHY instead of a bare 404.
    _spec_freq_ids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});

    // PER-CHANNEL COMB EXPORT (gnssRecord.hpp, 2026-08-14). ⚠️ THE CROSS-CHANNEL SUM BELOW IS A
    // LOSS WE SHOULD NEVER HAVE TAKEN -- it destroys the frequency axis a delay lives on, so
    // the broker can only FIT a per-instance constant where it should DERIVE one from the ramp.
    // KV: "purge the idea of summing across channels in each instance, that's *never* what we
    // want to do." This appends the UNSUMMED comb after the PRN records; the summed slots stay
    // for now so nothing downstream has to change on the same day.
    //
    // channel_ids is REQUIRED: the broker must know which SKY FREQUENCY each column is. A comb
    // whose columns are unlabelled is worse than no comb -- a delay fit across mislabelled
    // frequencies returns a confident wrong tau.
    _chan_export = config.get_default<bool>(unique_name, "chan_export", false);
    if (_chan_export && _spec_freq_ids.empty()) {
        FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: chan_export needs channel_ids -- an unlabelled "
                    "frequency axis makes every delay fit downstream confidently wrong",
                    unique_name);
        return;
    }
    if (_chan_export) {
        const size_t need =
            gnss::frame_floats(n, _n_elements, (int)_spec_freq_ids.size()) * sizeof(float);
        if ((size_t)out_buf->frame_size < need) {
            FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: chan_export needs a {:d} B frame ({:d} PRN "
                        "x {:d} record floats, then {:d} PRN x {:d} chan x {:d}); out_buf is "
                        "{:d} B. The generator sizes this -- regenerate the config.",
                        unique_name, need, n, rec_stride, n, (int)_spec_freq_ids.size(),
                        gnss::CHAN_FLOATS, (size_t)out_buf->frame_size);
            return;
        }
        INFO("GnssGpuRecordAssemble[{:s}]: exporting the UNSUMMED comb -- {:d} channels x {:d} "
             "floats per PRN per record, appended after the records ({:d} B frame)",
             unique_name, (int)_spec_freq_ids.size(), gnss::CHAN_FLOATS, need);
    }

    if (!_spec_freq_ids.empty()) {
        const size_t nc = _spec_freq_ids.size();
        // WINDOW QUANTISATION (task #53). The window index is floor(wstart / win_samples) on
        // the F-engine sample clock, so every instance assigns a record to the same window
        // WITHOUT talking to any other instance. Configure in SAMPLES, not seconds: the
        // division must be exact integer arithmetic or two instances can straddle a boundary
        // differently, which is the whole failure this replaces. Every instance must be given
        // the SAME value -- the generator writes it, and a mismatch shows up immediately as
        // disjoint window indices in the broker's first poll.
        //
        // Whole records land in exactly one window regardless of whether win_samples is a
        // record multiple (a record is assigned by its wstart, never split), so alignment does
        // not depend on that -- but making it a multiple keeps the per-window record count
        // constant, which is one less thing to explain in a residual.
        _spec_win_samples =
            (int64_t)config.get_default<double>(unique_name, "spectrum_window_samples", 0.0);
        const int depth = config.get_default<int>(unique_name, "spectrum_ring_depth", 8);
        if (_spec_win_samples > 0) {
            _spec_ring.resize((size_t)std::max(2, depth));
            for (auto& w : _spec_ring) {
                w.re.assign((size_t)n * nc, 0.0);
                w.im.assign((size_t)n * nc, 0.0);
                w.energy.assign((size_t)n * nc, 0.0);
                w.nrec.assign(n, 0);
                w.phi0.assign(n, 0.0);
                w.nreanchor.assign(n, 0);
            }
            INFO("per-channel spectrum export ON: {:d} channels, freq_id {:d}..{:d}; "
                 "ADDRESSABLE windows of {:d} samples ({:.3f} s), ring depth {:d}",
                 (int)nc, _spec_freq_ids.front(), _spec_freq_ids.back(),
                 (long long)_spec_win_samples, (double)_spec_win_samples / _sample_rate,
                 (int)_spec_ring.size());
        } else {
            // LEGACY: one open accumulator, reset on read. Kept so a node whose config predates
            // spectrum_window_samples keeps working until it is regenerated -- but it CANNOT be
            // aligned across instances, so the broker treats a legacy reply as unaddressable.
            _spec_ring.resize(1);
            _spec_ring[0].re.assign((size_t)n * nc, 0.0);
            _spec_ring[0].im.assign((size_t)n * nc, 0.0);
            _spec_ring[0].energy.assign((size_t)n * nc, 0.0);
            _spec_ring[0].nrec.assign(n, 0);
            _spec_ring[0].phi0.assign(n, 0.0);
            _spec_ring[0].nreanchor.assign(n, 0);
            WARN("per-channel spectrum export ON: {:d} channels -- but LEGACY reset-on-read "
                 "windows (no 'spectrum_window_samples' in this config). Instances CANNOT be "
                 "aligned; regenerate this node's config (task #53).",
                 (int)nc);
        }
        _spec_scratch.assign((size_t)std::max(1, _n_elements), {0.0, 0.0});
    }

    // ── THE BEAM CUBE (hpp): the (channel x element) axis, un-collapsed ──────────────────
    // Requires BOTH axes to exist: channel_ids (so a bin has a frequency) and an element axis
    // (so it has an antenna). Silent no-op otherwise -- this is opt-in and costs exactly
    // nothing when off, which is what makes it safe to ship dark and arm per chain.
    _cube_on = config.get_default<bool>(unique_name, "beam_cube", false) && !_spec_freq_ids.empty()
               && _n_elements > 0;
    if (_cube_on) {
        const int nc = (int)_spec_freq_ids.size();
        _cube_bin_width =
            std::max(0, config.get_default<int>(unique_name, "beam_cube_bin_width", 0));
        // 0 = one bin per channel. Otherwise ceil, so the last bin may be narrow rather than
        // silently dropping the top channels -- a truncated band edge is exactly the kind of
        // quiet loss that looks like a real beam feature later.
        _cube_bins = (_cube_bin_width > 0) ? (nc + _cube_bin_width - 1) / _cube_bin_width : nc;

        // WINDOW LENGTH IN SAMPLES, NOT SECONDS, for the reason #53 gave the spectrum ring:
        // the index is floor(wstart / win_samples) in integer arithmetic, and if that division
        // is not exact two instances can straddle a boundary differently -- which is the whole
        // failure being avoided. Every instance must be given the SAME value.
        //
        // ⚠️ AN EXACT 1.000 s WINDOW DOES NOT EXIST ON THIS CLOCK. The hop rate is 195312.5 Hz,
        // so one second is 195312.5 hops -- not an integer. 96 records = 24 frames = 1.00663 s
        // is an exact multiple of BOTH the record (2048 hops) and the frame (8192), so no
        // record is ever split and the per-window record count is constant, which is one less
        // thing to explain in a residual. Anyone wanting a round number in seconds is asking
        // for a boundary that lands mid-record on half the fleet.
        //
        // ⚠️⚠️ THE UNIT IS F-ENGINE SAMPLES, NOT HOPS -- wstart is in samples, and one hop is
        // fft_length (16384) of them. The default here READ 196608, which is 96 x 2048 HOPS:
        // the record arithmetic above, with the conversion left off. As a sample count that is
        // 12 hops, 61 us, and it would lap this 8-deep ring sixteen times per record while
        // every log line still looked plausible -- the window length is only ever printed as
        // win_samples/_sample_rate, which is small and self-consistently wrong. The generator
        // now always sets this explicitly (records x hops_per_record x fft_length, the same
        // expression spectrum_window_samples is derived from), because a value that must MATCH
        // across instances should never come from a default that agrees by luck.
        _cube_win_samples = (int64_t)config.get_default<double>(
            unique_name, "beam_cube_window_samples", 96.0 * 2048.0 * 16384.0);
        const int depth =
            std::max(2, config.get_default<int>(unique_name, "beam_cube_ring_depth", 8));
        _cube_ring.resize((size_t)depth);
        const size_t ncell = (size_t)n * _cube_bins * _n_elements;
        const size_t nbin = (size_t)n * _cube_bins;
        for (auto& W : _cube_ring) {
            W.coh_re.assign(ncell, 0.0);
            W.coh_im.assign(ncell, 0.0);
            W.incoh.assign(ncell, 0.0);
            W.w.assign(nbin, 0.0);
            W.energy.assign(nbin, 0.0);
            W.nrec.assign((size_t)n, 0);
            W.phi0.assign((size_t)n, 0.0);
            W.nreanchor.assign((size_t)n, 0);
        }
        // THE PUSH LEG (hpp). Optional: with no cube_buf the cube is REST-only, which is what
        // an interactive bench wants and is NOT an archive.
        // THE FULL ADDRESS: <hostname>/<stage>. The stage name is unique only within a node
        // and the far side has ONE bufferRecv for the whole fleet.
        {
            char hn[64] = {0};
            gethostname(hn, sizeof(hn) - 1);
            char* dot = std::strchr(hn, '.');
            if (dot)
                *dot = '\0'; // short name: "cx19", not "cx19.site.chord-observatory.ca"
            _cube_chain = config.get_default<std::string>(unique_name, "beam_cube_chain",
                                                          std::string(hn) + "/" + unique_name);
        }
        _cube_gpu = config.get_default<int>(unique_name, "beam_cube_gpu", -1);
        _cube_max_bins = std::max(
            _cube_bins, config.get_default<int>(unique_name, "beam_cube_max_bins", _cube_bins));
        _cube_max_prn = std::max(n, config.get_default<int>(unique_name, "beam_cube_max_prn", n));
        const std::string cbuf = config.get_default<std::string>(unique_name, "cube_buf", "");
        if (!cbuf.empty()) {
            _cube_out_buf = buffer_container.get_buffer(cbuf);
            _cube_out_buf->register_producer(unique_name);
            const size_t need = gnss::cube_frame_bytes(_cube_max_prn, _cube_max_bins, _n_elements);
            if ((size_t)_cube_out_buf->frame_size < need) {
                FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: cube_buf needs a {:d} B frame "
                            "({:d} PRN x {:d} bin x {:d} elem); {:s} is {:d} B. The generator "
                            "sizes this from gnss::cube_frame_bytes -- regenerate the config.",
                            unique_name, need, _cube_max_prn, _cube_max_bins, _n_elements, cbuf,
                            (size_t)_cube_out_buf->frame_size);
                return;
            }
            if (_cube_out_buf->metadata_pool == nullptr)
                WARN("GnssGpuRecordAssemble[{:s}]: cube_buf {:s} has NO metadata_pool. Frames "
                     "will carry no metadata; a bufferSend on this buffer DROPS them (it cannot "
                     "send a frame without one) and a bufferRecv far side expects the pool's "
                     "type. Give it metadata_pool: gnss_pool (#110).",
                     unique_name, cbuf);
            INFO("GnssGpuRecordAssemble[{:s}]: beam-cube PUSH leg on -> {:s} ({:d} B/frame, "
                 "~{:.2f} MB/s at this window length); backpressure DROPS and the loss is "
                 "counted into the next frame",
                 unique_name, cbuf, need, need / ((double)_cube_win_samples / _sample_rate) / 1e6);
        }
        INFO("GnssGpuRecordAssemble[{:s}]: BEAM CUBE on -- {:d} PRN x {:d} subband bin(s) "
             "({:d} channel(s), width {:d}) x {:d} element(s); COHERENT + incoherent, "
             "addressable windows of {:d} samples ({:.5f} s), ring depth {:d}; "
             "{:d} cell(s), {:.1f} MB",
             unique_name, n, _cube_bins, nc, _cube_bin_width ? _cube_bin_width : 1, _n_elements,
             (long long)_cube_win_samples, (double)_cube_win_samples / _sample_rate, depth, ncell,
             depth * (3.0 * ncell + 2.0 * nbin) * sizeof(double) / 1e6);
    } else if (config.get_default<bool>(unique_name, "beam_cube", false)) {
        WARN("GnssGpuRecordAssemble[{:s}]: beam_cube requested but {:s} -- the cube needs BOTH "
             "axes and is OFF.",
             unique_name,
             _spec_freq_ids.empty() ? "this config has no channel_ids (no frequency axis)"
                                    : "n_elements is 0 (no element axis)");
    }

    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_spectrum",
        std::bind(&GnssGpuRecordAssemble::spectrum_callback, this, _1));
    if (_cube_on)
        kotekan::restServer::instance().register_get_callback(
            unique_name + "/get_beam_cube",
            std::bind(&GnssGpuRecordAssemble::beam_cube_callback, this, _1));
    if (_elem_sum)
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_elem_gain",
            std::bind(&GnssGpuRecordAssemble::set_elem_gain_callback, this, _1, _2));
    // Registered whether or not steering is armed: the broker posts its sky to every
    // assembler it knows, and an unsteered one answering 404 every 30 s is indistinguishable
    // in its log from a dead one. Unsteered, the post is acknowledged and dropped.
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_sat_geometry",
        std::bind(&GnssGpuRecordAssemble::set_sat_geometry_callback, this, _1, _2));
    if (_elem_sum) {
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_elem_sum_adapt",
            std::bind(&GnssGpuRecordAssemble::set_elem_sum_adapt_callback, this, _1, _2));
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_elem_sum_shared",
            std::bind(&GnssGpuRecordAssemble::set_elem_sum_shared_callback, this, _1, _2));
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_elem_sum_shared_ref",
            std::bind(&GnssGpuRecordAssemble::set_elem_sum_shared_ref_callback, this, _1, _2));
        kotekan::restServer::instance().register_get_callback(
            unique_name + "/get_elem_cal",
            std::bind(&GnssGpuRecordAssemble::get_elem_cal_callback, this, _1));
        if (_proj_ready)
            kotekan::restServer::instance().register_post_callback(
                unique_name + "/set_elem_proj",
                std::bind(&GnssGpuRecordAssemble::set_elem_proj_callback, this, _1, _2));
    }
    // LIVE REFERENCE SWAP (KV, 2026-08-20). Registered whenever the element axis exists --
    // the header's correlation slots carry the reference even with elem_sum off, so the
    // swap is meaningful either way. See set_reference_element_callback.
    if (_n_elements > 0)
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_reference_element",
            std::bind(&GnssGpuRecordAssemble::set_reference_element_callback, this, _1, _2));
}

GnssGpuRecordAssemble::~GnssGpuRecordAssemble() {
    if (_chan_dump)
        std::fclose(_chan_dump);
    if (_phi_dump)
        std::fclose(_phi_dump);
}

void GnssGpuRecordAssemble::set_sat_geometry_callback(kotekan::connectionInstance& conn,
                                                      nlohmann::json& request) {
    // #102: {"<prn>": [az_deg, el_deg, az_rate_dps, el_rate_dps, t_utc], ...}. The rates and
    // the epoch let the steering follow the satellite between the ~30 s posts (gnssElemSteer.hpp:
    // held, a snapshot is ~0.2 m off by the next post on the 44 m baseline). A bare [az, el]
    // (older broker) is still accepted and held. Unknown PRNs are skipped (the broker posts its
    // whole sky; this stage steers the slots it owns).
    int n_up = 0;
    if (!_steer.enabled()) {
        conn.send_json_reply(nlohmann::json{{"updated", 0}, {"steering", "off"}});
        return;
    }
    try {
        const double now_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        // "_bore": [sep_deg, t_utc] -- the broker's POOLED nearest-to-boresight separation over
        // every constellation (a satellite of another system rails and leaks into this band
        // just the same). It gates the shared model's learning (see _elem_shared_freeze_deg).
        auto bi = request.find("_bore");
        if (bi != request.end() && bi->is_array() && !bi->empty()) {
            _bore_sep_deg.store((*bi)[0].get<double>());
            _bore_post_t.store(now_s);
        }
        std::lock_guard<std::mutex> lk(_steer_mtx);
        // "_probes": [prn, ...] -- this chain's noise probes (below-horizon PRNs) when the
        // broker names them; the projection's probe stack then uses exactly those slots.
        // Absent (older broker), a probe is inferred as a running slot with no geometry for
        // elem_proj_probe_since_s (geometry is only ever posted above the horizon).
        {
            auto pi = request.find("_probes");
            if (pi != request.end() && pi->is_array() && !_slot_probe_broker.empty()) {
                std::fill(_slot_probe_broker.begin(), _slot_probe_broker.end(), 0);
                for (const auto& v : *pi) {
                    if (!v.is_number_integer())
                        continue;
                    const int prn = v.get<int>();
                    for (size_t p = 0; p < _prns.size() && p < _slot_probe_broker.size(); ++p)
                        if (_prns[p] == prn)
                            _slot_probe_broker[p] = 1;
                }
                _probes_from_broker = true;
            }
        }
        for (auto it = request.begin(); it != request.end(); ++it) {
            const int prn = std::atoi(it.key().c_str());
            if (prn <= 0 || !it.value().is_array() || it.value().size() < 2)
                continue;
            for (size_t p = 0; p < _prns.size(); ++p)
                if (_prns[p] == prn) {
                    const auto& v = it.value();
                    const bool rated = v.size() >= 5;
                    _steer.update((int)p, v[0].get<double>(), v[1].get<double>(), now_s,
                                  rated ? v[2].get<double>() : 0.0,
                                  rated ? v[3].get<double>() : 0.0,
                                  rated ? v[4].get<double>() : 0.0);
                    ++n_up;
                    break;
                }
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_json_reply(nlohmann::json{{"updated", n_up}});
}

int GnssGpuRecordAssemble::follow_frame_prns(const void* pctl_v, int n_prn) {
    using namespace gnss_gpu;
    const PrnCtl* pctl = (const PrnCtl*)pctl_v;
    int n_changed = 0;
    for (int p = 0; p < n_prn && p < (int)_prns.size(); ++p) {
        // Record 0 speaks for the frame: the producer applies swaps at a FRAME boundary, so
        // every record in this frame carries the same slot->PRN map. A producer that predates
        // the field leaves prn = 0, and 0 is not a PRN -- read it as "no claim" and keep the
        // config's value, which is exactly the old behaviour.
        const int claimed = (int)pctl[(size_t)p].prn;
        if (claimed <= 0 || claimed == _prns[(size_t)p])
            continue;
        const int was = _prns[(size_t)p];
        _prns[(size_t)p] = claimed;
        ++n_changed;

        // COLD RESET, everything keyed to this slot, in ONE place. Each of these describes the
        // satellite that just left; carried across, each becomes the old satellite's state
        // attributed to the new one -- the accumulator-identity trap, and every one of them
        // would be silent (a warm cal is a plausible cal, a phase history is a plausible
        // history). The reference-element swap above resets the same set for the same reason.
        if (_elem_sum && (size_t)p < _cal.size()) {
            _cal[(size_t)p] =
                gnss::ElemCal(_n_elements, _reference_element, _elem_sum_tau_s, _elem_sum_min_w);
            _cal_shadow[(size_t)p] = _cal[(size_t)p];
            _cal_sim[(size_t)p] = -1.0;
            shared_reset_prn((size_t)p);
            proj_reset_slot((size_t)p, std::chrono::duration<double>(
                                           std::chrono::steady_clock::now().time_since_epoch())
                                           .count());
            if ((size_t)p < _anchor_warned.size())
                _anchor_warned[(size_t)p] = 0;
        }
        // The steer table is keyed by SLOT: the departed satellite's phasors must not steer
        // the newcomer for up to hold_s. Geometry for the new PRN arrives with the next post.
        if (_steer.enabled()) {
            std::lock_guard<std::mutex> lk(_steer_mtx);
            _steer.invalidate(p);
        }
        _phi[(size_t)p] = 0.0;
        _phi_cyc[(size_t)p] = 0.0;
        _phi_cmd_prev[(size_t)p] = 0.0;
        _phi_cmd_ok[(size_t)p] = 0;
        _fcar_prev[(size_t)p] = 0.0;
        _fnco_prev[(size_t)p] = 0.0;
        _fcar_prev_ok[(size_t)p] = 0;
        _a_prev[(size_t)p] = {0.0, 0.0};
        _a_prev_ok[(size_t)p] = 0;
        _elem_prev_ok[(size_t)p] = 0;
        _wstart_prev[(size_t)p] = 0;
        // The spectrum ring accumulates PER SLOT across a window. A window straddling the swap
        // would otherwise sum two satellites into one row and report the total as one PRN's
        // spectrum -- so drop this slot's partial accumulation in every open window. nrec = 0
        // makes the row read as absent rather than as a quiet satellite.
        {
            std::lock_guard<std::mutex> lk(_spec_mtx);
            for (auto& W : _spec_ring) {
                if (W.idx < 0 || (size_t)p >= W.nrec.size())
                    continue;
                const size_t nc = W.re.size() / std::max<size_t>(1, _prns.size());
                for (size_t c = 0; c < nc; ++c) {
                    W.re[(size_t)p * nc + c] = 0.0;
                    W.im[(size_t)p * nc + c] = 0.0;
                    W.energy[(size_t)p * nc + c] = 0.0;
                }
                W.nrec[(size_t)p] = 0;
                if ((size_t)p < W.phi0.size())
                    W.phi0[(size_t)p] = 0.0;
                if ((size_t)p < W.nreanchor.size())
                    W.nreanchor[(size_t)p] = 0;
            }
        }
        // THE BEAM CUBE accumulates per slot too, and for the same reason: carried across a
        // swap, the departing satellite's integrated power would be attributed to the arriving
        // one -- and a beam map is precisely a claim about where power came from, so this one
        // would not merely be wrong, it would be wrong in the shape of a feature. The COHERENT
        // sum is worse still: its phi0 refers to the OLD satellite's NCO history, so a
        // consumer would rotate the new satellite onto a reference that never described it.
        //
        // Clear the slot in EVERY OPEN WINDOW, not just the newest -- the ring holds several,
        // and a window straddling the swap would otherwise sum two satellites into one row and
        // report the total as one PRN. Zero the sums AND the weights together: a nonzero
        // weight over a zeroed sum reads as a real measurement of no power, which is worse
        // than absence.
        if (_cube_on) {
            std::lock_guard<std::mutex> ck(_cube_mtx);
            const size_t ne = (size_t)_n_elements;
            const size_t ncell = (size_t)_cube_bins * ne;
            for (auto& C : _cube_ring) {
                if (C.idx < 0 || (size_t)p >= C.nrec.size())
                    continue;
                std::fill_n(C.coh_re.begin() + (size_t)p * ncell, ncell, 0.0);
                std::fill_n(C.coh_im.begin() + (size_t)p * ncell, ncell, 0.0);
                std::fill_n(C.incoh.begin() + (size_t)p * ncell, ncell, 0.0);
                std::fill_n(C.w.begin() + (size_t)p * _cube_bins, _cube_bins, 0.0);
                std::fill_n(C.energy.begin() + (size_t)p * _cube_bins, _cube_bins, 0.0);
                C.nrec[(size_t)p] = 0;
                C.phi0[(size_t)p] = 0.0;
                C.nreanchor[(size_t)p] = 0;
            }
        }
        WARN("GnssGpuRecordAssemble: slot {:d} PRN {:d} -> {:d} (the producer swapped it). "
             "Every per-slot accumulator for this slot reset COLD -- element cal, NCO phase, "
             "arc continuity and any open spectrum window. Downstream sees a fresh "
             "acquisition on this slot, which is what it is.",
             p, was, claimed);
    }
    return n_changed;
}

void GnssGpuRecordAssemble::main_thread() {
    using namespace gnss_gpu;
    frameID frame_in(in_buf), frame_out(out_buf);
    const int n_prn = (int)_prns.size();

    while (!stop_thread) {
        const uint8_t* in = in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in == nullptr)
            return;
        FrameHdr hdr;
        std::memcpy(&hdr, in, sizeof(hdr));
        if (hdr.n_prn != n_prn) {
            FATAL_ERROR("GnssGpuRecordAssemble: frame n_prn {:d} != config {:d}", hdr.n_prn, n_prn);
            return;
        }
        // LIVE REFERENCE SWAP, applied at the frame boundary so every record in a frame
        // sees ONE reference. Rebuilds the cal set COLD: every learned per-element gain
        // (and any stored path-B prior) is phase-anchored to the OLD reference element and
        // does not transfer -- carrying them across would hand the combine a wrong phase
        // convention with full confidence. The header rides the new bare reference while
        // the cal re-warms (~3 tau); downstream sees one phase step, exactly as it does on
        // any fresh acquisition, and the WARN below is the operator's receipt.
        {
            int newref = -1;
            {
                std::lock_guard<std::mutex> lk(_gain_mtx);
                if (_pending_ref >= 0) {
                    newref = _pending_ref;
                    _pending_ref = -1;
                }
            }
            if (newref >= 0 && newref != _reference_element) {
                const int oldref = _reference_element;
                _reference_element = newref;
                if (_elem_sum) {
                    _cal.assign(_prns.size(), gnss::ElemCal(_n_elements, _reference_element,
                                                            _elem_sum_tau_s, _elem_sum_min_w));
                    _cal_shadow = _cal;
                    if (_proj_ready)
                        _cal_proj = _cal;
                    std::fill(_cal_sim.begin(), _cal_sim.end(), -1.0);
                    std::fill(_anchor_warned.begin(), _anchor_warned.end(), 0);
                    std::fill(_elem_prev_ok.begin(), _elem_prev_ok.end(), 0);
                    // The shared model is anchored to the old reference: relearn it.
                    _g_shared_warm = false;
                    _g_pin_ref_ok = false;
                    _g_shared_n = 0;
                    for (size_t p2 = 0; p2 < _cal.size(); ++p2)
                        shared_reset_prn(p2);
                }
                WARN("set_reference_element: header reference {:d} -> {:d} at the frame "
                     "boundary. Element cal rebuilt COLD ({} PRNs, re-warm ~{:.1f} s); "
                     "downstream sees one phase step on every PRN (fresh-acquisition class).",
                     oldref, newref, _cal.size(), 3.0 * _elem_sum_tau_s);
            }
        }
        // PATH B: a newly-injected per-element gain prior seeds every PRN's cal here, once,
        // before this frame's records are combined. seed() also stores it, so each subsequent
        // reset() (re-anchor) re-applies it rather than going cold -- the whole point.
        if (_elem_sum) {
            std::vector<std::complex<double>> pend;
            {
                std::lock_guard<std::mutex> lk(_gain_mtx);
                if (_pending_gain_set) {
                    pend.swap(_pending_gain);
                    _pending_gain_set = false;
                }
            }
            if (!pend.empty())
                WARN("set_elem_gain: consuming {:d} gains (n_elem={:d}, {:d} cals)",
                     (int)pend.size(), _n_elements, (int)_cal.size());
            if ((int)pend.size() == _n_elements) {
                for (auto& ec : _cal)
                    ec.seed(pend.data(), _n_elements);
                int nwarm = 0;
                for (auto& ec : _cal)
                    if (ec.warm())
                        ++nwarm;
                WARN("set_elem_gain: seeded, {:d}/{:d} cals warm right after seed", nwarm,
                     (int)_cal.size());
            }
        }
        const int n_chan = hdr.n_chan;
        // Output rows per spec: 4 normally, 6 when the writer peels (rows 4/5 = the peel
        // residual). 0 means a writer that predates the field -- read it as 4.
        const int n_rows_spec = (hdr.n_rows_spec > 0) ? hdr.n_rows_spec : ROWS_PLAIN;
        const int64_t* winstart = (const int64_t*)(in + off_winstart());
        const PrnCtl* pctl = (const PrnCtl*)(in + off_prnctl());
        // LIVE SLOT MEMBERSHIP: adopt the producer's map before anything in this frame is
        // labelled or accumulated. Costs one integer compare per slot per frame in steady
        // state.
        follow_frame_prns(pctl, n_prn);
        const double* corr = (const double*)(in + off_corr(n_prn)); // double2 rows
        // The energy block sits AFTER the corr block, whose size scales with the ELEMENT axis
        // -- the writer passes its n_elem (cudaGnssChordTrack: 32), so the reader must too, or
        // every "energy" lands inside the corr block: garbage when enough PRNs run to fill
        // that far (which is why sparse days flickered and broker-full days "worked"), zeros
        // when few do -- and a zero P_ENERGY makes the combiner treat the record as inactive,
        // so 2-PRN seeding produced structurally-perfect numerically-zero records
        // (found 2026-07-31, first trim-port test).
        const double* energy = (const double*)(in
                                               + off_energy(n_prn, n_chan, n_rows_spec,
                                                            (_n_elements > 0) ? _n_elements : 1));

        // NO ABSOLUTE ANCHOR? SAY SO, AND STILL EMIT A UNIFORM GRID. The old fallback stamped
        // system_clock::now() PER RECORD, and that silently broke every cross-record estimator
        // downstream: this stage emits hdr.n_rec records back to back, so on CHORD the four
        // sub-records of a frame were stamped microseconds apart instead of 10.49 ms apart.
        // GnssCoherentCombiner's rate search takes dt = the MINIMUM consecutive spacing and
        // builds an integer grid from it, so those microseconds became the grid and the records
        // were scattered across the transform. Cost, measured on cx19 2026-08-07: path B's
        // deep_rate_q read 4.5-11.4 against path A's 27-38 on IDENTICAL per-record data, failing
        // the q >= 10 gate, and coh_frac sat at 0.02-0.10 against 0.93-0.97 -- diagnosed for a
        // day and a half as a sensitivity problem because amp_snr, which uses no time base at
        // all, was a healthy 71-83% of path A throughout.
        //
        // Latching the anchor once and extrapolating by wstart keeps the same host-clock origin
        // (no worse than before for anything reading absolute time) and makes the grid exactly
        // uniform (no jitter at all for anything reading differences). The warning still fires,
        // because a host clock is not a GPS clock and the config should carry frame0_utc.
        if (hdr.utc0 <= 0.0 && hdr.n_rec > 0) {
            if (_wall_anchor == 0.0) {
                const double now = std::chrono::duration<double>(
                                       std::chrono::system_clock::now().time_since_epoch())
                                       .count();
                _wall_anchor = now - (double)winstart[0] / _sample_rate;
            }
            if ((_no_utc0_frames++ % 2000) == 0)
                WARN("GnssGpuRecordAssemble: producer sent no frame0_utc -- record UTC is the "
                     "HOST clock anchored once at startup, not GPS time ({:d} frames). Absolute "
                     "time is only as good as this host's clock; add frame0_utc to the producer "
                     "stage's config. ({:d} records/frame, so a per-record stamp would not even "
                     "be uniform.)",
                     _no_utc0_frames, hdr.n_rec);
        }

        for (int r = 0; r < hdr.n_rec && !stop_thread; ++r) {
            float* out = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            // ZERO THE WHOLE COMB BLOCK FIRST. The per-PRN record zeroing below covers only
            // record_stride floats; a PRN that does not run this window, or a channel outside
            // its covering mask, would otherwise hand the broker the PREVIOUS record's value as
            // if it were current -- the same "stale is not neutral" trap GnssChanAlignMerge
            // zeroes absent feeds for.
            if (_chan_export && out != nullptr)
                std::fill(out + gnss::chan_block_offset(n_prn, _n_elements),
                          out + gnss::frame_floats(n_prn, _n_elements, n_chan), 0.0f);
            if (out == nullptr)
                return;
            const int64_t wstart = winstart[r];
            const double utc =
                ((hdr.utc0 > 0.0) ? hdr.utc0 : _wall_anchor) + (double)wstart / _sample_rate;

            // BRIGHT-SATELLITE PROJECTION, per record (hpp note): feed the source trackers
            // from this record's RAW rows, build the per-channel basis, publish it to the
            // siblings. Inert (k = 0) unless armed and a source is in force.
            if (_proj_ready && _proj_mode.load() != 0)
                proj_prepare_record(corr, &pctl[(size_t)r * n_prn], n_chan, _n_elements, wstart,
                                    utc,
                                    std::chrono::duration<double>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
            else
                _proj_k_rec = 0;

            for (int p = 0; p < n_prn; ++p) {
                const int rec_stride = gnss::record_stride(_n_elements);
                float* rec = out + (size_t)p * rec_stride;
                // Zero the WHOLE record, element blocks included: a PRN that does not run this
                // window must not leave a previous window's per-antenna correlations behind.
                for (int f = 0; f < rec_stride; ++f)
                    rec[f] = 0.0f;
                rec[gnss::REC_PROJ_COST] = -1.0f; // measured below when a basis is in force
                const PrnCtl& c = pctl[(size_t)r * n_prn + p];
                // THE FRAME'S PRN WINS. _prns was reconciled to it at the top of this frame,
                // so the two agree -- but reading it from the record's own control word means
                // record slot 0 cannot be wrong even if a future producer moves a slot
                // mid-frame. 0 = a producer that does not stamp it; fall back to the list.
                rec[0] = (float)((c.prn > 0) ? (int)c.prn : _prns[p]);
                *reinterpret_cast<double*>(rec + gnss::RECORD_UTC_SLOT) = utc;
                if (!c.run) {
                    _a_prev_ok[p] = 0;
                    _elem_prev_ok[p] = 0;
                    _fcar_prev_ok[p] = 0;
                    _phi_cyc[p] = 0.0;
                    _phi_cmd_ok[p] = 0;
                    continue;
                }
                // Cross-channel sum over the covering mask, per correlator trial
                // (E, P, L, P_HEAD -- the prompt's head segment, gnssRecord.hpp slots 16-18),
                // plus, when the chain peels, the residual prompt and its head (rows 4/5 ->
                // gnssRecord.hpp slots 20-23). PrnCtl::job0 already carries the per-spec row
                // stride, so indexing is job0 + t either way.
                //
                // NB rows 0-3 hold the FULL, un-peeled correlation even when peeling: the
                // analytic add-back was applied on the device (docs/gnss_voltage_peel_live.md).
                // That is what leaves everything below this stage -- combiner, broker, viewer,
                // TEC -- unable to tell whether a peel happened.
                // With an element axis the correlation array is [rows][n_chan][n_elem] and the
                // covering-mask sum runs PER ANTENNA. The ENERGIES do not: one replica is
                // correlated against every antenna, so energy stays [rows][n_chan] and is summed
                // once (gnssRecord.hpp -- this is why energies live in the record header and only
                // the correlations go in the element blocks).
                //
                // n_e == 1 with n_elements == 0 makes the index expression below collapse to
                // exactly the single-antenna one, so that path is unchanged.
                const int n_e = (_n_elements > 0) ? _n_elements : 1;
                const int ref_e = (_n_elements > 0) ? _reference_element : 0;
                std::complex<double> g3[6]; // reference element, for the header + NCO/gain state
                double e3[6];               // element-independent
                _g_elem.assign((size_t)n_rows_spec * n_e, std::complex<double>(0.0, 0.0));
                // LIVE PROJECTION (hpp note): a non-source slot's rows are projected IN PLACE
                // in the input frame before anything reads them, so the sum below, the taps,
                // the cube and the element blocks all see one and the same projected data.
                if (_proj_k_rec > 0 && _proj_mode.load() == 2 && !_proj_isB[p])
                    proj_slot_inplace(const_cast<double*>(corr), &c, n_chan, n_e, n_rows_spec);
                // #102: steer this satellite's elements if geometry is fresh. The lock is
                // cheap here (one take per record row-set); the REST writer holds it only
                // while rebuilding one slot's table.
                // The slot's table is COPIED under the lock and read from the copy: a row
                // pointer read after the lock is released can be rewritten by a concurrent
                // /set_sat_geometry mid-combine (benign in value, undefined in principle).
                bool steered = false;
                if (_steer.enabled()) {
                    if (_steer.n_chan() != n_chan) {
                        // The table was built from the config's channel_ids; the frame says
                        // otherwise. Steering with a mismatched channel axis would apply one
                        // channel's phasor to another's data -- refuse, once and loudly.
                        if (!_steer_nchan_warned) {
                            WARN("elem steering DISABLED: steer table has {:d} channels, the "
                                 "frame {:d} -- channel_ids and the producer disagree",
                                 _steer.n_chan(), n_chan);
                            _steer_nchan_warned = 1;
                        }
                    } else {
                        std::lock_guard<std::mutex> lk(_steer_mtx);
                        const double now_s =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
                        if (_steer.warm((int)p, now_s)) {
                            // Follow the satellite to THIS record's time (the data's clock,
                            // not the host's: records lag the wall by the pipeline depth).
                            _steer.refresh((int)p, utc);
                            _steer_buf.resize((size_t)n_chan * n_e);
                            _steer.copy_slot((int)p, _steer_buf.data());
                            steered = true;
                        }
                    }
                }
                for (int t = 0; t < n_rows_spec; ++t) {
                    const size_t row = (size_t)(c.job0 + t) * n_chan;
                    double e = 0.0;
                    for (int ch = 0; ch < n_chan; ++ch)
                        if ((c.chan_mask >> ch) & 1ULL) {
                            e += energy[row + ch];
                            const size_t base = (row + ch) * n_e;
                            const gnss::ElemSteer::cf* st =
                                steered ? &_steer_buf[(size_t)ch * n_e] : nullptr;
                            for (int el = 0; el < n_e; ++el) {
                                std::complex<double> v(corr[2 * (base + el)],
                                                       corr[2 * (base + el) + 1]);
                                if (st)
                                    v *= std::complex<double>(st[el].real(), st[el].imag());
                                _g_elem[(size_t)t * n_e + el] += v;
                            }
                        }
                    g3[t] = _g_elem[(size_t)t * n_e + ref_e];
                    e3[t] = e;
                }
                // SHADOW PROJECTION (hpp note): the same prompt, projected per channel before
                // steering and summing, into _g_proj -- for the projected learner and the
                // capture diagnostics only. The frame, _g_elem and everything live are untouched.
                const bool proj_this = _proj_k_rec > 0 && _proj_mode.load() == 1 && !_proj_isB[p];
                if (proj_this)
                    proj_slot_shadow(corr, &c, n_chan, n_e, steered);
                // SELF-CALIBRATED ELEMENT SUM (hpp note). Once this PRN's cal is warm, every
                // header row (E/P/L/PH/RES -- same weights: same antennas) becomes the
                // calibrated weighted mean instead of the bare reference element: reference-
                // anchored phase, "one element" scale, noise down ~sqrt(N_healthy). The cal
                // updates AFTER the combine (causal: this record's weights come from previous
                // records only), from the PROMPT row against the header actually emitted --
                // the bootstrap that starts on the reference element and converges to MRC.
                // The element BLOCKS are untouched: they are the per-antenna measurement.
                std::complex<double> g_sky(0.0, 0.0); // LOO sky-phase-corrected prompt (slot 24/25)
                if (_elem_sum) {
                    gnss::ElemCal& ec = _cal[p];
                    if (c.reanchored == 1 && !_elem_hold_on_reanchor)
                        ec.reset(); // fresh acquisition: full reset (legacy behavior)
                    // HOLD across a carrier re-anchor by default: the per-element cal is
                    // E_e*conj(E_ref), which the COMMON carrier cancels out of -- so a carrier
                    // re-pin (reanchored==1) does not change the element gains, and resetting
                    // them there is exactly what kept ElemCal from ever warming under the
                    // near-every-record re-anchoring (measured 0%% warm). The L5 array is phase-
                    // coherent, so the held gains stay valid; update() refreshes any drift.
                    // NB: do NOT reset _anchor_warned on re-anchor -- the trackers can re-anchor
                    // every record, and re-arming the WARN there ballooned the node log to tens
                    // of GB. The "reference too weak" message is worth exactly once per PRN.
                    // SHARED MODEL: this record's weights come from the model as it stood
                    // (previous records' consensus and coefficient -- causal), installed as
                    // held weights so combine/combine_split run unchanged.
                    if (_elem_shared.load())
                        shared_hold(p);
                    if (ec.warm()) {
                        for (int t = 0; t < n_rows_spec; ++t)
                            g3[t] = ec.combine(&_g_elem[(size_t)t * n_e]);
                        // The deep fold's input: same prompt, each element derotated by the phase
                        // of the OTHERS. Separate slot -- the header keeps the raw phase because
                        // that IS the ADR observable (gnssRecord.hpp REC_SKY_RE).
                        g_sky = ec.combine_split(&_g_elem[(size_t)1 * n_e]);
                    }
                    if (_elem_prev_ok[p] && wstart > _wstart_prev[p]) {
                        const double dt_s = (double)(wstart - _wstart_prev[p]) / _sample_rate;
                        if (_elem_adapt) {
                            ec.update(&_g_elem[(size_t)1 * n_e], dt_s);
                        } else {
                            // HELD: the live weights stay what the prior (or the last adapting
                            // record) left them. The shadow keeps learning from this PRN alone,
                            // and its agreement with the held weights is the capture detector:
                            // sim ~1 = this satellite's data still say the same phases; sim -> 0
                            // = something else is teaching it (a transit).
                            gnss::ElemCal& sh = _cal_shadow[p];
                            if (c.reanchored == 1 && !_elem_hold_on_reanchor)
                                sh.reset();
                            sh.update(&_g_elem[(size_t)1 * n_e], dt_s);
                            if (sh.warm() && ec.warm()) {
                                const auto& wl = ec.weights();
                                const auto& ws = sh.weights();
                                std::complex<double> x(0.0, 0.0);
                                double nl = 0.0, ns = 0.0;
                                for (int e2 = 0; e2 < n_e; ++e2) {
                                    x += std::conj(wl[(size_t)e2]) * ws[(size_t)e2];
                                    nl += std::norm(wl[(size_t)e2]);
                                    ns += std::norm(ws[(size_t)e2]);
                                }
                                _cal_sim[p] =
                                    (nl > 0.0 && ns > 0.0) ? std::norm(x) / (nl * ns) : -1.0;
                            }
                            if (_elem_shared.load())
                                shared_pol_update(p, &_g_elem[(size_t)1 * n_e], dt_s);
                        }
                        // THE PROJECTED LEARNER (hpp note): fed the projected prompt while a
                        // source is in force and the plain prompt otherwise, so out of transit
                        // it is the plain shadow's twin and in transit their difference is the
                        // capture. Diagnostics per slot: cap_plain / cap_proj = how much of
                        // each learner's weight vector lies in the interferer subspace,
                        // b_cos2 = the same for the held (live) weights = the projection's
                        // predicted cost on this satellite, sim_pp = the two learners' agreement.
                        if (_proj_ready && _proj_mode.load() != 0) {
                            gnss::ElemCal& pj = _cal_proj[p];
                            if (c.reanchored == 1 && !_elem_hold_on_reanchor)
                                pj.reset();
                            pj.update(proj_this ? _g_proj.data() : &_g_elem[(size_t)1 * n_e], dt_s);
                            proj_slot_diag(p, &c, n_chan, n_e, steered);
                            rec[gnss::REC_PROJ_COST] = (float)_b_cos2[p];
                        }
                    }
                    if (ec.anchor_moved()) {
                        if (!_anchor_warned[p]) {
                            WARN("elem_sum PRN {:d}: reference element {:d} too weak -- phase "
                                 "anchor moved to the strongest element (one-time phase step "
                                 "downstream)",
                                 _prns[p], _reference_element);
                            _anchor_warned[p] = 1;
                        }
                    }
                }
                // Per-channel PROMPT dump (diagnostic, see hpp): raw pre-rotation per-channel
                // correlations -- the cross-channel relative phases are the observable.
                if (_chan_dump && _prns[p] == _chan_dump_prn
                    && (++_chan_dump_ctr % _chan_dump_decim) == 0) {
                    const size_t prow = (size_t)(c.job0 + 1) * n_chan; // trial 1 = PROMPT
                    for (int ch = 0; ch < n_chan; ++ch)
                        if ((c.chan_mask >> ch) & 1ULL)
                            std::fprintf(_chan_dump, "%.6f %d %.6e %.6e %.6e\n", utc, ch,
                                         corr[2 * ((prow + ch) * n_e + ref_e)],
                                         corr[2 * ((prow + ch) * n_e + ref_e) + 1],
                                         energy[prow + ch]);
                }
                rec[1] = c.fcar_report;
                rec[2] = (float)c.cp_seed;
                rec[6] = c.n_owned;
                rec[gnss::REC_E_RE] = (float)g3[0].real();
                rec[gnss::REC_E_IM] = (float)g3[0].imag();
                rec[gnss::REC_E_ENERGY] = (float)e3[0];
                rec[gnss::REC_L_RE] = (float)g3[2].real();
                rec[gnss::REC_L_IM] = (float)g3[2].imag();
                rec[gnss::REC_L_ENERGY] = (float)e3[2];

                // Carrier NCO (pass-2 half): the command's fence re-anchor resets the phase
                // history exactly like the tracker's in-place reset; f_nco (ctrim + ff ramp)
                // changes the slope of phi, never jumps it.
                // reanchored 2 and 3 are the SAME event; they differ only in who subtracted.
                // 3 carries the step in c.dcyc because the assembler cannot compute it
                // accurately from fcar (see gnssGpuChain.hpp: differencing two 1.176 GHz
                // doubles leaves 0.4 rad on the table, the same order as the term itself).
                const bool repin = (c.reanchored == 2 || c.reanchored == 3);
                const double phi_before_fold = _phi[p];
                if (c.reanchored == 1 || (repin && !_fcar_prev_ok[p])) {
                    // FRESH acquisition: no phase history to preserve. Break the arc.
                    _phi[p] = 0.0;
                    _phi_cyc[p] = 0.0;
                    _a_prev_ok[p] = 0;
                    if (!_elem_hold_on_reanchor)
                        _elem_prev_ok[p] = 0; // legacy: also break element continuity
                    _phi_cmd_ok[p] = 0;
                } else if (repin) {
                    // PHASE-CONTINUOUS RE-PIN. Re-pinning f_ref steps the ABSOLUTELY-ANCHORED
                    // replica phase by df*t_abs -- thousands of cycles at soak age. The old code
                    // folded that step into an EXPORT-ONLY offset and then ZEROED the NCO, so the
                    // exported ADR survived but the DATA did not: the rotation applied to A moved
                    // by the step, i.e. every re-pin punched an effectively random phase jump into
                    // the very correlation the combiner deep-integrates. That is the residual C/N0
                    // sawtooth -- and it showed exactly where this explanation says it must, in the
                    // COHERENT estimator at precisely max_anchor_age_s (30.0 s on every strong sat,
                    // 0.2 dB on GPS to 1.6 dB on B1C) and NOT in the phase-blind incoherent one
                    // (2026-07-14, on-sky).
                    //
                    // The step is FOLDABLE: an NCO absorbs an arbitrary constant phase. Put it in
                    // the NCO instead of the export and the despread output is continuous THROUGH
                    // the re-pin -- the commanded phase stays exactly as continuous as before (the
                    // export algebra is unchanged: fcar*t - phi_cyc is invariant under this fold),
                    // so the validated ADR arcs are untouched.
                    //
                    // This is what makes a re-pin CHEAP, which is the real prize: with the phase
                    // step folded and the code-currency step already translated above, the replica
                    // can be re-pinned as often as we like. max_anchor_age_s: 0 re-pins EVERY
                    // record, so f_ref never goes stale and the within-record decoherence that
                    // grows with anchor age (the OTHER half of the sawtooth,
                    // ~(dop_rate*age*t_rec)^2
                    // -- negligible on GPS's 1 ms record, ~1 dB on B1C's 10 ms) never accumulates.
                    //
                    // ⚠️ THIS BRANCH WAS DEAD CODE ON CHORD UNTIL 2026-08-13. Both CHORD producers
                    // hardcoded `reanchored` = 0 (cudaGnssChordTrack.cpp, cudaGnssInject.cpp), so
                    // the step below -- 109 to 1127 CYCLES per record at 3.37 days of uptime,
                    // measured on gal_e5a -- went straight into every correlation. With
                    // carrier-gain 0.0 making f_nco zero as well, _phi was identically zero and
                    // this stage applied NO derotation at all. That is the whole of the "per-record
                    // common phase is white in time" folklore: it is not the sky, it is this
                    // subtraction never being performed. See task #52.
                    const double t_pin = (double)wstart / _sample_rate;
                    const double dcyc =
                        (c.reanchored == 3) ? c.dcyc : (c.fcar - _fcar_prev[p]) * t_pin;
                    _phi_cyc[p] += dcyc;
                    _phi[p] = std::remainder(_phi[p] + 2.0 * M_PI * dcyc, 2.0 * M_PI);
                }
                // #72 DIAGNOSTIC PASS-THROUGH. Both are recorded by the despread as the values
                // it actually handed the kernel; this stage only copies them. Unconditional --
                // NOT inside the dt/_a_prev_ok guard below -- because the question they answer
                // is asked per record, including the first of an arc.
                rec[gnss::REC_ANG0] = (float)c.ang0;
                rec[gnss::REC_PHI_DDOP] = (float)c.phi_ddop;
                const double dt = (double)(wstart - _wstart_prev[p]) / _sample_rate;
                if (_a_prev_ok[p] && dt > 0.0) {
                    // Commanded-trim increment (slot 19, see gnssRecord.hpp): the trim the
                    // producer ACTUALLY applied, exported honestly in PrnCtl::ctrim_hz and
                    // integrated over this record so downstream can subtract the loop's
                    // transients from the ADR exactly. This used to be reconstructed from the
                    // identity ctrim = (f_nco + fcar_report - fcar)/2 -- valid only for the
                    // airspy tracker's f_nco = ctrim + ff convention; on the CHORD producers
                    // it evaluated to (ctrim - f_offset)/2, MHz-scale garbage (see the
                    // PrnCtl::ctrim_hz doc, gnssGpuChain.hpp).
                    rec[gnss::REC_TRIM_INC] = (float)(c.ctrim_hz * dt);
                    // ⚠️ THE MIDPOINT of the previous and current f_nco (e2e [4e],
                    // 2026-08-21, MEASURED): neither endpoint is right. dt spans
                    // [t_prev, t_now]; charging the NEW slope over that gap left a
                    // persistent |dctrim|*dt offset per command step (the "[4e] drip"),
                    // but charging the OLD slope leaves the same-size residue with the
                    // opposite lineage, because a step also moves the record's PHASE
                    // CENTROID (the prompt integrates over the window; its phase sits at
                    // window CENTER, the argument anchor at window START). The average is
                    // exact: bench commanded rms 7.8 vs control 7.5 mcyc under a
                    // 4x-live staircase, from ~20-26 for either endpoint. REC_TRIM_INC
                    // above deliberately keeps THIS record's ctrim: it documents what the
                    // producer applied to this record's despread -- a different (and
                    // correct) statement.
                    _phi[p] += 2.0 * M_PI * 0.5 * (_fnco_prev[p] + c.f_nco) * dt;
                    // Keep the ROTATION phase bounded, but track the NCO phase UNWRAPPED (in
                    // cycles) for the carrier-phase export. remainder() slides phi by whole
                    // multiples of 2*pi, which a rotation cannot see and a mod-1 phase export
                    // cannot see either -- but a per-record INCREMENT sees every one of them as
                    // a spurious +-1 cycle. The NCO wraps ~f_nco times a second, so the ADR bled
                    // about one cycle per wrap: measured as a per-satellite ADR-rate error of a
                    // few Hz, scaling with each satellite's carrier trim (2026-07-13).
                    _phi[p] = std::remainder(_phi[p], 2.0 * M_PI);
                    _phi_cyc[p] += 0.5 * (_fnco_prev[p] + c.f_nco) * dt;
                }
                _fnco_prev[p] = c.f_nco; // AFTER the charge: next record's gap rides this value
                const std::complex<double> rot = std::polar(1.0, -_phi[p]);
                // #72: PUBLISH THE CURRENCY. `rot` is about to be applied to slots 3/4 and to
                // every comb column, and its exponent has a per-instance ARBITRARY origin. Ship
                // it so a consumer can rotate every instance onto one common reference; see
                // gnss::REC_PHI0, and the SpecWindow's W.phi0[p] which does the same for
                // /get_spectrum. Written BEFORE the rotation is used, so it can never describe
                // a different record's phase than the one it rode out with.
                rec[gnss::REC_PHI0] = (float)_phi[p];
                const std::complex<double> g_corr = g3[1] * rot;
                if (_phi_dump && _prns[p] == _phi_dump_prn) {
                    std::fprintf(
                        _phi_dump,
                        "%d %lld %d %.17g %.17g %.17g %.17g %.10g %.17g %.17g %.10g %.10g\n", r,
                        (long long)wstart, (int)c.reanchored, c.dcyc, phi_before_fold, _phi[p],
                        c.f_nco, (double)(wstart - _wstart_prev[p]) / _sample_rate, c.ang0, c.fcar,
                        std::arg(g3[1]), std::arg(g_corr));
                    if (--_phi_dump_left <= 0) {
                        std::fclose(_phi_dump);
                        _phi_dump = nullptr;
                        INFO("GnssGpuRecordAssemble: phi dump for PRN {:d} complete",
                             _phi_dump_prn);
                    }
                }
                rec[3] = (float)g_corr.real();
                rec[4] = (float)g_corr.imag();
                rec[5] = (float)e3[1];
                // PER-CHANNEL PROMPT SPECTRUM accumulation (hpp note; task #32). Same source
                // as the chan_dump above (trial 1 = PROMPT, per channel, pre-sum), same
                // element combination as the header (cal when warm, else the reference
                // element), and the SAME `rot` the record's prompt just got -- one common
                // phase per record, so the accumulation doesn't wind with the residual
                // carrier while the cross-channel RELATIVE phases (the delay observable)
                // pass through untouched. Only the covering-mask channels accumulate;
                // masked-off ones stay identically zero and the consumer skips zero-energy.
                if (!_spec_freq_ids.empty() && (int)_spec_freq_ids.size() == n_chan) {
                    const size_t prow = (size_t)(c.job0 + 1) * n_chan;
                    std::lock_guard<std::mutex> lk(_spec_mtx);
                    SpecWindow& W = spec_window_for(wstart);
                    // PUBLISH THE PHASE CURRENCY (task #52). The rotation applied below is
                    // exp(-i*_phi[p]), and _phi[p] is an accumulator that is ZEROED on a fresh
                    // acquisition and STEPPED on a continuous re-pin -- and the broker re-pins
                    // seeds every ~2 minutes. So the export's phase reference MOVES UNDERNEATH
                    // the consumer, on exactly the minutes timescale where the measured
                    // window-to-window coherence collapses to its random floor.
                    //
                    // Neither of my two earlier attempts was right: leaving it implicit (the
                    // original) hands the consumer a reference it cannot know, and subtracting
                    // a per-window origin (45fe3a438) throws away the within-window continuity
                    // that is the whole point. KV: "the original was also wrong."
                    //
                    // So publish it. A window's sum is exp(-i*phi0) * SUM_r G_r*exp(-i*dphi_r),
                    // so a consumer that knows phi0 can rotate any two windows onto a common
                    // reference exactly; and n_reanchor > 0 says the accumulator was reset or
                    // stepped MID-window, which no single constant can undo -- drop those.
                    // This is #45's (value, currency, epoch) discipline applied to a phase:
                    // the spectrum was a value whose currency was never transported.
                    if (W.nrec[p] == 0)
                        W.phi0[p] = _phi[p];
                    // ⚠️ ONLY AN **UNFOLDED** RESET BREAKS THE WINDOW (2026-08-14, #53).
                    // This counted `reanchored != 0`, but reanchored 2 and 3 are the
                    // CONTINUOUS re-pins whose phase step the block immediately above has
                    // just folded INTO _phi -- exactly so the accumulator stays continuous
                    // across them. Only 1 is a fresh acquisition, where there is no phase
                    // history, the NCO is reset and the arc genuinely breaks.
                    //
                    // Counting the folded ones made n_reanchor a constant: cudaGnssInject
                    // sets `reanchored = have_hist ? 3 : 1` on EVERY record (the Doppler is
                    // propagated per record, so the reference legitimately moves every
                    // record), so path B reported n_reanchor > 0 for every PRN in every
                    // window, forever. Measured on sky: 100% of (PRN, window) on all four
                    // chains, before AND after the seed-cadence work -- which is what
                    // finally showed the counter, not the receiver, was the problem.
                    // fleet_spectrum_aligned drops any PRN with n_reanchor > 0, so #53's
                    // aligned gather discarded 100% of its points on a benign condition.
                    if (c.reanchored == 1)
                        W.nreanchor[p] += 1;
                    // ⚠️ DO **NOT** REFERENCE THIS TO THE WINDOW'S OWN FIRST RECORD.
                    // I tried exactly that (45fe3a438) to remove the arbitrary per-instance
                    // origin of _phi[p], and it was a mistake: _phi[p] is CONTINUOUS IN TIME
                    // (it only resets on re-acquisition), so subtracting a per-window origin
                    // gives every window its own arbitrary constant and the exported phase
                    // becomes random FROM WINDOW TO WINDOW. Measured after that change:
                    // coherence of a fixed (PRN,instance,channel) across 7 consecutive windows
                    // was 0.38 against a 1/sqrt(7) = 0.378 random baseline -- i.e. destroyed.
                    // KV: "we'd be adding flat but random phases time-to-time."
                    // The cross-INSTANCE concern that motivated it was real but is not fixed
                    // here; a continuous common reference is worth more than a tidy origin.
                    // KEEP the raw _phi[p]: continuous in time, and its INCREMENTS are common
                    // across instances (same commanded carrier, and since #53 the same
                    // records), which is what a cross-window combine needs.
                    // (task #52).
                    // `_phi[p]` is an ACCUMULATOR whose zero is set at a per-instance
                    // re-acquisition ("FRESH acquisition: break the arc"), so its absolute
                    // value is arbitrary and DIFFERENT on every instance. Exporting
                    // g * exp(-i*_phi) therefore stamped each instance's spectrum with its own
                    // arbitrary constant -- which left INTRA-instance coherence untouched
                    // (a constant does not tilt a ramp) while destroying INTER-instance
                    // coherence. Measured on sky 2026-08-12 before this fix: intra 0.48-0.98
                    // (0.86-0.98 on strong satellites) against inter 0.05-0.46. I first read
                    // that as a physical per-instance instrumental phase; it is not -- the
                    // instances are compute assignments over ONE PFB, interleaved stride-16
                    // combs over the SAME 5972-6076 span, with no separate signal path to
                    // carry a calibration term. KV caught it.
                    //
                    // The accumulator itself is NECESSARY: without it the prompt winds with the
                    // residual carrier across the ~100 records in a window (1 Hz of residual is
                    // 6.6 rad over 1.05 s) and the window sum decoheres. Only its ORIGIN is
                    // wrong. Computing an absolute phase instead is not an option either --
                    // f_car * t_abs at days of uptime is ~1e8 cycles, the same precision trap
                    // as the code-phase transport disease.
                    //
                    // So subtract the phase at the window's OWN first record. The INCREMENT
                    // from there is identical on every instance (same commanded carrier, and
                    // since #53 the same records), so the arbitrary origin cancels exactly and
                    // every instance lands on one common reference. The record path keeps the
                    // raw _phi[p] and is untouched.
                    // ROWS: trial 0 = EARLY, 1 = PROMPT, 2 = LATE (see prow above). One lambda
                    // so all three taps get IDENTICALLY the same element combine and the same
                    // NCO rotation -- a discriminator built from taps combined even slightly
                    // differently is measuring the difference between the combines, not the
                    // code offset.
                    const size_t erow = (size_t)(c.job0 + 0) * n_chan;
                    const size_t lrow = (size_t)(c.job0 + 2) * n_chan;
                    const bool warm = _elem_sum && _cal[p].warm();
                    // STEERED, EXACTLY AS THE HEADER IS (#102). The cal's weights are learned
                    // from the STEERED prompt above, so they carry the instrument phase only.
                    // Applied to the raw per-channel correlations they leave every element's
                    // geometric phase in the sum, and a combine spread over a dozen dishes then
                    // reads 5-20 dB below the header -- the per-channel mean of
                    // |sum_e w_e exp(i 2 pi f_ch tau_e)|^2, satellite by satellite -- while the
                    // unsteered noise probes do not, so the served C/N0 and the comb DLL's own
                    // taps both lose exactly that. The per-PRN learners hid it by putting nearly
                    // all their weight on the reference element; a shared model does not. The
                    // reference element is the phase centre, so the steered combine keeps the
                    // comb's code and phase currency. The beam cube below stays raw: per element
                    // is its axis.
                    const gnss::ElemSteer::cf* st_tab = steered ? _steer_buf.data() : nullptr;
                    auto tap = [&](size_t row, int ch) -> std::complex<double> {
                        const size_t b = (row + ch) * n_e;
                        std::complex<double> v;
                        if (warm) {
                            const gnss::ElemSteer::cf* st =
                                st_tab ? st_tab + (size_t)ch * n_e : nullptr;
                            for (int el = 0; el < n_e; ++el) {
                                std::complex<double> x(corr[2 * (b + el)], corr[2 * (b + el) + 1]);
                                if (st)
                                    x *= std::complex<double>(st[el].real(), st[el].imag());
                                _spec_scratch[el] = x;
                            }
                            v = _cal[p].combine(_spec_scratch.data());
                        } else {
                            v = {corr[2 * (b + ref_e)], corr[2 * (b + ref_e) + 1]};
                        }
                        return v * rot;
                    };
                    // ── BEAM CUBE (hpp): the (subband bin x element) axis, un-collapsed ───
                    // The joint quantity, taken from `corr` BEFORE either collapse: the
                    // element blocks below sum this over channels, the comb block sums it over
                    // elements, and neither keeps the product.
                    //
                    // BOTH SUMS ARE KEPT. The incoherent one is the beam (no phase model, no
                    // nav-bit cap, grows like sqrt(K)); the coherent one is the arc, and it is
                    // the one that cannot be reconstructed afterwards. `rot` is the SAME
                    // rotation the record's prompt and the spectrum ring just got -- one
                    // common phase per record -- so the accumulation does not wind with the
                    // residual carrier while the cross-element and cross-channel RELATIVE
                    // phases pass through untouched. Its origin, _phi[p], is published as
                    // phi0 below; without that the coherent sum is a number with an unknowable
                    // reference (see the hpp).
                    //
                    // Normalised by the channel's OWN replica energy, once, for every element:
                    // one replica is correlated against all of them, so A_e = G_e/E_c keeps the
                    // antenna ratios comparable -- the property the whole beam map rests on. A
                    // zero-energy channel contributes nothing rather than an infinity.
                    if (_cube_on) {
                        std::lock_guard<std::mutex> ck(_cube_mtx);
                        CubeWindow& C = cube_window_for(wstart);
                        // The SAME epoch the record's own `utc` was stamped from (above), so
                        // the cube and the record stream agree on what time a wstart is.
                        // 0.0 when the producer sent no frame0_utc AND the fallback is unset.
                        C.utc0 = (hdr.utc0 > 0.0) ? hdr.utc0 : _wall_anchor;
                        // PUBLISH THE PHASE CURRENCY at the window's first record, and count
                        // only UNFOLDED resets: reanchored 2 and 3 are the CONTINUOUS re-pins
                        // whose phase step has already been folded into _phi, so counting them
                        // would mark every window broken forever (measured: 100% of (PRN,
                        // window) on all four chains, which is how that counter was found to be
                        // the problem rather than the receiver). Only 1 is a fresh acquisition,
                        // where the NCO is reset and the arc genuinely breaks.
                        if (C.nrec[p] == 0)
                            C.phi0[p] = _phi[p];
                        if (c.reanchored == 1)
                            C.nreanchor[p] += 1;
                        C.nrec[p] += 1;
                        for (int ch = 0; ch < n_chan; ++ch) {
                            if (!((c.chan_mask >> ch) & 1ULL))
                                continue;
                            const double ec = energy[prow + ch];
                            if (!(ec > 0.0))
                                continue;
                            const int bin = (_cube_bin_width > 0) ? (ch / _cube_bin_width) : ch;
                            const size_t b = (prow + ch) * n_e;
                            const size_t k0 = ((size_t)p * _cube_bins + bin) * n_e;
                            const double inv = 1.0 / ec;
                            for (int el = 0; el < n_e; ++el) {
                                const std::complex<double> v =
                                    std::complex<double>(corr[2 * (b + el)] * inv,
                                                         corr[2 * (b + el) + 1] * inv)
                                    * rot;
                                C.coh_re[k0 + el] += v.real();
                                C.coh_im[k0 + el] += v.imag();
                                // |v|^2 == |A|^2: rot is unit modulus, so the incoherent sum is
                                // blind to the rotation, which is exactly why it survives a
                                // broken arc when the coherent sum does not.
                                C.incoh[k0 + el] += std::norm(v);
                            }
                            const size_t kb = (size_t)p * _cube_bins + bin;
                            C.w[kb] += 1.0;
                            C.energy[kb] += ec;
                        }
                    }
                    for (int ch = 0; ch < n_chan; ++ch) {
                        if (!((c.chan_mask >> ch) & 1ULL))
                            continue;
                        std::complex<double> g = tap(prow, ch);
                        const size_t k = (size_t)p * n_chan + ch;
                        W.re[k] += g.real();
                        W.im[k] += g.imag();
                        W.energy[k] += energy[prow + ch];
                        // THE SAME VALUE, PER RECORD (gnssRecord.hpp comb block). The window
                        // ring above is what /get_spectrum serves; this is the identical
                        // quantity written out un-accumulated, because a cross-RECORD rate fit
                        // cannot be done on a window sum -- the sum is exactly what a residual
                        // rate destroys.
                        if (_chan_export) {
                            float* cb = out + gnss::chan_offset(p, ch, n_prn, n_chan, _n_elements);
                            const std::complex<double> ge = tap(erow, ch);
                            const std::complex<double> gl = tap(lrow, ch);
                            cb[gnss::CHAN_RE] = (float)g.real();
                            cb[gnss::CHAN_IM] = (float)g.imag();
                            cb[gnss::CHAN_ENERGY] = (float)energy[prow + ch];
                            cb[gnss::CHAN_E_RE] = (float)ge.real();
                            cb[gnss::CHAN_E_IM] = (float)ge.imag();
                            cb[gnss::CHAN_E_ENERGY] = (float)energy[erow + ch];
                            cb[gnss::CHAN_L_RE] = (float)gl.real();
                            cb[gnss::CHAN_L_IM] = (float)gl.imag();
                            cb[gnss::CHAN_L_ENERGY] = (float)energy[lrow + ch];
                        }
                    }
                    W.nrec[p] += 1;
                    if (W.w0 < 0)
                        W.w0 = wstart;
                    W.w1 = wstart;
                }
                // Head segment: SAME NCO rotation as the prompt (it is the same correlation,
                // restricted to the hops before the code-period boundary), so head + tail
                // reconstructs P exactly and the combiner can wipe each side with its own
                // overlay chip.
                // SKY-PHASE-CORRECTED PROMPT: same NCO rotation as the prompt it is a version of,
                // so the combiner can deep-integrate it in exactly the prompt's currency. Zero
                // when the cal is cold / elem_sum is off, which consumers read as "absent".
                if (std::norm(g_sky) > 0.0) {
                    const std::complex<double> gs = g_sky * rot;
                    rec[gnss::REC_SKY_RE] = (float)gs.real();
                    rec[gnss::REC_SKY_IM] = (float)gs.imag();
                }
                const std::complex<double> gh_corr = g3[3] * rot;
                rec[gnss::REC_PH_RE] = (float)gh_corr.real();
                rec[gnss::REC_PH_IM] = (float)gh_corr.imag();
                rec[gnss::REC_PH_ENERGY] = (float)e3[3];
                // PEEL RESIDUAL (slots 20-23): the same prompt correlation taken on the voltage
                // after this PRN's own waveform was subtracted, and its head segment. SAME NCO
                // rotation as the prompt -- it is the same correlation, so the combiner can
                // deep-integrate it with the same per-segment overlay wipe. Left at zero when the
                // chain does not peel, which every existing consumer already reads as "absent".
                if (n_rows_spec > ROW_RES_PH) {
                    const std::complex<double> gr = g3[ROW_RES_P] * rot;
                    const std::complex<double> grh = g3[ROW_RES_PH] * rot;
                    rec[gnss::REC_RES_RE] = (float)gr.real();
                    rec[gnss::REC_RES_IM] = (float)gr.imag();
                    rec[gnss::REC_RES_PH_RE] = (float)grh.real();
                    rec[gnss::REC_RES_PH_IM] = (float)grh.imag();
                }
                // ELEMENT BLOCKS. Every antenna gets the SAME NCO rotation: the NCO is a per-PRN
                // model of the code/carrier, not a per-antenna quantity, so rotating each element
                // by `rot` preserves exactly the inter-element phase differences -- which are the
                // measurement. Rotating per element would divide out the very thing being mapped.
                if (_n_elements > 0) {
                    for (int el = 0; el < _n_elements; ++el) {
                        float* eb = out + gnss::elem_offset(p, el, _n_elements);
                        const std::complex<double> ge = _g_elem[(size_t)1 * n_e + el] * rot;
                        const std::complex<double> gE = _g_elem[(size_t)0 * n_e + el] * rot;
                        const std::complex<double> gL = _g_elem[(size_t)2 * n_e + el] * rot;
                        const std::complex<double> gH = _g_elem[(size_t)3 * n_e + el] * rot;
                        eb[gnss::ELEM_P_RE] = (float)ge.real();
                        eb[gnss::ELEM_P_IM] = (float)ge.imag();
                        eb[gnss::ELEM_E_RE] = (float)gE.real();
                        eb[gnss::ELEM_E_IM] = (float)gE.imag();
                        eb[gnss::ELEM_L_RE] = (float)gL.real();
                        eb[gnss::ELEM_L_IM] = (float)gL.imag();
                        eb[gnss::ELEM_PH_RE] = (float)gH.real();
                        eb[gnss::ELEM_PH_IM] = (float)gH.imag();
                        if (n_rows_spec > ROW_RES_PH) {
                            const std::complex<double> gr =
                                _g_elem[(size_t)ROW_RES_P * n_e + el] * rot;
                            const std::complex<double> grh =
                                _g_elem[(size_t)ROW_RES_PH * n_e + el] * rot;
                            eb[gnss::ELEM_RES_RE] = (float)gr.real();
                            eb[gnss::ELEM_RES_IM] = (float)gr.imag();
                            eb[gnss::ELEM_RES_PH_RE] = (float)grh.real();
                            eb[gnss::ELEM_RES_PH_IM] = (float)grh.imag();
                        }
                    }
                }

                _a_prev[p] = (e3[1] > 0.0) ? g_corr / e3[1] : std::complex<double>(0.0, 0.0);
                _a_prev_ok[p] = 1;
                _elem_prev_ok[p] = 1;
                _wstart_prev[p] = wstart;

                // COMMANDED CARRIER PHASE (cycles mod 1), the GPU twin of the CPU tracker's
                // export -- see gnssRecord.hpp: replica f_ref*t_abs + the NCO's phi. Adding
                // the combiner's measured arg(A) reconstructs the received carrier phase,
                // with the re-pin's replica-phase step cancelling instead of slipping.
                // phi enters NEGATED: f_ref is physical-signed, the NCO is in the r2c-flipped
                // internal convention (settled on sky by a satellite with its trim pinned at
                // the clamp).
                // (no _phi_fix term any more: the re-pin step is folded into _phi_cyc itself,
                // which leaves fcar*t_abs - phi_cyc continuous on its own.)
                const double t_abs_w = (double)wstart / _sample_rate;
                const double phi_cmd_cyc = c.fcar * t_abs_w - _phi_cyc[p];
                // the INCREMENT (gnssRecord.hpp): bounded, float-exact, unwrap-free
                rec[gnss::REC_CPHASE] =
                    _phi_cmd_ok[p] ? (float)(phi_cmd_cyc - _phi_cmd_prev[p]) : 0.0f;
                _phi_cmd_prev[p] = phi_cmd_cyc;
                _phi_cmd_ok[p] = 1;
                _fcar_prev[p] = c.fcar;
                _fcar_prev_ok[p] = 1;
            }

            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->sample_seq = wstart;
            }
            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out++;
        }
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in++;
    }
}

GnssGpuRecordAssemble::SpecWindow& GnssGpuRecordAssemble::spec_window_for(int64_t wstart) {
    // Caller holds _spec_mtx.
    if (_spec_win_samples <= 0)
        return _spec_ring[0]; // legacy single accumulator: one open window, reset on read
    // FLOOR division, not C's truncation-toward-zero. wstart should never be negative, but a
    // silent sign convention change upstream must not quietly put two instances on different
    // sides of zero -- that is precisely the class of bug this endpoint exists to remove.
    int64_t idx = wstart / _spec_win_samples;
    if (wstart < 0 && idx * _spec_win_samples != wstart)
        --idx;
    SpecWindow& W =
        _spec_ring[(size_t)(((idx % (int64_t)_spec_ring.size()) + (int64_t)_spec_ring.size())
                            % (int64_t)_spec_ring.size())];
    if (W.idx != idx) {
        // First record of this window (or the ring wrapped past the old occupant): clear.
        std::fill(W.re.begin(), W.re.end(), 0.0);
        std::fill(W.im.begin(), W.im.end(), 0.0);
        std::fill(W.energy.begin(), W.energy.end(), 0.0);
        std::fill(W.nrec.begin(), W.nrec.end(), 0);
        std::fill(W.phi0.begin(), W.phi0.end(), 0.0);
        std::fill(W.nreanchor.begin(), W.nreanchor.end(), 0);
        W.idx = idx;
        W.w0 = W.w1 = -1;
    }
    if (idx > _spec_max_idx)
        _spec_max_idx = idx;
    return W;
}

GnssGpuRecordAssemble::CubeWindow& GnssGpuRecordAssemble::cube_window_for(int64_t wstart) {
    // Caller holds _cube_mtx. Same window arithmetic as spec_window_for, deliberately: the two
    // rings must agree on where a boundary is, or a consumer joining the cube to the spectrum
    // would be pairing different slices of time and nothing would say so.
    int64_t idx = wstart / _cube_win_samples;
    if (wstart < 0 && idx * _cube_win_samples != wstart)
        --idx; // FLOOR, not C's truncation toward zero (see spec_window_for)
    const int64_t n = (int64_t)_cube_ring.size();
    CubeWindow& C = _cube_ring[(size_t)(((idx % n) + n) % n)];
    if (C.idx != idx) {
        // First record of this window, or the ring wrapped past the old occupant: clear it.
        std::fill(C.coh_re.begin(), C.coh_re.end(), 0.0);
        std::fill(C.coh_im.begin(), C.coh_im.end(), 0.0);
        std::fill(C.incoh.begin(), C.incoh.end(), 0.0);
        std::fill(C.w.begin(), C.w.end(), 0.0);
        std::fill(C.energy.begin(), C.energy.end(), 0.0);
        std::fill(C.nrec.begin(), C.nrec.end(), 0);
        std::fill(C.phi0.begin(), C.phi0.end(), 0.0);
        std::fill(C.nreanchor.begin(), C.nreanchor.end(), 0);
        C.nrec_seen = 0;
        C.idx = idx;
        C.w0 = C.w1 = -1;
        C.utc0 = 0.0;
    }
    if (C.w0 < 0)
        C.w0 = wstart;
    C.w1 = wstart;
    // ⚠️ A WINDOW SHORTER THAN A RECORD IS A UNIT ERROR, AND IT IS INVISIBLE OTHERWISE. It
    // produces one window per record, each holding exactly one, so every array is well formed,
    // every index is consistent, and the archive is simply the record stream at 1/96 of the
    // integration it claims. The tell is structural rather than numeric -- a window that never
    // accumulates a second record -- so check it here, once, where consecutive wstarts are the
    // thing in hand. (beam_cube_window_samples is in F-ENGINE SAMPLES; a value in HOPS is off
    // by fft_length and lands exactly here.)
    if (!_cube_win_warned && C.nrec_seen == 0 && idx == _cube_max_idx + 1 && _cube_max_idx >= 0) {
        if (++_cube_singleton_windows >= 8) {
            _cube_win_warned = true;
            WARN("GnssGpuRecordAssemble[{:s}]: beam_cube_window_samples {:d} gives ONE RECORD "
                 "PER WINDOW ({:.6f} s) -- 8 in a row. That is what a window length given in "
                 "HOPS rather than F-engine SAMPLES looks like (they differ by fft_length); "
                 "the cube is being recorded at the record cadence, not the window cadence it "
                 "reports. Expected many records per window.",
                 unique_name, (long long)_cube_win_samples,
                 (double)_cube_win_samples / _sample_rate);
        }
    } else if (C.nrec_seen > 0) {
        _cube_singleton_windows = 0;
    }
    ++C.nrec_seen;
    if (idx > _cube_max_idx) {
        // The PREVIOUS window is now provably complete -- a later one has opened, which is the
        // only local evidence that no more records are coming for it. Push it. Done here, on
        // the boundary crossing, so completeness is a property of the clock and not of any
        // consumer's timing.
        if (_cube_out_buf != nullptr && _cube_max_idx >= 0) {
            const int64_t n2 = (int64_t)_cube_ring.size();
            const CubeWindow& prev = _cube_ring[(size_t)(((_cube_max_idx % n2) + n2) % n2)];
            if (prev.idx == _cube_max_idx)
                emit_cube_window(prev);
        }
        _cube_max_idx = idx;
    }
    return C;
}

void GnssGpuRecordAssemble::emit_cube_window(const CubeWindow& C) {
    // Caller holds _cube_mtx.
    //
    // NON-BLOCKING ACQUIRE (hpp): this stage is in the real-time path, so a full buffer must
    // cost the ARCHIVE a window, never cost the tracker a record. is_frame_empty() is a safe
    // pre-check here because this stage is the buffer's only producer -- nothing else can take
    // the slot between the test and the call.
    if (!_cube_out_buf->is_frame_empty(_cube_out_id)) {
        ++_cube_dropped;
        // First drop, then every hundredth: a wedged far side would otherwise print at ~1 Hz
        // per instance forever and drown the log -- and the count rides in the data anyway,
        // so the log is a courtesy here, not the record.
        if (_cube_dropped == 1 || _cube_dropped % 100 == 0)
            WARN("beam-cube window {:d} DROPPED, output buffer full ({:d} lost since start). "
                 "The next emitted frame's dropped_windows carries this, so the gap is SIZED "
                 "in the archive rather than inferred from a hole in the index.",
                 (long long)C.idx, (long long)_cube_dropped);
        return;
    }
    uint8_t* f = _cube_out_buf->wait_for_empty_frame(unique_name, _cube_out_id);
    if (f == nullptr)
        return; // shutting down
    const int mp = _cube_max_prn, mb = _cube_max_bins, ne = _n_elements;
    std::memset(f, 0, gnss::cube_frame_bytes(mp, mb, ne)); // pad is ZERO, not stale bytes

    auto i64 = [&](size_t off, int64_t v) { std::memcpy(f + off, &v, sizeof(v)); };
    auto f64 = [&](size_t off, double v) { std::memcpy(f + off, &v, sizeof(v)); };
    auto i32 = [&](size_t off, int32_t v) { std::memcpy(f + off, &v, sizeof(v)); };
    i64(0, gnss::CUBE_VERSION);
    i64(8, C.idx);
    i64(16, C.w0);
    i64(24, C.w1);
    i64(32, _cube_dropped);
    i64(40, (int64_t)_prns.size());
    i64(48, (int64_t)_cube_bins);
    i64(56, (int64_t)ne);
    f64(64, (double)_cube_win_samples);
    f64(72, _sample_rate);
    i32(80, _cube_gpu);
    i32(84, _cube_bin_width ? _cube_bin_width : 1);
    // THE MAXIMA THIS FRAME WAS SIZED FOR. n_prn/n_bin above are the actual extents; the arrays
    // below are laid out by these, so they are what makes the frame self-delimiting on disk.
    i32(88, mp);
    i32(92, mb);
    // v3: the epoch of sample 0, so the archive owns its own time axis (gnssRecord.hpp). The
    // window's UTC is utc0 + w0 / sample_rate; a reader that has only this frame can date it.
    f64(gnss::CUBE_UTC0_OFFSET, C.utc0);
    std::strncpy((char*)f + gnss::CUBE_CHAIN_OFFSET, _cube_chain.c_str(),
                 gnss::CUBE_CHAIN_CHARS - 1);

    size_t off = gnss::CUBE_HEADER_BYTES;
    // Doubles FIRST -- see gnssRecord.hpp: their alignment must not depend on max_bins/max_prn.
    double* phi0 = (double*)(f + off);
    off += (size_t)mp * sizeof(double);
    int32_t* fid_lo = (int32_t*)(f + off);
    off += (size_t)mb * sizeof(int32_t);
    int32_t* fid_hi = (int32_t*)(f + off);
    off += (size_t)mb * sizeof(int32_t);
    int32_t* prn = (int32_t*)(f + off);
    off += (size_t)mp * sizeof(int32_t);
    int32_t* nrec = (int32_t*)(f + off);
    off += (size_t)mp * sizeof(int32_t);
    int32_t* nre = (int32_t*)(f + off);
    off += (size_t)mp * sizeof(int32_t);
    float* w = (float*)(f + off);
    off += (size_t)mp * mb * sizeof(float);
    float* en = (float*)(f + off);
    off += (size_t)mp * mb * sizeof(float);
    float* coh_re = (float*)(f + off);
    off += (size_t)mp * mb * ne * sizeof(float);
    float* coh_im = (float*)(f + off);
    off += (size_t)mp * mb * ne * sizeof(float);
    float* incoh = (float*)(f + off);

    const int nc = (int)_spec_freq_ids.size();
    for (int b = 0; b < _cube_bins && b < mb; ++b) {
        const int c0 = _cube_bin_width ? b * _cube_bin_width : b;
        const int c1 = _cube_bin_width ? std::min(nc, (b + 1) * _cube_bin_width) - 1 : b;
        fid_lo[b] = _spec_freq_ids[c0];
        fid_hi[b] = _spec_freq_ids[c1];
    }
    for (size_t p = 0; p < _prns.size() && (int)p < mp; ++p) {
        // A slot with no records this window is left at prn = 0 -- SILENCE, not a row of zeros.
        // Zeros would read downstream as a real measurement of no power, which is the one
        // reading that must never be manufactured by a transport.
        if (C.nrec[p] <= 0)
            continue;
        prn[p] = _prns[p];
        nrec[p] = C.nrec[p];
        nre[p] = C.nreanchor[p];
        phi0[p] = C.phi0[p];
        for (int b = 0; b < _cube_bins && b < mb; ++b) {
            const size_t kb = p * _cube_bins + b;
            w[p * mb + b] = (float)C.w[kb];
            en[p * mb + b] = (float)C.energy[kb];
            const size_t src = ((size_t)p * _cube_bins + b) * ne;
            const size_t dst = ((size_t)p * mb + b) * ne;
            for (int el = 0; el < ne; ++el) {
                coh_re[dst + el] = (float)C.coh_re[src + el];
                coh_im[dst + el] = (float)C.coh_im[src + el];
                incoh[dst + el] = (float)C.incoh[src + el];
            }
        }
    }
    // ⚠️⚠️ THE METADATA OBJECT IS NOT OPTIONAL ON A FRAME THAT FEEDS bufferSend (#110,
    // 2026-09-05). This leg shipped without these four lines and every node in the fleet
    // SEGFAULTED 45-75 s after start: bufferSend does buf->get_metadata(id)->get_serialized_size()
    // on every frame it sends, get_metadata() returns an EMPTY shared_ptr for a frame whose
    // producer never allocated one, and that is a null dereference in the sender thread -- taken
    // on the FIRST window emitted, which is why the archiver saw every connection and zero frames,
    // and why the delay was the broker's seeding time (no window opens before a PRN is despread).
    // Same stamp as the record leg above and GnssTelemPack: the window's first sample, so a
    // kotekan consumer downstream of bufferRecv can address the frame without parsing it.
    if (_cube_out_buf->metadata_pool) {
        _cube_out_buf->allocate_new_metadata_object(_cube_out_id);
        if (auto* m = get_gnss_chan_metadata(_cube_out_buf, _cube_out_id))
            m->sample_seq = C.w0;
    }
    _cube_out_buf->mark_frame_full(unique_name, _cube_out_id);
    _cube_out_id = (_cube_out_id + 1) % _cube_out_buf->num_frames;
}


// THE BEAM CUBE endpoint (hpp): the (subband x element) axis, coherent AND incoherent.
//
// ADDRESSABLE AND IDEMPOTENT, like /get_spectrum and for the same reason: `?window=N` returns
// exactly window N, no argument returns the newest COMPLETE one, nothing is reset. An earlier
// draft of this endpoint was reset-on-read, which is defensible only while the payload carries
// no phase -- the moment the coherent sum was added it stopped being defensible, because two
// consumers would then split a coherent integration between them and each would see a sum over
// a random subset of the records with no way to know it.
//
// EMITTED RAW: no debias, no range normalisation, no combine, no dB. All of them are the
// reader's job. Debias in the POWER domain from the probe PRNs, per (instance, element,
// subband, time bin), never medianed across elements.
void GnssGpuRecordAssemble::beam_cube_callback(kotekan::connectionInstance& conn) {
    if (!_cube_on) {
        conn.send_error("beam cube is OFF for this stage (needs 'beam_cube: true' plus both "
                        "axes: 'channel_ids' for frequency and 'n_elements' for element)",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    bool want_specific = false;
    int64_t want = -1;
    for (const auto& kv : conn.get_query()) {
        if (kv.first != "window")
            continue;
        want_specific = true;
        try {
            want = std::stoll(kv.second);
        } catch (const std::exception&) {
            conn.send_error("get_beam_cube: 'window' must be an integer window index",
                            kotekan::HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    }

    nlohmann::json reply;
    const int ne = _n_elements;
    std::vector<double> coh_re, coh_im, incoh, w, energy, phi0;
    std::vector<int> nrec, nre;
    int64_t idx = -1, w0 = -1, w1 = -1, lo = -1, hi = -1;
    const char* status = "ok";
    {
        std::lock_guard<std::mutex> ck(_cube_mtx);
        // A window is COMPLETE once a LATER one has been opened -- the only local evidence
        // that no more records are coming for it; there is no end-of-window event to wait on.
        hi = _cube_max_idx - 1;
        lo = _cube_max_idx - (int64_t)_cube_ring.size() + 1;
        if (lo < 0)
            lo = 0;
        idx = want_specific ? want : hi;
        if (_cube_max_idx < 0)
            status = "not_yet";
        else if (idx > hi)
            status = "not_yet"; // NORMAL: instances lag each other; simply re-ask this one
        else if (idx < lo)
            status = "too_old"; // those records are gone -- the caller must JUMP, and say so
        else {
            const int64_t n = (int64_t)_cube_ring.size();
            const CubeWindow& C = _cube_ring[(size_t)(((idx % n) + n) % n)];
            if (C.idx != idx)
                status = "too_old";
            else {
                coh_re = C.coh_re;
                coh_im = C.coh_im;
                incoh = C.incoh;
                w = C.w;
                energy = C.energy;
                phi0 = C.phi0;
                nrec = C.nrec;
                nre = C.nreanchor;
                w0 = C.w0;
                w1 = C.w1;
            }
        }
    }

    const int nc = (int)_spec_freq_ids.size();
    reply["n_bin"] = _cube_bins;
    reply["n_elem"] = ne;
    reply["n_chan"] = nc;
    reply["bin_width"] = _cube_bin_width ? _cube_bin_width : 1;
    reply["freq_ids"] = _spec_freq_ids;
    reply["window"] = idx;
    reply["window_samples"] = (double)_cube_win_samples;
    reply["sample_rate"] = _sample_rate;
    reply["available"] = nlohmann::json::array({lo, hi});
    reply["status"] = status;
    reply["wstart0"] = w0; // the cross-instance alignment key, same clock as fleet hops
    reply["wstart1"] = w1;
    reply["reference_element"] = _reference_element;
    // Per-bin freq_id span, so a reader never has to reproduce the binning arithmetic -- the
    // one place a resolution change could silently re-point every archived row.
    nlohmann::json bins = nlohmann::json::array();
    for (int b = 0; b < _cube_bins; ++b) {
        const int c0 = _cube_bin_width ? b * _cube_bin_width : b;
        const int c1 = _cube_bin_width ? std::min(nc, (b + 1) * _cube_bin_width) - 1 : b;
        bins.push_back({_spec_freq_ids[c0], _spec_freq_ids[c1]});
    }
    reply["bin_freq_ids"] = bins;
    if (std::string(status) != "ok") {
        reply["prns"] = nlohmann::json::array();
        conn.send_json_reply(reply);
        return;
    }

    nlohmann::json prns = nlohmann::json::array();
    for (size_t p = 0; p < _prns.size(); ++p) {
        if (nrec[p] <= 0)
            continue; // silence, not zeros -- the get_records/get_spectrum convention
        nlohmann::json wb = nlohmann::json::array(), eb = nlohmann::json::array();
        nlohmann::json cb = nlohmann::json::array(), ib = nlohmann::json::array();
        for (int b = 0; b < _cube_bins; ++b) {
            const size_t kb = p * _cube_bins + b;
            wb.push_back(w[kb]);
            eb.push_back(energy[kb]);
            nlohmann::json cr = nlohmann::json::array(), ir = nlohmann::json::array();
            const size_t k0 = ((size_t)p * _cube_bins + b) * ne;
            for (int el = 0; el < ne; ++el) {
                cr.push_back({coh_re[k0 + el], coh_im[k0 + el]});
                ir.push_back(incoh[k0 + el]);
            }
            cb.push_back(cr);
            ib.push_back(ir);
        }
        // coh[bin][elem] = [re, im] and incoh[bin][elem], with w[bin] the term count: the SUMS
        // and their weight, so windows, instances and days all combine by addition and nothing
        // has to be reprocessed to add a night. phi0 is the coherent sum's phase reference and
        // n_reanchor > 0 means the arc broke INSIDE this window -- no constant can undo that,
        // so drop those rather than rotating them onto anything.
        prns.push_back({{"prn", _prns[p]},
                        {"n_rec", nrec[p]},
                        {"phi0", phi0[p]},
                        {"n_reanchor", nre[p]},
                        {"w", wb},
                        {"energy", eb},
                        {"coh", cb},
                        {"incoh", ib}});
    }
    reply["prns"] = prns;
    conn.send_json_reply(reply);
}


void GnssGpuRecordAssemble::spectrum_callback(kotekan::connectionInstance& conn) {
    // ADDRESSABLE, IDEMPOTENT READ (task #53). `?window=N` returns exactly window N; no
    // argument returns the newest COMPLETE one. Nothing is reset, so polling is free of side
    // effects and a second consumer no longer steals anyone's data.
    //
    // EVERY reply -- success or refusal -- carries available:[lo,hi], because that is what
    // lets the broker resynchronise without guessing. `too_old` means those records are gone
    // and the broker must JUMP (and log how many windows it skipped, so falling behind is
    // visible rather than a silent permanent resync); `not_yet` is NORMAL, not an error --
    // instances lag each other by a few records and the caller should simply re-ask that
    // instance, which now works because the ring still holds the window when it completes.
    if (_spec_freq_ids.empty()) {
        conn.send_error("per-channel spectrum export is OFF: this stage's config has no "
                        "'channel_ids' (regenerate the node config with a "
                        "gen_chord_gnss_config.py that wires it -- task #32)",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    const size_t nc = _spec_freq_ids.size();
    const bool legacy = (_spec_win_samples <= 0);

    // Requested index: absent -> newest COMPLETE window. "Complete" means a LATER window has
    // been opened, which is the only evidence available locally that no more records are
    // coming for this one -- there is no end-of-window event to wait on.
    bool want_specific = false;
    int64_t want = -1;
    for (const auto& kv : conn.get_query()) {
        if (kv.first != "window")
            continue;
        want_specific = true;
        try {
            want = std::stoll(kv.second);
        } catch (const std::exception&) {
            conn.send_error("get_spectrum: 'window' must be an integer window index",
                            kotekan::HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    }

    std::vector<double> re, im, en, phi0;
    std::vector<int> nrec, nre;
    int64_t w0 = -1, w1 = -1, idx = -1, lo = -1, hi = -1;
    const char* status = "ok";
    {
        std::lock_guard<std::mutex> lk(_spec_mtx);
        if (legacy) {
            // Legacy path keeps the old reset-on-read semantics exactly, so a not-yet-
            // regenerated node behaves as it always did rather than half-changing.
            SpecWindow& W = _spec_ring[0];
            re.swap(W.re);
            im.swap(W.im);
            en.swap(W.energy);
            nrec.swap(W.nrec);
            phi0 = W.phi0;
            nre = W.nreanchor;
            w0 = W.w0;
            w1 = W.w1;
            W.re.assign((size_t)_prns.size() * nc, 0.0);
            W.im.assign((size_t)_prns.size() * nc, 0.0);
            W.energy.assign((size_t)_prns.size() * nc, 0.0);
            W.nrec.assign(_prns.size(), 0);
            W.w0 = W.w1 = -1;
        } else {
            hi = _spec_max_idx - 1; // newest COMPLETE
            for (const auto& W : _spec_ring)
                if (W.idx >= 0 && W.idx <= hi && (lo < 0 || W.idx < lo))
                    lo = W.idx;
            idx = want_specific ? want : hi;
            if (_spec_max_idx < 0 || hi < 0 || lo < 0) {
                status = "not_yet"; // nothing has completed since start-up
            } else if (idx > hi) {
                status = "not_yet";
            } else if (idx < lo) {
                status = "too_old";
            } else {
                const SpecWindow& W = _spec_ring[(
                    size_t)(((idx % (int64_t)_spec_ring.size()) + (int64_t)_spec_ring.size())
                            % (int64_t)_spec_ring.size())];
                if (W.idx != idx) {
                    // In range but the slot holds someone else: a gap (no records landed in
                    // that window at all). Report it as too_old rather than inventing zeros --
                    // "I never had it" and "here is an empty window" are different facts.
                    status = "too_old";
                } else {
                    re = W.re;
                    im = W.im;
                    en = W.energy;
                    nrec = W.nrec;
                    phi0 = W.phi0;
                    nre = W.nreanchor;
                    w0 = W.w0;
                    w1 = W.w1;
                }
            }
        }
    }

    nlohmann::json reply;
    reply["freq_ids"] = _spec_freq_ids;
    reply["sample_rate"] = _sample_rate;
    reply["status"] = status;
    reply["addressable"] = !legacy; // false -> this node predates #53; do NOT trust alignment
    reply["window_samples"] = (double)_spec_win_samples;
    reply["window"] = idx;
    reply["available"] = nlohmann::json::array({lo, hi});
    reply["wstart0"] = w0; // absolute sample of the first / last record in this window --
    reply["wstart1"] = w1; // the cross-instance alignment key, same clock as fleet hops
    if (std::string(status) != "ok") {
        reply["prns"] = nlohmann::json::array();
        conn.send_json_reply(reply);
        return;
    }
    nlohmann::json prns = nlohmann::json::array();
    for (size_t p = 0; p < _prns.size(); ++p) {
        if (nrec[p] <= 0)
            continue; // silence, not zeros (get_records convention)
        nlohmann::json chans = nlohmann::json::array();
        for (size_t ch = 0; ch < nc; ++ch) {
            const size_t k = p * nc + ch;
            // Energy-normalized like get_records' re/im: A = G/E is directly comparable
            // across instances, and E rides along as the ML combining weight.
            if (en[k] > 0.0)
                chans.push_back({re[k] / en[k], im[k] / en[k], en[k]});
            else
                chans.push_back({0.0, 0.0, 0.0}); // masked-off channel: weight 0, skipped
        }
        prns.push_back({{"prn", _prns[p]},
                        {"n_rec", nrec[p]},
                        {"phi0", phi0.empty() ? 0.0 : phi0[p]},
                        {"n_reanchor", nre.empty() ? 0 : nre[p]},
                        {"chan", chans}});
    }
    reply["prns"] = prns;
    conn.send_json_reply(reply);
}


// PATH B endpoint: inject a per-element complex gain prior into every PRN's ElemCal so the
// coherent combine starts warm on (re)acquisition. Parse FIRST, lock LAST: main_thread takes
// _gain_mtx only for the swap, never across a parse.
void GnssGpuRecordAssemble::set_elem_gain_callback(kotekan::connectionInstance& conn,
                                                   nlohmann::json& request) {
    if (!_elem_sum) {
        conn.send_error("elem_sum is OFF on this stage: no per-element cal to seed",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    std::vector<std::complex<double>> gains;
    try {
        auto& arr = request.at("gain"); // [[re,im], ...], length n_elements; [] or all-zero clears
        if (!arr.is_array())
            throw std::runtime_error("'gain' must be an array of [re, im] pairs");
        gains.reserve(arr.size());
        for (auto& e : arr) {
            if (!e.is_array() || e.size() != 2)
                throw std::runtime_error("each gain must be [re, im]");
            gains.emplace_back(e[0].get<double>(), e[1].get<double>());
        }
    } catch (const std::exception& ex) {
        conn.send_error(std::string("set_elem_gain: ") + ex.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    if (!gains.empty() && (int)gains.size() != _n_elements) {
        conn.send_error("set_elem_gain: gain length != n_elements",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    if (gains.empty()) // explicit "clear the prior": a full-length zero vector -> seed() clears
        gains.assign((size_t)_n_elements, std::complex<double>(0.0, 0.0));
    {
        std::lock_guard<std::mutex> lk(_gain_mtx);
        _pending_gain.swap(gains);
        _pending_gain_set = true;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssGpuRecordAssemble::set_elem_sum_adapt_callback(kotekan::connectionInstance& conn,
                                                        nlohmann::json& request) {
    // {"adapt": true|false}. A plain flag read by the per-record loop: no lock, no frame
    // boundary needed -- a record combined with the old value is as valid as the next.
    bool adapt;
    try {
        adapt = request.at("adapt").get<bool>();
    } catch (const std::exception& ex) {
        conn.send_error(std::string("set_elem_sum_adapt: expected {\"adapt\": bool}: ") + ex.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    const bool was = _elem_adapt.exchange(adapt);
    if (was != adapt)
        WARN("elem_sum_adapt {:s} -> {:s}: per-PRN element weights now {:s}",
             was ? "true" : "false", adapt ? "true" : "false",
             adapt ? "LEARNING from each PRN's own data" : "HELD (shadow cal keeps learning)");
    conn.send_json_reply(nlohmann::json{{"adapt", adapt}, {"was", was}});
}

void GnssGpuRecordAssemble::set_elem_sum_shared_callback(kotekan::connectionInstance& conn,
                                                         nlohmann::json& request) {
    // {"shared": true|false}. On -> the per-PRN weights are rebuilt from the shared model
    // every record and the learners are held (adapt forced false). Off -> back to held
    // per-PRN weights (whatever the model last installed); re-enable adapt separately.
    bool shared;
    try {
        shared = request.at("shared").get<bool>();
    } catch (const std::exception& ex) {
        conn.send_error(std::string("set_elem_sum_shared: expected {\"shared\": bool}: ")
                            + ex.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    const bool was = _elem_shared.exchange(shared);
    bool adapt_was = _elem_adapt.load();
    if (shared)
        adapt_was = _elem_adapt.exchange(false);
    if (was != shared)
        WARN("elem_sum_shared {:s} -> {:s}{:s}", was ? "true" : "false", shared ? "true" : "false",
             (shared && adapt_was) ? " (adapt forced false)" : "");
    conn.send_json_reply(
        nlohmann::json{{"shared", shared}, {"was", was}, {"adapt", _elem_adapt.load()}});
}

namespace {
/// The steady clock in seconds -- the same clock every freshness decision here uses.
double steady_now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

void GnssGpuRecordAssemble::shared_reset_prn(size_t p) {
    if (p >= _pol_num.size())
        return;
    _pol_num[p] = std::complex<double>(0.0, 0.0);
    _pol_den[p] = 0.0;
    _pol_warmth[p] = 0.0;
    _pol_c[p] = std::complex<double>(0.0, 0.0);
}

void GnssGpuRecordAssemble::shared_consensus(double now_s) {
    // One instrument per pol from the warm shadow cals: each PRN's per-pol weight vector is
    // normalised to sum|w| = 1 (its overall MRC scale is the satellite's strength, not the
    // instrument), rotated onto the current model (its overall phase is the satellite's), and
    // averaged with EQUAL weight per satellite. A slow EMA then follows the instrument; the
    // first model is one satellite alone (a blind average of unaligned vectors could cancel).
    // Re-anchored so the reference element's pol-0 weight is real positive (the header's
    // phase convention) and the pol-1 anchor element likewise (a convention only: the per-PRN
    // coefficient carries pol-1's phase).
    // ⚠️ THE REFERENCE ELEMENT'S MAGNITUDE IS CAPPED at the median of the other live elements'
    // BEFORE normalising, and satellites are NOT weighted by their MRC weight sum. A cal
    // whose reference weight has degenerated (its variance collapses when the leave-one-out
    // sum correlates strongly; measured thousands against ~10) is that satellite's own
    // problem in the per-PRN sum, but weighted by its (huge) weight sum and normalised it is
    // a unit vector on the reference element that outvotes every other satellite: the model
    // became the bare reference element and every satellite lost the array gain.
    using cd = std::complex<double>;
    const int n = _n_elements;
    if (n <= 0 || _g_shared.size() != (size_t)n)
        return;
    const double dt = now_s - _g_shared_t;
    _g_shared_t = now_s;
    fleet_ref_apply_pending();
    // Transit: the model in force stays exactly as it was, and none is FORMED from captured
    // learners either (before a model exists the PRNs ride their own, as always).
    if (shared_frozen(now_s))
        return;
    const int h = (n % 2 == 0) ? n / 2 : n; // an odd count is one pol
    const int n_pol = (h < n) ? 2 : 1;
    std::vector<cd> acc((size_t)n, cd(0.0, 0.0));
    std::vector<cd> v((size_t)n, cd(0.0, 0.0));
    std::vector<double> mags;
    double A = 0.0;
    int cnt = 0;
    // Seed choice: the warm shadow with the most live elements (a degenerate cal has few).
    size_t q_best = 0;
    int live_best = -1;
    for (size_t q = 0; q < _cal_shadow.size(); ++q) {
        if (!_cal_shadow[q].warm())
            continue;
        int live = 0;
        for (const auto& w : _cal_shadow[q].weights())
            live += (std::abs(w) > 0.0);
        if (live > live_best) {
            live_best = live;
            q_best = q;
        }
    }
    if (live_best <= 1)
        return;
    for (size_t q = 0; q < _cal_shadow.size(); ++q) {
        const auto& sh = _cal_shadow[q];
        if (!sh.warm() || (!_g_shared_warm && q != q_best))
            continue;
        const auto& w = sh.weights();
        bool any = false;
        for (int pol = 0; pol < n_pol; ++pol) {
            const int e0 = pol * h, e1 = std::min(n, e0 + h);
            const int anchor = std::min(e1 - 1, e0 + _reference_element);
            // Cap the anchor element at the median live magnitude of the others (the same
            // convention rebuild_weights uses when it detects the degeneracy).
            mags.clear();
            for (int e = e0; e < e1; ++e)
                if (e != anchor && std::abs(w[(size_t)e]) > 0.0)
                    mags.push_back(std::abs(w[(size_t)e]));
            if (mags.size() < 2)
                continue; // one live element besides the anchor is not an instrument
            std::sort(mags.begin(), mags.end());
            const double cap = mags[mags.size() / 2];
            for (int e = e0; e < e1; ++e)
                v[(size_t)e] = w[(size_t)e];
            if (std::abs(v[(size_t)anchor]) > cap)
                v[(size_t)anchor] *= cap / std::abs(v[(size_t)anchor]);
            double s = 0.0;
            cd x(0.0, 0.0);
            for (int e = e0; e < e1; ++e) {
                s += std::abs(v[(size_t)e]);
                if (_g_shared_warm)
                    x += std::conj(_g_shared[(size_t)e]) * v[(size_t)e];
            }
            if (s <= 0.0)
                continue;
            const cd rot = (std::abs(x) > 0.0) ? std::conj(x) / std::abs(x) : cd(1.0, 0.0);
            for (int e = e0; e < e1; ++e)
                acc[(size_t)e] += v[(size_t)e] * rot / s;
            any = true;
        }
        if (any) {
            A += 1.0;
            ++cnt;
        }
    }
    if (cnt == 0 || A <= 0.0)
        return;
    const double beta =
        _g_shared_warm ? std::min(0.25, 1.0 - std::exp(-std::max(dt, 0.0) / _elem_shared_tau_s))
                       : 1.0;
    for (int e = 0; e < n; ++e)
        _g_shared[(size_t)e] += beta * (acc[(size_t)e] / A - _g_shared[(size_t)e]);
    // #154: pinned to the FLEET reference where it applies (live mode, F describes this half)
    // -- in full for a first model, slewed for a warm one -- else to the instance's own pin.
    const bool fleet_live = _fleet_mode.load() == 2 && !_g_fleet_ref.empty();
    bool fleet_used[2] = {false, false};
    for (int pol = 0; pol < n_pol; ++pol) {
        const int e0 = pol * h, e1 = std::min(n, e0 + h);
        const int anchor = std::min(e1 - 1, e0 + _reference_element);
        cd pin(1.0, 0.0);
        if (fleet_live) {
            const auto o = gnss::ref_offset(_g_fleet_ref.data(), _g_shared.data(), e0, e1);
            if (o.ok && o.sim >= _fleet_min_sim) {
                pin = gnss::pin_rotation(o, _g_shared_warm,
                                         _fleet_slew_rad_s * std::min(std::max(dt, 0.0), 2.0));
                fleet_used[pol] = true;
            }
        }
        if (!fleet_used[pol]) {
            cd y(0.0, 0.0);
            if (_g_pin_ref_ok)
                for (int e = e0; e < e1; ++e)
                    y += std::conj(_g_pin_ref[(size_t)e]) * _g_shared[(size_t)e];
            if (std::abs(y) > 0.0)
                pin = std::conj(y) / std::abs(y); // <ref, G*pin> real positive, whole pol
            else if (std::abs(_g_shared[(size_t)anchor]) > 0.0)
                pin = std::conj(_g_shared[(size_t)anchor]) / std::abs(_g_shared[(size_t)anchor]);
        }
        double s = 0.0;
        for (int e = e0; e < e1; ++e)
            s += std::abs(_g_shared[(size_t)e]);
        if (s <= 0.0)
            continue;
        double gmax = 0.0;
        for (int e = e0; e < e1; ++e) {
            _g_shared[(size_t)e] = _g_shared[(size_t)e] * pin / s;
            gmax = std::max(gmax, std::abs(_g_shared[(size_t)e]));
        }
        // A model with half its weight on one element is not an instrument (a healthy array
        // spreads it over a dozen); refuse it and let every PRN ride its own learner until
        // the consensus is sane again. Logged once per collapse.
        if (gmax > 0.5) {
            if (_g_shared_warm || !_g_shared_collapsed)
                WARN("elem_sum_shared: pol-{:d} model collapsed onto one element ({:.0f}% of "
                     "the weight) -- not installed, PRNs ride their own learners",
                     pol, 100.0 * gmax);
            _g_shared_collapsed = 1;
            _g_shared_warm = false;
            _g_shared_n = 0;
            _g_pin_ref_ok = false;
            return;
        }
    }
    _g_shared_collapsed = 0;
    _g_shared_warm = true;
    _g_shared_n = cnt;
    if (!_g_pin_ref_ok) {
        _g_pin_ref = _g_shared;
        _g_pin_ref_ok = true;
    }
    // The own pin follows every half the fleet reference moved, so that a fall back to it (F
    // cleared, mode changed, a shape F no longer describes) holds the phase the model HAS.
    for (int pol = 0; pol < n_pol; ++pol)
        if (fleet_used[pol])
            for (int e = pol * h, e1 = std::min(n, pol * h + h); e < e1; ++e)
                _g_pin_ref[(size_t)e] = _g_shared[(size_t)e];
    if (_fleet_mode.load() != 0 && !_g_fleet_ref.empty()) {
        const double slew_deg = _fleet_slew_rad_s * 180.0 / M_PI;
        for (int pol = 0; pol < n_pol && pol < 2; ++pol) {
            const int e0 = pol * h, e1 = std::min(n, e0 + h);
            const auto o = gnss::ref_offset(_g_fleet_ref.data(), _g_shared.data(), e0, e1);
            const double err = o.ok ? o.err_rad * 180.0 / M_PI : 0.0;
            _fleet_err_deg[pol] = err;
            _fleet_sim[pol] = o.ok ? o.sim : 0.0;
            _fleet_applied[pol] = fleet_used[pol];
            if (fleet_used[pol] && std::abs(err) > 5.0 && !_fleet_slewing[pol]) {
                _fleet_slewing[pol] = true;
                WARN("elem_sum_shared: pol-{:d} model {:+.0f} deg off the fleet reference (sim "
                     "{:.2f}) -- slewing at {:.2f} deg/s, ~{:.0f} s",
                     pol, err, o.sim, slew_deg, slew_deg > 0.0 ? std::abs(err) / slew_deg : 0.0);
            } else if (_fleet_slewing[pol] && (!fleet_used[pol] || std::abs(err) < 1.0)) {
                _fleet_slewing[pol] = false;
                WARN("elem_sum_shared: pol-{:d} {:s} ({:+.1f} deg, sim {:.2f})", pol,
                     fleet_used[pol] ? "on the fleet reference"
                                     : "slew stopped: the reference no longer applies",
                     err, o.sim);
            }
        }
    }
}

namespace {
const char* fleet_mode_name(int m) {
    return m == 2 ? "live" : (m == 1 ? "log" : "off");
}
} // namespace

void GnssGpuRecordAssemble::fleet_ref_apply_pending() {
    // REST staging -> state, on the main thread (shared_consensus).
    std::vector<std::complex<double>> ref;
    bool ref_set = false;
    int mode = -1;
    double slew = -1.0;
    {
        std::lock_guard<std::mutex> lk(_gain_mtx);
        if (_pending_fleet_ref_set) {
            ref.swap(_pending_fleet_ref);
            ref_set = true;
            _pending_fleet_ref_set = false;
        }
        mode = _pending_fleet_mode;
        _pending_fleet_mode = -1;
        slew = _pending_fleet_slew_deg_s;
        _pending_fleet_slew_deg_s = -1.0;
    }
    if (!ref_set && mode < 0 && slew < 0.0)
        return;
    const int was = _fleet_mode.load();
    if (ref_set) {
        _g_fleet_ref.swap(ref);
        _fleet_ref_present = !_g_fleet_ref.empty();
    }
    if (mode >= 0)
        _fleet_mode = mode;
    if (slew >= 0.0)
        _fleet_slew_rad_s = slew * M_PI / 180.0;
    // Leaving live, or losing F: the own pin takes over from the model's CURRENT phase, so the
    // switch itself never steps it.
    if (((was == 2 && _fleet_mode.load() != 2) || (ref_set && _g_fleet_ref.empty()))
        && _g_shared_warm) {
        _g_pin_ref = _g_shared;
        _g_pin_ref_ok = true;
    }
    _fleet_slewing[0] = _fleet_slewing[1] = false;
    WARN("elem_sum_shared: fleet reference {:s} -> {:s}{:s}, slew {:.2f} deg/s",
         fleet_mode_name(was), fleet_mode_name(_fleet_mode.load()),
         ref_set ? (_g_fleet_ref.empty() ? " (reference CLEARED: own pin)" : " (new reference)")
                 : "",
         _fleet_slew_rad_s * 180.0 / M_PI);
}

void GnssGpuRecordAssemble::set_elem_sum_shared_ref_callback(kotekan::connectionInstance& conn,
                                                             nlohmann::json& request) {
    // {"ref": [[re, im], ...] (n_elements pairs; [] clears), "mode": "off|log|live",
    //  "slew_deg_s": x} -- every field optional. Staged; the next consensus update (~1 s, not
    // during a transit freeze) applies it. python/scripts/gnss/elem_shared_ref.py post.
    std::vector<std::complex<double>> ref;
    bool ref_set = false;
    int mode = -1;
    double slew = -1.0;
    try {
        if (request.contains("ref")) {
            auto& arr = request.at("ref");
            if (!arr.is_array())
                throw std::runtime_error("'ref' must be an array of [re, im] pairs");
            for (auto& e : arr) {
                if (!e.is_array() || e.size() != 2)
                    throw std::runtime_error("each ref element must be [re, im]");
                ref.emplace_back(e[0].get<double>(), e[1].get<double>());
            }
            if (!ref.empty() && (int)ref.size() != _n_elements)
                throw std::runtime_error(fmt::format("'ref' has {:d} elements, n_elements is {:d}",
                                                     ref.size(), _n_elements));
            ref_set = true;
        }
        if (request.contains("mode")) {
            const std::string m = request.at("mode").get<std::string>();
            mode = (m == "off") ? 0 : (m == "log") ? 1 : (m == "live") ? 2 : -2;
            if (mode == -2)
                throw std::runtime_error("'mode' must be off | log | live");
        }
        if (request.contains("slew_deg_s")) {
            slew = request.at("slew_deg_s").get<double>();
            if (!(slew >= 0.0) || slew > 180.0)
                throw std::runtime_error("'slew_deg_s' must be in [0, 180]");
        }
    } catch (const std::exception& ex) {
        conn.send_error(std::string("set_elem_sum_shared_ref: ") + ex.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(_gain_mtx);
        if (ref_set) {
            _pending_fleet_ref.swap(ref);
            _pending_fleet_ref_set = true;
        }
        if (mode >= 0)
            _pending_fleet_mode = mode;
        if (slew >= 0.0)
            _pending_fleet_slew_deg_s = slew;
    }
    conn.send_json_reply(nlohmann::json{
        {"mode", fleet_mode_name(_fleet_mode.load())},
        {"present", _fleet_ref_present.load()},
        {"staged", {{"ref", ref_set}, {"mode", mode >= 0}, {"slew_deg_s", slew >= 0.0}}},
        {"applies", "next consensus update"}});
}

bool GnssGpuRecordAssemble::shared_frozen(double now_s) {
    // A post older than 120 s is not evidence of anything (broker down or an older broker
    // that sends no "_bore"): no freeze from it, and an armed hold still runs out.
    if (now_s - _bore_post_t.load() <= 120.0 && _bore_sep_deg.load() < _elem_shared_freeze_deg)
        _freeze_until = now_s + _elem_shared_freeze_hold_s;
    return now_s < _freeze_until;
}

void GnssGpuRecordAssemble::shared_hold(size_t p) {
    // Install this PRN's weights from the model. Until a model exists, the PRN rides its own
    // learner's weights (the held-mode behaviour; outside a transit freeze, see below), so
    // nothing is lost while the consensus forms; until its inter-pol coefficient is warm it
    // combines pol-0 alone (coherent from the first record, 3 dB short of both pols).
    using cd = std::complex<double>;
    const int n = _n_elements;
    if (p >= _cal.size() || n <= 0)
        return;
    const double now_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now_s - _g_shared_t >= 1.0)
        shared_consensus(now_s);
    gnss::ElemCal& ec = _cal[p];
    if (!_g_shared_warm) {
        // ⚠️ NOT INSIDE A TRANSIT FREEZE. The shadow is exactly the per-PRN learner the freeze
        // exists to keep out of the combine: near boresight a bright satellite's leakage
        // teaches it that satellite's phases, and a full-array combine on those weights puts
        // the array gain on the leak (worse than the bare reference element). So while frozen
        // the live cal keeps what it holds -- the shadow's weights from before the freeze, or
        // nothing, which leaves the header on the reference element.
        const auto& sh = _cal_shadow[p];
        if (sh.warm() && !shared_frozen(now_s))
            ec.hold(sh.weights().data(), n);
        return;
    }
    const int h = (n % 2 == 0) ? n / 2 : n;
    for (int e = 0; e < h; ++e)
        _w_scratch[(size_t)e] = _g_shared[(size_t)e];
    cd c(0.0, 0.0);
    if (h < n && _pol_warmth[p] > 0.95 && _pol_den[p] > 0.0)
        c = _pol_num[p] / _pol_den[p];
    _pol_c[p] = c;
    for (int e = h; e < n; ++e)
        _w_scratch[(size_t)e] = c * _g_shared[(size_t)e];
    ec.hold(_w_scratch.data(), n);
}

void GnssGpuRecordAssemble::shared_pol_update(size_t p, const std::complex<double>* g_prompt,
                                              double dt_s) {
    // The inter-pol coefficient from the two SUB-BEAMS of this record: B0 = pol-0 model dotted
    // with the prompt, B1 = pol-1 likewise; c = <B1 conj(B0)> / <|B0|^2>. Each sub-beam has
    // the half-array's gain, so a bright satellite's leakage is rejected here as in the beam
    // itself -- that is what makes this one number safe to learn per PRN when 32 were not.
    using cd = std::complex<double>;
    const int n = _n_elements;
    if (!_g_shared_warm || p >= _pol_num.size() || n < 2 || (n % 2) != 0 || !(dt_s > 0.0))
        return;
    // Transit: hold every coefficient too. The sub-beams reject leakage by the half-array's
    // gain, but a boresight transit is 20-30 dB above a weak satellite and 3 s forgets fast.
    if (shared_frozen(steady_now_s()))
        return;
    const int h = n / 2;
    cd b0(0.0, 0.0), b1(0.0, 0.0);
    for (int e = 0; e < h; ++e)
        b0 += std::conj(_g_shared[(size_t)e]) * g_prompt[e];
    for (int e = h; e < n; ++e)
        b1 += std::conj(_g_shared[(size_t)e]) * g_prompt[e];
    const double alpha = std::min(0.25, 1.0 - std::exp(-dt_s / _elem_pol_tau_s));
    _pol_num[p] += alpha * (b1 * std::conj(b0) - _pol_num[p]);
    _pol_den[p] += alpha * (std::norm(b0) - _pol_den[p]);
    _pol_warmth[p] += alpha * (1.0 - _pol_warmth[p]);
}

void GnssGpuRecordAssemble::get_elem_cal_callback(kotekan::connectionInstance& conn) {
    // Per PRN slot: is the live cal warm, is the anchor off the reference, and -- when held --
    // how well the shadow (still learning) agrees with the held weights (-1 = not measured).
    // The doubles are written by main_thread and read here without a lock: a torn read of one
    // diagnostic double is acceptable, a lock on the per-record path is not.
    nlohmann::json out;
    out["adapt"] = _elem_adapt.load();
    out["n_elements"] = _n_elements;
    out["shared"] = _elem_shared.load();
    out["shared_warm"] = _g_shared_warm;
    out["shared_n"] = _g_shared_n;
    {
        const double now_s = steady_now_s();
        out["shared_frozen"] = now_s < _freeze_until;
        out["bore_sep_deg"] = _bore_sep_deg.load();
        out["bore_post_age_s"] = now_s - _bore_post_t.load();
        out["pin_ref_ok"] = _g_pin_ref_ok;
    }
    out["fleet_ref"] = {{"mode", fleet_mode_name(_fleet_mode.load())},
                        {"present", _fleet_ref_present.load()},
                        {"slew_deg_s", _fleet_slew_rad_s * 180.0 / M_PI},
                        {"min_sim", _fleet_min_sim},
                        {"err_deg", {_fleet_err_deg[0], _fleet_err_deg[1]}},
                        {"sim", {_fleet_sim[0], _fleet_sim[1]}},
                        {"applied", {_fleet_applied[0], _fleet_applied[1]}}};
    nlohmann::json gs = nlohmann::json::array();
    for (const auto& g : _g_shared)
        gs.push_back({g.real(), g.imag()});
    out["g_shared"] = gs;
    nlohmann::json rows = nlohmann::json::array();
    for (size_t p = 0; p < _cal.size() && p < _prns.size(); ++p) {
        nlohmann::json row = {{"prn", _prns[p]},
                              {"warm", _cal[p].warm()},
                              {"anchor_moved", _cal[p].anchor_moved()},
                              {"shadow_warm", _cal_shadow[p].warm()},
                              {"sim", _cal_sim[p]},
                              {"pol_c", {_pol_c[p].real(), _pol_c[p].imag()}},
                              {"pol_warm", p < _pol_warmth.size() && _pol_warmth[p] > 0.95}};
        if (_proj_ready && p < _cal_proj.size()) {
            row["proj_warm"] = _cal_proj[p].warm();
            row["cap_plain"] = _cap_plain[p];
            row["cap_proj"] = _cap_proj[p];
            row["b_cos2"] = _b_cos2[p];
            row["sim_pp"] = _sim_pp[p];
            row["is_src"] = _proj_isB[p] != 0;
            row["is_probe"] = _proj_isProbe[p] != 0;
        }
        rows.push_back(row);
    }
    out["prns"] = rows;
    if (_proj_ready) {
        // The projection's state (hpp note): mode and tunables, what is in force this record
        // (k per channel, which source kinds), the probe-stack trigger level per channel, the
        // CPU it costs, and a human summary of the sources.
        const int mode = _proj_mode.load();
        nlohmann::json pj;
        pj["mode"] = (mode == 2) ? "live" : (mode == 1) ? "shadow" : "off";
        pj["deg"] = _proj_deg.load();
        pj["rank_max"] = _proj_rank_max.load();
        pj["max_age_s"] = _proj_max_age_s.load();
        pj["probe_frac_min"] = _proj_probe_frac_min.load();
        pj["group"] = _proj_group;
        pj["sys"] = std::string(1, _proj_sys);
        pj["probes_from_broker"] = _probes_from_broker.load();
        pj["k"] = _proj_k_rec;
        pj["us_per_record"] = _proj_us;
        pj["active_records"] = _proj_active_records;
        {
            std::lock_guard<std::mutex> lk(_proj_mtx);
            pj["sources"] = _proj_desc;
        }
        nlohmann::json chans = nlohmann::json::array();
        for (size_t ch = 0; ch < _proj_fids.size(); ++ch)
            chans.push_back(
                {{"fid", _proj_fids[ch]},
                 {"k", ch < _proj_k_ch.size() ? _proj_k_ch[ch] : 0},
                 {"src", ch < _proj_src_ch.size() ? _proj_src_ch[ch] : 0},
                 {"probe_frac", ch < _proj_probe_frac.size() ? _proj_probe_frac[ch] : 0.0}});
        pj["channels"] = chans;
        out["proj"] = pj;
    }
    conn.send_json_reply(out);
}

void GnssGpuRecordAssemble::set_reference_element_callback(kotekan::connectionInstance& conn,
                                                           nlohmann::json& request) {
    // Body: {"element": N}. Staged only; main_thread applies at the next frame boundary
    // (see the swap block there for the exact semantics). Validation mirrors the
    // construction-time FATAL: the same bounds, answered as a 400 instead of a crash.
    int el = -1;
    try {
        el = request.at("element").get<int>();
    } catch (const std::exception& e) {
        conn.send_error(std::string("set_reference_element: bad payload (want {\"element\": N}): ")
                            + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    if (el < 0 || el >= _n_elements) {
        conn.send_error(fmt::format("set_reference_element: element {:d} outside "
                                    "[0, n_elements={:d})",
                                    el, _n_elements),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    int prev;
    {
        std::lock_guard<std::mutex> lk(_gain_mtx);
        _pending_ref = el;
        prev = _reference_element;
    }
    nlohmann::json reply{{"reference_element", prev},
                         {"pending", el},
                         {"applies", "next frame boundary"},
                         {"rewarm_s", _elem_sum ? 3.0 * _elem_sum_tau_s : 0.0}};
    conn.send_json_reply(reply);
}

// ── BRIGHT-SATELLITE PROJECTION (hpp note; gnssProjSubspace.hpp; PROJECTION_PLAN.md) ────────

void GnssGpuRecordAssemble::proj_reset_slot(size_t p, double now_s) {
    if (!_proj_ready || p >= _proj_own.size())
        return;
    _proj_own[p].reset();
    _cal_proj[p] = gnss::ElemCal(_n_elements, _reference_element, _elem_sum_tau_s, _elem_sum_min_w);
    _cap_plain[p] = _cap_proj[p] = _b_cos2[p] = _sim_pp[p] = -1.0;
    _proj_isB[p] = 0;
    _proj_isProbe[p] = 0;
    _proj_dropped[p] = 0;
    _slot_was_run[p] = 0;
    _slot_run_since[p] = now_s;
    {
        // The REST geometry handler writes this vector under _steer_mtx; so do we.
        std::lock_guard<std::mutex> lk(_steer_mtx);
        _slot_probe_broker[p] = 0;
    }
}

void GnssGpuRecordAssemble::proj_prepare_record(const double* corr, const void* pctl_rec_v,
                                                int n_chan, int n_e, int64_t wstart, double utc,
                                                double now_s) {
    using namespace gnss_gpu;
    using cd = std::complex<double>;
    const PrnCtl* pc = (const PrnCtl*)pctl_rec_v;
    const int n_prn = (int)_prns.size();
    _proj_k_rec = 0;
    for (auto& Q : _proj_Q)
        Q.clear();
    std::fill(_proj_isB.begin(), _proj_isB.end(), 0);
    std::fill(_proj_isProbe.begin(), _proj_isProbe.end(), 0);
    std::fill(_proj_dropped.begin(), _proj_dropped.end(), 0);
    std::fill(_proj_k_ch.begin(), _proj_k_ch.end(), 0);
    std::fill(_proj_src_ch.begin(), _proj_src_ch.end(), 0);
    if (n_chan != (int)_proj_fids.size()) {
        // The basis is keyed by channel_ids; a frame with another channel count would apply
        // one channel's subspace to another's data. Refuse, once and loudly.
        if (!_proj_nchan_warned) {
            WARN("elem projection DISABLED: channel_ids has {:d} channels, the frame {:d}",
                 (int)_proj_fids.size(), n_chan);
            _proj_nchan_warned = 1;
        }
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    _proj_corr_rec = corr;
    const double dt_s = (_proj_wstart_prev > 0 && wstart > _proj_wstart_prev)
                            ? (double)(wstart - _proj_wstart_prev) / _sample_rate
                            : 0.0;
    _proj_wstart_prev = wstart;
    for (int p = 0; p < n_prn; ++p) {
        const bool run = pc[p].run != 0;
        if (run && !_slot_was_run[(size_t)p])
            _slot_run_since[(size_t)p] = now_s;
        _slot_was_run[(size_t)p] = run ? 1 : 0;
    }

    // 1. SOURCES AMONG THE OWN SLOTS: running, geometry fresh, inside elem_proj_deg of
    //    boresight -- nearest first, at most rank_max. Everything else that runs and has had
    //    no geometry since it started (a below-horizon probe) feeds the probe stack.
    const double az = _bore_az_deg * M_PI / 180.0, el = _bore_el_deg * M_PI / 180.0;
    const double eb[3] = {std::cos(el) * std::sin(az), std::cos(el) * std::cos(az), std::sin(el)};
    std::vector<std::pair<double, int>> cand;
    std::vector<uint8_t> probe_flag((size_t)n_prn, 0);
    const bool broker_probes = _probes_from_broker.load();
    const double deg = _proj_deg.load();
    {
        std::lock_guard<std::mutex> lk(_steer_mtx);
        for (int p = 0; p < n_prn; ++p) {
            if (!pc[p].run)
                continue;
            double e[3];
            const bool fresh = _steer.warm(p, now_s);
            if (fresh && _steer.direction(p, utc, e)) {
                const double c =
                    std::max(-1.0, std::min(1.0, e[0] * eb[0] + e[1] * eb[1] + e[2] * eb[2]));
                const double sep = std::acos(c) * 180.0 / M_PI;
                if (sep < deg)
                    cand.emplace_back(sep, p);
            } else if (broker_probes
                           ? (_slot_probe_broker[(size_t)p] != 0)
                           : (!fresh
                              && now_s - _slot_run_since[(size_t)p] >= _proj_probe_since_s)) {
                probe_flag[(size_t)p] = 1;
            }
        }
    }
    std::sort(cand.begin(), cand.end());
    const int kmax = std::max(1, std::min(_proj_kmax_alloc, _proj_rank_max.load()));
    if ((int)cand.size() > kmax)
        cand.resize((size_t)kmax);
    for (int p = 0; p < n_prn; ++p) {
        bool is = false;
        for (const auto& c : cand)
            is = is || (c.second == p);
        if (!is && _proj_own[(size_t)p])
            _proj_own[(size_t)p].reset(); // no longer a source: forget its tracker
    }
    // The sources' steering tables at THIS record's time (hpp note: the own-row covariance is
    // accumulated in the source's steered frame and un-steered at use with the same table).
    {
        std::lock_guard<std::mutex> lk(_steer_mtx);
        // A steer table wider than this record's channel axis would write past _proj_steer:
        // impossible while both come from channel_ids, and cheap to refuse if that ever drifts.
        if ((size_t)_steer.n_chan() * _steer.n_elem() > (size_t)n_chan * n_e) {
            if (!_proj_nchan_warned) {
                WARN(
                    "elem projection DISABLED: steer table {:d}x{:d} exceeds the frame's {:d}x{:d}",
                    _steer.n_chan(), _steer.n_elem(), n_chan, n_e);
                _proj_nchan_warned = 1;
            }
            cand.clear();
        }
        for (size_t ci = 0; ci < cand.size(); ++ci) {
            const int p = cand[ci].second;
            _steer.refresh(p, utc);
            _steer.copy_slot(p, &_proj_steer[ci * (size_t)n_chan * n_e]);
        }
    }
    auto steer_of = [&](size_t ci, int ch) -> const gnss::ElemSteer::cf* {
        return &_proj_steer[(ci * (size_t)n_chan + (size_t)ch) * n_e];
    };
    // q in the raw frame from a tracker's steered-frame vector: q_raw = conj(D) o q_steered.
    auto unsteer = [&](size_t ci, int ch, const cd* qs, cd* out) {
        const gnss::ElemSteer::cf* st = steer_of(ci, ch);
        for (int i = 0; i < n_e; ++i)
            out[i] = std::conj(cd(st[i].real(), st[i].imag())) * qs[i];
    };

    // 2. OWN-ROW TRACKERS: this record's PROMPT row, STEERED by the source's own table, per
    //    covering channel; rank 1 (a satellite's own row is rank one by construction),
    //    warm-started power iteration.
    for (size_t ci = 0; ci < cand.size(); ++ci) {
        const int p = cand[ci].second;
        _proj_isB[(size_t)p] = 1;
        auto& T = _proj_own[(size_t)p];
        if (!T)
            T = std::make_unique<gnss::ProjSubspace>(n_e, n_chan, _proj_tau_s, 1);
        const PrnCtl& c = pc[p];
        const size_t prow = (size_t)(c.job0 + ROW_P) * n_chan;
        for (int ch = 0; ch < n_chan; ++ch) {
            if (!((c.chan_mask >> ch) & 1ULL))
                continue;
            const double* v = corr + 2 * (prow + ch) * n_e;
            const gnss::ElemSteer::cf* st = steer_of(ci, ch);
            for (int i = 0; i < n_e; ++i)
                _v_scratch[(size_t)i] = cd(v[2 * i], v[2 * i + 1]) * cd(st[i].real(), st[i].imag());
            T->push(ch, _v_scratch.data(), dt_s);
            T->solve(ch, T->warm(ch) ? 1 : 2);
        }
    }
    // Every steered slot's table at this record's time, for the identity check below (a
    // direction the probe stack or a sibling reports may be one of OUR tracked satellites,
    // which must then be treated as a source, not a victim). ~45 kB of copies per record.
    std::fill(_proj_steered.begin(), _proj_steered.end(), 0);
    {
        std::lock_guard<std::mutex> lk(_steer_mtx);
        for (int p = 0; p < n_prn; ++p)
            if (pc[p].run && _steer.warm(p, now_s))
                _proj_steered[(size_t)p] = 1;
    }

    // 3. THE PROBE STACK: every probe row's prompt into one covariance per channel; solved
    //    every 4th record (the direction moves on the second scale, the solve is n^2 per
    //    component). Its component-0 energy fraction is the trigger level, served per channel.
    for (int p = 0; p < n_prn; ++p) {
        if (!probe_flag[(size_t)p] || _proj_isB[(size_t)p])
            continue;
        _proj_isProbe[(size_t)p] = 1;
        const PrnCtl& c = pc[p];
        const size_t prow = (size_t)(c.job0 + ROW_P) * n_chan;
        for (int ch = 0; ch < n_chan; ++ch) {
            if (!((c.chan_mask >> ch) & 1ULL))
                continue;
            const double* v = corr + 2 * (prow + ch) * n_e;
            for (int i = 0; i < n_e; ++i)
                _v_scratch[(size_t)i] = cd(v[2 * i], v[2 * i + 1]);
            _proj_probe.push(ch, _v_scratch.data(), dt_s);
        }
    }
    const bool do_solve = (++_proj_solve_ctr % 4) == 0;
    for (int ch = 0; ch < n_chan; ++ch) {
        if (do_solve && _proj_probe.warmth(ch) > 0.0)
            _proj_probe.solve(ch, 2);
        _proj_probe_frac[(size_t)ch] = _proj_probe.warm(ch) ? _proj_probe.frac(ch, 0) : 0.0;
    }

    // 4. THE BASIS PER CHANNEL: own rows, then the siblings' entries from the board, then the
    //    probe stack. Gram-Schmidt drops a direction already spanned (the same emitter seen
    //    through two sources), rank capped at rank_max.
    //
    //    ⚠️ A PROBE-STACK DIRECTION IS USED ONLY WHEN IT IS ACCOUNTED FOR. Overnight 09-28/29 the
    //    stack triggered 40 % of the time far from any transit, on a direction carrying up to
    //    99 % of the probe rows' cross energy that matched no tracked satellite, and when it did
    //    latch onto a real satellite at 5-10 deg that satellite's own held weights sat at cos^2
    //    0.7-0.9 with the direction: projected live, it would have been nulled out of its own
    //    row. So a probe direction (ours or a sibling's) is taken only while the broker reports a
    //    satellite inside elem_proj_deg of boresight, and then: a direction a row source on this
    //    channel already spans is the same emitter seen twice (skipped, and never put through
    //    the identity, see below); a direction that IS one tracked satellite's own signature
    //    (proj_identify) is a column for nothing when that satellite is steered -- inside the
    //    window its row is the source, outside it it is a sidelobe leaker worth 1/26 of every
    //    victim -- so it is dropped; the same for a tracked satellite WITHOUT geometry makes
    //    that slot a source (its own rows spared) and serves the direction to the others; and a
    //    direction that names no single row is the row-less emitter the stack exists for.
    //    ⚠️ C42 09-29 (third lesson): the identity used to run on every probe-class direction
    //    and MARK its answer a source. On the six chains that do not track the emitter every row
    //    along the direction is a victim's, so it named the most captured victim on 40 % of the
    //    records, whose projected learner was then fed the plain prompt (cap_proj = cap_plain,
    //    sim_pp 0.99 on e5a/e5b/e6/l5 -- and live, its rows would have gone unprojected); and on
    //    L2C, where C42 has no signal, the stack's direction was the brightest GPS satellite's
    //    own signature, accepted as "unnamed" and charged to every other GPS row (b_cos2 > 0.3 on
    //    16 % of the polls).
    const double frac_min = _proj_probe_frac_min.load();
    const double max_age = _proj_max_age_s.load();
    const bool bore_near = (now_s - _bore_post_t.load() <= 120.0) && _bore_sep_deg.load() < deg;
    int k_rec = 0;
    int n_sib = 0, n_probe_ch = 0, n_ident = 0, n_drop = 0;
    double probe_frac_max = 0.0;
    std::fill(_proj_probe_used.begin(), _proj_probe_used.end(), 0);
    for (int ch = 0; ch < n_chan; ++ch) {
        gnss::ProjBasis& Q = _proj_Q[(size_t)ch];
        Q.kmax = kmax;
        int src = 0;
        for (size_t ci = 0; ci < cand.size(); ++ci) {
            const auto& T = *_proj_own[(size_t)cand[ci].second];
            if (!T.warm(ch) || T.k(ch) <= 0)
                continue;
            unsteer(ci, ch, T.q(ch, 0), _v_scratch.data());
            if (Q.add(_v_scratch.data()))
                src |= 1;
        }
        // A probe-class direction (the stack's, ours or a sibling's): see the note above.
        auto accept_probe_dir = [&](const cd* q) -> bool {
            if (!bore_near)
                return false;
            // Already spanned by a row source on this channel (Gram-Schmidt would drop it at
            // the same 0.5): the same emitter seen twice, nothing to name.
            if (Q.k > 0 && Q.cos2(q) >= 0.5)
                return false;
            int n_along = 0;
            const int who = proj_identify(q, ch, n_e, pctl_rec_v, &n_along);
            if (who < 0) {
                // Unnamed: used only when it captures someone, i.e. at least two of our rows lie
                // along it (a common interferer nobody here tracks). A direction along NO row of
                // ours is a satellite this chain does not track, seen through the sidelobes:
                // during G32's pass (09-29 15:2x) the 1207/1268/1278 chains, where G32 has no
                // signal, accepted such directions on 1-6 channels and charged the victims
                // b_cos2 > 0.1 on 15-19 % of the polls (7 % on the chains with a row source),
                // for nothing. A direction along ONE row is that row's own signature, named above.
                return n_along >= 2;
            }
            if (_proj_steered[(size_t)who]) {
                // One steered satellite's own signature: never a column (its row is the source
                // when it is inside the window; a sidelobe leaker when it is not).
                if (!_proj_dropped[(size_t)who]) {
                    _proj_dropped[(size_t)who] = 1;
                    ++n_drop;
                }
                return false;
            }
            if (!_proj_isB[(size_t)who]) {
                // Tracked without geometry: its own rows are spared, the direction serves the rest.
                _proj_isB[(size_t)who] = 1;
                ++n_ident;
            }
            return true;
        };
        gnss::ProjBoard::instance().for_each(
            _proj_group, _proj_fids[(size_t)ch], unique_name, now_s, max_age,
            [&](const gnss::ProjEntry& e) {
                if (e.n != n_e || e.q.size() < (size_t)e.k * (size_t)n_e)
                    return; // another element count, or a malformed entry: never index it
                for (int j = 0; j < e.k; ++j) {
                    for (int i = 0; i < n_e; ++i)
                        _v_scratch[(size_t)i] =
                            cd(e.q[(size_t)j * n_e + i].real(), e.q[(size_t)j * n_e + i].imag());
                    if (e.src == 1 && !accept_probe_dir(_v_scratch.data()))
                        continue;
                    if (Q.add(_v_scratch.data())) {
                        src |= (e.src == 1) ? 4 : 2;
                        if (e.src == 1)
                            _proj_probe_used[(size_t)ch] = 1;
                        else
                            ++n_sib;
                    }
                }
            });
        // The probe-stack trigger has hysteresis: a channel enters at frac_min and stays until
        // 0.8 frac_min, so a level hovering at the threshold does not flicker the basis (and
        // the log) record by record.
        const bool was_on = (_proj_probe_on[(size_t)ch] != 0);
        const bool probe_on =
            _proj_probe.warm(ch) && _proj_probe.frac(ch, 0) >= (was_on ? 0.8 * frac_min : frac_min);
        _proj_probe_on[(size_t)ch] = probe_on ? 1 : 0;
        if (probe_on) {
            bool any = false;
            for (int j = 0; j < _proj_probe.k(ch); ++j) {
                if (!accept_probe_dir(_proj_probe.q(ch, j)))
                    continue;
                any = Q.add(_proj_probe.q(ch, j)) || any;
            }
            if (any) {
                src |= 4;
                _proj_probe_used[(size_t)ch] = 1;
                ++n_probe_ch;
                probe_frac_max = std::max(probe_frac_max, _proj_probe.frac(ch, 0));
            }
        }
        _proj_k_ch[(size_t)ch] = Q.k;
        _proj_src_ch[(size_t)ch] = src;
        k_rec = std::max(k_rec, Q.k);
    }
    _proj_k_rec = k_rec;
    if (k_rec > 0)
        ++_proj_active_records;

    // 5. PUBLISH to the siblings: our own rows (the nearest source, or both when two are
    //    inside the window) and our probe stack, per channel; retire them when they go away.
    gnss::ProjBoard& board = gnss::ProjBoard::instance();
    bool pub_own = false, pub_probe = false;
    for (int ch = 0; ch < n_chan; ++ch) {
        gnss::ProjEntry e;
        e.owner = unique_name;
        e.sys = _proj_sys;
        e.src = 0;
        e.wstart = wstart;
        e.t_pub = now_s;
        e.n = n_e;
        e.k = 0;
        for (size_t ci = 0; ci < cand.size(); ++ci) {
            const int p = cand[ci].second;
            const auto& T = *_proj_own[(size_t)p];
            if (!T.warm(ch) || T.k(ch) <= 0 || e.k >= kmax)
                continue;
            if (e.k == 0) {
                e.prn = _prns[(size_t)p];
                e.sep_deg = cand[ci].first;
                e.frac = T.frac(ch, 0);
            }
            unsteer(ci, ch, T.q(ch, 0), _v_scratch.data());
            for (int i = 0; i < n_e; ++i)
                e.q.emplace_back((float)_v_scratch[(size_t)i].real(),
                                 (float)_v_scratch[(size_t)i].imag());
            ++e.k;
        }
        if (e.k > 0) {
            board.publish(_proj_group, _proj_fids[(size_t)ch], e);
            pub_own = true;
        }
        if (_proj_probe_used[(size_t)ch] && _proj_probe_on[(size_t)ch] && _proj_probe.k(ch) > 0) {
            gnss::ProjEntry pe;
            pe.owner = _proj_owner_probe;
            pe.sys = _proj_sys;
            pe.src = 1;
            pe.frac = _proj_probe.frac(ch, 0);
            pe.wstart = wstart;
            pe.t_pub = now_s;
            pe.n = n_e;
            pe.k = 0;
            for (int j = 0; j < _proj_probe.k(ch) && pe.k < kmax; ++j, ++pe.k) {
                const cd* q = _proj_probe.q(ch, j);
                for (int i = 0; i < n_e; ++i)
                    pe.q.emplace_back((float)q[i].real(), (float)q[i].imag());
            }
            board.publish(_proj_group, _proj_fids[(size_t)ch], pe);
            pub_probe = true;
        }
    }
    if (!pub_own && _proj_pub_own)
        board.retire(_proj_group, unique_name);
    if (!pub_probe && _proj_pub_probe)
        board.retire(_proj_group, _proj_owner_probe);
    _proj_pub_own = pub_own;
    _proj_pub_probe = pub_probe;

    // 6. A one-line description of the sources, rebuilt (and logged) only when the SET changes:
    //    the signature is compared every record, the string is formatted only on a change.
    {
        std::vector<int> sig;
        sig.reserve(cand.size() + 4);
        for (const auto& c : cand)
            sig.push_back(_prns[(size_t)c.second]);
        sig.push_back(-1);
        sig.push_back(n_sib > 0);
        sig.push_back(n_probe_ch > 0);
        sig.push_back(n_ident);
        sig.push_back(n_drop);
        for (int p = 0; p < n_prn; ++p)
            if (_proj_dropped[(size_t)p])
                sig.push_back(p);
        if (sig != _proj_sig) {
            _proj_sig = sig;
            std::string d;
            for (const auto& [sep, p] : cand)
                d += fmt::format("{}{:c}{:02d} row {:.2f} deg", d.empty() ? "" : "; ", _proj_sys,
                                 _prns[(size_t)p], sep);
            if (n_sib > 0)
                d += fmt::format("{}sibling x{:d}", d.empty() ? "" : "; ", n_sib);
            if (n_probe_ch > 0)
                d += fmt::format("{}probe stack on {:d} ch (frac {:.2f})", d.empty() ? "" : "; ",
                                 n_probe_ch, probe_frac_max);
            if (n_ident > 0) {
                std::string who;
                for (int p = 0; p < n_prn; ++p)
                    if (_proj_isB[(size_t)p]
                        && std::none_of(
                            cand.begin(), cand.end(),
                            [&](const std::pair<double, int>& c) { return c.second == p; }))
                        who += fmt::format("{}{:c}{:02d}", who.empty() ? "" : ",", _proj_sys,
                                           _prns[(size_t)p]);
                d += fmt::format("{}identified {:s}", d.empty() ? "" : "; ", who);
            }
            if (n_drop > 0) {
                std::string who;
                for (int p = 0; p < n_prn; ++p)
                    if (_proj_dropped[(size_t)p])
                        who += fmt::format("{}{:c}{:02d}", who.empty() ? "" : ",", _proj_sys,
                                           _prns[(size_t)p]);
                d += fmt::format("{}probe dir dropped ({:s})", d.empty() ? "" : "; ", who);
            }
            {
                std::lock_guard<std::mutex> lk(_proj_mtx);
                _proj_desc = d;
            }
            // Logged on change, at most every 5 s (a source set that flips faster than that is
            // itself the message, and one line says so).
            if (now_s - _proj_log_t >= 5.0) {
                _proj_log_t = now_s;
                INFO("elem projection [{:s}]: {:s} (k {:d}, mode {:s})", unique_name,
                     d.empty() ? "no source in force" : d, k_rec,
                     _proj_mode.load() == 2 ? "live" : "shadow");
            }
        }
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    _proj_us += 0.01 * (us - _proj_us);
}

void GnssGpuRecordAssemble::proj_slot_inplace(double* corr_rw, const void* pctl_slot, int n_chan,
                                              int n_e, int n_rows) {
    using namespace gnss_gpu;
    using cd = std::complex<double>;
    const PrnCtl& c = *(const PrnCtl*)pctl_slot;
    for (int t = 0; t < n_rows; ++t) {
        const size_t row = (size_t)(c.job0 + t) * n_chan;
        for (int ch = 0; ch < n_chan; ++ch) {
            if (!((c.chan_mask >> ch) & 1ULL) || _proj_Q[(size_t)ch].k == 0)
                continue;
            double* v = corr_rw + 2 * (row + ch) * n_e;
            for (int i = 0; i < n_e; ++i)
                _v_scratch[(size_t)i] = cd(v[2 * i], v[2 * i + 1]);
            _proj_Q[(size_t)ch].project(_v_scratch.data());
            for (int i = 0; i < n_e; ++i) {
                v[2 * i] = _v_scratch[(size_t)i].real();
                v[2 * i + 1] = _v_scratch[(size_t)i].imag();
            }
        }
    }
}

void GnssGpuRecordAssemble::proj_slot_shadow(const double* corr, const void* pctl_slot, int n_chan,
                                             int n_e, bool steered) {
    using namespace gnss_gpu;
    using cd = std::complex<double>;
    const PrnCtl& c = *(const PrnCtl*)pctl_slot;
    std::fill(_g_proj.begin(), _g_proj.end(), cd(0.0, 0.0));
    const size_t row = (size_t)(c.job0 + ROW_P) * n_chan;
    for (int ch = 0; ch < n_chan; ++ch) {
        if (!((c.chan_mask >> ch) & 1ULL))
            continue;
        const double* v = corr + 2 * (row + ch) * n_e;
        for (int i = 0; i < n_e; ++i)
            _v_scratch[(size_t)i] = cd(v[2 * i], v[2 * i + 1]);
        if (_proj_Q[(size_t)ch].k > 0)
            _proj_Q[(size_t)ch].project(_v_scratch.data());
        const gnss::ElemSteer::cf* st = steered ? &_steer_buf[(size_t)ch * n_e] : nullptr;
        for (int i = 0; i < n_e; ++i) {
            cd x = _v_scratch[(size_t)i];
            if (st)
                x *= cd(st[i].real(), st[i].imag());
            _g_proj[(size_t)i] += x;
        }
    }
}

void GnssGpuRecordAssemble::proj_slot_diag(size_t p, const void* pctl_slot, int n_chan, int n_e,
                                           bool steered) {
    using namespace gnss_gpu;
    using cd = std::complex<double>;
    // The two learners' agreement, always (1 out of transit; their divergence is the capture).
    {
        const auto& ws = _cal_shadow[p].weights();
        const auto& wp = _cal_proj[p].weights();
        if (_cal_shadow[p].warm() && _cal_proj[p].warm()) {
            cd x(0.0, 0.0);
            double ns = 0.0, np = 0.0;
            for (int i = 0; i < n_e; ++i) {
                x += std::conj(ws[(size_t)i]) * wp[(size_t)i];
                ns += std::norm(ws[(size_t)i]);
                np += std::norm(wp[(size_t)i]);
            }
            _sim_pp[p] = (ns > 0.0 && np > 0.0) ? std::norm(x) / (ns * np) : -1.0;
        } else {
            _sim_pp[p] = -1.0;
        }
    }
    // Capture and cost: the fraction of a weight vector lying in the interferer subspace, as
    // this satellite's steered channel sum sees it -- per channel cos^2(w, D_ch q_ch), i.e.
    // cos^2(conj(D_ch) w, q_ch), averaged over the covering channels with a basis in force.
    // Control for a random direction: 1/n_live (~0.04). Only while a basis is in force and the
    // slot is steered (a probe is never steered and never learns anything meaningful).
    // A SOURCE is excluded: its own learner is degenerate anyway (every element correlates
    // with the reference at rho^2 > 0.99, so ElemCal's self-reference guard collapses it) and
    // its capture would read as a bright satellite captured by itself.
    if (_proj_k_rec == 0 || !steered || _proj_isB[p]) {
        _cap_plain[p] = _cap_proj[p] = _b_cos2[p] = -1.0;
        return;
    }
    const PrnCtl& c = *(const PrnCtl*)pctl_slot;
    const std::vector<cd>* w[3] = {&_cal_shadow[p].weights(), &_cal_proj[p].weights(),
                                   &_cal[p].weights()};
    const bool ok[3] = {_cal_shadow[p].warm(), _cal_proj[p].warm(), _cal[p].warm()};
    double acc[3] = {0.0, 0.0, 0.0};
    int nch = 0;
    for (int ch = 0; ch < n_chan; ++ch) {
        if (!((c.chan_mask >> ch) & 1ULL) || _proj_Q[(size_t)ch].k == 0)
            continue;
        const gnss::ElemSteer::cf* st = &_steer_buf[(size_t)ch * n_e];
        for (int m = 0; m < 3; ++m) {
            if (!ok[m])
                continue;
            for (int i = 0; i < n_e; ++i)
                _v_scratch[(size_t)i] =
                    std::conj(cd(st[i].real(), st[i].imag())) * (*w[m])[(size_t)i];
            const double c2 = _proj_Q[(size_t)ch].cos2(_v_scratch.data());
            if (c2 >= 0.0)
                acc[m] += c2;
        }
        ++nch;
    }
    _cap_plain[p] = (nch > 0 && ok[0]) ? acc[0] / nch : -1.0;
    _cap_proj[p] = (nch > 0 && ok[1]) ? acc[1] / nch : -1.0;
    _b_cos2[p] = (nch > 0 && ok[2]) ? acc[2] / nch : -1.0;
}

void GnssGpuRecordAssemble::set_elem_proj_callback(kotekan::connectionInstance& conn,
                                                   nlohmann::json& request) {
    // Body: any of {"mode": "off"|"shadow"|"live" (or 0|1|2), "deg": D, "rank_max": K,
    // "max_age_s": S, "probe_frac_min": F}. Applied at once (atomics read per record); the
    // reply is the state in force. rank_max cannot exceed the columns allocated at
    // construction (config elem_proj_rank_max).
    try {
        if (request.contains("mode")) {
            const auto& m = request["mode"];
            int mode = -1;
            if (m.is_number_integer())
                mode = m.get<int>();
            else if (m.is_string()) {
                const std::string s = m.get<std::string>();
                mode = (s == "off") ? 0 : (s == "shadow") ? 1 : (s == "live") ? 2 : -1;
            }
            if (mode < 0 || mode > 2) {
                conn.send_error("set_elem_proj: mode must be off|shadow|live",
                                kotekan::HTTP_RESPONSE::BAD_REQUEST);
                return;
            }
            const int prev = _proj_mode.exchange(mode);
            if (prev != mode)
                WARN("set_elem_proj[{:s}]: mode {:d} -> {:d} ({:s})", unique_name, prev, mode,
                     mode == 2   ? "LIVE: rows projected in place"
                     : mode == 1 ? "SHADOW: projected learner + diagnostics only"
                                 : "off");
        }
        if (request.contains("deg"))
            _proj_deg = std::max(0.0, request["deg"].get<double>());
        if (request.contains("rank_max"))
            _proj_rank_max =
                std::max(1, std::min(_proj_kmax_alloc, request["rank_max"].get<int>()));
        if (request.contains("max_age_s"))
            _proj_max_age_s = std::max(0.0, request["max_age_s"].get<double>());
        if (request.contains("probe_frac_min")) // > 1 = the probe stack never triggers
            _proj_probe_frac_min =
                std::max(0.0, std::min(2.0, request["probe_frac_min"].get<double>()));
    } catch (const std::exception& e) {
        conn.send_error(std::string("set_elem_proj: bad payload: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    const int mode = _proj_mode.load();
    conn.send_json_reply(nlohmann::json{{"mode", (mode == 2)   ? "live"
                                                 : (mode == 1) ? "shadow"
                                                               : "off"},
                                        {"deg", _proj_deg.load()},
                                        {"rank_max", _proj_rank_max.load()},
                                        {"rank_alloc", _proj_kmax_alloc},
                                        {"max_age_s", _proj_max_age_s.load()},
                                        {"probe_frac_min", _proj_probe_frac_min.load()},
                                        {"group", _proj_group},
                                        {"sys", std::string(1, _proj_sys)}});
}

int GnssGpuRecordAssemble::proj_identify(const std::complex<double>* q, int ch, int n_e,
                                         const void* pctl_rec, int* n_along) {
    // THE EMITTER IS THE BRIGHTEST ROW ALONG q, NOT THE LEARNER THAT AGREES WITH q. The first
    // version matched q against the slots' learned weights and, during C34's pass (09-29
    // 11:5x), "identified" C37, C21 and C23 -- captured victims whose learners point along the
    // interferer -- while the real source's own learner is degenerate (rho^2 > 0.99 collapses
    // it). A victim so identified was then excluded from the projection: the most captured
    // satellite escaped the fix. So: among the running slots, take this record's RAW prompt
    // row per channel; the interferer's row lies along q with cos^2 ~ 1 and is the strongest
    // by 10-20 dB; a captured victim's row lies along q only in proportion to the leak, and
    // its power is that of a victim. Score = cos^2 x |row|^2, threshold cos^2 > 0.5.
    // AND THE BRIGHTEST VICTIM IS STILL A VICTIM (C42 09-29 13:5x, the sibling chains): where
    // the emitter has no row, every row along q is a victim's and "the brightest" is just the
    // most captured. So the COUNT of rows along q decides: exactly one -> that slot's own
    // signature; several -> the emitter only if it is 10x brighter than the next (its power
    // against its victims' leak), else nobody -- a common interferer in every row, tracked by
    // no slot here.
    using namespace gnss_gpu;
    using cd = std::complex<double>;
    const PrnCtl* pc = (const PrnCtl*)pctl_rec;
    const int n_prn = (int)_prns.size();
    const int n_chan = (int)_proj_fids.size();
    int best = -1;
    double sbest = 0.0, ssecond = 0.0;
    int n = 0;
    *n_along = 0;
    double qn = 0.0;
    for (int i = 0; i < n_e; ++i)
        qn += std::norm(q[i]);
    if (!(qn > 0.0) || _proj_corr_rec == nullptr)
        return -1;
    for (int p = 0; p < n_prn; ++p) {
        const PrnCtl& c = pc[p];
        if (!c.run || _proj_isProbe[(size_t)p] || !((c.chan_mask >> ch) & 1ULL))
            continue;
        const double* v = _proj_corr_rec + 2 * ((size_t)(c.job0 + ROW_P) * n_chan + ch) * n_e;
        cd sdot(0.0, 0.0);
        double vn = 0.0;
        for (int i = 0; i < n_e; ++i) {
            const cd x(v[2 * i], v[2 * i + 1]);
            sdot += std::conj(q[i]) * x;
            vn += std::norm(x);
        }
        if (!(vn > 0.0))
            continue;
        const double c2 = std::norm(sdot) / (vn * qn);
        if (c2 < 0.5)
            continue;
        ++n;
        const double score = c2 * vn;
        if (score > sbest) {
            ssecond = sbest;
            sbest = score;
            best = p;
        } else if (score > ssecond) {
            ssecond = score;
        }
    }
    *n_along = n;
    if (n <= 1)
        return best;
    return (sbest >= 10.0 * ssecond) ? best : -1;
}
