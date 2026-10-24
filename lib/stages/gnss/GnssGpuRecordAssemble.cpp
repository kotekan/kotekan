#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "gnssChanPower.hpp" // for ChanPower, chan_norm_weights
#include "gnssGpuChain.hpp"
#include "gnssRecord.hpp"
#include "gnssSharedPin.hpp" // for ref_offset, pin_rotation
#include "visUtil.hpp"       // for frameID

#include "json.hpp" // for the REST requests and replies

#include <algorithm>
#include <chrono>
#include <cmath>
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
    _sample_rate = config.get<double>(unique_name, "sample_rate");
    // Slot 2 is a float32. A code phase over a 2,046,000-chip baked code (GLONASS L2OC with OC2)
    // has a 0.125-chip ulp there, a standing bias rather than noise, since the argument moves
    // only with the seed and the trim. Every reader wraps slot 2 mod the primary code
    // (trkresid.phys_chips), so the generator sets this to the primary length on such a chain
    // and the record keeps the phase at 0.004 chip. 0 leaves slot 2 as it was.
    _rec_cp_mod_chips = config.get_default<double>(unique_name, "record_cp_mod_chips", 0.0);
    if (_rec_cp_mod_chips < 0.0) {
        FATAL_ERROR("GnssGpuRecordAssemble: record_cp_mod_chips {:g} must be >= 0",
                    _rec_cp_mod_chips);
        return;
    }
    // Element axis: n_elements per-antenna blocks follow every PRN record (gnssRecord.hpp), and
    // the correlation array is [rows][n_chan][n_elem].
    _n_elements = config.get<int>(unique_name, "n_elements");
    if (_n_elements < 1) {
        FATAL_ERROR("GnssGpuRecordAssemble: n_elements {:d} must be >= 1", _n_elements);
        return;
    }
    // The element the record header's correlation slots carry. The broker closes its DLL and
    // carrier loops on those slots, so this must be a healthy, high-gain feed.
    _reference_element = config.get_default<int>(unique_name, "reference_element", 0);
    if (_reference_element < 0 || _reference_element >= _n_elements) {
        FATAL_ERROR("GnssGpuRecordAssemble: reference_element {:d} outside [0, n_elements={:d})",
                    _reference_element, _n_elements);
        return;
    }
    // Self-calibrated element sum (see the header and gnssElemCal.hpp).
    _elem_sum = config.get_default<bool>(unique_name, "elem_sum", false);
    _elem_sum_tau_s = config.get_default<double>(unique_name, "elem_sum_tau_s", 5.0);
    // Hold vs adapt. Each per-PRN cal learns its element phases from that PRN's own data, so a
    // satellite near boresight, whose leakage dominates every weak satellite's per-element
    // despread, teaches every cal its phases and the array gain is lost. With geometry steered
    // and a shared instrumental prior seeded, adapt = false freezes the weights at the prior; a
    // shadow cal keeps learning per PRN so the capture stays visible (/get_elem_cal).
    _elem_adapt = config.get_default<bool>(unique_name, "elem_sum_adapt", true);
    // Shared model (see the header): one instrument per pol plus one inter-pol coefficient per
    // PRN. Shared implies held: a learner that also combined would defeat the point.
    _elem_shared = config.get_default<bool>(unique_name, "elem_sum_shared", false);
    _elem_shared_tau_s = config.get_default<double>(unique_name, "elem_sum_shared_tau_s", 300.0);
    _elem_pol_tau_s = config.get_default<double>(unique_name, "elem_sum_pol_tau_s", 3.0);
    _elem_shared_freeze_deg =
        config.get_default<double>(unique_name, "elem_sum_shared_freeze_deg", 6.0);
    _chan_norm = config.get_default<bool>(unique_name, "chan_noise_norm", false);
    if (_chan_norm) {
        // The frame inputs that are this stage's elements.
        _chan_norm_elems = config.get<std::vector<int>>(unique_name, "chan_noise_norm_elems");
        if (_chan_norm_elems.empty()) {
            FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: chan_noise_norm needs a non-empty "
                        "chan_noise_norm_elems",
                        unique_name);
            return;
        }
        INFO("GnssGpuRecordAssemble[{:s}]: chan_noise_norm ON -- channels weighted by 1/rms of "
             "their measured power over {:d} frame inputs ({:d}..{:d}; GnssChordVoltageTap "
             "band-power pass)",
             unique_name, (int)_chan_norm_elems.size(), _chan_norm_elems.front(),
             _chan_norm_elems.back());
    }
    if (_elem_shared) {
        _elem_adapt = false;
        INFO("GnssGpuRecordAssemble[{:s}]: SHARED element model ON (consensus tau {:.0f} s, "
             "inter-pol tau {:.1f} s); per-PRN weights held to it",
             unique_name, _elem_shared_tau_s, _elem_pol_tau_s);
    }
    // Fleet reference (see the header): flat [re0, im0, re1, im1, ...] over n_elements, the
    // generator's copy of one band's entry in the --elem-shared-ref snapshot.
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
        else if (mode == "live")
            _fleet_mode = 1;
        else {
            FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: elem_sum_shared_ref_mode '{:s}' is not "
                        "off | live",
                        unique_name, mode);
            return;
        }
        if (_fleet_mode != 0)
            INFO("GnssGpuRecordAssemble[{:s}]: fleet reference {:s} ({:s}), slew {:.2f} deg/s, "
                 "min sim {:.2f}",
                 unique_name, mode, _g_fleet_ref.empty() ? "NONE configured: own pin" : "loaded",
                 _fleet_slew_rad_s * 180.0 / M_PI, _fleet_min_sim);
    }
    // Element steering: geometric per-(satellite, channel, element) phasors. Off unless
    // elem_positions_enu is given (flat [n_elements][3], metres ENU of any fixed array point).
    // The positions are re-referenced to the configured reference element, so its phasor is
    // unity and steering leaves the header slots and the ADR observable unchanged. A live
    // /set_reference_element does not re-reference them. The sign is a measured config parameter
    // (gnssElemSteer.hpp).
    {
        auto pos = config.get_default<std::vector<double>>(unique_name, "elem_positions_enu", {});
        const double st_sign = config.get_default<double>(unique_name, "elem_steer_sign", 1.0);
        const double st_hold = 120.0; // how long a post's geometry stays valid, s
        if (!pos.empty()) {
            if ((int)pos.size() != 3 * _n_elements)
                FATAL_ERROR("GnssGpuRecordAssemble: elem_positions_enu has {:d} values, "
                            "want 3*n_elements = {:d}",
                            (int)pos.size(), 3 * _n_elements);
            // Read channel_ids locally: _spec_freq_ids is not set until further down.
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
            INFO("GnssGpuRecordAssemble[{:s}]: element steering READY ({:d} elements, "
                 "{:d} channels, sign {:+.0f}) -- awaiting /set_sat_geometry posts",
                 unique_name, _n_elements, (int)_st_fids.size(), st_sign);
        }
    }
    const int n = (int)_prns.size();
    // The record width is a C++ constant but the frame is sized in yaml (n_prn * record_floats *
    // sizeof(float)), with nothing linking the two: refuse an under-sized frame here rather than
    // write past its end.
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
    // #190 lane directions (see the header): off by default; the generator's
    // --elem-dir-publish all turns it on with GnssN2Project's source lanes.
    {
        const std::string pub =
            config.get_default<std::string>(unique_name, "elem_dir_publish", "off");
        _lane_on = (pub == "all");
        const double tau = config.get_default<double>(unique_name, "elem_dir_tau_s", 0.5);
        _lane_every = std::max(1, config.get_default<int>(unique_name, "elem_dir_every", 4));
        _lane_coh_min = config.get_default<double>(unique_name, "elem_dir_coh_min", 0.05);
        _lane_fids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});
        _lane_group = unique_name.substr(0, unique_name.find('_'));
        const int np = (int)_prns.size();
        _lane = gnss::LaneDirection(_n_elements, (int)_lane_fids.size(), np, tau);
        _lane_scratch.assign((size_t)_n_elements, std::complex<double>(0.0, 0.0));
        _lane_t_prev.assign((size_t)np, 0.0);
        _lane_prn.assign((size_t)np, 0);
        _lane_pub.assign((size_t)np, 0);
        _lane_cnt.assign((size_t)np, 0);
        _lane_owner.assign((size_t)np, std::string());
        _lane_xfer_fids =
            config.get_default<std::vector<int>>(unique_name, "lane_transfer_freq_ids", {});
        for (const int fid : _lane_xfer_fids)
            if (std::find(_lane_fids.begin(), _lane_fids.end(), fid) != _lane_fids.end())
                FATAL_ERROR("lane_transfer_freq_ids: {:d} is one of channel_ids", fid);
        _lane_ok.assign(_lane_fids.size(), 0);
        _lane_pubdir.assign(_lane_fids.size() * (size_t)_n_elements,
                            std::complex<double>(0.0, 0.0));
        _lane_shift.assign((size_t)_n_elements, gnss::ElemSteer::cf(1.0f, 0.0f));
        if (_lane_on && _lane_fids.empty()) {
            WARN("elem_dir_publish all needs channel_ids: lane directions OFF");
            _lane_on = false;
        }
        if (_lane_on)
            INFO("lane directions: publishing every running non-probe slot's prompt-row "
                 "direction per channel to the board group {:s}, tau {:.2f} s, every {:d} "
                 "records, coherence >= {:.3f}, transferred to {:d} more channels",
                 _lane_group, tau, _lane_every, _lane_coh_min, (int)_lane_xfer_fids.size());
    }
    // Bright-satellite projection (see the header and gnssProjSubspace.hpp). Needs channel_ids
    // (the board is keyed on freq_id) and elem_sum (the shadow learners). Off by default;
    // /set_elem_proj changes the mode live.
    {
        auto fids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});
        const std::string mode_s =
            config.get_default<std::string>(unique_name, "elem_proj_mode", "off");
        _proj_mode = (mode_s == "live") ? 2 : (mode_s == "shadow") ? 1 : 0;
        _proj_deg = config.get_default<double>(unique_name, "elem_proj_deg", 4.0);
        _proj_group = unique_name.substr(0, unique_name.find('_'));
        // The constellation, from the generator's band tokens in the stage name.
        auto has = [&](const char* t) { return unique_name.find(t) != std::string::npos; };
        _proj_sys = (has("_e5a") || has("_e5b") || has("_e6") || has("_e1"))         ? 'E'
                    : (has("_b2a") || has("_b2b") || has("_b3i") || has("_b1"))      ? 'C'
                    : (has("_l1of") || has("_l2of") || has("_l2oc") || has("_l3oc")) ? 'R'
                                                                                     : 'G';
        _proj_ready = _elem_sum && !fids.empty();
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
                 "needs elem_sum and channel_ids -- OFF",
                 unique_name, mode_s);
            _proj_mode = 0;
        }
    }
    _phi.assign(n, 0.0);
    _phi_cyc.assign(n, 0.0);
    _phi_cmd_prev.assign(n, 0.0);
    _phi_cmd_ok.assign(n, 0);
    _fnco_prev.assign(n, 0.0);
    _fcar_prev_ok.assign(n, 0);
    _a_prev_ok.assign(n, 0);
    _elem_prev_ok.assign(n, 0);
    _wstart_prev.assign(n, 0);
    // Per-channel prompt spectrum (see the header), enabled by channel_ids: the per-GPU freq_id
    // list the despread runs, which labels each channel with its sky frequency. Without it
    // nothing accumulates, and /get_spectrum answers with an error rather than a 404 so a
    // mis-wired poller sees why.
    _spec_freq_ids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});

    // Per-channel comb export (gnssRecord.hpp): the unsummed per-channel taps appended after the
    // PRN records. The cross-channel sum destroys the frequency axis a delay lives on; the comb
    // keeps it. channel_ids is required: a delay fit across mislabelled frequencies returns a
    // confident wrong tau.
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
        // Window index = floor(wstart / spectrum_window_samples) on the F-engine sample clock, so
        // every instance assigns a record to the same window without coordination. The length
        // is in samples so the division is exact integer arithmetic, and every instance must be
        // given the same value (the generator derives it). A record is assigned whole by its
        // wstart; a length that is a whole number of records keeps the per-window count constant.
        _spec_win_samples = (int64_t)config.get<double>(unique_name, "spectrum_window_samples");
        if (_spec_win_samples <= 0) {
            FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: spectrum_window_samples {:d} must be > 0",
                        unique_name, (long long)_spec_win_samples);
            return;
        }
        const int depth = config.get_default<int>(unique_name, "spectrum_ring_depth", 8);
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
             (int)nc, _spec_freq_ids.front(), _spec_freq_ids.back(), (long long)_spec_win_samples,
             (double)_spec_win_samples / _sample_rate, (int)_spec_ring.size());
        _spec_scratch.assign((size_t)_n_elements, {0.0, 0.0});
    }

    // Beam cube (see the header): the (channel x element) axis, un-collapsed. Needs
    // channel_ids (a frequency per bin).
    _cube_on = config.get_default<bool>(unique_name, "beam_cube", false) && !_spec_freq_ids.empty();
    if (_cube_on) {
        const int nc = (int)_spec_freq_ids.size();
        // One bin per channel.
        const int bin_width = config.get_default<int>(unique_name, "beam_cube_bin_width", 0);
        if (bin_width > 1) {
            FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: beam_cube_bin_width {:d} is not supported: "
                        "the cube has one bin per channel",
                        unique_name, bin_width);
            return;
        }
        _cube_bins = nc;

        // Window length in F-engine samples (wstart's unit; one hop is fft_length samples), for
        // the same reason as spectrum_window_samples: the index is an exact integer division,
        // and every instance must be given the same value (the generator derives it). The
        // default, 96 records x 2048 hops x 16384 samples = 1.00663 s, is a whole number of
        // records and of 8192-hop frames, so no record is split; at a 195312.5 Hz hop rate no
        // exact 1 s window exists.
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
        // Push leg (see the header). Frames are addressed <short hostname>/<stage> by default: a
        // stage name is unique only within a node, and one bufferRecv receives the whole fleet.
        {
            char hn[64] = {0};
            gethostname(hn, sizeof(hn) - 1);
            char* dot = std::strchr(hn, '.');
            if (dot)
                *dot = '\0'; // short host name, without the domain
            _cube_chain = config.get_default<std::string>(unique_name, "beam_cube_chain",
                                                          std::string(hn) + "/" + unique_name);
        }
        _cube_gpu = config.get_default<int>(unique_name, "beam_cube_gpu", -1);
        _cube_max_bins = std::max(
            _cube_bins, config.get_default<int>(unique_name, "beam_cube_max_bins", _cube_bins));
        _cube_max_prn = std::max(n, config.get_default<int>(unique_name, "beam_cube_max_prn", n));
        const std::string cbuf = config.get<std::string>(unique_name, "cube_buf");
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
            WARN("GnssGpuRecordAssemble[{:s}]: cube_buf {:s} has no metadata_pool. Frames "
                 "will carry no metadata; a bufferSend on this buffer drops them (it cannot "
                 "send a frame without one) and a bufferRecv far side expects the pool's "
                 "type. Give it metadata_pool: gnss_pool.",
                 unique_name, cbuf);
        INFO("GnssGpuRecordAssemble[{:s}]: beam-cube PUSH leg on -> {:s} ({:d} B/frame, "
             "~{:.2f} MB/s at this window length); backpressure DROPS and the loss is "
             "counted into the next frame",
             unique_name, cbuf, need, need / ((double)_cube_win_samples / _sample_rate) / 1e6);
        INFO("GnssGpuRecordAssemble[{:s}]: BEAM CUBE on -- {:d} PRN x {:d} channel(s) x {:d} "
             "element(s); coherent + incoherent, "
             "windows of {:d} samples ({:.5f} s), ring depth {:d}; "
             "{:d} cell(s), {:.1f} MB",
             unique_name, n, nc, _n_elements, (long long)_cube_win_samples,
             (double)_cube_win_samples / _sample_rate, depth, ncell,
             depth * (3.0 * ncell + 2.0 * nbin) * sizeof(double) / 1e6);
    } else if (config.get_default<bool>(unique_name, "beam_cube", false)) {
        WARN("GnssGpuRecordAssemble[{:s}]: beam_cube requested but this config has no "
             "channel_ids (no frequency axis) -- the cube is OFF.",
             unique_name);
    }

    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_spectrum",
        std::bind(&GnssGpuRecordAssemble::spectrum_callback, this, _1));
    if (_elem_sum)
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_elem_gain",
            std::bind(&GnssGpuRecordAssemble::set_elem_gain_callback, this, _1, _2));
    // Registered whether or not steering is configured: the broker posts its sky to every
    // assembler, and a 404 every 30 s would look like a dead stage in its log. Unsteered, the
    // post is acknowledged and dropped, "_bore" included.
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
    // Registered with elem_sum off too: the header slots carry the reference element.
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_reference_element",
        std::bind(&GnssGpuRecordAssemble::set_reference_element_callback, this, _1, _2));
}

void GnssGpuRecordAssemble::set_sat_geometry_callback(kotekan::connectionInstance& conn,
                                                      nlohmann::json& request) {
    // {"<prn>": [az_deg, el_deg, az_rate_dps, el_rate_dps, t_utc], ...}. The rates and epoch
    // let the steering follow the satellite between the ~30 s posts (gnssElemSteer.hpp); a bare
    // [az, el] is held as a snapshot. PRNs this stage does not hold are skipped.
    int n_up = 0;
    if (!_steer.enabled()) {
        conn.send_json_reply(nlohmann::json{{"updated", 0}, {"steering", "off"}});
        return;
    }
    try {
        const double now_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        // "_bore": [sep_deg, t_utc], the broker's nearest-to-boresight separation pooled over
        // every constellation (a satellite of another system leaks into this band just the
        // same). It gates the shared model's learning (see _elem_shared_freeze_deg).
        auto bi = request.find("_bore");
        if (bi != request.end() && bi->is_array() && !bi->empty()) {
            _bore_sep_deg.store((*bi)[0].get<double>());
            _bore_post_t.store(now_s);
        }
        std::lock_guard<std::mutex> lk(_steer_mtx);
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
        // Record 0 speaks for the frame: the producer applies swaps at a frame boundary, so
        // every record in it carries the same slot -> PRN map.
        const int claimed = (int)pctl[(size_t)p].prn;
        if (claimed == _prns[(size_t)p])
            continue;
        const int was = _prns[(size_t)p];
        _prns[(size_t)p] = claimed;
        ++n_changed;

        // Cold-reset everything keyed to this slot. Each describes the satellite that left, and
        // carried across it would be attributed to the new one, silently: a warm cal or a phase
        // history looks plausible. main_thread's reference-element swap also rebuilds the cals
        // cold.
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
        // The steer table is keyed by slot: the departed satellite's phasors must not steer the
        // newcomer for up to hold_s. Geometry for the new PRN arrives with the next post.
        if (_steer.enabled()) {
            std::lock_guard<std::mutex> lk(_steer_mtx);
            _steer.invalidate(p);
        }
        _phi[(size_t)p] = 0.0;
        _phi_cyc[(size_t)p] = 0.0;
        _phi_cmd_prev[(size_t)p] = 0.0;
        _phi_cmd_ok[(size_t)p] = 0;
        _fnco_prev[(size_t)p] = 0.0;
        _fcar_prev_ok[(size_t)p] = 0;
        _a_prev_ok[(size_t)p] = 0;
        _elem_prev_ok[(size_t)p] = 0;
        _wstart_prev[(size_t)p] = 0;
        // The spectrum ring accumulates per slot: clear this slot in every window, or one
        // straddling the swap sums two satellites into one row. nrec = 0 makes the row read as
        // absent rather than as a quiet satellite.
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
        // The beam cube likewise, in every window: carried across, the departed satellite's
        // power would be attributed to the newcomer, and its phi0 refers to the old satellite's
        // NCO history. Zero the sums and the weights together: a nonzero weight over a zeroed
        // sum reads as a measurement of no power.
        if (_cube_on) {
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
        // Reference-element swap, applied at the frame boundary so every record in a frame
        // sees one reference. The cals are rebuilt cold: every learned gain and any injected
        // prior is phase-anchored to the old reference and does not transfer. The header rides
        // the new reference element while the cals re-warm (~3 tau); downstream sees one phase
        // step, as on a fresh acquisition.
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
        // A newly injected gain prior (/set_elem_gain) seeds every PRN's cal once, before this
        // frame's records are combined. seed() also stores it, so each later reset() re-applies
        // it rather than going cold.
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
        if (hdr.n_rows_spec != ROWS_PLAIN) {
            FATAL_ERROR("GnssGpuRecordAssemble: frame n_rows_spec {:d} != ROWS_PLAIN ({:d})",
                        hdr.n_rows_spec, ROWS_PLAIN);
            return;
        }
        const int64_t* winstart = (const int64_t*)(in + off_winstart());
        const PrnCtl* pctl = (const PrnCtl*)(in + off_prnctl());
        // Adopt the producer's slot -> PRN map before anything in this frame is labelled or
        // accumulated.
        follow_frame_prns(pctl, n_prn);
        const double* corr = (const double*)(in + off_corr(n_prn)); // double2 rows
        // The energy block follows the corr block, whose size scales with the element count:
        // pass the writer's n_elem, or every energy read lands inside the corr block (and a zero
        // P energy makes the combiner treat the record as inactive).
        const double* energy =
            (const double*)(in + off_energy(n_prn, n_chan, ROWS_PLAIN, _n_elements));
        // Per-channel weights for the cross-channel sums (gnssChanPower.hpp): 1/rms of each
        // channel's measured power, scaled to mean 1, so a channel enters by its signal-to-noise
        // and not by the F-engine digital gain. Unit weights until every channel is measured.
        _chan_w.assign((size_t)n_chan, 1.0);
        bool chan_applied = false;
        if (_chan_norm && (int)_spec_freq_ids.size() == n_chan) {
            _chan_pw.resize((size_t)n_chan);
            gnss::ChanPower::instance().lookup(_spec_freq_ids.data(), n_chan,
                                               _chan_norm_elems.data(),
                                               (int)_chan_norm_elems.size(), _chan_pw.data());
            chan_applied = gnss::chan_norm_weights(_chan_pw.data(), n_chan, _chan_w.data());
            if (!chan_applied)
                ++_chan_norm_miss;
        }
        _chan_norm_applied.store(chan_applied);
        if (chan_applied) {
            std::lock_guard<std::mutex> lk(_chan_w_mtx);
            _chan_w_pub = _chan_w;
        }

        // Record UTC is utc0 + wstart / sample_rate. utc0 comes from the producer's
        // frame0_utc (the telescope's GPS time0, or the config override).
        if (!(hdr.utc0 > 0.0)) {
            FATAL_ERROR("GnssGpuRecordAssemble: the producer sent no frame0_utc (utc0 {:g}): "
                        "give the telescope GPS time or set frame0_utc on the producer",
                        hdr.utc0);
            return;
        }

        for (int r = 0; r < hdr.n_rec && !stop_thread; ++r) {
            float* out = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            // Zero the whole comb block first: the per-PRN zeroing below covers only
            // record_stride floats, and a PRN that does not run, or a channel outside its
            // covering mask, must not carry the previous record's value.
            if (_chan_export && out != nullptr)
                std::fill(out + gnss::chan_block_offset(n_prn, _n_elements),
                          out + gnss::frame_floats(n_prn, _n_elements, n_chan), 0.0f);
            if (out == nullptr)
                return;
            const int64_t wstart = winstart[r];
            const double utc = hdr.utc0 + (double)wstart / _sample_rate;

            // Bright-satellite projection, per record (see the header): feed the source trackers
            // from this record's raw rows, build the per-channel basis and publish it to the
            // siblings. k = 0 unless armed and a source is in force.
            if (_proj_ready && _proj_mode.load() != 0)
                proj_prepare_record(corr, &pctl[(size_t)r * n_prn], n_chan, _n_elements, wstart,
                                    utc,
                                    std::chrono::duration<double>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
            else
                _proj_k_rec = 0;
            int proj_src = 0; // the OR of the basis sources over this instance's channels
            if (_proj_k_rec > 0)
                for (int src : _proj_src_ch)
                    proj_src |= src;

            for (int p = 0; p < n_prn; ++p) {
                const int rec_stride = gnss::record_stride(_n_elements);
                float* rec = out + (size_t)p * rec_stride;
                // Zero the whole record, element blocks included: a PRN that does not run this
                // window must not leave a previous window's values behind.
                for (int f = 0; f < rec_stride; ++f)
                    rec[f] = 0.0f;
                rec[gnss::REC_PROJ_COST] = -1.0f; // set below when the projection runs
                rec[gnss::REC_PROJ_K] = (float)_proj_k_rec;
                rec[gnss::REC_PROJ_SRC] = (float)proj_src;
                const PrnCtl& c = pctl[(size_t)r * n_prn + p];
                // The record's own PRN stamp wins, so slot 0 stays right even if a producer
                // moves a slot mid-frame; 0 = not stamped, use the list.
                rec[0] = (float)((c.prn > 0) ? (int)c.prn : _prns[p]);
                *reinterpret_cast<double*>(rec + gnss::RECORD_UTC_SLOT) = utc;
                if (!c.run) {
                    if (_lane_on) {
                        if (_lane_pub[p])
                            gnss::ProjBoard::instance().retire(_lane_group, _lane_owner[p]);
                        _lane_pub[p] = 0;
                        _lane_prn[p] = 0;
                        _lane_t_prev[p] = 0.0;
                        _lane.reset(p);
                    }
                    _a_prev_ok[p] = 0;
                    _elem_prev_ok[p] = 0;
                    _fcar_prev_ok[p] = 0;
                    _phi_cyc[p] = 0.0;
                    _phi_cmd_ok[p] = 0;
                    continue;
                }
                // Cross-channel sum over the covering mask, per correlator row: E, P, L and
                // P_HEAD (the prompt's head segment, gnssRecord.hpp slots 16-18). PrnCtl::job0
                // already carries the per-spec row stride, so the row is job0 + t.
                //
                // The correlation array is [rows][n_chan][n_elem] and the sum runs per antenna.
                // The energies are [rows][n_chan]: one replica is correlated against every
                // antenna, so they are summed once and live in the record header only.
                const int n_e = _n_elements;
                const int ref_e = _reference_element;
                std::complex<double> g3[ROWS_PLAIN]; // reference element: header, NCO/gain state
                double e3[ROWS_PLAIN];               // element-independent
                _g_elem.assign((size_t)ROWS_PLAIN * n_e, std::complex<double>(0.0, 0.0));
                // Live projection (see the header): a non-source slot's rows are projected in
                // place in the input frame before anything reads them, so the sum below, the
                // taps, the cube and the element blocks all see the same projected data.
                if (_proj_k_rec > 0 && _proj_mode.load() == 2 && !_proj_isB[p])
                    proj_slot_inplace(const_cast<double*>(corr), &c, n_chan, n_e, ROWS_PLAIN);
                // Steer this satellite's elements if its geometry is fresh. The slot's table is
                // copied under the lock and read from the copy, so a concurrent
                // /set_sat_geometry cannot rewrite it mid-combine.
                bool steered = false;
                if (_steer.enabled()) {
                    if (_steer.n_chan() != n_chan) {
                        // The table was built from the config's channel_ids and the frame
                        // disagrees: steering would apply one channel's phasor to another's
                        // data. Refuse, and warn once.
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
                            // Evaluate the track at this record's time (the data clock:
                            // records lag the wall clock by the pipeline depth).
                            _steer.refresh((int)p, utc);
                            _steer_buf.resize((size_t)n_chan * n_e);
                            _steer.copy_slot((int)p, _steer_buf.data());
                            steered = true;
                        }
                    }
                }
                // #190: this record's steered prompt row feeds the slot's lane direction
                // (one EMA per channel); a PRN change on the slot resets it.
                const int lane_prn = (c.prn > 0) ? (int)c.prn : _prns[p];
                const bool lane_rec = _lane_on && steered && !_proj_isProbe[(size_t)p]
                                      && (int)_lane_fids.size() == n_chan;
                double lane_dt = 0.0;
                if (lane_rec) {
                    if (_lane_prn[p] != lane_prn) {
                        if (_lane_pub[p])
                            gnss::ProjBoard::instance().retire(_lane_group, _lane_owner[p]);
                        _lane_pub[p] = 0;
                        _lane.reset(p);
                        _lane_prn[p] = lane_prn;
                        _lane_owner[p] = unique_name + "/lane/" + std::to_string(lane_prn);
                        _lane_t_prev[p] = 0.0;
                    }
                    lane_dt = _lane_t_prev[p] > 0.0 ? utc - _lane_t_prev[p] : 0.0;
                    _lane_t_prev[p] = utc;
                }
                for (int t = 0; t < ROWS_PLAIN; ++t) {
                    const size_t row = (size_t)(c.job0 + t) * n_chan;
                    double e = 0.0;
                    const bool lane_row = lane_rec && t == ROW_P;
                    for (int ch = 0; ch < n_chan; ++ch)
                        if ((c.chan_mask >> ch) & 1ULL) {
                            const double w = _chan_w[ch]; // 1 unless chan_noise_norm
                            e += w * energy[row + ch];
                            const size_t base = (row + ch) * n_e;
                            const gnss::ElemSteer::cf* st =
                                steered ? &_steer_buf[(size_t)ch * n_e] : nullptr;
                            for (int el = 0; el < n_e; ++el) {
                                std::complex<double> v(corr[2 * (base + el)],
                                                       corr[2 * (base + el) + 1]);
                                if (st)
                                    v *= std::complex<double>(st[el].real(), st[el].imag());
                                _g_elem[(size_t)t * n_e + el] += w * v;
                                if (lane_row)
                                    _lane_scratch[(size_t)el] = v;
                            }
                            if (lane_row)
                                _lane.push(p, ch, _lane_scratch.data(), lane_dt);
                        }
                    g3[t] = _g_elem[(size_t)t * n_e + ref_e];
                    e3[t] = e;
                }
                // #190: publish the slot's lane directions once per elem_dir_every records,
                // per covering channel whose EMA is warm and coherent, in the raw frame.
                if (lane_rec && (++_lane_cnt[p] % _lane_every) == 0) {
                    gnss::ProjBoard& board = gnss::ProjBoard::instance();
                    const double now_s =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
                    std::fill(_lane_ok.begin(), _lane_ok.end(), 0);
                    for (int ch = 0; ch < n_chan; ++ch) {
                        if (!((c.chan_mask >> ch) & 1ULL) || !_lane.warm(p, ch))
                            continue;
                        const double coh = _lane.coherence(p, ch);
                        std::complex<double>* dir = &_lane_pubdir[(size_t)ch * n_e];
                        if (coh < _lane_coh_min
                            || !_lane.direction(p, ch, &_steer_buf[(size_t)ch * n_e], dir))
                            continue;
                        _lane_ok[(size_t)ch] = 1;
                        gnss::ProjEntry e;
                        e.owner = _lane_owner[p];
                        e.src = 2;
                        e.t_pub = now_s;
                        e.power = _lane.power(p, ch);
                        e.coh = coh;
                        e.n = n_e;
                        e.k = 1;
                        e.q.reserve((size_t)n_e);
                        for (int el = 0; el < n_e; ++el)
                            e.q.emplace_back((float)dir[el].real(), (float)dir[el].imag());
                        board.publish(_lane_group, _lane_fids[(size_t)ch], e);
                        _lane_pub[p] = 1;
                    }
                    double los[3];
                    bool have_los = false;
                    if (!_lane_xfer_fids.empty()) {
                        std::lock_guard<std::mutex> lk(_steer_mtx);
                        have_los = _steer.direction((int)p, utc, los);
                    }
                    if (have_los) {
                        for (const int fid : _lane_xfer_fids) {
                            int src = -1;
                            for (int ch = 0; ch < n_chan; ++ch)
                                if (_lane_ok[(size_t)ch]
                                    && (src < 0
                                        || std::abs(_lane_fids[(size_t)ch] - fid)
                                               < std::abs(_lane_fids[(size_t)src] - fid)))
                                    src = ch;
                            if (src < 0)
                                break;
                            _steer.shift(los, (fid - _lane_fids[(size_t)src]) * 0.1953125e6,
                                         _lane_shift.data());
                            const std::complex<double>* dir = &_lane_pubdir[(size_t)src * n_e];
                            gnss::ProjEntry e;
                            e.owner = _lane_owner[p];
                            e.src = 2;
                            e.t_pub = now_s;
                            e.power = _lane.power(p, src);
                            e.coh = _lane.coherence(p, src);
                            e.n = n_e;
                            e.k = 1;
                            e.q.reserve((size_t)n_e);
                            for (int el = 0; el < n_e; ++el) {
                                const gnss::ElemSteer::cf sh = _lane_shift[(size_t)el];
                                const std::complex<double> v =
                                    dir[el] * std::complex<double>(sh.real(), -sh.imag());
                                e.q.emplace_back((float)v.real(), (float)v.imag());
                            }
                            board.publish(_lane_group, fid, e);
                        }
                    }
                }
                // Shadow projection (see the header): the same prompt, projected per channel
                // before steering and summing, into _g_proj for the projected learner and the
                // capture diagnostics only. Nothing live changes.
                const bool proj_this = _proj_k_rec > 0 && _proj_mode.load() == 1 && !_proj_isB[p];
                if (proj_this)
                    proj_slot_shadow(corr, &c, n_chan, n_e, steered);
                // Self-calibrated element sum (see the header). Once this PRN's cal is warm,
                // every header row (same weights: same antennas) is the calibrated weighted mean
                // instead of the bare reference element: reference-anchored phase, one-element
                // scale, noise down ~sqrt(N_healthy). The cal updates after the combine, from the
                // prompt row, so this record's header weights come from earlier records only.
                // The element blocks are untouched: they are the per-antenna measurement.
                std::complex<double> g_sky(0.0, 0.0); // split-aperture prompt (slots 24/25)
                if (_elem_sum) {
                    gnss::ElemCal& ec = _cal[p];
                    // The cal holds across a re-anchor (reanchored 1): the per-element cal is
                    // E_e*conj(E_ref), from which the common carrier cancels.
                    // _anchor_warned is not re-armed on a re-anchor either, so the warning cannot
                    // repeat at the re-anchor rate.
                    // Shared model: this record's weights come from the model as it stood
                    // (causal), installed as held weights so combine/combine_split run unchanged.
                    if (_elem_shared.load())
                        shared_hold(p);
                    if (ec.warm()) {
                        for (int t = 0; t < ROWS_PLAIN; ++t)
                            g3[t] = ec.combine(&_g_elem[(size_t)t * n_e]);
                        // The deep fold's input: the same prompt by split aperture, one half of
                        // the elements derotated by the other half's phase (combine_split). A
                        // separate slot: the header keeps the raw phase, which is the ADR
                        // observable (gnssRecord.hpp REC_SKY_RE).
                        g_sky = ec.combine_split(&_g_elem[(size_t)1 * n_e]);
                    }
                    if (_elem_prev_ok[p] && wstart > _wstart_prev[p]) {
                        const double dt_s = (double)(wstart - _wstart_prev[p]) / _sample_rate;
                        if (_elem_adapt) {
                            ec.update(&_g_elem[(size_t)1 * n_e], dt_s);
                        } else {
                            // Held: the live weights stay as the prior, the shared model or the
                            // last adapting record left them. The shadow keeps learning from this
                            // PRN alone, and its agreement with the held weights is the capture
                            // detector: sim ~1 = this satellite's data still give the same
                            // phases; sim -> 0 = something else is teaching it (a transit).
                            gnss::ElemCal& sh = _cal_shadow[p];
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
                        // The projected learner (see the header): fed the projected prompt
                        // while a basis is in force (in live mode the prompt itself is) and the
                        // plain prompt otherwise, so out of transit it is the shadow's twin and
                        // in transit their difference is the capture. Per slot: cap_plain and
                        // cap_proj = how much of each learner's weight vector lies in the
                        // interferer subspace, b_cos2 = the same for the live weights (the
                        // projection's cost on this satellite), sim_pp = the learners' agreement.
                        if (_proj_ready && _proj_mode.load() != 0) {
                            gnss::ElemCal& pj = _cal_proj[p];
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
                rec[1] = c.fcar_report;
                {
                    double cp = c.cp_seed;
                    if (_rec_cp_mod_chips > 0.0) {
                        cp = std::fmod(cp, _rec_cp_mod_chips);
                        if (cp < 0.0)
                            cp += _rec_cp_mod_chips;
                    }
                    rec[2] = (float)cp;
                }
                rec[6] = c.n_owned;
                rec[gnss::REC_E_RE] = (float)g3[0].real();
                rec[gnss::REC_E_IM] = (float)g3[0].imag();
                rec[gnss::REC_E_ENERGY] = (float)e3[0];
                rec[gnss::REC_L_RE] = (float)g3[2].real();
                rec[gnss::REC_L_IM] = (float)g3[2].imag();
                rec[gnss::REC_L_ENERGY] = (float)e3[2];

                // Carrier NCO. reanchored 1 is a fresh acquisition and resets the phase history;
                // f_nco changes the slope of phi, never steps it. reanchored 3 is a continuous
                // re-pin whose step the producer carries in c.dcyc (gnssGpuChain.hpp
                // PrnCtl::dcyc).
                const bool repin = (c.reanchored == 3);
                if (c.reanchored == 1 || (repin && !_fcar_prev_ok[p])) {
                    // Fresh acquisition, or a re-pin with no previous fcar: no phase history to
                    // keep. Break the arc.
                    _phi[p] = 0.0;
                    _phi_cyc[p] = 0.0;
                    _a_prev_ok[p] = 0;
                    _phi_cmd_ok[p] = 0;
                } else if (repin) {
                    // Phase-continuous re-pin. Re-pinning f_ref steps the absolutely anchored
                    // replica phase by df*t_abs, many cycles at days of uptime. An NCO absorbs
                    // any constant phase, so fold the step into both _phi (the rotation) and
                    // _phi_cyc (the export): the derotated correlation stays continuous through
                    // the re-pin, and the commanded phase fcar*t - phi_cyc is unchanged. A re-pin
                    // is then free, so a producer can re-pin every record and the within-record
                    // decoherence that grows with anchor age, ~(dop_rate*age*t_rec)^2, never
                    // builds up.
                    _phi_cyc[p] += c.dcyc;
                    _phi[p] = std::remainder(_phi[p] + 2.0 * M_PI * c.dcyc, 2.0 * M_PI);
                }
                // Diagnostic pass-through: the values the despread actually handed the kernel
                // (gnssRecord.hpp REC_ANG0, REC_PHI_DDOP). Written for every record, including
                // the first of an arc, so outside the dt guard below.
                rec[gnss::REC_ANG0] = (float)c.ang0;
                rec[gnss::REC_PHI_DDOP] = (float)c.phi_ddop;
                const double dt = (double)(wstart - _wstart_prev[p]) / _sample_rate;
                if (_a_prev_ok[p] && dt > 0.0) {
                    // Commanded-trim increment (slot 19): the trim the producer applied
                    // (PrnCtl::ctrim_hz), integrated over this record so downstream can subtract
                    // the loop's transients from the ADR exactly. It cannot be reconstructed
                    // from f_nco, fcar_report and fcar (see PrnCtl::ctrim_hz).
                    rec[gnss::REC_TRIM_INC] = (float)(c.ctrim_hz * dt);
                    // Charge the midpoint of the previous and current f_nco over dt. Either
                    // endpoint alone leaves a |dctrim|*dt residue per command step, because a
                    // step also moves the record's phase centroid (the prompt's phase sits at the
                    // window centre, the argument anchor at its start).
                    // REC_TRIM_INC above keeps this record's own ctrim on purpose: it states
                    // what the producer applied to this record.
                    _phi[p] += 2.0 * M_PI * 0.5 * (_fnco_prev[p] + c.f_nco) * dt;
                    // Keep the rotation phase bounded but track the NCO phase unwrapped, in
                    // cycles, for the carrier-phase export: remainder() slides phi by whole
                    // turns, which a rotation cannot see but a per-record increment would read
                    // as a +-1 cycle slip.
                    _phi[p] = std::remainder(_phi[p], 2.0 * M_PI);
                    _phi_cyc[p] += 0.5 * (_fnco_prev[p] + c.f_nco) * dt;
                }
                _fnco_prev[p] = c.f_nco; // after the charge: the next midpoint uses it
                const std::complex<double> rot = std::polar(1.0, -_phi[p]);
                // Publish the rotation's phase (gnss::REC_PHI0): `rot` is applied to slots 3/4
                // and every comb column, and _phi[p] has an arbitrary per-instance origin, so a
                // consumer needs it to put instances on one common reference. The spectrum and
                // cube windows publish the same as phi0.
                rec[gnss::REC_PHI0] = (float)_phi[p];
                const std::complex<double> g_corr = g3[1] * rot;
                rec[3] = (float)g_corr.real();
                rec[4] = (float)g_corr.imag();
                rec[5] = (float)e3[1];
                // Per-channel prompt spectrum (see the header): the prompt row per channel,
                // before the cross-channel sum, combined over elements by the cal when warm, else
                // the reference element, and rotated by the same `rot` as the record's prompt.
                // In adapt mode the cal here already includes this record's update, which the
                // header combine did not. One common phase per record stops the sum winding with
                // the residual carrier while the cross-channel relative phases (the delay
                // observable) pass through. Only covering-mask channels accumulate; the rest stay
                // zero and the consumer skips zero energy.
                if (!_spec_freq_ids.empty() && (int)_spec_freq_ids.size() == n_chan) {
                    const size_t prow = (size_t)(c.job0 + 1) * n_chan;
                    std::lock_guard<std::mutex> lk(_spec_mtx);
                    SpecWindow& W = spec_window_for(wstart);
                    // Publish the phase reference rather than subtract it. The rotation is
                    // exp(-i*_phi[p]); _phi[p] has an arbitrary per-instance origin, is zeroed
                    // only when the arc breaks, and takes each re-pin step so that the derotated
                    // prompt stays continuous. A window's sum is exp(-i*phi0) * sum_r
                    // G_r*exp(-i*dphi_r), with phi0 = _phi[p] at the window's first record, so a
                    // consumer can rotate any two windows onto a common reference. Referencing
                    // each window to its own first record instead would give every window its own
                    // arbitrary constant. The accumulator itself is needed: without it the prompt
                    // winds with the residual carrier across the window (1 Hz is 6.6 rad over
                    // 1.05 s).
                    if (W.nrec[p] == 0)
                        W.phi0[p] = _phi[p];
                    // n_reanchor counts only fresh acquisitions (reanchored 1), which reset _phi
                    // mid-window where no constant can undo it; consumers drop those windows.
                    // Re-pins (3) were folded into _phi above and keep the window
                    // continuous, and a producer may re-pin every record.
                    if (c.reanchored == 1)
                        W.nreanchor[p] += 1;
                    // Rows: 0 = early, 1 = prompt, 2 = late. One lambda gives all three taps the
                    // identical element combine and rotation; a discriminator built from taps
                    // combined differently measures the difference between the combines, not
                    // the code offset.
                    const size_t erow = (size_t)(c.job0 + 0) * n_chan;
                    const size_t lrow = (size_t)(c.job0 + 2) * n_chan;
                    const bool warm = _elem_sum && _cal[p].warm();
                    // Steered, as the header is. The cal's weights are learned from the steered
                    // prompt, so they carry only the instrument phase; applied to unsteered
                    // per-channel correlations they would leave each element's geometric phase in
                    // the sum and the combine would lose coherence.
                    // Steering is referenced to the reference element configured at
                    // construction, so the steered combine keeps the comb's code and phase
                    // currency. The cold path takes the reference element unsteered, which
                    // matches the header only while that is the configured element. The beam
                    // cube below stays unsteered: per element is its axis.
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
                    // Beam cube (see the header): the (subband bin x element) prompt, taken from
                    // `corr` before either collapse (the element blocks sum over channels, the
                    // comb over elements). Both sums are kept: the incoherent one is the beam (no
                    // phase model); the coherent one cannot be reconstructed afterwards. `rot` is
                    // the same per-record rotation as the prompt and the spectrum, and its origin
                    // _phi[p] is published as phi0. Each channel is normalised by its own replica
                    // energy, A_e = G_e/E_c for every element (one replica is correlated against
                    // all of them), so antenna ratios stay comparable; a zero-energy channel is
                    // skipped.
                    if (_cube_on) {
                        CubeWindow& C = cube_window_for(wstart);
                        // The epoch the record's `utc` was stamped from, so the cube and the
                        // record stream agree on the time of a wstart.
                        C.utc0 = hdr.utc0;
                        // phi0 at the window's first record, and only fresh acquisitions
                        // counted, as for the spectrum ring.
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
                            const size_t b = (prow + ch) * n_e;
                            const size_t k0 = ((size_t)p * _cube_bins + ch) * n_e;
                            const double inv = 1.0 / ec;
                            for (int el = 0; el < n_e; ++el) {
                                const std::complex<double> v =
                                    std::complex<double>(corr[2 * (b + el)] * inv,
                                                         corr[2 * (b + el) + 1] * inv)
                                    * rot;
                                C.coh_re[k0 + el] += v.real();
                                C.coh_im[k0 + el] += v.imag();
                                // |v|^2 == |A|^2: rot is unit modulus, so the incoherent sum
                                // ignores the rotation and survives a broken arc.
                                C.incoh[k0 + el] += std::norm(v);
                            }
                            const size_t kb = (size_t)p * _cube_bins + ch;
                            C.w[kb] += 1.0;
                            C.energy[kb] += ec;
                        }
                    }
                    for (int ch = 0; ch < n_chan; ++ch) {
                        if (!((c.chan_mask >> ch) & 1ULL))
                            continue;
                        // The same per-channel weight as the record's sums, so the comb and
                        // the record agree channel by channel.
                        const double w = _chan_w[ch];
                        std::complex<double> g = tap(prow, ch) * w;
                        const size_t k = (size_t)p * n_chan + ch;
                        W.re[k] += g.real();
                        W.im[k] += g.imag();
                        W.energy[k] += w * energy[prow + ch];
                        // The same values per record (gnssRecord.hpp comb block): /get_spectrum
                        // serves window sums, and a cross-record rate fit needs the unsummed
                        // records.
                        if (_chan_export) {
                            float* cb = out + gnss::chan_offset(p, ch, n_prn, n_chan, _n_elements);
                            const std::complex<double> ge = tap(erow, ch) * w;
                            const std::complex<double> gl = tap(lrow, ch) * w;
                            cb[gnss::CHAN_RE] = (float)g.real();
                            cb[gnss::CHAN_IM] = (float)g.imag();
                            cb[gnss::CHAN_ENERGY] = (float)(w * energy[prow + ch]);
                            cb[gnss::CHAN_E_RE] = (float)ge.real();
                            cb[gnss::CHAN_E_IM] = (float)ge.imag();
                            cb[gnss::CHAN_E_ENERGY] = (float)(w * energy[erow + ch]);
                            cb[gnss::CHAN_L_RE] = (float)gl.real();
                            cb[gnss::CHAN_L_IM] = (float)gl.imag();
                            cb[gnss::CHAN_L_ENERGY] = (float)(w * energy[lrow + ch]);
                        }
                    }
                    W.nrec[p] += 1;
                    if (W.w0 < 0)
                        W.w0 = wstart;
                    W.w1 = wstart;
                }
                // Sky-phase-corrected prompt: the same rotation as the prompt, so the combiner
                // can deep-integrate it in the prompt's currency. Zero when the cal is cold or
                // elem_sum is off, which consumers read as absent.
                if (std::norm(g_sky) > 0.0) {
                    const std::complex<double> gs = g_sky * rot;
                    rec[gnss::REC_SKY_RE] = (float)gs.real();
                    rec[gnss::REC_SKY_IM] = (float)gs.imag();
                }
                // Head segment: the same rotation as the prompt (it is the same correlation,
                // restricted to the hops before the code-period boundary), so head + tail
                // reconstructs P exactly and the combiner can wipe each side with its own
                // overlay chip.
                const std::complex<double> gh_corr = g3[3] * rot;
                rec[gnss::REC_PH_RE] = (float)gh_corr.real();
                rec[gnss::REC_PH_IM] = (float)gh_corr.imag();
                rec[gnss::REC_PH_ENERGY] = (float)e3[3];
                // Element blocks. Every antenna gets the same rotation: the NCO is a per-PRN
                // model, not a per-antenna one, so the inter-element phase differences, which are
                // the measurement, are preserved.
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
                }

                _a_prev_ok[p] = 1;
                _elem_prev_ok[p] = 1;
                _wstart_prev[p] = wstart;

                // Commanded carrier phase, cycles (gnssRecord.hpp REC_CPHASE): replica f_ref*t_abs
                // minus the NCO phase. Adding the combiner's measured arg(A) reconstructs the
                // received carrier phase; a re-pin step cancels because it is folded into
                // _phi_cyc. phi enters negated: f_ref is physical-signed, the NCO is in the
                // r2c-flipped internal convention.
                const double t_abs_w = (double)wstart / _sample_rate;
                const double phi_cmd_cyc = c.fcar * t_abs_w - _phi_cyc[p];
                // Exported as the increment since the previous record. fcar includes the RF
                // offset, so the increment is ~1e7 cycles and float32 keeps only whole cycles.
                rec[gnss::REC_CPHASE] =
                    _phi_cmd_ok[p] ? (float)(phi_cmd_cyc - _phi_cmd_prev[p]) : 0.0f;
                _phi_cmd_prev[p] = phi_cmd_cyc;
                _phi_cmd_ok[p] = 1;
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
    // Floor division, not C's truncation toward zero, so even a negative wstart lands every
    // instance in the same window.
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
    // The same floor arithmetic as spec_window_for, on beam_cube_window_samples (a separate
    // length, so the two rings' boundaries need not match).
    int64_t idx = wstart / _cube_win_samples;
    if (wstart < 0 && idx * _cube_win_samples != wstart)
        --idx; // floor, not truncation toward zero (see spec_window_for)
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
    // Unit check. A length in hops rather than F-engine samples (off by fft_length) gives
    // windows shorter than a record, and every array stays well formed. The count rises when a
    // window opens at the index after the newest, resets on any second call into an open window
    // (nrec_seen counts calls, one per running PRN per record), and warns once at 8.
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
        // A later window has opened, the only local evidence that no more records are coming
        // for the previous one: push it now, on the boundary crossing.
        if (_cube_max_idx >= 0) {
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
    // Never blocks (see the header): this stage is in the real-time path, so a full buffer
    // costs the archive a window, never the tracker a record.
    // is_frame_empty() is a safe pre-check because this stage is the buffer's only producer.
    if (!_cube_out_buf->is_frame_empty(_cube_out_id)) {
        ++_cube_dropped;
        // Log the first drop and every hundredth; the count rides in the next frame anyway.
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
    std::memset(f, 0, gnss::cube_frame_bytes(mp, mb, ne)); // padding is zero, not stale bytes

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
    i32(84, 1); // bin_width: one channel per bin
    // The maxima the frame is sized for: n_prn/n_bin above are the actual extents, and the
    // arrays below are laid out by these, which makes the frame self-delimiting.
    i32(88, mp);
    i32(92, mb);
    // The epoch of sample 0, so a frame carries its own time axis: the window's UTC is
    // utc0 + w0 / sample_rate.
    f64(gnss::CUBE_UTC0_OFFSET, C.utc0);
    std::strncpy((char*)f + gnss::CUBE_CHAIN_OFFSET, _cube_chain.c_str(),
                 gnss::CUBE_CHAIN_CHARS - 1);

    size_t off = gnss::CUBE_HEADER_BYTES;
    // Doubles first (gnssRecord.hpp): their alignment must not depend on max_bins/max_prn.
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

    for (int b = 0; b < _cube_bins && b < mb; ++b)
        fid_lo[b] = fid_hi[b] = _spec_freq_ids[b];
    for (size_t p = 0; p < _prns.size() && (int)p < mp; ++p) {
        // A slot with no records this window is left at prn = 0: absent, not a row of zeros
        // that would read as a measurement of no power.
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
    // A frame that feeds bufferSend needs a metadata object: bufferSend drops a frame without
    // one. Stamped like the record leg: the window's first sample, so a consumer after
    // bufferRecv can address the frame without parsing it.
    if (_cube_out_buf->metadata_pool) {
        _cube_out_buf->allocate_new_metadata_object(_cube_out_id);
        if (auto* m = get_gnss_chan_metadata(_cube_out_buf, _cube_out_id))
            m->sample_seq = C.w0;
    }
    _cube_out_buf->mark_frame_full(unique_name, _cube_out_id);
    _cube_out_id = (_cube_out_id + 1) % _cube_out_buf->num_frames;
}


void GnssGpuRecordAssemble::spectrum_callback(kotekan::connectionInstance& conn) {
    // Addressable, idempotent read: `?window=N` returns window N, no argument the newest
    // complete one. Nothing is reset, so a second poller is harmless. Every reply carries
    // available: [lo, hi] so the broker can resynchronise: `too_old` means the records are gone
    // and the caller must skip ahead; `not_yet` is normal (instances lag each other by a few
    // records) and the caller re-asks.
    if (_spec_freq_ids.empty()) {
        conn.send_error("per-channel spectrum export is off: this stage's config has no "
                        "'channel_ids' (regenerate the node config with "
                        "config/gen_chord_gnss_config.py)",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    const size_t nc = _spec_freq_ids.size();

    // A window is complete once a later one has opened: there is no end-of-window event.
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
        hi = _spec_max_idx - 1; // newest complete
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
                // In range but the slot holds another index: no records landed in that
                // window. Report too_old rather than an empty window.
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

    nlohmann::json reply;
    reply["freq_ids"] = _spec_freq_ids;
    reply["sample_rate"] = _sample_rate;
    reply["status"] = status;
    reply["addressable"] = true; // the broker's aligned spectrum gather requires it
    reply["window_samples"] = (double)_spec_win_samples;
    reply["window"] = idx;
    reply["available"] = nlohmann::json::array({lo, hi});
    reply["wstart0"] = w0; // absolute sample of the first / last record in this window:
    reply["wstart1"] = w1; // the cross-instance alignment key, same clock as fleet hops
    if (std::string(status) != "ok") {
        reply["prns"] = nlohmann::json::array();
        conn.send_json_reply(reply);
        return;
    }
    nlohmann::json prns = nlohmann::json::array();
    for (size_t p = 0; p < _prns.size(); ++p) {
        if (nrec[p] <= 0)
            continue; // absent, not zeros (as in /get_records)
        nlohmann::json chans = nlohmann::json::array();
        for (size_t ch = 0; ch < nc; ++ch) {
            const size_t k = p * nc + ch;
            // Energy-normalised like /get_records: A = G/E is comparable across instances,
            // and E is the ML combining weight.
            if (en[k] > 0.0)
                chans.push_back({re[k] / en[k], im[k] / en[k], en[k]});
            else
                chans.push_back({0.0, 0.0, 0.0}); // masked-off channel: weight 0, skipped
        }
        prns.push_back({{"prn", _prns[p]},
                        {"n_rec", nrec[p]},
                        {"phi0", phi0[p]},
                        {"n_reanchor", nre[p]},
                        {"chan", chans}});
    }
    reply["prns"] = prns;
    conn.send_json_reply(reply);
}


// Inject a per-element complex gain prior into every PRN's ElemCal so the combine starts warm
// on (re)acquisition. Parse first, lock last: main_thread takes _gain_mtx only for the swap.
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
    // {"adapt": true|false}. An atomic read per record: no frame boundary needed, since a
    // record combined with the old value is as valid as the next.
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
    // {"shared": true|false}. On: the per-PRN weights are rebuilt from the shared model every
    // record and the learners are held (adapt forced false). Off: the weights stay as the model
    // last installed them; re-enable adapt separately.
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
/// The steady clock in seconds, the clock every freshness decision here uses.
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
    // One instrument per pol from the warm shadow cals. Each PRN's per-pol weight vector is
    // normalised to sum|w| = 1 (its overall scale is the satellite's strength, not the
    // instrument), rotated onto the current model (its overall phase is the satellite's) and
    // averaged with equal weight per satellite; a slow EMA then follows the instrument. The
    // first model is one satellite alone, since a blind average of unaligned vectors could
    // cancel. Each pol's global phase is then pinned: to the fleet reference in live mode,
    // else to the instance's own pin, else to its anchor element (the per-PRN coefficient
    // carries pol 1's phase relative to pol 0).
    //
    // The anchor element's magnitude is capped at the median of the other live elements before
    // normalising, and satellites are not weighted by their MRC weight sum: a cal whose
    // reference weight has degenerated (its variance collapses when the leave-one-out sum
    // correlates strongly) would otherwise outvote every other satellite with a unit vector on
    // the reference element, and the model would become the bare reference element.
    using cd = std::complex<double>;
    const int n = _n_elements;
    if (_g_shared.size() != (size_t)n)
        return;
    const double dt = now_s - _g_shared_t;
    _g_shared_t = now_s;
    fleet_ref_apply_pending();
    // Transit: the model in force stays as it is, and none is formed from captured learners.
    if (shared_frozen(now_s))
        return;
    // No first model inside the wider formation cone either; a warm model keeps adapting.
    if (!_g_shared_warm
        && gnss::formation_frozen(now_s, _bore_post_t.load(), _bore_sep_deg.load(),
                                  _elem_shared_form_deg))
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
            // Cap the anchor element at the median live magnitude of the others (the convention
            // ElemCal::rebuild_weights uses for a self-referenced element).
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
    // Pin to the fleet reference where it applies (live mode, and F describes this half): in
    // full for a first model, slewed for a warm one. Otherwise pin to the instance's own pin.
    const bool fleet_live = _fleet_mode.load() == 1 && !_g_fleet_ref.empty();
    bool fleet_used[2] = {false, false};
    gnss::RefOffset offs[2];
    if (fleet_live)
        for (int pol = 0; pol < n_pol && pol < 2; ++pol)
            offs[pol] = gnss::ref_offset(_g_fleet_ref.data(), _g_shared.data(), pol * h,
                                         std::min(n, pol * h + h));
    for (int pol = 0; pol < n_pol; ++pol) {
        const int e0 = pol * h, e1 = std::min(n, e0 + h);
        const int anchor = std::min(e1 - 1, e0 + _reference_element);
        cd pin(1.0, 0.0);
        if (fleet_live) {
            const auto o = offs[pol];
            if (o.ok && o.sim >= _fleet_min_sim) {
                pin = gnss::pin_rotation(o, _g_shared_warm,
                                         _fleet_slew_rad_s * std::min(std::max(dt, 0.0), 2.0));
                fleet_used[pol] = true;
            }
            // A warm half F has not described for _fleet_reseed_s takes F's shape (already in
            // the fleet's phase, so no pin). Its records were on a mis-shaped model, so the step
            // costs nothing; adaptation continues from here.
            const double low_s =
                _fleet_reseed[pol].low_since < 0.0 ? 0.0 : now_s - _fleet_reseed[pol].low_since;
            const gnss::RefOffset& sib = offs[1 - pol];
            const bool sib_ok = n_pol == 2 && sib.ok && sib.sim >= _fleet_reseed_sib_sim;
            if (_g_shared_warm
                && gnss::reseed_due(_fleet_reseed[pol], o.ok, o.sim, _fleet_min_sim, sib_ok, now_s,
                                    _fleet_reseed_s)) {
                for (int e = e0; e < e1; ++e)
                    _g_shared[(size_t)e] = _g_fleet_ref[(size_t)e];
                fleet_used[pol] = true;
                INFO("elem_sum_shared: pol-{:d} model re-seeded from the fleet reference (shape "
                     "similarity {:.2f} below {:.2f} for {:.0f} s, sibling pol {:.2f}; re-seed "
                     "{:d})",
                     pol, o.sim, _fleet_min_sim, low_s, sib.sim, _fleet_reseed[pol].n);
            }
        } else {
            _fleet_reseed[pol].low_since = -1.0;
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
        // spreads it over a dozen): refuse it and let every PRN ride its own learner until the
        // consensus recovers. Logged once per collapse.
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
    // The own pin follows every half the fleet reference moved, so falling back to it (F
    // cleared, mode changed, a shape F no longer describes) holds the model's current phase.
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
    return m == 1 ? "live" : "off";
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
    // Leaving live, or losing F: the own pin takes over from the model's current phase, so the
    // switch itself never steps it.
    if (((was == 1 && _fleet_mode.load() != 1) || (ref_set && _g_fleet_ref.empty()))
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
    // {"ref": [[re, im], ...] (n_elements pairs; [] clears), "mode": "off|live",
    //  "slew_deg_s": x}, every field optional. Staged; applied at the next consensus update
    // (~1 s, while elem_sum_shared is on). Posted by python/scripts/gnss/elem_shared_ref.py.
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
            mode = (m == "off") ? 0 : (m == "live") ? 1 : -2;
            if (mode == -2)
                throw std::runtime_error("'mode' must be off | live");
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
    // A post older than 120 s is no evidence (broker down, or posts without "_bore"): no new
    // freeze from it, and an armed hold still runs out.
    if (now_s - _bore_post_t.load() <= 120.0 && _bore_sep_deg.load() < _elem_shared_freeze_deg)
        _freeze_until = now_s + _elem_shared_freeze_hold_s;
    return now_s < _freeze_until;
}

void GnssGpuRecordAssemble::shared_hold(size_t p) {
    // Install this PRN's weights from the model. Until a model exists the PRN rides its shadow
    // learner's weights (outside a transit freeze, see below); until its inter-pol coefficient
    // is warm it combines pol 0 alone (coherent from the first record, 3 dB short of both pols).
    using cd = std::complex<double>;
    const int n = _n_elements;
    if (p >= _cal.size())
        return;
    const double now_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now_s - _g_shared_t >= 1.0)
        shared_consensus(now_s);
    gnss::ElemCal& ec = _cal[p];
    if (!_g_shared_warm) {
        // Not inside a transit freeze: near boresight a bright satellite's leakage teaches the
        // shadow that satellite's phases, and a full-array combine on those weights puts the
        // array gain on the leak. While frozen the live cal keeps what it holds: the shadow's
        // weights from before the freeze, or nothing, which leaves the header on the reference
        // element.
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
    // The inter-pol coefficient from this record's two sub-beams: B0 = pol-0 model dotted with
    // the prompt, B1 = pol-1 likewise; c = <B1 conj(B0)> / <|B0|^2>. Each sub-beam has the
    // half-array's gain, so a bright satellite's leakage is rejected here as in the beam
    // itself, which makes one number per PRN safe to learn where a per-element vector is not.
    using cd = std::complex<double>;
    const int n = _n_elements;
    if (!_g_shared_warm || p >= _pol_num.size() || n < 2 || (n % 2) != 0 || !(dt_s > 0.0))
        return;
    // Transit: hold every coefficient too. The sub-beams reject leakage by the half-array's
    // gain, but a boresight transit is 20-30 dB above a weak satellite and the coefficient's
    // EMA (elem_sum_pol_tau_s) is short.
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
        out["form_frozen"] = gnss::formation_frozen(now_s, _bore_post_t.load(),
                                                    _bore_sep_deg.load(), _elem_shared_form_deg);
        out["form_deg"] = _elem_shared_form_deg;
        out["bore_sep_deg"] = _bore_sep_deg.load();
        out["bore_post_age_s"] = now_s - _bore_post_t.load();
        out["pin_ref_ok"] = _g_pin_ref_ok;
    }
    {
        std::lock_guard<std::mutex> lk(_chan_w_mtx);
        out["chan_norm"] = {{"on", _chan_norm},
                            {"applied", _chan_norm_applied.load()},
                            {"miss", _chan_norm_miss.load()},
                            {"w", _chan_w_pub}};
    }
    out["fleet_ref"] = {{"mode", fleet_mode_name(_fleet_mode.load())},
                        {"present", _fleet_ref_present.load()},
                        {"slew_deg_s", _fleet_slew_rad_s * 180.0 / M_PI},
                        {"min_sim", _fleet_min_sim},
                        {"err_deg", {_fleet_err_deg[0], _fleet_err_deg[1]}},
                        {"sim", {_fleet_sim[0], _fleet_sim[1]}},
                        {"applied", {_fleet_applied[0], _fleet_applied[1]}},
                        {"reseed_s", _fleet_reseed_s},
                        {"reseed_sib_sim", _fleet_reseed_sib_sim},
                        {"reseeds", {_fleet_reseed[0].n, _fleet_reseed[1].n}}};
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
        // The projection's state: mode and tunables, what is in force this record (k and
        // source kinds per channel), the probe-stack trigger level per channel, the CPU it
        // costs, and a summary of the sources.
        const int mode = _proj_mode.load();
        nlohmann::json pj;
        pj["mode"] = (mode == 2) ? "live" : (mode == 1) ? "shadow" : "off";
        pj["deg"] = _proj_deg.load();
        pj["rank_max"] = _proj_rank_max.load();
        pj["max_age_s"] = _proj_max_age_s.load();
        pj["probe_frac_min"] = _proj_probe_frac_min.load();
        pj["group"] = _proj_group;
        pj["sys"] = std::string(1, _proj_sys);
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
    // Body: {"element": N}. Staged only; main_thread applies it at the next frame boundary.
    // The bounds are the constructor's, answered with a 400 instead of FATAL_ERROR.
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

// ---- Bright-satellite projection (see the header and gnssProjSubspace.hpp) ----

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
        // one channel's subspace to another's data. Refuse, and warn once.
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

    // 1. Sources among the own slots: running, geometry fresh, inside elem_proj_deg of
    //    boresight; nearest first, at most rank_max. A probe (a slot running for at least
    //    _proj_probe_since_s with no fresh geometry; geometry is posted only above the
    //    horizon) feeds the probe stack.
    const double az = _bore_az_deg * M_PI / 180.0, el = _bore_el_deg * M_PI / 180.0;
    const double eb[3] = {std::cos(el) * std::sin(az), std::cos(el) * std::cos(az), std::sin(el)};
    std::vector<std::pair<double, int>> cand;
    std::vector<uint8_t> probe_flag((size_t)n_prn, 0);
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
            } else if (!fresh && now_s - _slot_run_since[(size_t)p] >= _proj_probe_since_s) {
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
    // The sources' steering tables at this record's time: the own-row covariance is
    // accumulated in the source's steered frame and un-steered at use with the same table.
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

    // 2. Own-row trackers: this record's prompt row, steered by the source's own table, per
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
    // Which running slots have fresh geometry this record, for the identity check below: a
    // direction the probe stack or a sibling reports may be one of our own satellites.
    std::fill(_proj_steered.begin(), _proj_steered.end(), 0);
    {
        std::lock_guard<std::mutex> lk(_steer_mtx);
        for (int p = 0; p < n_prn; ++p)
            if (pc[p].run && _steer.warm(p, now_s))
                _proj_steered[(size_t)p] = 1;
    }

    // 3. The probe stack: every probe row's prompt into one covariance per channel, solved
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

    // 4. The basis per channel: own rows, then the siblings' entries from the board, then the
    //    probe stack. Gram-Schmidt drops a direction already spanned (the same emitter seen
    //    through two sources); rank is capped at rank_max.
    //
    //    A probe-class direction (ours or a sibling's) is used only when it can be accounted
    //    for: the stack also triggers far from any transit, and a direction that is a real
    //    satellite's own signature would be nulled out of that satellite's row. So it is taken
    //    only while the broker reports a satellite inside elem_proj_deg of boresight, and then
    //    (accept_probe_dir): a direction already in the basis is skipped; one that is a steered
    //    satellite's own signature (proj_identify) is dropped, since inside the window its row
    //    is already the source and outside it is a sidelobe leaker; one that is an unsteered
    //    tracked satellite's signature makes that slot a source (its rows spared) and serves
    //    the others; one that names no single row is used only when at least two of our rows
    //    lie along it (a common interferer no slot here tracks).
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
            // Already in the basis (Gram-Schmidt would drop it at the same 0.5): the same
            // emitter seen twice, nothing to name.
            if (Q.k > 0 && Q.cos2(q) >= 0.5)
                return false;
            int n_along = 0;
            const int who = proj_identify(q, ch, n_e, pctl_rec_v, &n_along);
            if (who < 0) {
                // Unnamed: used only when at least two of our rows lie along it (a common
                // interferer nobody here tracks). A direction along no row of ours is a satellite
                // this chain does not track, seen through the sidelobes, and projecting it would
                // cost the other rows for nothing. A direction along one row is that row's own
                // signature and is named.
                return n_along >= 2;
            }
            if (_proj_steered[(size_t)who]) {
                // A steered satellite's own signature: never a column (its row is the source
                // inside the window, a sidelobe leaker outside it).
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

    // 5. Publish to the siblings, per channel: our own rows (up to rank_max sources, nearest
    //    first) and our probe stack; retire them when they go away.
    gnss::ProjBoard& board = gnss::ProjBoard::instance();
    bool pub_own = false, pub_probe = false;
    for (int ch = 0; ch < n_chan; ++ch) {
        gnss::ProjEntry e;
        e.owner = unique_name;
        e.src = 0;
        e.t_pub = now_s;
        e.n = n_e;
        e.k = 0;
        for (size_t ci = 0; ci < cand.size(); ++ci) {
            const int p = cand[ci].second;
            const auto& T = *_proj_own[(size_t)p];
            if (!T.warm(ch) || T.k(ch) <= 0 || e.k >= kmax)
                continue;
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
            pe.src = 1;
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

    // 6. A one-line description of the sources, rebuilt (and logged) only when the set
    //    changes: the signature is compared every record, the string formatted only on a change.
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
            // Logged on change, at most every 5 s.
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
    // this satellite's steered channel sum sees it: per channel cos^2(w, D_ch q_ch), i.e.
    // cos^2(conj(D_ch) w, q_ch), averaged over the covering channels with a basis in force.
    // A random direction gives 1/n_live. Only while a basis is in force and the slot is steered
    // (a probe is never steered). A source is excluded: its own learner is degenerate (every
    // element correlates with the reference at rho^2 > 0.99, which ElemCal's self-reference
    // guard collapses) and its capture would read as a satellite captured by itself.
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
    // "max_age_s": S, "probe_frac_min": F}. Applied at once (atomics read per record); the reply
    // is the state in force. rank_max cannot exceed the columns allocated at construction.
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
    // The emitter of direction q is the brightest row along q, not the learner that agrees with
    // q: a captured victim's learner points along the interferer, while the source's own
    // learner is degenerate. Among the running non-probe slots, take this record's raw prompt
    // row on channel ch. The interferer's row lies along q with cos^2 ~ 1 and is the strongest
    // by 10-20 dB; a victim's row lies along q only in proportion to the leak. Rows with
    // cos^2 >= 0.5 count as along q, scored cos^2 x |row|^2, and their count decides: one is
    // that slot's own signature; of several, the brightest only if it is 10x the next, else
    // -1 (where the emitter has no row here, every row along q is a victim's).
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
