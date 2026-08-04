#include "GnssVoltagePeel.hpp"

#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "GnssChanMetadata.hpp"        // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSignal.hpp"              // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"          // for FATAL_ERROR
#include "pfbPrototype.hpp"            // for window_from_string

#include <algorithm> // for fill
#include <cmath>     // for fmod, nan
#include <complex>
#include <cstring>   // for memcpy
#include <set>

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssVoltagePeel);

GnssVoltagePeel::GnssVoltagePeel(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssVoltagePeel::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr) {
        FATAL_ERROR("GnssVoltagePeel: unknown signal '{:s}'", signal);
        return;
    }

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    _f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    _doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);

    _N = config.get<int>(unique_name, "spectrum_length");
    _fft_len = 2 * _N;
    _chan_offset = config.get_default<int>(unique_name, "channel_offset", 0);
    _n_chan = config.get<int>(unique_name, "n_channels");
    if (_chan_offset < 0 || _n_chan <= 0 || _chan_offset + _n_chan > _N) {
        FATAL_ERROR("GnssVoltagePeel: channel slice [{:d},{:d}) invalid for N={:d}", _chan_offset,
                    _chan_offset + _n_chan, _N);
        return;
    }
    const int num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");

    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty()) {
        FATAL_ERROR("GnssVoltagePeel: 'prns' is empty");
        return;
    }
    const int n_prn = (int)_prns.size();

    auto seed = [&](const std::string& key, std::vector<double>& out) {
        const auto v = config.get_default<std::vector<double>>(unique_name, key, {});
        out.assign(n_prn, 0.0);
        for (int i = 0; i < n_prn; ++i)
            out[i] = v.empty() ? 0.0 : (v.size() == 1 ? v[0] : v.at(i));
    };
    seed("doppler_hz", _doppler);
    seed("code_phase_chips", _code_phase);
    _code_phase_rate.assign(n_prn, 0.0);
    seed("code_phase_rate", _code_phase_rate); // chips/hop; usually broker-set, config for offline tests
    _doppler_rate.assign(n_prn, 0.0);
    seed("doppler_rate_hz_s", _doppler_rate); // Hz/s; broker-set (v2 FF), config for offline tests
    _ref_hop.assign(n_prn, 0);
    _pre_amp.assign(n_prn, 0.0);
    _post_amp.assign(n_prn, 0.0);

    try {
        _replica = std::make_unique<gnss::ChannelizedReplicaBank>(
            *sig, _sample_rate, _f_offset, _N, num_taps, dsp::window_from_string(win), _prns);
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssVoltagePeel: {:s}", e.what());
        return;
    }
    // FDMA (GLONASS L1OF/L2OF): every satellite shares one code and is separated by CARRIER, so
    // each PRN needs its own offset from band centre. The table is built by the config generator
    // from the live GLONASS frequency plan and travels in the yaml, where it is readable next to
    // the PRN list. Absent/empty -> CDMA, i.e. every other signal, unchanged.
    _replica->set_prn_freq_offsets(
        config.get_default<std::vector<double>>(unique_name, "prn_freq_offset_hz", {}));
    if (_replica->prn_freq_offsets_set() > 0)
        INFO("GnssVoltagePeel: FDMA carrier offsets applied to {:d} of {:d} PRNs",
             _replica->prn_freq_offsets_set(), (int)_prns.size());

    _hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", _replica->repl_period_hops());
    _replica->code_doppler_sign = config.get_default<double>(unique_name, "code_doppler_sign", 1.0);
    _pullin_chips = config.get_default<double>(unique_name, "pullin_chips", 0.5);
    _pullin_step = config.get_default<double>(unique_name, "pullin_step", 0.25);
    _fll_gain = config.get_default<double>(unique_name, "fll_gain", 0.0);
    // DEFAULT 0 = v1 (raw per-segment gain) -- the true old on-sky behaviour. The 2026-07-23
    // removal of the `segs.size()==1` gate made this knob ACTIVE for the first time on streaming
    // windows (every one of which straddles the period boundary), so configs carrying a nonzero
    // gain_alpha silently changed path. The calibrated twin-chain bench could not separate v1 from
    // any v2 variant -- all three floor the instrument at >=26-35 dB -- so there is no measured
    // reason to prefer smoothing here. The fused live peel gets its gain feed-forward instead
    // (docs/gnss_voltage_peel_live.md); this stage is now the offline reference.
    _gain_alpha = config.get_default<double>(unique_name, "gain_alpha", 0.0);
    _fll_lock_amp = config.get_default<double>(unique_name, "fll_lock_amp", 0.0);
    _fll_max_gap = config.get_default<double>(unique_name, "fll_max_gap_s", 0.005);

    const auto active_prns = config.get_default<std::vector<int>>(unique_name, "active_prns", {});
    _active.assign(n_prn, 1);
    if (!active_prns.empty()) {
        const std::set<int> on(active_prns.begin(), active_prns.end());
        for (int i = 0; i < n_prn; ++i)
            _active[i] = on.count(_prns[i]) ? 1 : 0;
    }
}

void GnssVoltagePeel::set_seeds_callback(kotekan::connectionInstance& conn,
                                         nlohmann::json& request) {
    try {
        std::lock_guard<std::mutex> lk(_seed_mtx);
        std::fill(_active.begin(), _active.end(), (uint8_t)0);
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            for (size_t i = 0; i < _prns.size(); ++i)
                if (_prns[i] == prn) {
                    _doppler[i] = s.at("doppler_hz").get<double>();
                    _code_phase[i] = s.at("code_phase_chips").get<double>();
                    _code_phase_rate[i] = s.value("code_phase_rate", 0.0);
                    _doppler_rate[i] = s.value("doppler_rate_hz_s", 0.0);
                    _ref_hop[i] = s.value("ref_hop", (long long)0);
                    _active[i] = 1;
                }
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssVoltagePeel::get_status_callback(kotekan::connectionInstance& conn) {
    nlohmann::json out = nlohmann::json::array();
    std::lock_guard<std::mutex> lk(_diag_mtx);
    for (size_t i = 0; i < _prns.size(); ++i) {
        const double pre = _pre_amp[i], post = _post_amp[i];
        out.push_back({{"prn", _prns[i]},
                       {"amplitude", pre},
                       {"residual", post},
                       {"peel_db", (pre > 0 && post > 0) ? 20.0 * std::log10(pre / post) : 0.0}});
    }
    conn.send_json_reply(out);
}

void GnssVoltagePeel::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_seeds",
        std::bind(&GnssVoltagePeel::set_seeds_callback, this, _1, _2));
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_status", std::bind(&GnssVoltagePeel::get_status_callback, this, _1));

    int frame_in = 0;
    int frame_out = 0;
    const int n_prn = (int)_prns.size();
    const double L = (double)_replica->code_length();

    // v2 smooth-gain state (per PRN): an FLL tracks the residual carrier phase so the gain can be
    // derotated; gain_ema is a complex EMA of the DEROTATED, de-bitted gain. The decision-directed
    // sign + FLL NCO carry the fast nav-bit/overlay and carrier phase, so only the SLOW dish gain is
    // averaged -> the subtracted gain's noise (hence the peel self-noise) drops ~sqrt(EMA window).
    std::vector<double> f_ref(n_prn, std::nan(""));
    std::vector<double> t_anchor(n_prn, 0.0);  ///< abs time (s) f_ref was pinned; ramp integrates from here
    std::vector<double> f_track(n_prn, 0.0);
    std::vector<double> phi_track(n_prn, 0.0);
    std::vector<cf> a_prev(n_prn, cf(0.0f, 0.0f));
    std::vector<uint8_t> a_prev_ok(n_prn, 0);
    std::vector<long long> hop_prev(n_prn, 0);
    std::vector<cf> gain_ema(n_prn, cf(0.0f, 0.0f));
    const double dt_hop = (double)_fft_len / _sample_rate;

    std::vector<cf> window((size_t)_hops_per_record * _n_chan);
    int hops_filled = 0;
    long long seen_hops = 0;
    long long window_start_hop = 0;

    while (!stop_thread) {
        auto* in_local = (cf*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        const int frame_hops = in_buf->frame_size / (int)sizeof(cf) / _n_chan;

        long long frame_first_hop = seen_hops;
        if (metadata_is_gnss_chan(in_buf)) {
            auto* mi = get_gnss_chan_metadata(in_buf, frame_in);
            if (mi && mi->sample_seq >= 0)
                frame_first_hop = mi->sample_seq / _fft_len;
        }

        for (int in_pos = 0; in_pos < frame_hops; ++in_pos) {
            if (hops_filled == 0)
                window_start_hop = frame_first_hop + in_pos;
            std::memcpy(&window[(size_t)hops_filled * _n_chan], &in_local[(size_t)in_pos * _n_chan],
                        sizeof(cf) * _n_chan);
            if (++hops_filled < _hops_per_record)
                continue;
            hops_filled = 0;
            const long long window_start = window_start_hop * (long long)_fft_len;

            // Snapshot the (REST-updatable) seeds for this window.
            std::vector<double> dop, cp, cp_rate, dop_rate;
            std::vector<long long> ref_hop;
            std::vector<uint8_t> active;
            {
                std::lock_guard<std::mutex> lk(_seed_mtx);
                dop = _doppler;
                cp = _code_phase;
                cp_rate = _code_phase_rate;
                dop_rate = _doppler_rate;
                ref_hop = _ref_hop;
                active = _active;
            }

            cf* out = (cf*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            if (out == nullptr)
                return;
            // Residual starts as the input window; each PRN is peeled off it in turn.
            std::memcpy(out, window.data(), sizeof(cf) * window.size());

            for (int p = 0; p < n_prn; ++p) {
                if (!active[p])
                    continue;
                const double fcar = dop[p];
                const auto cover = _replica->covering_bins(fcar, _doppler_margin_hz);
                std::vector<int> owned;
                for (int c : cover)
                    if (c >= _chan_offset && c < _chan_offset + _n_chan)
                        owned.push_back(c);
                if (owned.empty())
                    continue; // carrier not in this subband

                double cp_seed = cp[p] + cp_rate[p] * (double)(window_start_hop - ref_hop[p]);
                cp_seed = std::fmod(cp_seed, L);
                if (cp_seed < 0.0)
                    cp_seed += L;

                // Extract the covering channels from the RUNNING residual (successive peel: prior
                // sats already removed, so Gold cross-talk doesn't floor this despread).
                std::vector<std::vector<cf>> data_ch(owned.size(),
                                                     std::vector<cf>(_hops_per_record));
                for (size_t ci = 0; ci < owned.size(); ++ci) {
                    const int local = owned[ci] - _chan_offset;
                    for (int m = 0; m < _hops_per_record; ++m)
                        data_ch[ci][m] = out[(size_t)m * _n_chan + local];
                }

                // PERIOD-BOUNDARY SEGMENTATION: the stream-aligned window straddles the code-
                // period boundary at a cp-determined hop, and EVERY overlay/nav sign flip our
                // signals have (C/A nav bits, L5 NH, E1C CS25, B1C per-period secondary) happens
                // exactly THERE. A single per-window gain then models a sign-flipping signal
                // with a constant one: in a flip window the net gain ~cancels and the
                // "projection" removes almost nothing (2026-07-12 bench: B1C C21 peeled 0.9 dB
                // while the per-window residual read exactly 0 -- the projection was perfect,
                // onto the wrong basis). Estimate + subtract the gain PER SEGMENT instead:
                // the projection onto span{R*1_A, R*1_B} captures any boundary sign flip
                // without knowing the overlay. (GPS's bit edges also live on period
                // boundaries -- this removes the old bench's bit-straddle depth cap too.)
                int h_b = (int)llround((L - cp_seed) / L * (double)_hops_per_record);
                if (h_b < 0)
                    h_b = 0;
                if (h_b > _hops_per_record)
                    h_b = _hops_per_record;
                std::vector<std::pair<int, int>> segs;
                if (h_b > 0)
                    segs.emplace_back(0, h_b);
                if (h_b < _hops_per_record)
                    segs.emplace_back(h_b, _hops_per_record);
                auto slice = [](const std::vector<std::vector<cf>>& v, int m0, int m1) {
                    std::vector<std::vector<cf>> s(v.size());
                    for (size_t i = 0; i < v.size(); ++i)
                        s[i].assign(v[i].begin() + m0, v[i].begin() + m1);
                    return s;
                };

                // Code pull-in: lock to the peak over a small cp window so a residual code
                // error still peels on-peak (the broker cp_rate folds in the l-a clock bias).
                // The pull-in metric is the length-weighted SUM of segment powers -- flip-blind
                // (a whole-window |A| would read ~0 on a flip window and pick garbage).
                gnss::DespreadResult best{};
                double best_pw = -1.0, best_off = 0.0;
                std::vector<std::vector<cf>> best_repl;
                std::vector<std::complex<double>> best_gain;
                for (double off = -_pullin_chips; off <= _pullin_chips + 1e-9;
                     off += (_pullin_step > 0.0 ? _pullin_step : 1.0)) {
                    const auto repl =
                        _replica->channels(p, window_start, cp_seed + off, fcar, _hops_per_record);
                    std::vector<std::vector<cf>> repl_ch;
                    repl_ch.reserve(owned.size());
                    for (int c : owned)
                        repl_ch.push_back(repl[c]);
                    double pw = 0.0;
                    std::vector<std::complex<double>> gains;
                    gnss::DespreadResult res0{};
                    for (size_t si = 0; si < segs.size(); ++si) {
                        const auto r = gnss::channelized_despread(
                            slice(data_ch, segs[si].first, segs[si].second),
                            slice(repl_ch, segs[si].first, segs[si].second));
                        gains.push_back(r.amplitude);
                        pw += std::norm(r.amplitude)
                              * (double)(segs[si].second - segs[si].first);
                        if (si == 0)
                            res0 = r;
                    }
                    if (pw > best_pw) {
                        best_pw = pw;
                        best = res0;
                        best_off = off;
                        best_repl = std::move(repl_ch);
                        best_gain = std::move(gains);
                    }
                    if (_pullin_chips <= 0.0)
                        break;
                }
                // Code-delay carrier phase: a signal delayed by the pull-in offset carries a phase
                // 2*pi*(f_offset+dop)*off/chip_rate. Reference the despread phase back to the SMOOTH
                // cp track (remove best_off's phase) so the FLL/gain see a continuous carrier despite
                // the per-record pull-in re-pick, then re-apply it for the subtraction at best_repl.
                const double dphi_code = 2.0 * M_PI * (_f_offset + fcar) * best_off
                                         / _replica->chip_rate_hz();

                // Gain(s) to subtract, PER SEGMENT. v1 (fll_gain==0 && gain_alpha==0): the raw
                // per-segment despread (noisy -> ~3 dB self-noise, no cross-record averaging).
                // v2: reference each segment's gain to a fixed per-PRN f_ref by removing the KNOWN
                // model carrier phase -- the broker dop-refresh's absolute-anchor jump (f_ref-fcar)*t
                // AND the Doppler-rate ramp 0.5*drate*(t-t_anchor)^2 -- so the FLL/EMA see only the
                // small UNMODELED residual (no rate-lag, coherent across broker polls). A decision-
                // directed sign folds the per-segment overlay flip, so BOTH segments feed ONE slow
                // gain EMA: the gain-estimate noise (hence the subtracted self-noise) drops
                // ~sqrt(window). Runs on ALL windows now (the old segs==1 gate left v2 INERT on-sky,
                // since every streaming window straddles the period boundary). The replica is
                // regenerated per window at fcar, so unlike the tracker no fence/re-anchor is needed
                // -- the analytic corr bridges the cross-window gain phase. See diag/peel_ff_harness.py.
                std::vector<cf> a_sub_seg(segs.size());
                if (_fll_gain > 0.0 || _gain_alpha > 0.0) {
                    using cd = std::complex<double>;
                    const double t = (double)window_start_hop * dt_hop; // absolute time (s)
                    if (std::isnan(f_ref[p])) {
                        f_ref[p] = fcar;
                        t_anchor[p] = t;
                        f_track[p] = 0.0;
                        phi_track[p] = 0.0;
                        a_prev_ok[p] = 0;
                        gain_ema[p] = cf(0.0f, 0.0f);
                    }
                    // KNOWN model phase relative to f_ref. (fcar - f_ref)*t bridges the absolute-
                    // anchor jump at each broker dop refresh; 0.5*drate*(t-t_anchor)^2 is the ramp
                    // curvature the FLL used to lag. Any residual constant/linear error is absorbed
                    // by the EMA / FLL.
                    // SIGN: the despread residual lives in the r2c-flipped INTERNAL convention --
                    // gnssRecord.hpp gives arg(A) = 2*pi*(fcar*t_abs - Phi_rx), so the MODEL side
                    // enters NEGATED relative to the naive (Phi_rx - fcar*t). Same trap the tracker
                    // documents when it negates doppler_rate_hz_s on entry (355e7636). Hence
                    // (fcar - f_ref)*t and MINUS 0.5*drate*dtau^2.
                    const double dtau = t - t_anchor[p];
                    const double corr = 2.0 * M_PI
                                        * ((fcar - f_ref[p]) * t - 0.5 * dop_rate[p] * dtau * dtau);
                    const double dt =
                        a_prev_ok[p] ? (double)(window_start_hop - hop_prev[p]) * dt_hop : 0.0;
                    if (a_prev_ok[p] && dt > 0.0) {
                        phi_track[p] += 2.0 * M_PI * f_track[p] * dt;
                        phi_track[p] = std::remainder(phi_track[p], 2.0 * M_PI);
                    }
                    // Total derotation to the smooth common-reference frame (model + pull-in + FLL).
                    const double derot = corr + dphi_code + phi_track[p];
                    // FLL from the head segment's common-reference gain (sign-blind -- it squares).
                    const cd a_head = cd(best.amplitude) * std::polar(1.0, -derot);
                    if (_fll_gain > 0.0 && a_prev_ok[p] && dt > 0.0 && dt <= _fll_max_gap
                        && std::abs(a_head) > _fll_lock_amp && std::abs(a_prev[p]) > _fll_lock_amp) {
                        const cd prod = a_head * std::conj(cd(a_prev[p]));
                        f_track[p] += _fll_gain * std::arg(prod * prod) / 2.0 / (2.0 * M_PI * dt);
                    }
                    a_prev[p] = (cf)a_head;
                    a_prev_ok[p] = 1;
                    hop_prev[p] = window_start_hop;
                    // Per segment: common-reference, decision-directed sign into ONE EMA, re-apply.
                    for (size_t si = 0; si < segs.size(); ++si) {
                        const cd a_ref = cd(best_gain[si]) * std::polar(1.0, -derot);
                        cd a_smooth = a_ref;
                        if (_gain_alpha > 0.0 && std::abs(a_ref) > _fll_lock_amp) {
                            const double sign =
                                (std::abs(gain_ema[p]) == 0.0f
                                 || std::real(a_ref * std::conj(cd(gain_ema[p]))) >= 0.0)
                                    ? 1.0
                                    : -1.0;
                            const cd g_deb = a_ref * sign;
                            gain_ema[p] = (std::abs(gain_ema[p]) == 0.0f)
                                              ? (cf)g_deb
                                              : (cf)((1.0 - _gain_alpha) * cd(gain_ema[p])
                                                     + _gain_alpha * g_deb);
                            a_smooth = cd(gain_ema[p]) * sign; // smoothed gain, overlay sign re-applied
                        }
                        a_sub_seg[si] = (cf)(a_smooth * std::polar(1.0, derot)); // back to best_repl frame
                    }
                } else {
                    for (size_t si = 0; si < segs.size(); ++si)
                        a_sub_seg[si] = (cf)best_gain[si]; // v1: raw per-segment gain
                }

                // SUBTRACT the gain(s)*R from the residual over the covering channels --
                // per segment, so a sign flip at the period boundary is peeled exactly.
                for (size_t si = 0; si < segs.size(); ++si) {
                    const cf a_s = a_sub_seg[si];
                    for (size_t ci = 0; ci < owned.size(); ++ci) {
                        const int local = owned[ci] - _chan_offset;
                        for (int m = segs[si].first; m < segs[si].second; ++m)
                            out[(size_t)m * _n_chan + local] -= a_s * best_repl[ci][m];
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(_diag_mtx);
                    const double amp = std::abs(best.amplitude);
                    _pre_amp[p] = _pre_amp[p] > 0 ? 0.98 * _pre_amp[p] + 0.02 * amp : amp;
                    // post = despread of the residual with the just-subtracted replica (per-record
                    // it is ~0 by construction; the true deep depth is measured downstream on the
                    // residual, but this confirms the subtraction fired).
                    gnss::DespreadResult r2 = gnss::channelized_despread(
                        [&] {
                            std::vector<std::vector<cf>> d(owned.size(),
                                                           std::vector<cf>(_hops_per_record));
                            for (size_t ci = 0; ci < owned.size(); ++ci) {
                                const int local = owned[ci] - _chan_offset;
                                for (int m = 0; m < _hops_per_record; ++m)
                                    d[ci][m] = out[(size_t)m * _n_chan + local];
                            }
                            return d;
                        }(),
                        best_repl);
                    const double res = std::abs(r2.amplitude);
                    _post_amp[p] = _post_amp[p] > 0 ? 0.98 * _post_amp[p] + 0.02 * res : res;
                }
            }

            // Stamp the absolute sample reference so a downstream search/monitor code-phase-references
            // the residual the same way, then emit the residual voltage window.
            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->sample_seq = window_start;
            }
            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out = (frame_out + 1) % out_buf->num_frames;
        }

        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
        seen_hops += frame_hops;
    }
}
