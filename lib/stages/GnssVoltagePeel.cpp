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
    _hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", _replica->repl_period_hops());
    _replica->code_doppler_sign = config.get_default<double>(unique_name, "code_doppler_sign", 1.0);
    _pullin_chips = config.get_default<double>(unique_name, "pullin_chips", 0.5);
    _pullin_step = config.get_default<double>(unique_name, "pullin_step", 0.25);
    _fll_gain = config.get_default<double>(unique_name, "fll_gain", 0.0);
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
            std::vector<double> dop, cp, cp_rate;
            std::vector<long long> ref_hop;
            std::vector<uint8_t> active;
            {
                std::lock_guard<std::mutex> lk(_seed_mtx);
                dop = _doppler;
                cp = _code_phase;
                cp_rate = _code_phase_rate;
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

                // Code pull-in: lock to the peak |A| over a small cp window so a residual code
                // error still peels on-peak (the broker cp_rate folds in the l-a clock bias).
                gnss::DespreadResult best{};
                double best_pw = -1.0, best_off = 0.0;
                std::vector<std::vector<cf>> best_repl;
                for (double off = -_pullin_chips; off <= _pullin_chips + 1e-9;
                     off += (_pullin_step > 0.0 ? _pullin_step : 1.0)) {
                    const auto repl =
                        _replica->channels(p, window_start, cp_seed + off, fcar, _hops_per_record);
                    std::vector<std::vector<cf>> repl_ch;
                    repl_ch.reserve(owned.size());
                    for (int c : owned)
                        repl_ch.push_back(repl[c]);
                    const auto res = gnss::channelized_despread(data_ch, repl_ch);
                    const double pw = std::norm(res.amplitude);
                    if (pw > best_pw) {
                        best_pw = pw;
                        best = res;
                        best_off = off;
                        best_repl = std::move(repl_ch);
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

                // Gain to subtract. v1: the raw per-record despread (noisy -> ~3 dB self-noise).
                // v2 (fll_gain>0 / gain_alpha>0): FLL-track the residual carrier phase, derotate the
                // gain, EMA the SLOW (de-bitted) gain, re-apply the bit + carrier phase -> the gain
                // noise averages down so much less noise is subtracted back along R (deeper peel).
                cf a_use = (cf)best.amplitude;
                if (_fll_gain > 0.0 || _gain_alpha > 0.0) {
                    using cd = std::complex<double>;
                    if (std::isnan(f_ref[p])) {
                        f_ref[p] = fcar;
                        f_track[p] = 0.0;
                        phi_track[p] = 0.0;
                        a_prev_ok[p] = 0;
                        gain_ema[p] = cf(0.0f, 0.0f);
                    }
                    const double dt =
                        a_prev_ok[p] ? (double)(window_start_hop - hop_prev[p]) * dt_hop : 0.0;
                    if (a_prev_ok[p] && dt > 0.0) {
                        phi_track[p] += 2.0 * M_PI * f_track[p] * dt;
                        phi_track[p] = std::remainder(phi_track[p], 2.0 * M_PI);
                    }
                    const cd a_raw = cd(best.amplitude) * std::polar(1.0, -dphi_code); // -> smooth cp ref
                    const cd a_corr = a_raw * std::polar(1.0, -phi_track[p]);          // derotated
                    if (_fll_gain > 0.0 && a_prev_ok[p] && dt > 0.0 && dt <= _fll_max_gap
                        && std::abs(a_corr) > _fll_lock_amp && std::abs(a_prev[p]) > _fll_lock_amp) {
                        const cd prod = a_corr * std::conj(cd(a_prev[p]));
                        f_track[p] += _fll_gain * std::arg(prod * prod) / 2.0 / (2.0 * M_PI * dt);
                    }
                    a_prev[p] = (cf)a_corr;
                    a_prev_ok[p] = 1;
                    hop_prev[p] = window_start_hop;
                    cd a_smooth = a_corr;
                    if (_gain_alpha > 0.0 && std::abs(a_corr) > _fll_lock_amp) {
                        // decision-directed sign vs the running gain (nav bit / NH overlay / pilot=+1)
                        const double sign =
                            (std::abs(gain_ema[p]) == 0.0f
                             || std::real(a_corr * std::conj(cd(gain_ema[p]))) >= 0.0)
                                ? 1.0
                                : -1.0;
                        const cd g_deb = a_corr * sign;
                        gain_ema[p] = (std::abs(gain_ema[p]) == 0.0f)
                                          ? (cf)g_deb
                                          : (cf)((1.0 - _gain_alpha) * cd(gain_ema[p])
                                                 + _gain_alpha * g_deb);
                        a_smooth = cd(gain_ema[p]) * sign; // smoothed gain, bit re-applied
                    }
                    // back to the model frame AND to best_cp (re-apply the code-delay phase) so the
                    // subtraction matches best_repl.
                    a_use = (cf)(a_smooth * std::polar(1.0, phi_track[p] + dphi_code));
                }

                // SUBTRACT a_use*R from the residual over the covering channels.
                for (size_t ci = 0; ci < owned.size(); ++ci) {
                    const int local = owned[ci] - _chan_offset;
                    for (int m = 0; m < _hops_per_record; ++m)
                        out[(size_t)m * _n_chan + local] -= a_use * best_repl[ci][m];
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
