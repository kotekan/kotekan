#include "GnssChannelizedTracker.hpp"

#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "GnssChanMetadata.hpp"        // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSignal.hpp"              // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"          // for FATAL_ERROR
#include "pfbPrototype.hpp"            // for window_from_string

#include <algorithm> // for fill
#include <chrono>    // for system_clock
#include <cmath>     // for nan, arg, norm, isnan, fabs
#include <complex>   // for complex, arg, norm
#include <cstring>   // for memcpy
#include <set>       // for set

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssChannelizedTracker);

GnssChannelizedTracker::GnssChannelizedTracker(Config& config, const std::string& unique_name,
                                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChannelizedTracker::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr) {
        FATAL_ERROR("GnssChannelizedTracker: unknown signal '{:s}'", signal);
        return;
    }

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    const double f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    _capture_utc0 = config.get_default<double>(unique_name, "capture_utc0", 0.0);
    _doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);

    _N = config.get<int>(unique_name, "spectrum_length");
    _fft_len = 2 * _N;
    _chan_offset = config.get_default<int>(unique_name, "channel_offset", 0);
    _n_chan = config.get<int>(unique_name, "n_channels");
    if (_chan_offset < 0 || _n_chan <= 0 || _chan_offset + _n_chan > _N) {
        FATAL_ERROR("GnssChannelizedTracker: channel slice [{:d},{:d}) invalid for N={:d}",
                    _chan_offset, _chan_offset + _n_chan, _N);
        return;
    }
    const int num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");

    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty()) {
        FATAL_ERROR("GnssChannelizedTracker: 'prns' is empty");
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
    _doppler_rate.assign(n_prn, 0.0);
    _ref_hop.assign(n_prn, 0);

    try {
        _replica = std::make_unique<gnss::ChannelizedReplicaBank>(
            *sig, _sample_rate, f_offset, _N, num_taps, dsp::window_from_string(win), _prns);
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssChannelizedTracker: {:s}", e.what());
        return;
    }
    _hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", _replica->repl_period_hops());
    _replica->code_doppler_sign = config.get_default<double>(unique_name, "code_doppler_sign", 1.0);
    _dll_spacing = config.get_default<double>(unique_name, "dll_spacing_chips", 0.5);
    _fll_gain = config.get_default<double>(unique_name, "fll_gain", 0.0);
    _fll_reacq_hz = config.get_default<double>(unique_name, "fll_reacq_hz", 200.0);
    _fll_max_gap = config.get_default<double>(unique_name, "fll_max_gap_s", 0.005);
    // Freeze the FLL discriminator when the despread |A| drops below this -- so a signal dropout
    // (radar sweep, the broker coasting the seed) doesn't drive f_track on noise; the NCO keeps
    // coasting at the held frequency and the loop resumes cleanly on return. 0 = never freeze.
    _fll_lock_amp = config.get_default<double>(unique_name, "fll_lock_amp", 0.0);

    const auto active_prns = config.get_default<std::vector<int>>(unique_name, "active_prns", {});
    _active.assign(n_prn, 1);
    if (!active_prns.empty()) {
        const std::set<int> on(active_prns.begin(), active_prns.end());
        for (int i = 0; i < n_prn; ++i)
            _active[i] = on.count(_prns[i]) ? 1 : 0;
    }
}

void GnssChannelizedTracker::set_seeds_callback(kotekan::connectionInstance& conn,
                                                nlohmann::json& request) {
    try {
        std::lock_guard<std::mutex> lk(_seed_mtx);
        std::fill(_active.begin(), _active.end(), (uint8_t)0); // list is authoritative
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            for (size_t i = 0; i < _prns.size(); ++i)
                if (_prns[i] == prn) {
                    _doppler[i] = s.at("doppler_hz").get<double>();
                    _code_phase[i] = s.at("code_phase_chips").get<double>();
                    // Optional first-order code-phase model: cp0 drifts linearly with the
                    // capture hop (residual code-rate error), so the broker can hand a
                    // slope + anchor hop and the tracker extrapolates to its own window
                    // hop -- removing the seed-staleness bias rather than searching it out.
                    _code_phase_rate[i] = s.value("code_phase_rate", 0.0);
                    // Optional 2nd-order carrier model: the almanac Doppler rate (Hz/s), used to
                    // ramp the replica frequency so the coherent-integration residual stays flat.
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

void GnssChannelizedTracker::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_seeds",
        std::bind(&GnssChannelizedTracker::set_seeds_callback, this, _1, _2));

    int frame_in = 0;
    int frame_out = 0;
    const int n_prn = (int)_prns.size();

    // Carrier FLL state (per PRN). The replica stays at a FIXED reference Doppler f_ref
    // (NaN = not yet acquired) -- it is anchored to the absolute sample index, so retuning
    // it mid-stream would jump the phase by 2*pi*df*t_abs (huge). Instead an NCO accumulates
    // a phase correction phi_track (= integral of 2*pi*f_track) that derotates the despread
    // output, and the discriminator drives the residual frequency f_track. a_prev holds the
    // previous (derotated) amplitude for the record-to-record discriminator.
    std::vector<double> f_ref(n_prn, std::nan(""));
    std::vector<double> f_track(n_prn, 0.0);   // FLL residual-frequency estimate (Hz)
    std::vector<double> phi_track(n_prn, 0.0); // NCO accumulated carrier phase (rad)
    std::vector<cf> a_prev(n_prn, cf(0.0f, 0.0f));
    std::vector<uint8_t> a_prev_ok(n_prn, 0);
    std::vector<long long> hop_prev(n_prn, 0);
    std::vector<long long> reacq_hop(n_prn, 0); // hop where f_ref was last (re)fixed: anchor for the Doppler-rate feed-forward

    // Rolling measurement window of [hop][subband-channel] voltages.
    std::vector<cf> window((size_t)_hops_per_record * _n_chan);
    int hops_filled = 0;
    long long seen_hops = 0;        // fallback absolute hop counter
    long long window_start_hop = 0; // absolute hop index of the current window's first hop

    while (!stop_thread) {
        auto* in_local = (cf*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        const int frame_hops = in_buf->frame_size / (int)sizeof(cf) / _n_chan;

        // Absolute hop index of this frame's first hop, from sample_seq (shared across
        // nodes so every tracker's window_start agrees); fall back to a local count.
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

            // Snapshot the (REST-updatable) seeds + active set for this window.
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

            float* out = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            if (out == nullptr)
                return;
            const double utc = (_capture_utc0 > 0.0)
                                   ? _capture_utc0 + (double)window_start / _sample_rate
                                   : std::chrono::duration<double>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count();

            for (int p = 0; p < n_prn; ++p) {
                float* rec = out + (size_t)p * RECORD_FLOATS;
                for (int f = 0; f < RECORD_FLOATS; ++f)
                    rec[f] = 0.0f;
                rec[0] = (float)_prns[p];
                *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = utc;
                if (!active[p]) {
                    f_ref[p] = std::nan(""); // forget the loop; re-acquire when it returns
                    a_prev_ok[p] = 0;
                    continue;
                }

                // Effective carrier Doppler for the replica = the broker seed. Covering
                // channels follow it.
                const double dop_eff = dop[p];
                const auto cover = _replica->covering_bins(dop_eff, _doppler_margin_hz);
                std::vector<int> owned;
                for (int c : cover)
                    if (c >= _chan_offset && c < _chan_offset + _n_chan)
                        owned.push_back(c);
                // Code phase at this window: first-order extrapolation of the seeded cp0 from
                // its anchor hop (removes seed staleness). The broker's DLL trims the residual
                // at ~Hz from the shipped E/L correlators -- the tracker itself never moves it.
                const double L = (double)_replica->code_length();
                double cp_seed = cp[p] + cp_rate[p] * (double)(window_start_hop - ref_hop[p]);
                cp_seed = std::fmod(cp_seed, L);
                if (cp_seed < 0.0)
                    cp_seed += L;

                // FLL reference Doppler: (re)acquire from the broker seed when uninitialised
                // or far adrift (loss of lock / sat change), else hold it FIXED so the
                // absolute-anchored replica phase stays continuous; the NCO tracks the rest.
                if (_fll_gain > 0.0
                    && (std::isnan(f_ref[p]) || std::fabs(f_ref[p] - dop[p]) > _fll_reacq_hz)) {
                    f_ref[p] = dop[p];
                    f_track[p] = 0.0;
                    phi_track[p] = 0.0;
                    a_prev_ok[p] = 0;
                    reacq_hop[p] = window_start_hop; // anchor the Doppler-rate feed-forward here
                }
                const double fcar = (_fll_gain > 0.0) ? f_ref[p] : dop[p];
                // 2nd-order carrier feed-forward: ramp the replica frequency by the almanac Doppler
                // RATE so the despread residual stays ~constant across a multi-100 ms coherent window.
                // A 1st-order FLL alone velocity-lags the ramp -> a quadratic carrier phase that
                // cancels the deep sum, worst at zenith (max Doppler acceleration). Anchor at the hop
                // where the base carrier was last fixed: reacq_hop (FLL ref) or the fresh broker
                // seed anchor ref_hop (no FLL). With the ramp in the replica, the FLL sees only
                // the small constant clock/fit residual -- no loop-logic change needed.
                const long long ff_anchor = (_fll_gain > 0.0) ? reacq_hop[p] : ref_hop[p];
                const double fcar_eff =
                    fcar + dop_rate[p] * (double)(window_start_hop - ff_anchor) * (double)_fft_len
                               / _sample_rate;
                rec[1] = (float)(fcar_eff + ((_fll_gain > 0.0) ? f_track[p] : 0.0));
                rec[2] = (float)cp_seed;
                rec[6] = (float)owned.size();
                if (owned.empty())
                    continue; // carrier not in this subband

                // The window data is fixed; only the replica varies with the cp trial.
                std::vector<std::vector<cf>> data_ch;
                data_ch.reserve(owned.size());
                for (int c : owned) {
                    std::vector<cf> col(_hops_per_record);
                    const int local = c - _chan_offset;
                    for (int m = 0; m < _hops_per_record; ++m)
                        col[m] = window[(size_t)m * _n_chan + local];
                    data_ch.push_back(std::move(col));
                }

                // Early/Prompt/Late despread -- three DUMB correlators at the COMMANDED
                // code phase (R1, docs/gnss_architecture_audit.md). The Prompt despreads
                // exactly at cp_seed: it is the science output and is phase-continuous by
                // construction, because the tracker makes NO alignment decisions (the
                // retired per-record max-power pull-in re-phased the despread and killed
                // deep coherence; the retired hold_lock false-armed on noise statistics).
                // Early/Late sit at -+_dll_spacing chips on the correlation triangle and
                // exist ONLY so the low-rate DLL downstream can measure the sub-chip
                // tracking error: the combiner aggregates |E|^2 / |L|^2 across subbands and
                // windows, and the BROKER closes the code loop at ~Hz through set_seeds.
                // Only E/L powers are consumed, so they ship un-derotated.
                auto despread_at = [&](double cpx) {
                    const auto repl =
                        _replica->channels(p, window_start, cpx, fcar_eff, _hops_per_record);
                    std::vector<std::vector<cf>> repl_ch;
                    repl_ch.reserve(owned.size());
                    for (int c : owned)
                        repl_ch.push_back(repl[c]);
                    return gnss::channelized_despread(data_ch, repl_ch);
                };
                const auto early = despread_at(cp_seed - _dll_spacing);
                const auto best = despread_at(cp_seed); // prompt
                const auto late = despread_at(cp_seed + _dll_spacing);
                rec[gnss::REC_E_RE] = (float)early.correlation.real();
                rec[gnss::REC_E_IM] = (float)early.correlation.imag();
                rec[gnss::REC_E_ENERGY] = (float)early.replica_energy;
                rec[gnss::REC_L_RE] = (float)late.correlation.real();
                rec[gnss::REC_L_IM] = (float)late.correlation.imag();
                rec[gnss::REC_L_ENERGY] = (float)late.replica_energy;

                if (_fll_gain <= 0.0) {
                    rec[3] = (float)best.correlation.real();
                    rec[4] = (float)best.correlation.imag();
                    rec[5] = (float)best.replica_energy;
                } else {
                    // Carrier NCO + FLL. The replica is at the fixed f_ref, so the despread
                    // carries the residual carrier (true - f_ref). Advance the phase NCO by
                    // the tracked residual over the inter-record gap and derotate the output
                    // -- this corrects frequency WITHOUT retuning the absolute-anchored
                    // replica. The bit-robust discriminator (square the phase product to
                    // cancel the +-1 nav-bit pi flip, valid while |err*dt|<1/4) then drives
                    // f_track toward the residual; skip it across a frame gap (it would alias).
                    const double dt = (double)(window_start_hop - hop_prev[p]) * (double)_fft_len
                                      / _sample_rate;
                    if (a_prev_ok[p] && dt > 0.0) {
                        phi_track[p] += 2.0 * M_PI * f_track[p] * dt; // NCO (advances over gaps too)
                        phi_track[p] = std::remainder(phi_track[p], 2.0 * M_PI); // keep bounded
                    }
                    const std::complex<double> rot = std::polar(1.0, -phi_track[p]);
                    const std::complex<double> g_corr = std::complex<double>(best.correlation) * rot;
                    const cf a_corr = (cf)(std::complex<double>(best.amplitude) * rot);
                    rec[3] = (float)g_corr.real();
                    rec[4] = (float)g_corr.imag();
                    rec[5] = (float)best.replica_energy;

                    if (a_prev_ok[p] && dt > 0.0 && dt <= _fll_max_gap
                        && std::abs(a_corr) > (float)_fll_lock_amp
                        && std::abs(a_prev[p]) > (float)_fll_lock_amp) {
                        const std::complex<double> prod =
                            std::complex<double>(a_corr) * std::conj(std::complex<double>(a_prev[p]));
                        const double dphi = std::arg(prod * prod) / 2.0; // bit-robust phase walk
                        f_track[p] += _fll_gain * dphi / (2.0 * M_PI * dt); // += gain * (residual Hz)
                    }
                    a_prev[p] = a_corr;
                    a_prev_ok[p] = 1;
                    hop_prev[p] = window_start_hop;
                }
            }

            // Stamp the window's absolute sample on the record frame: it survives
            // bufferSend/Recv (records ship back to the combiner across nodes) and lets
            // the combiner verify per-subband window alignment before summing.
            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->sample_seq = window_start;
            }
            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out = (frame_out + 1) % out_buf->num_frames;
        }

        seen_hops += frame_hops;
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
    }
}
