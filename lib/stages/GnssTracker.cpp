#include "GnssTracker.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp" // for make_input_desc
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssRecord.hpp" // for RECORD_FLOATS, RECORD_UTC_SLOT
#include "gnssSignal.hpp" // for signal_by_name
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <set>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using gnss::RECORD_FLOATS;
using gnss::RECORD_UTC_SLOT;

REGISTER_KOTEKAN_STAGE(GnssTracker);

// FLL loop-filter gain (per block) for the tracked-Doppler estimate.
static constexpr double FLL_ALPHA = 0.1;

static inline double parab_offset(double l, double c, double r) {
    const double denom = l - 2.0 * c + r;
    if (denom >= 0.0)
        return 0.0;
    double d = 0.5 * (l - r) / denom;
    return std::max(-0.5, std::min(0.5, d));
}

GnssTracker::GnssTracker(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssTracker::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr)
        FATAL_ERROR("GnssTracker: unknown signal '{:s}'.", signal);
    _sig = *sig;

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    _f_offset = config.get_default<double>(unique_name, "f_offset", 1e6);
    _capture_utc0 = config.get_default<double>(unique_name, "capture_utc0", 0.0);
    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty())
        FATAL_ERROR("GnssTracker: 'prns' list is empty.");

    _track_step = config.get_default<double>(unique_name, "track_doppler_step", 50.0);
    _track_guard = config.get_default<int>(unique_name, "track_doppler_guard", 3);
    _lock_snr = config.get_default<double>(unique_name, "lock_snr", 12.0);
    _drop_after = config.get_default<int>(unique_name, "drop_after", 200);
    _record_decimate = config.get_default<int>(unique_name, "record_decimate", 1);
    if (_track_step <= 0 || _track_guard < 0 || _record_decimate < 1)
        FATAL_ERROR("GnssTracker: invalid track grid / record_decimate.");

    const std::string group = config.get_default<std::string>(unique_name, "state_group", "gnss");
    _state = &gnss::track_state(group);

    core = std::make_unique<gnss::CorrelatorCore>(_sig, _sample_rate, _f_offset, _prns);
    _Ns = core->Ns();

    const int n_prn = (int)_prns.size();
    const size_t record_bytes = (size_t)n_prn * RECORD_FLOATS * sizeof(float);
    if ((size_t)out_buf->frame_size != record_bytes)
        FATAL_ERROR("GnssTracker: out_buf frame_size ({:d} B) must equal n_prn*RECORD_FLOATS*4 = "
                    "{:d} B.",
                    out_buf->frame_size, record_bytes);
    in_buf->set_frame_desc(kotekan_airspy::make_input_desc(in_buf->frame_size / sizeof(int16_t)));

    INFO("GnssTracker: signal={:s}, Fs={:.3f} MHz, Ns={:d}, f_offset={:.3f} kHz, {:d} PRNs, "
         "track +-{:d} bins x {:.0f} Hz, lock_snr={:.1f}, drop_after={:d}, group='{:s}'",
         _sig.name, _sample_rate / 1e6, _Ns, _f_offset / 1e3, n_prn, _track_guard, _track_step,
         _lock_snr, _drop_after, group);

    block.assign(_Ns, 0.0f);
    in_local = nullptr;
    in_pos = 0;
    in_samples_per_frame = in_buf->frame_size / sizeof(int16_t);

    trk_active.assign(n_prn, 0);
    trk_locked.assign(n_prn, 0);
    trk_below.assign(n_prn, 0);
    trk_sign.assign(n_prn, 1.0f);
    trk_prev.assign(n_prn, std::complex<float>(0.0f, 0.0f));
    trk_cont_phase.assign(n_prn, 0.0);
    trk_prev_wrapped.assign(n_prn, 0.0);
    trk_prev_doppler.assign(n_prn, 0.0);
    trk_prev_code_phase.assign(n_prn, 0.0);
    trk_doppler.assign(n_prn, 0.0);
    last_amp.assign(n_prn, 0.0f);
    last_snr.assign(n_prn, 0.0f);
    last_code_phase.assign(n_prn, 0.0f);
    last_cpx.assign(n_prn, std::complex<float>(0.0f, 0.0f));
}

bool GnssTracker::fill_block() {
    int block_fill = 0;
    while (block_fill < _Ns) {
        if (in_local == nullptr || in_pos >= in_samples_per_frame) {
            if (in_local != nullptr) {
                in_buf->mark_frame_empty(unique_name, frame_in);
                frame_in = (frame_in + 1) % in_buf->num_frames;
            }
            in_local = (int16_t*)in_buf->wait_for_full_frame(unique_name, frame_in);
            if (in_local == nullptr)
                return false;
            in_pos = 0;
        }
        const int take = std::min(_Ns - block_fill, in_samples_per_frame - in_pos);
        for (int i = 0; i < take; ++i)
            block[block_fill + i] = (float)in_local[in_pos + i];
        block_fill += take;
        in_pos += take;
    }
    return true;
}

void GnssTracker::track_phase_update(int p, std::complex<float> raw, float snr, double doppler_bin,
                                     double doppler_seed, double code_phase_chips, float& nav_sign,
                                     float& phase_cont, float& doppler_out) {
    auto wrap = [](double x) {
        while (x > M_PI)
            x -= 2.0 * M_PI;
        while (x <= -M_PI)
            x += 2.0 * M_PI;
        return x;
    };

    constexpr double CODE_TOL_CHIPS = 3.0;
    const double dopp_tol = 1.5 * _track_step;
    const double dt = _sig.code_period_s; // tracking runs every block

    if (snr < _lock_snr) {
        trk_locked[p] = 0;
        nav_sign = 0.0f;
        phase_cont = 0.0f;
        doppler_out = (float)doppler_seed;
        return;
    }

    double code_jump = std::fabs(code_phase_chips - trk_prev_code_phase[p]);
    code_jump = std::min(code_jump, (double)_sig.code_length - code_jump);
    const bool consistent = trk_locked[p]
                            && std::fabs(doppler_bin - trk_prev_doppler[p]) <= dopp_tol
                            && code_jump <= CODE_TOL_CHIPS;

    if (!consistent) {
        trk_sign[p] = 1.0f;
        const double w = std::atan2(raw.imag(), raw.real());
        trk_cont_phase[p] = w;
        trk_prev_wrapped[p] = w;
        trk_prev[p] = raw;
        trk_doppler[p] = doppler_seed;
    } else {
        const double pred = trk_prev_wrapped[p] + 2.0 * M_PI * (trk_doppler[p] - doppler_bin) * dt;
        std::complex<float> corr = trk_sign[p] * raw;
        double w = std::atan2(corr.imag(), corr.real());
        if (std::fabs(wrap(w - pred)) > M_PI / 2.0) {
            trk_sign[p] = -trk_sign[p];
            corr = -corr;
            w = std::atan2(corr.imag(), corr.real());
        }
        const double dphi = wrap(w - trk_prev_wrapped[p]);
        trk_cont_phase[p] += dphi;
        const double meas_doppler = doppler_bin + dphi / (2.0 * M_PI * dt);
        trk_doppler[p] = (1.0 - FLL_ALPHA) * trk_doppler[p] + FLL_ALPHA * meas_doppler;
        trk_prev_wrapped[p] = w;
        trk_prev[p] = corr;
    }

    trk_locked[p] = 1;
    trk_prev_doppler[p] = doppler_bin;
    trk_prev_code_phase[p] = code_phase_chips;
    nav_sign = trk_sign[p];
    phase_cont = (float)trk_cont_phase[p];
    doppler_out = (float)trk_doppler[p];
}

void GnssTracker::get_locked_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::array();
    const int n_prn = (int)_prns.size();
    for (int p = 0; p < n_prn; ++p) {
        if (!trk_active[p])
            continue;
        reply.push_back({{"prn", _prns[p]},
                         {"locked", (bool)trk_locked[p]},
                         {"doppler_hz", trk_doppler[p]},
                         {"snr", last_snr[p]},
                         {"amp", last_amp[p]}});
    }
    conn.send_json_reply(reply);
}

void GnssTracker::main_thread() {
    frame_in = 0;
    frame_out = 0;
    const int n_prn = (int)_prns.size();
    const int n_bins = 2 * _track_guard + 1;
    const long long fs_int = std::llround(_sample_rate);

    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_locked", std::bind(&GnssTracker::get_locked_callback, this, _1));

    // Per-bin scratch for one PRN's narrow Doppler search.
    std::vector<float> mag2((size_t)n_bins * _Ns);
    std::vector<std::complex<float>> cpx((size_t)n_bins * _Ns);
    std::vector<double> bin_fd(n_bins);

    long long block_index = 0;
    while (!stop_thread) {
        if (!fill_block())
            break;
        const long long block_start = block_index * (long long)_Ns;
        block_index++;

        // Drain search hand-offs: (re)seed each promoted PRN's Doppler.
        for (const gnss::Seed& s : _state->take_pending()) {
            for (int p = 0; p < n_prn; ++p) {
                if (_prns[p] != s.prn)
                    continue;
                trk_doppler[p] = s.doppler_hz;
                trk_active[p] = 1;
                trk_locked[p] = 0; // force a fresh continuous-phase record
                trk_below[p] = 0;
                INFO("GnssTracker: acquired PRN {:d} (Doppler {:+.0f} Hz, acq snr {:.1f})", s.prn,
                     s.doppler_hz, s.snr);
                break;
            }
        }

        core->to_analytic(block.data());

        for (int p = 0; p < n_prn; ++p) {
            if (!trk_active[p])
                continue;

            // Narrow Doppler window centred on the tracked value.
            std::fill(mag2.begin(), mag2.end(), 0.0f);
            const double center = trk_doppler[p];
            for (int b = 0; b < n_bins; ++b) {
                bin_fd[b] = center + (b - _track_guard) * _track_step;
                core->prepare_doppler(bin_fd[b]);
                core->correlate_into(p, &mag2[(size_t)b * _Ns], &cpx[(size_t)b * _Ns]);
            }

            // Peak over (bin, lag); mean for SNR.
            float peak = -1.0f;
            int best_b = 0, best_k = 0;
            double sum = 0.0;
            for (int b = 0; b < n_bins; ++b) {
                const float* m = &mag2[(size_t)b * _Ns];
                for (int k = 0; k < _Ns; ++k) {
                    sum += m[k];
                    if (m[k] > peak) {
                        peak = m[k];
                        best_b = b;
                        best_k = k;
                    }
                }
            }
            const double mean = sum / ((double)n_bins * _Ns);
            const float snr = (mean > 0.0) ? (float)(peak / mean) : 0.0f;

            // Sub-bin Doppler + sub-sample code-phase refinement.
            double dop_refined = bin_fd[best_b];
            if (best_b > 0 && best_b < n_bins - 1) {
                const double dd = parab_offset(mag2[(size_t)(best_b - 1) * _Ns + best_k], peak,
                                               mag2[(size_t)(best_b + 1) * _Ns + best_k]);
                dop_refined += dd * _track_step;
            }
            const int km1 = (best_k - 1 + _Ns) % _Ns, kp1 = (best_k + 1) % _Ns;
            const double dk = parab_offset(mag2[(size_t)best_b * _Ns + km1], peak,
                                           mag2[(size_t)best_b * _Ns + kp1]);
            const float code_phase_chips =
                (float)(((best_k + dk) * (double)_sig.code_length) / _Ns);

            // De-rotate the kept peak to undo the per-block wipeoff phase reset,
            // keeping carrier phase continuous across blocks.
            const double fwipe = _f_offset + bin_fd[best_b];
            double cyc = fwipe * (double)(block_start % fs_int) / _sample_rate;
            cyc -= std::floor(cyc);
            const double carg = -2.0 * M_PI * cyc;
            const std::complex<float> peak_cpx =
                cpx[(size_t)best_b * _Ns + best_k]
                * std::complex<float>((float)std::cos(carg), (float)std::sin(carg));

            float nav_sign = 0.0f, phase_cont = 0.0f, doppler_out = (float)dop_refined;
            track_phase_update(p, peak_cpx, snr, bin_fd[best_b], dop_refined, code_phase_chips,
                               nav_sign, phase_cont, doppler_out);

            last_amp[p] = std::sqrt(std::max(0.0f, peak));
            last_snr[p] = snr;
            last_code_phase[p] = code_phase_chips;
            last_cpx[p] = peak_cpx;

            // Lock hysteresis: release to the search after a run of misses.
            if (snr < _lock_snr) {
                if (++trk_below[p] >= _drop_after) {
                    trk_active[p] = 0;
                    trk_locked[p] = 0;
                    INFO("GnssTracker: dropped PRN {:d} (lost lock).", _prns[p]);
                }
            } else {
                trk_below[p] = 0;
            }
        }

        // Publish the owned set so the search skips these PRNs.
        std::set<int> owned;
        for (int p = 0; p < n_prn; ++p)
            if (trk_active[p])
                owned.insert(_prns[p]);
        _state->set_owned(owned);

        // Emit a record frame on the decimation boundary.
        if (block_index % _record_decimate != 0)
            continue;

        float* out_local = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
        if (out_local == nullptr)
            break;
        const double utc =
            (_capture_utc0 > 0.0)
                ? _capture_utc0 + (double)block_start / _sample_rate
                : std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                      .count();
        for (int p = 0; p < n_prn; ++p) {
            float* rec = out_local + (size_t)p * RECORD_FLOATS;
            rec[0] = (float)_prns[p];
            if (trk_active[p]) {
                rec[1] = (float)trk_doppler[p];
                rec[2] = last_code_phase[p];
                rec[3] = last_amp[p];
                rec[4] = last_cpx[p].real();
                rec[5] = last_cpx[p].imag();
                rec[6] = last_snr[p];
                rec[7] = trk_sign[p];
                rec[8] = (float)trk_cont_phase[p];
            } else {
                for (int f = 1; f < RECORD_UTC_SLOT; ++f)
                    rec[f] = 0.0f;
            }
            *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = utc;
        }
        out_buf->mark_frame_full(unique_name, frame_out);
        frame_out = (frame_out + 1) % out_buf->num_frames;
    }
}
