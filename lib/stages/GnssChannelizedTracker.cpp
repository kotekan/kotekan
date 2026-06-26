#include "GnssChannelizedTracker.hpp"

#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "GnssChanMetadata.hpp"        // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSignal.hpp"              // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"          // for FATAL_ERROR
#include "pfbPrototype.hpp"            // for window_from_string

#include <algorithm> // for fill
#include <chrono>    // for system_clock
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
    _pullin_chips = config.get_default<double>(unique_name, "pullin_chips", 0.0);
    _pullin_step = config.get_default<double>(unique_name, "pullin_step", 0.5);

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

        // Absolute hop index of this frame's first hop, from fpga_seq (shared across
        // nodes so every tracker's window_start agrees); fall back to a local count.
        long long frame_first_hop = seen_hops;
        if (metadata_is_gnss_chan(in_buf)) {
            auto* mi = get_gnss_chan_metadata(in_buf, frame_in);
            if (mi && mi->fpga_seq >= 0)
                frame_first_hop = mi->fpga_seq / _fft_len;
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
            std::vector<double> dop, cp;
            std::vector<uint8_t> active;
            {
                std::lock_guard<std::mutex> lk(_seed_mtx);
                dop = _doppler;
                cp = _code_phase;
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
                if (!active[p])
                    continue;

                // Covering channels for this carrier that fall in this subband.
                const auto cover = _replica->covering_bins(dop[p], _doppler_margin_hz);
                std::vector<int> owned;
                for (int c : cover)
                    if (c >= _chan_offset && c < _chan_offset + _n_chan)
                        owned.push_back(c);
                rec[1] = (float)dop[p];
                rec[2] = (float)cp[p];
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

                // Code pull-in: the seed cp is ~1 chip stale (search latency + residual
                // cp drift), so despread over a small cp window and lock to the peak |A|
                // -- a DLL-style search that makes the lock drift/latency-independent
                // (this tracker spans all covering channels, so one peak cp is coherent
                // across them). pullin_chips=0 -> a single despread at the seed.
                gnss::DespreadResult best{};
                double best_pw = -1.0, best_cp = cp[p];
                for (double off = -_pullin_chips; off <= _pullin_chips + 1e-9;
                     off += (_pullin_step > 0.0 ? _pullin_step : 1.0)) {
                    const auto repl =
                        _replica->channels(p, window_start, cp[p] + off, dop[p], _hops_per_record);
                    std::vector<std::vector<cf>> repl_ch;
                    repl_ch.reserve(owned.size());
                    for (int c : owned)
                        repl_ch.push_back(repl[c]);
                    const auto res = gnss::channelized_despread(data_ch, repl_ch);
                    const double pw = std::norm(res.amplitude);
                    if (pw > best_pw) {
                        best_pw = pw;
                        best = res;
                        best_cp = cp[p] + off;
                    }
                    if (_pullin_chips <= 0.0)
                        break;
                }
                rec[2] = (float)best_cp; // the locked (pulled-in) code phase
                rec[3] = (float)best.correlation.real();
                rec[4] = (float)best.correlation.imag();
                rec[5] = (float)best.replica_energy;
            }

            // Stamp the window's absolute sample on the record frame: it survives
            // bufferSend/Recv (records ship back to the combiner across nodes) and lets
            // the combiner verify per-subband window alignment before summing.
            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->fpga_seq = window_start;
            }
            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out = (frame_out + 1) % out_buf->num_frames;
        }

        seen_hops += frame_hops;
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
    }
}
