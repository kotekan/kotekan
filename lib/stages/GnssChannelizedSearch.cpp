#include "GnssChannelizedSearch.hpp"

#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "GnssChanMetadata.hpp"        // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSignal.hpp"              // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"          // for INFO, FATAL_ERROR
#include "pfbPrototype.hpp"            // for window_from_string

#include <algorithm>  // for min, max
#include <cmath>      // for fabs, fmod, norm
#include <cstring>    // for memcpy
#include <functional> // for bind
#include "json.hpp"   // for json

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssChannelizedSearch);

GnssChannelizedSearch::GnssChannelizedSearch(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChannelizedSearch::main_thread, this)),
    _snap_ready(false), _worker_busy(false), _worker_stop(false) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr) {
        FATAL_ERROR("GnssChannelizedSearch: unknown signal '{:s}'", signal);
        return;
    }

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    const double f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    _doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);

    _N = config.get<int>(unique_name, "spectrum_length");
    _fft_len = 2 * _N;
    _chan_offset = config.get_default<int>(unique_name, "channel_offset", 0);
    _n_chan = config.get<int>(unique_name, "n_channels");
    if (_chan_offset < 0 || _n_chan <= 0 || _chan_offset + _n_chan > _N) {
        FATAL_ERROR("GnssChannelizedSearch: channel slice [{:d},{:d}) invalid for N={:d}",
                    _chan_offset, _chan_offset + _n_chan, _N);
        return;
    }
    const int num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");

    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty()) {
        FATAL_ERROR("GnssChannelizedSearch: 'prns' is empty");
        return;
    }
    _acquire_snr = config.get_default<double>(unique_name, "acquire_snr", 12.0);
    _acquire_windows = config.get_default<int>(unique_name, "acquire_windows", 64);
    _hold_snapshots = config.get_default<int>(unique_name, "hold_snapshots", 5);
    const double dmin = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    const double dmax = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    const double dstep = config.get_default<double>(unique_name, "doppler_step", 500.0);
    for (double f = dmin; f <= dmax + 1e-6; f += dstep)
        _doppler_grid.push_back(f);

    try {
        _replica = std::make_unique<gnss::ChannelizedReplicaBank>(
            *sig, _sample_rate, f_offset, _N, num_taps, dsp::window_from_string(win), _prns);
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssChannelizedSearch: {:s}", e.what());
        return;
    }
    _hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", _replica->repl_period_hops());

    _detections.assign(_prns.size(), Detection{});
    _snap_hops = (size_t)_acquire_windows * _hops_per_record;
    _snapshot.assign(_snap_hops * (size_t)_n_chan, cf(0.0f, 0.0f));
}

GnssChannelizedSearch::~GnssChannelizedSearch() {
    {
        std::lock_guard<std::mutex> lk(_m);
        _worker_stop = true;
        _cv.notify_one();
    }
    if (_worker.joinable())
        _worker.join();
}

void GnssChannelizedSearch::search_snapshot() {
    const int n_prn = (int)_prns.size();
    const int Mp = _replica->repl_period_hops();
    const int hpr = _hops_per_record;
    const long long anchor = (long long)Mp * _fft_len; // warm-up reads periodic code
    const double cps = _replica->chip_rate_hz() / _sample_rate;
    const double dmax = _doppler_grid.empty()
                            ? 0.0
                            : std::max(std::fabs(_doppler_grid.front()),
                                       std::fabs(_doppler_grid.back()));
    const int nwin = std::min(_acquire_windows, (int)(_snap_hops / (size_t)hpr));

    // Covering channels (global) for this carrier that fall in this subband.
    const auto cover = _replica->covering_bins(dmax, _doppler_margin_hz);
    std::vector<int> cov_local, cov_global;
    for (int c : cover)
        if (c >= _chan_offset && c < _chan_offset + _n_chan) {
            cov_local.push_back(c - _chan_offset);
            cov_global.push_back(c);
        }

    for (int p = 0; p < n_prn; ++p) {
        if (cov_local.empty()) { // carrier not in this subband: nothing to find
            std::lock_guard<std::mutex> lk(_det_mtx);
            _detections[p] = Detection{};
            continue;
        }

        // repl0 (code 0, Doppler 0), sliced to this subband's local channels.
        const auto repl_full = _replica->channels(p, anchor, 0.0, 0.0, Mp); // [N][Mp]
        std::vector<std::vector<cf>> repl0(_n_chan);
        for (int lc = 0; lc < _n_chan; ++lc)
            repl0[lc] = repl_full[_chan_offset + lc];

        // Integrate the |D|^2 surface over the snapshot windows (incoherent).
        std::vector<double> surf;
        gnss::AcquisitionSurface dims{};
        std::vector<std::vector<cf>> dch(_n_chan, std::vector<cf>(hpr));
        for (int w = 0; w < nwin; ++w) {
            for (int lc = 0; lc < _n_chan; ++lc)
                for (int m = 0; m < hpr; ++m)
                    dch[lc][m] = _snapshot[((size_t)(w * hpr + m)) * _n_chan + lc];
            dims = gnss::channelized_accumulate(dch, repl0, cov_local, _doppler_grid, _sample_rate,
                                                _n_chan, surf, _acq_ws, cov_global, _fft_len);
        }
        const auto a = gnss::channelized_peak(surf, dims, _doppler_grid, _sample_rate,
                                              _replica->chip_rate_hz(), _replica->code_length());

        const bool detected = a.snr >= _acquire_snr;
        Detection det;
        det.snr = (float)a.snr;
        if (detected) {
            // r2c fold conjugates the channel frequency axis -> flip Doppler sign.
            const double dop = -a.doppler_hz;
            // Refine code phase to sample resolution: ±1 hop exact-despread scan on
            // window 0, absorbing the coarse-lag (hop) quantization of the acquire.
            const int span = _fft_len;
            double best_cp = a.code_phase_chips, best_pw = -1.0;
            for (int off = -span; off <= span; ++off) {
                const double cp = a.code_phase_chips + off * cps;
                const auto repl = _replica->channels(p, anchor, cp, dop, hpr);
                std::vector<std::vector<cf>> d, r;
                for (size_t i = 0; i < cov_local.size(); ++i) {
                    std::vector<cf> col(hpr);
                    for (int m = 0; m < hpr; ++m)
                        col[m] = _snapshot[((size_t)m) * _n_chan + cov_local[i]];
                    d.push_back(std::move(col));
                    r.push_back(repl[cov_global[i]]);
                }
                const double pw = std::norm(gnss::channelized_despread(d, r).amplitude);
                if (pw > best_pw) {
                    best_pw = pw;
                    best_cp = cp;
                }
            }
            // best_cp is the satellite code phase at the snapshot's absolute start
            // (S_snap = snap_start_hop * fft_len). Reference it to absolute sample 0
            // so the tracker (on the same hop*2N grid) can seed directly:
            // cp0 = best_cp - S_snap*chip_per_sample (mod L). Reduce the hop offset
            // modulo the replica period first to keep float precision.
            const double L = (double)_replica->code_length();
            const double off = std::fmod((double)(_snap_start_hop % Mp) * (double)_fft_len * cps, L);
            double cp = std::fmod(best_cp - off, L);
            if (cp < 0.0)
                cp += L;
            det.doppler_hz = dop;
            det.code_phase_chips = cp;
            det.valid = true;
            INFO("GnssChannelizedSearch[{:s}]: PRN {:d} detected (Doppler {:+.0f} Hz, cp {:.1f} "
                 "chips, snr {:.1f})",
                 unique_name, _prns[p], dop, cp, a.snr);
        }
        // Latch: a fresh detection replaces the held one; a miss keeps the last valid
        // detection for _hold_snapshots snapshots (a momentary miss must not drop the
        // seed), then expires it.
        std::lock_guard<std::mutex> lk(_det_mtx);
        if (detected) {
            _detections[p] = det;
        } else {
            _detections[p].snr = det.snr;
            if (_detections[p].valid && ++_detections[p].misses > _hold_snapshots)
                _detections[p].valid = false;
        }
    }
}

void GnssChannelizedSearch::search_worker() {
    while (true) {
        {
            std::unique_lock<std::mutex> lk(_m);
            _cv.wait(lk, [&] { return _snap_ready || _worker_stop; });
            if (_worker_stop && !_snap_ready)
                return;
            _snap_ready = false;
        }
        search_snapshot(); // snapshot stable: _worker_busy keeps the drain off it
        {
            std::lock_guard<std::mutex> lk(_m);
            _worker_busy = false;
        }
    }
}

void GnssChannelizedSearch::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_detections",
        std::bind(&GnssChannelizedSearch::get_detections_callback, this, _1));

    _worker = std::thread(&GnssChannelizedSearch::search_worker, this);

    int frame_in = 0;
    bool filling = false;
    size_t fill_hops = 0;
    long long abs_hops = 0; // total hops consumed from the stream (absolute reference)

    while (!stop_thread) {
        auto* in_local = (cf*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        const int frame_hops = in_buf->frame_size / (int)sizeof(cf) / _n_chan;

        // Absolute hop index of this frame's first hop, from the F-engine's fpga_seq
        // metadata (shared across nodes); fall back to a local count if absent.
        long long frame_first_hop = abs_hops;
        if (metadata_is_gnss_chan(in_buf)) {
            auto* mi = get_gnss_chan_metadata(in_buf, frame_in);
            if (mi && mi->fpga_seq >= 0)
                frame_first_hop = mi->fpga_seq / _fft_len;
        }

        if (!filling) {
            std::lock_guard<std::mutex> lk(_m);
            if (!_worker_busy) {
                filling = true;
                fill_hops = 0;
                _snap_start_hop = frame_first_hop; // snapshot hop 0 is this frame's hop 0
            }
        }
        if (filling) {
            const size_t take = std::min((size_t)frame_hops, _snap_hops - fill_hops);
            std::memcpy(&_snapshot[fill_hops * (size_t)_n_chan], in_local,
                        take * (size_t)_n_chan * sizeof(cf));
            fill_hops += take;
            if (fill_hops >= _snap_hops) {
                filling = false;
                std::lock_guard<std::mutex> lk(_m);
                _snap_ready = true;
                _worker_busy = true;
                _cv.notify_one();
            }
        }

        // Always release the frame immediately -- never backpressure.
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
        abs_hops += frame_hops;
    }

    {
        std::lock_guard<std::mutex> lk(_m);
        _worker_stop = true;
        _cv.notify_one();
    }
    if (_worker.joinable())
        _worker.join();
}

void GnssChannelizedSearch::get_detections_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::array();
    std::lock_guard<std::mutex> lk(_det_mtx);
    for (size_t p = 0; p < _prns.size(); ++p) {
        const Detection& d = _detections[p];
        if (!d.valid)
            continue;
        reply.push_back({{"prn", _prns[p]},
                         {"doppler_hz", d.doppler_hz},
                         {"code_phase_chips", d.code_phase_chips},
                         {"snr", d.snr}});
    }
    conn.send_json_reply(reply);
}
