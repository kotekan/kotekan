#include "GnssAcquire.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp" // for make_input_desc
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssSignal.hpp" // for signal_by_name
#include "kotekanLogging.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssAcquire);

static inline double parab_offset(double l, double c, double r) {
    const double denom = l - 2.0 * c + r;
    if (denom >= 0.0)
        return 0.0;
    double d = 0.5 * (l - r) / denom;
    return std::max(-0.5, std::min(0.5, d));
}

GnssAcquire::GnssAcquire(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssAcquire::main_thread, this)),
    snap_ready(false), worker_busy(false), worker_stop(false) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr)
        FATAL_ERROR("GnssAcquire: unknown signal '{:s}'.", signal);
    _sig = *sig;

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    const double f_offset = config.get_default<double>(unique_name, "f_offset", 1e6);
    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty())
        FATAL_ERROR("GnssAcquire: 'prns' list is empty.");

    _doppler_min = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    _doppler_max = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    _doppler_step = config.get_default<double>(unique_name, "doppler_step", 250.0);
    _search_blocks = config.get_default<int>(unique_name, "search_blocks", 200);
    _acquire_snr = config.get_default<double>(unique_name, "acquire_snr", 9.0);
    if (_doppler_step <= 0 || _doppler_max < _doppler_min || _search_blocks < 1)
        FATAL_ERROR("GnssAcquire: invalid Doppler grid / search_blocks.");

    const std::string group = config.get_default<std::string>(unique_name, "state_group", "gnss");
    _state = &gnss::track_state(group);

    core = std::make_unique<gnss::CorrelatorCore>(_sig, _sample_rate, f_offset, _prns);
    _Ns = core->Ns();

    for (double f = _doppler_min; f <= _doppler_max + 1e-6; f += _doppler_step)
        _doppler_grid.push_back(f);

    in_samples_per_frame = in_buf->frame_size / sizeof(int16_t);
    in_buf->set_frame_desc(kotekan_airspy::make_input_desc(in_samples_per_frame));

    snap_len = (size_t)_search_blocks * _Ns;
    snapshot.assign(snap_len, 0.0f);
    accum.assign((size_t)_prns.size() * _doppler_grid.size() * _Ns, 0.0f);

    INFO("GnssAcquire: signal={:s}, Fs={:.3f} MHz, Ns={:d}, {:d} PRNs, {:d} Doppler bins "
         "({:.0f}..{:.0f} step {:.0f} Hz), snapshot {:d} blocks, acquire_snr={:.1f}, group='{:s}'",
         _sig.name, _sample_rate / 1e6, _Ns, (int)_prns.size(), (int)_doppler_grid.size(),
         _doppler_min, _doppler_max, _doppler_step, _search_blocks, _acquire_snr, group);
}

void GnssAcquire::search_snapshot() {
    const std::set<int> owned = _state->owned();
    const int n_prn = (int)_prns.size();
    const int n_dop = (int)_doppler_grid.size();

    std::vector<uint8_t> cand(n_prn);
    int n_cand = 0;
    for (int p = 0; p < n_prn; ++p) {
        cand[p] = owned.count(_prns[p]) ? 0 : 1;
        n_cand += cand[p];
    }
    if (n_cand == 0)
        return; // tracker owns everything; nothing to search

    std::fill(accum.begin(), accum.end(), 0.0f);

    // Hoisted batch correlation: data FFT once per (block, Doppler), reused
    // across all candidate PRNs.
    for (int b = 0; b < _search_blocks; ++b) {
        core->to_analytic(&snapshot[(size_t)b * _Ns]);
        for (int di = 0; di < n_dop; ++di) {
            core->prepare_doppler(_doppler_grid[di]);
            for (int p = 0; p < n_prn; ++p) {
                if (!cand[p])
                    continue;
                core->correlate_into(p, &accum[((size_t)p * n_dop + di) * _Ns], nullptr);
            }
        }
    }

    // Peak per candidate; promote detections.
    for (int p = 0; p < n_prn; ++p) {
        if (!cand[p])
            continue;
        const float* slab = &accum[(size_t)p * n_dop * _Ns];
        float peak = -1.0f;
        int best_di = 0, best_k = 0;
        double sum = 0.0;
        for (int di = 0; di < n_dop; ++di) {
            const float* m = &slab[(size_t)di * _Ns];
            for (int k = 0; k < _Ns; ++k) {
                sum += m[k];
                if (m[k] > peak) {
                    peak = m[k];
                    best_di = di;
                    best_k = k;
                }
            }
        }
        const double mean = sum / ((double)n_dop * _Ns);
        const float snr = (mean > 0.0) ? (float)(peak / mean) : 0.0f;
        if (snr < _acquire_snr)
            continue;

        double dop = _doppler_grid[best_di];
        if (best_di > 0 && best_di < n_dop - 1) {
            const double dd = parab_offset(slab[(size_t)(best_di - 1) * _Ns + best_k], peak,
                                           slab[(size_t)(best_di + 1) * _Ns + best_k]);
            dop += dd * _doppler_step;
        }
        INFO("GnssAcquire: found PRN {:d} (Doppler {:+.0f} Hz, snr {:.1f}) -> promote", _prns[p],
             dop, snr);
        _state->promote(gnss::Seed{_prns[p], dop, snr});
    }
}

void GnssAcquire::search_worker() {
    while (true) {
        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&] { return snap_ready || worker_stop; });
            if (worker_stop && !snap_ready)
                return;
            snap_ready = false;
        }
        search_snapshot(); // snapshot is stable: worker_busy keeps the drain off it
        {
            std::lock_guard<std::mutex> lk(m);
            worker_busy = false;
        }
    }
}

void GnssAcquire::main_thread() {
    frame_in = 0;
    worker = std::thread(&GnssAcquire::search_worker, this);

    bool filling = false;
    size_t fill_pos = 0;

    while (!stop_thread) {
        int16_t* in_local = (int16_t*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;

        // Start a fresh snapshot only when the worker is free; otherwise drop.
        if (!filling) {
            std::lock_guard<std::mutex> lk(m);
            if (!worker_busy) {
                filling = true;
                fill_pos = 0;
            }
        }
        if (filling) {
            const size_t take = std::min((size_t)in_samples_per_frame, snap_len - fill_pos);
            for (size_t i = 0; i < take; ++i)
                snapshot[fill_pos + i] = (float)in_local[i];
            fill_pos += take;
            if (fill_pos >= snap_len) {
                filling = false;
                std::lock_guard<std::mutex> lk(m);
                snap_ready = true;
                worker_busy = true;
                cv.notify_one();
            }
        }

        // Always release the frame immediately -- never backpressure.
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
    }

    {
        std::lock_guard<std::mutex> lk(m);
        worker_stop = true;
        cv.notify_one();
    }
    if (worker.joinable())
        worker.join();
}
