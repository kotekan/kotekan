#include "GnssSearchAggregator.hpp"

#include "GnssCorrFrame.hpp"  // for CorrFrameHeader, corr_*
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "gnssChannelizedAcquire.hpp" // for aggregate_accumulate, channelized_peak
#include "gnssSignal.hpp"     // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp" // for INFO, WARN, FATAL_ERROR
#include "pfbPrototype.hpp"   // for window_from_string
#include "visUtil.hpp"        // for frameID

#include <cmath>      // for fmod
#include <functional> // for bind
#include "json.hpp"   // for json

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssSearchAggregator);

GnssSearchAggregator::GnssSearchAggregator(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssSearchAggregator::main_thread, this)) {

    in_bufs = get_buffer_array("in_bufs");
    for (Buffer* b : in_bufs)
        b->register_consumer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr) {
        FATAL_ERROR("GnssSearchAggregator: unknown signal '{:s}'", signal);
        return;
    }

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    const double f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    const int N = config.get<int>(unique_name, "spectrum_length");
    _fft_len = 2 * N;
    const int num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");

    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty()) {
        FATAL_ERROR("GnssSearchAggregator: 'prns' is empty");
        return;
    }
    _acquire_snr = config.get_default<double>(unique_name, "acquire_snr", 12.0);
    _hold_snapshots = config.get_default<int>(unique_name, "hold_snapshots", 5);
    const double dmin = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    const double dmax = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    const double dstep = config.get_default<double>(unique_name, "doppler_step", 500.0);
    for (double f = dmin; f <= dmax + 1e-6; f += dstep)
        _doppler_grid.push_back(f);

    try {
        _replica = std::make_unique<gnss::ChannelizedReplicaBank>(
            *sig, _sample_rate, f_offset, N, num_taps, dsp::window_from_string(win), _prns);
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssSearchAggregator: {:s}", e.what());
        return;
    }

    _detections.assign(_prns.size(), Detection{});
}

void GnssSearchAggregator::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_detections",
        std::bind(&GnssSearchAggregator::get_detections_callback, this, _1));

    std::vector<frameID> in_ids;
    for (Buffer* b : in_bufs)
        in_ids.emplace_back(b);

    const int n_prn = (int)_prns.size();
    const int Mp = _replica->repl_period_hops();
    const double cps = _replica->chip_rate_hz() / _sample_rate;
    const double L = (double)_replica->code_length();

    while (!stop_thread) {
        // Gather one P_c frame from every channel (same snapshot, lock-step).
        std::vector<char*> frames;
        bool stopping = false;
        for (size_t i = 0; i < in_bufs.size(); ++i) {
            char* in = (char*)in_bufs[i]->wait_for_full_frame(unique_name, in_ids[i]);
            if (in == nullptr) {
                stopping = true;
                break;
            }
            frames.push_back(in);
        }
        if (stopping)
            break;

        // Validate + pick dimensions/snapshot from the first frame (all match).
        auto* h0 = reinterpret_cast<gnss::CorrFrameHeader*>(frames[0]);
        const int nwin = h0->nwin, nd = h0->nd, sph = h0->sph;
        const long long snap_start_hop = h0->snap_start_hop;
        bool ok = (h0->magic == gnss::CORR_FRAME_MAGIC && h0->Mp == Mp && h0->n_prn == n_prn
                   && nd == (int)_doppler_grid.size());
        for (char* f : frames) {
            auto* h = reinterpret_cast<gnss::CorrFrameHeader*>(f);
            if (h->magic != gnss::CORR_FRAME_MAGIC || h->snap_start_hop != snap_start_hop
                || h->n_prn != n_prn || h->nwin != nwin || h->nd != nd || h->Mp != Mp)
                ok = false;
        }
        if (!ok) {
            WARN("GnssSearchAggregator[{:s}]: inconsistent P_c frames, skipping snapshot",
                 unique_name);
            for (size_t i = 0; i < in_bufs.size(); ++i)
                in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);
            continue;
        }

        // Covering channels present this snapshot (carrier geometry, PRN-independent).
        std::vector<char*> cov_frames;
        std::vector<int> cov_freq;
        for (char* f : frames) {
            auto* h = reinterpret_cast<gnss::CorrFrameHeader*>(f);
            if (h->covers) {
                cov_frames.push_back(f);
                cov_freq.push_back(h->chan_freq);
            }
        }
        const int nc = (int)cov_frames.size();

        // Absolute-reference offset: cp reported is the satellite code phase at the
        // snapshot's absolute start; subtract the start's code advance so the seed
        // is node-independent (same convention as GnssChannelizedSearch).
        const double off = std::fmod((double)(snap_start_hop % Mp) * (double)sph * cps, L);

        for (int p = 0; p < n_prn; ++p) {
            if (nc == 0) { // carrier not covered anywhere this snapshot
                std::lock_guard<std::mutex> lk(_det_mtx);
                if (_detections[p].valid && ++_detections[p].misses > _hold_snapshots)
                    _detections[p].valid = false;
                continue;
            }

            // Cross-channel combine: for each window, gather P_c[ci][d][q] and
            // accumulate |D|^2 into the surface (incoherent over windows).
            std::vector<double> surf;
            gnss::AcquisitionSurface dims{};
            std::vector<std::vector<std::vector<cf>>> Pw(nc,
                std::vector<std::vector<cf>>(nd, std::vector<cf>(Mp)));
            for (int w = 0; w < nwin; ++w) {
                for (int ci = 0; ci < nc; ++ci) {
                    const cf* pay =
                        reinterpret_cast<const cf*>(cov_frames[ci] + gnss::corr_payload_offset());
                    for (int d = 0; d < nd; ++d)
                        for (int q = 0; q < Mp; ++q)
                            Pw[ci][d][q] = pay[gnss::corr_index(p, w, d, q, nwin, nd, Mp)];
                }
                dims = gnss::aggregate_accumulate(Pw, cov_freq, sph, surf);
            }
            const auto a = gnss::channelized_peak(surf, dims, _doppler_grid, _sample_rate,
                                                  _replica->chip_rate_hz(), _replica->code_length());

            const bool detected = a.snr >= _acquire_snr;
            Detection det;
            det.snr = (float)a.snr;
            if (detected) {
                const double dop = -a.doppler_hz; // r2c fold conjugates freq axis
                double cp = std::fmod(a.code_phase_chips - off, L);
                if (cp < 0.0)
                    cp += L;
                det.doppler_hz = dop;
                det.code_phase_chips = cp;
                det.valid = true;
                INFO("GnssSearchAggregator[{:s}]: PRN {:d} detected (Doppler {:+.0f} Hz, cp {:.1f} "
                     "chips, snr {:.1f}, {:d} chan)",
                     unique_name, _prns[p], dop, cp, a.snr, nc);
            }
            std::lock_guard<std::mutex> lk(_det_mtx);
            if (detected) {
                _detections[p] = det;
            } else {
                _detections[p].snr = det.snr;
                if (_detections[p].valid && ++_detections[p].misses > _hold_snapshots)
                    _detections[p].valid = false;
            }
        }

        for (size_t i = 0; i < in_bufs.size(); ++i)
            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);
    }
}

void GnssSearchAggregator::get_detections_callback(kotekan::connectionInstance& conn) {
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
