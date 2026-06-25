#include "GnssChannelCorrelator.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "GnssCorrFrame.hpp"    // for CorrFrameHeader, corr_*
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "gnssSignal.hpp"       // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"   // for INFO, FATAL_ERROR
#include "pfbPrototype.hpp"     // for window_from_string
#include "visUtil.hpp"          // for frameID

#include <algorithm> // for min, max
#include <cmath>     // for fabs
#include <cstring>   // for memcpy
#include <functional> // for bind

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssChannelCorrelator);

GnssChannelCorrelator::GnssChannelCorrelator(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChannelCorrelator::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr) {
        FATAL_ERROR("GnssChannelCorrelator: unknown signal '{:s}'", signal);
        return;
    }

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    const double f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    _doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);

    _N = config.get<int>(unique_name, "spectrum_length");
    _fft_len = 2 * _N;
    _chan_freq = config.get_default<int>(unique_name, "channel_offset", 0);
    if (_chan_freq < 0 || _chan_freq >= _N) {
        FATAL_ERROR("GnssChannelCorrelator: channel {:d} out of range [0,{:d})", _chan_freq, _N);
        return;
    }
    const int num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");

    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty()) {
        FATAL_ERROR("GnssChannelCorrelator: 'prns' is empty");
        return;
    }
    _acquire_windows = config.get_default<int>(unique_name, "acquire_windows", 4);
    const double dmin = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    const double dmax = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    const double dstep = config.get_default<double>(unique_name, "doppler_step", 500.0);
    for (double f = dmin; f <= dmax + 1e-6; f += dstep)
        _doppler_grid.push_back(f);

    try {
        _replica = std::make_unique<gnss::ChannelizedReplicaBank>(
            *sig, _sample_rate, f_offset, _N, num_taps, dsp::window_from_string(win), _prns);
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssChannelCorrelator: {:s}", e.what());
        return;
    }
    _hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", _replica->repl_period_hops());

    // Covering is carrier geometry, PRN-independent: does this channel fall in the
    // carrier's passband over the searched Doppler span?
    const double dabs = _doppler_grid.empty()
                            ? 0.0
                            : std::max(std::fabs(_doppler_grid.front()),
                                       std::fabs(_doppler_grid.back()));
    const auto cover = _replica->covering_bins(dabs, _doppler_margin_hz);
    _covers = std::find(cover.begin(), cover.end(), _chan_freq) != cover.end();

    _snap_hops = (size_t)_acquire_windows * _hops_per_record;
    _snapshot.assign(_snap_hops, cf(0.0f, 0.0f));

    INFO("GnssChannelCorrelator[{:s}]: channel {:d} {:s} the carrier ({:d} PRNs, {:d} windows)",
         unique_name, _chan_freq, _covers ? "covers" : "does not cover", (int)_prns.size(),
         _acquire_windows);
}

void GnssChannelCorrelator::correlate_snapshot(char* out, long long snap_start_hop) {
    const int Mp = _replica->repl_period_hops();
    const int nd = (int)_doppler_grid.size();
    const int hpr = _hops_per_record;
    const int nwin = _acquire_windows;
    const int n_prn = (int)_prns.size();
    const long long anchor = (long long)Mp * _fft_len; // warm-up reads periodic code

    auto* hdr = reinterpret_cast<gnss::CorrFrameHeader*>(out);
    hdr->magic = gnss::CORR_FRAME_MAGIC;
    hdr->chan_freq = _chan_freq;
    hdr->snap_start_hop = snap_start_hop;
    hdr->covers = _covers ? 1 : 0;
    hdr->n_prn = n_prn;
    hdr->nwin = nwin;
    hdr->nd = nd;
    hdr->Mp = Mp;
    hdr->sph = _fft_len;
    if (!_covers)
        return; // header-only frame keeps the aggregator's gather aligned

    cf* payload = reinterpret_cast<cf*>(out + gnss::corr_payload_offset());
    std::vector<cf> data(hpr);
    for (int p = 0; p < n_prn; ++p) {
        // repl0 (code 0, Doppler 0) for this single channel.
        const auto repl_full = _replica->channels(p, anchor, 0.0, 0.0, Mp); // [N][Mp]
        const std::vector<cf>& repl0 = repl_full[_chan_freq];
        for (int w = 0; w < nwin; ++w) {
            for (int m = 0; m < hpr; ++m)
                data[m] = _snapshot[(size_t)(w * hpr + m)];
            // P_c[d][q] for this channel/window.
            const auto P = gnss::channel_correlate(data, repl0, _doppler_grid, _sample_rate,
                                                   _fft_len, _acq_ws);
            for (int d = 0; d < nd; ++d)
                for (int q = 0; q < Mp; ++q)
                    payload[gnss::corr_index(p, w, d, q, nwin, nd, Mp)] = P[d][q];
        }
    }
}

void GnssChannelCorrelator::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf);

    bool filling = false;
    size_t fill_hops = 0;
    long long abs_hops = 0;        // total hops consumed (local fallback reference)
    long long snap_start_hop = 0;  // absolute hop of the snapshot's hop 0

    while (!stop_thread) {
        auto* in_local = (cf*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in_local == nullptr)
            break;
        const int frame_hops = in_buf->frame_size / (int)sizeof(cf); // single channel

        long long frame_first_hop = abs_hops;
        if (metadata_is_gnss_chan(in_buf)) {
            auto* mi = get_gnss_chan_metadata(in_buf, in_id);
            if (mi && mi->fpga_seq >= 0)
                frame_first_hop = mi->fpga_seq / _fft_len;
        }

        int consumed = 0;
        while (consumed < frame_hops) {
            if (!filling) {
                filling = true;
                fill_hops = 0;
                snap_start_hop = frame_first_hop + consumed;
            }
            const size_t take =
                std::min((size_t)(frame_hops - consumed), _snap_hops - fill_hops);
            std::memcpy(&_snapshot[fill_hops], in_local + consumed, take * sizeof(cf));
            fill_hops += take;
            consumed += (int)take;

            if (fill_hops >= _snap_hops) {
                // Snapshot complete -> emit one P_c frame (loss-less back-pressure).
                char* out = (char*)out_buf->wait_for_empty_frame(unique_name, out_id);
                if (out == nullptr) {
                    in_buf->mark_frame_empty(unique_name, in_id);
                    return;
                }
                correlate_snapshot(out, snap_start_hop);
                out_buf->mark_frame_full(unique_name, out_id++);
                filling = false;
            }
        }

        in_buf->mark_frame_empty(unique_name, in_id++);
        abs_hops += frame_hops;
    }
}
