#include "GnssQuantize44.hpp"

#include "gnss44.hpp"
#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata, metadata_is_gnss_chan

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssQuantize44);

GnssQuantize44::GnssQuantize44(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssQuantize44::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _N = config.get<int>(unique_name, "spectrum_length");
    _target_lsb = config.get_default<float>(unique_name, "target_lsb", 2.1f); // CHORD: complex rms 2.1 = 1.5/component
    _measure_frames = config.get_default<int>(unique_name, "measure_frames", 64);
    if (_N <= 0)
        throw std::runtime_error("GnssQuantize44: spectrum_length must be > 0");

    _sumsq.assign(_N, 0.0);
    _inv_scale.assign(_N, 0.0f);
    _scale.assign(_N, 0.0f);
    _rails = std::vector<std::atomic<long long>>(_N);
    for (auto& r : _rails)
        r.store(0);

    // Optional fixed scales: skip the measurement entirely (replay/bench determinism).
    const auto fixed = config.get_default<std::vector<float>>(unique_name, "fixed_scales", {});
    if (!fixed.empty()) {
        if ((int)fixed.size() != _N)
            throw std::runtime_error("GnssQuantize44: fixed_scales must have spectrum_length "
                                     "entries");
        for (int c = 0; c < _N; ++c) {
            _inv_scale[c] = fixed[c];
            _scale[c] = (fixed[c] != 0.0f) ? 1.0f / fixed[c] : 0.0f;
        }
        _frozen = true;
        INFO("GnssQuantize44: using {:d} fixed channel scales (bandpass measurement skipped)", _N);
    }
}

void GnssQuantize44::freeze_scales() {
    // inv_scale[c] = target_lsb / rms[c], with rms the COMPLEX-SAMPLE rms sqrt(E|v|^2) -- the
    // convention CHORD states its operating point in: 2.1 = 1.5 * sqrt(2), i.e. 1.5 lsb PER
    // COMPONENT. (Getting this convention wrong runs the quantizer sqrt(2) hot or cold.) At the
    // CHORD point each component rails at ~5 sigma, so the Gaussian railfrac floor is ~1e-6 --
    // any visible railfrac is real RFI/CW, not design clipping. A dead channel (rms 0) would
    // divide by zero -> leave its scale at 0, which encodes everything to the offset zero
    // rather than to a rail.
    int dead = 0;
    double lo = 1e30, hi = 0.0;
    for (int c = 0; c < _N; ++c) {
        const double rms = std::sqrt(_sumsq[c] / std::max(1LL, _nsamp));
        if (!(rms > 0.0) || !std::isfinite(rms)) {
            _inv_scale[c] = 0.0f;
            _scale[c] = 0.0f;
            dead++;
            continue;
        }
        _inv_scale[c] = (float)(_target_lsb / rms);
        _scale[c] = (float)(rms / _target_lsb);
        lo = std::min(lo, rms);
        hi = std::max(hi, rms);
    }
    _frozen = true;
    INFO("GnssQuantize44: bandpass frozen over {:d} frames ({:d} samples/chan): rms {:.3e}..{:.3e} "
         "-> {:.1f} lsb; {:d} dead channels",
         _frames_measured, (int)_nsamp, lo, hi, _target_lsb, dead);
    if (hi > 0.0 && lo > 0.0)
        INFO("GnssQuantize44: bandpass dynamic range {:.1f} dB across {:d} channels -- this is why "
             "the scale is PER CHANNEL",
             20.0 * std::log10(hi / lo), _N);
}

void GnssQuantize44::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf);
    long long frames = 0;

    while (!stop_thread) {
        auto* in = (std::complex<float>*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;
        auto* out = (unsigned char*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;

        const int hops = in_buf->frame_size / (int)sizeof(std::complex<float>) / _N;
        if (out_buf->frame_size < (size_t)hops * (size_t)_N)
            throw std::runtime_error("GnssQuantize44: out_buf frame too small for [hop][N] bytes");

        if (!_frozen) {
            // MEASURE. Emit the offset zero meanwhile -- NOT silence-by-memset: 0x00 decodes to
            // (-8,-8), a full-scale DC spike that would slam every downstream stage during warmup.
            for (int m = 0; m < hops; ++m)
                for (int c = 0; c < _N; ++c) {
                    const std::complex<float>& v = in[(size_t)m * _N + c];
                    _sumsq[c] += (double)v.real() * v.real() + (double)v.imag() * v.imag();
                }
            _nsamp += hops;
            std::memset(out, gnss44::ZERO, (size_t)hops * _N);
            if (++_frames_measured >= _measure_frames)
                freeze_scales();
        } else {
            long long railed_here = 0;
            for (int m = 0; m < hops; ++m)
                for (int c = 0; c < _N; ++c) {
                    const std::complex<float>& v = in[(size_t)m * _N + c];
                    int railed = 0;
                    out[(size_t)m * _N + c] =
                        gnss44::pack(v.real(), v.imag(), _inv_scale[c], &railed);
                    if (railed) {
                        _rails[c].fetch_add(1, std::memory_order_relaxed);
                        railed_here++;
                    }
                }
            _rail_total.fetch_add((long long)hops * _N, std::memory_order_relaxed);
            // A rising railfrac is RFI/CW. At CHORD's operating point (complex rms 2.1 =
            // 1.5 lsb/component) the rail sits ~5 sigma out -> Gaussian floor ~1e-6, so ANY
            // sustained railfrac is real interference, not design clipping. (For reference:
            // per-component 2.1 rails ~1.2%, measured on the real L1 front end; 5% is far
            // above every sane operating point.)
            if (railed_here > (long long)hops * _N * 5 / 100 && (frames % 100) == 0)
                WARN("GnssQuantize44: railfrac {:.2f}% this frame -- far above the Gaussian "
                     "floor for target_lsb {:.1f} (RFI/CW? check the front end)",
                     100.0 * railed_here / ((double)hops * _N), _target_lsb);
        }

        // Metadata: the 4+4b frame is the same hops at the same absolute samples, PLUS the
        // scales those bytes were encoded with. The scales ride EVERY frame (empty until the
        // freeze) so a consumer can never despread bytes against gains other than these --
        // GnssChanMetadata explains why frame-carriage beats any side channel here.
        if (metadata_is_gnss_chan(in_buf)) {
            out_buf->allocate_new_metadata_object(out_id);
            auto* meta = get_gnss_chan_metadata(out_buf, out_id);
            meta->sample_seq = get_gnss_chan_metadata(in_buf, in_id)->sample_seq;
            if (_frozen)
                meta->chan_scale = _scale; // lsb -> volts, per channel
        }

        in_buf->mark_frame_empty(unique_name, in_id++);
        out_buf->mark_frame_full(unique_name, out_id++);
        ++frames;
    }
}
