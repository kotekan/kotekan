#include "GnssQuantize44.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "gnss44.hpp"           // for pack
#include "kotekanLogging.hpp"   // for FATAL_ERROR, INFO, WARN
#include "visUtil.hpp"          // for frameID

#include <algorithm>  // for max, min
#include <cmath>      // for sqrt, log10
#include <complex>    // for complex
#include <cstring>    // for memset
#include <functional> // for bind

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

    _spectrum_length = config.get<int>(unique_name, "spectrum_length");
    _target_lsb = config.get_default<float>(unique_name, "target_lsb", 2.1f);
    _measure_frames = config.get_default<int>(unique_name, "measure_frames", 64);

    const size_t hop_bytes = (size_t)_spectrum_length * sizeof(std::complex<float>);
    if (_spectrum_length < 1 || in_buf->frame_size % hop_bytes != 0
        || out_buf->frame_size * sizeof(std::complex<float>) != in_buf->frame_size) {
        FATAL_ERROR("GnssQuantize44: in_buf frames ({:d} B) must hold whole hops of {:d} "
                    "channels, and out_buf frames ({:d} B) one byte per sample.",
                    in_buf->frame_size, _spectrum_length, out_buf->frame_size);
        return;
    }
    _sumsq.assign(_spectrum_length, 0.0);
}

void GnssQuantize44::freeze_scales() {
    _inv_scale.assign(_spectrum_length, 0.0f);
    double lo = 0.0, hi = 0.0;
    int dead = 0;
    for (int c = 0; c < _spectrum_length; ++c) {
        const double rms = std::sqrt(_sumsq[c] / _num_measured);
        // A channel without power keeps a scale of 0, which encodes it as zeros.
        if (!(rms > 0.0)) {
            dead++;
            continue;
        }
        _inv_scale[c] = _target_lsb / rms;
        lo = (lo == 0.0) ? rms : std::min(lo, rms);
        hi = std::max(hi, rms);
    }
    INFO("GnssQuantize44: scales set from {:d} frames: channel rms {:.3g} to {:.3g} ({:.1f} dB), "
         "{:d} channels without power",
         _frames_measured, lo, hi, lo > 0.0 ? 20.0 * std::log10(hi / lo) : 0.0, dead);
}

void GnssQuantize44::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf);
    const int num_hops = in_buf->frame_size / sizeof(std::complex<float>) / _spectrum_length;
    const size_t num_samples = (size_t)num_hops * _spectrum_length;
    uint64_t frames = 0;

    while (!stop_thread) {
        auto* in = (const std::complex<float>*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;
        uint8_t* out = out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;

        if (_inv_scale.empty()) {
            for (int m = 0; m < num_hops; ++m)
                for (int c = 0; c < _spectrum_length; ++c)
                    _sumsq[c] += std::norm(in[(size_t)m * _spectrum_length + c]);
            _num_measured += num_hops;
            memset(out, 0x88, num_samples); // 0 + 0j
            if (++_frames_measured >= _measure_frames)
                freeze_scales();
        } else {
            size_t num_railed = 0;
            for (int m = 0; m < num_hops; ++m)
                for (int c = 0; c < _spectrum_length; ++c) {
                    const size_t k = (size_t)m * _spectrum_length + c;
                    int railed = 0;
                    out[k] = gnss44::pack(in[k].real(), in[k].imag(), _inv_scale[c], &railed);
                    num_railed += railed;
                }
            // At 1.5 lsb per component a rail is 5 sigma out, so clipping this common is
            // interference or a changed gain.
            if (num_railed * 20 > num_samples && frames % 100 == 0)
                WARN("GnssQuantize44: {:.1f}% of this frame's samples clipped",
                     100.0 * num_railed / num_samples);
        }

        if (metadata_is_gnss_chan(in_buf) && metadata_is_gnss_chan(out_buf)) {
            out_buf->allocate_new_metadata_object(out_id);
            get_gnss_chan_metadata(out_buf, out_id)->sample_seq =
                get_gnss_chan_metadata(in_buf, in_id)->sample_seq;
        }

        in_buf->mark_frame_empty(unique_name, in_id++);
        out_buf->mark_frame_full(unique_name, out_id++);
        frames++;
    }
}
