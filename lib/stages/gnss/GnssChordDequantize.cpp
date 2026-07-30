#include "GnssChordDequantize.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp" // for frameID

#include <algorithm>
#include <complex>
#include <functional>

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssChordDequantize);

GnssChordDequantize::GnssChordDequantize(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChordDequantize::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _n_chan = config.get<int>(unique_name, "n_channels");
    _n_elem = config.get_default<int>(unique_name, "n_elements", 1);
    _element = config.get_default<int>(unique_name, "element", 0);
    _n_hops = config.get<int>(unique_name, "samples_per_data_set");
    _scale = config.get_default<float>(unique_name, "scale", 1.0f);
    _conjugate = config.get_default<bool>(unique_name, "conjugate", false);

    // Zero-fill to a contiguous span (see the header note on why the FFT search needs it).
    _out_chan = config.get_default<int>(unique_name, "out_channels", _n_chan);
    const int out_offset = config.get_default<int>(unique_name, "out_offset", 0);
    const auto chan_ids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});
    _out_idx.resize((size_t)_n_chan);
    if (_out_chan == _n_chan && chan_ids.empty()) {
        for (int c = 0; c < _n_chan; ++c)
            _out_idx[(size_t)c] = c; // pass-through
    } else {
        if ((int)chan_ids.size() != _n_chan) {
            FATAL_ERROR("GnssChordDequantize: channel_ids has {:d} entries but n_channels is {:d}",
                        chan_ids.size(), _n_chan);
            return;
        }
        for (int c = 0; c < _n_chan; ++c) {
            const int slot = chan_ids[(size_t)c] - out_offset;
            if (slot < 0 || slot >= _out_chan) {
                FATAL_ERROR("GnssChordDequantize: channel {:d} maps to slot {:d}, outside the "
                            "[0,{:d}) output span at out_offset {:d}",
                            chan_ids[(size_t)c], slot, _out_chan, out_offset);
                return;
            }
            _out_idx[(size_t)c] = slot;
        }
        INFO("GnssChordDequantize[{:s}]: {:d} measured channels zero-filled into a contiguous "
             "{:d}-channel span from global {:d} (fill {:.1f}%)",
             unique_name, _n_chan, _out_chan, out_offset,
             100.0 * (double)_n_chan / (double)_out_chan);
    }

    if (_element < 0 || _element >= _n_elem)
        FATAL_ERROR("GnssChordDequantize: element {:d} outside [0, {:d})", _element, _n_elem);

    const size_t in_need = (size_t)_n_hops * _n_chan * _n_elem;
    const size_t out_need = (size_t)_n_hops * _out_chan * sizeof(std::complex<float>);
    if ((size_t)in_buf->frame_size < in_need)
        FATAL_ERROR("GnssChordDequantize: in_buf frame {:d} B < {:d}", (size_t)in_buf->frame_size,
                    in_need);
    if ((size_t)out_buf->frame_size < out_need)
        FATAL_ERROR("GnssChordDequantize: out_buf frame {:d} B < {:d} ({:d} hops x {:d} chan x "
                    "cfloat32)",
                    (size_t)out_buf->frame_size, out_need, _n_hops, _out_chan);
}

void GnssChordDequantize::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf);

    while (!stop_thread) {
        uint8_t* in = in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;
        auto* out = (std::complex<float>*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;

        // Zero the whole frame first: the un-measured channels of the span MUST be zero, and
        // a stale previous frame there would be correlated as if it were data.
        if (_out_chan != _n_chan)
            std::fill(out, out + (size_t)_n_hops * _out_chan, std::complex<float>(0.f, 0.f));
        for (int m = 0; m < _n_hops; ++m) {
            const uint8_t* src = in + (size_t)m * _n_chan * _n_elem + _element;
            std::complex<float>* dst = out + (size_t)m * _out_chan;
            for (int c = 0; c < _n_chan; ++c) {
                const uint8_t b = src[(size_t)c * _n_elem];
                // HIGH nibble = REAL, LOW = IMAG, each stored as value+8. See the header note:
                // swapping these is invisible in magnitude and inverts the Doppler sign.
                const float re = (float)(int((b & 0xf0) >> 4) - 8) * _scale;
                const float im = (float)(int(b & 0x0f) - 8) * _scale;
                // conjugate: MEASURED ON SKY 2026-07-30. The CHORD F-engine's channelized
                // output is CONJUGATED relative to this decode's convention (equivalently:
                // the nibbles may carry imag-high -- indistinguishable, since swap = i*conj
                // and a constant phase is invisible to every downstream product). The
                // X-engine only forms |.|^2, so nothing in the production system could ever
                // see this, and gpuSimulate's labeling was never load-bearing evidence.
                // Found by running the acquire offline on captured sky both ways: as-is,
                // PRN 32 gave 10.1 (noise, ceiling ~14); conjugated, 22.5 -- and the
                // measured Dopplers then match BRDC to ~6 Hz on two satellites with a
                // common +5.8 Hz receiver clock bias. FIRST LIGHT was behind this flag.
                dst[_out_idx[(size_t)c]] =
                    std::complex<float>(re, _conjugate ? -im : im);
            }
        }

        // The search anchors its reported code phase to the absolute sample index, so the
        // metadata has to ride along or every detection it publishes is meaningless.
        if (metadata_is_gnss_chan(in_buf)) {
            out_buf->allocate_new_metadata_object(out_id);
            get_gnss_chan_metadata(out_buf, out_id)->sample_seq =
                get_gnss_chan_metadata(in_buf, in_id)->sample_seq;
        }

        in_buf->mark_frame_empty(unique_name, in_id++);
        out_buf->mark_frame_full(unique_name, out_id++);
    }
}
