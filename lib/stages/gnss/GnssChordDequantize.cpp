#include "GnssChordDequantize.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp" // for frameID

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

    if (_element < 0 || _element >= _n_elem)
        FATAL_ERROR("GnssChordDequantize: element {:d} outside [0, {:d})", _element, _n_elem);

    const size_t in_need = (size_t)_n_hops * _n_chan * _n_elem;
    const size_t out_need = (size_t)_n_hops * _n_chan * sizeof(std::complex<float>);
    if ((size_t)in_buf->frame_size < in_need)
        FATAL_ERROR("GnssChordDequantize: in_buf frame {:d} B < {:d}", (size_t)in_buf->frame_size,
                    in_need);
    if ((size_t)out_buf->frame_size < out_need)
        FATAL_ERROR("GnssChordDequantize: out_buf frame {:d} B < {:d} ({:d} hops x {:d} chan x "
                    "cfloat32)",
                    (size_t)out_buf->frame_size, out_need, _n_hops, _n_chan);
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

        for (int m = 0; m < _n_hops; ++m) {
            const uint8_t* src = in + (size_t)m * _n_chan * _n_elem + _element;
            std::complex<float>* dst = out + (size_t)m * _n_chan;
            for (int c = 0; c < _n_chan; ++c) {
                const uint8_t b = src[(size_t)c * _n_elem];
                // HIGH nibble = REAL, LOW = IMAG, each stored as value+8. See the header note:
                // swapping these is invisible in magnitude and inverts the Doppler sign.
                const float re = (float)(int((b & 0xf0) >> 4) - 8) * _scale;
                const float im = (float)(int(b & 0x0f) - 8) * _scale;
                dst[c] = std::complex<float>(re, im);
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
