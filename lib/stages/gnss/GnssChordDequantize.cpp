#include "GnssChordDequantize.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "gnss44.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp" // for frameID

#include <complex>
#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;
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
    _n_hops = config.get<int>(unique_name, "samples_per_data_set");
    // Only the single-element conjugated decode exists; refuse a config asking for another.
    if (config.get_default<int>(unique_name, "n_elements", 1) != 1
        || config.get_default<int>(unique_name, "element", 0) != 0
        || !config.get_default<bool>(unique_name, "conjugate", true)) {
        FATAL_ERROR("GnssChordDequantize[{:s}]: only n_elements 1, element 0 and conjugate true "
                    "are supported",
                    unique_name);
        return;
    }
    // The search anchors its code phases to sample_seq, which must ride along on every frame.
    if (!metadata_is_gnss_chan(in_buf) || !metadata_is_gnss_chan(out_buf)) {
        FATAL_ERROR("GnssChordDequantize[{:s}]: in_buf and out_buf need a GnssChanMetadata pool",
                    unique_name);
        return;
    }

    const size_t in_need = (size_t)_n_hops * _n_chan;
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

        const size_t n = (size_t)_n_hops * _n_chan;
        for (size_t k = 0; k < n; ++k) {
            float re, im;
            gnss44::unpack(in[k], 1.0f, &re, &im);
            // The CHORD F-engine's output is conjugated relative to this decode (or its nibbles
            // are swapped; the two differ by a constant phase, which no downstream product
            // sees). Without the conjugate the GNSS despread matches no satellite.
            out[k] = std::complex<float>(re, -im);
        }

        // The search anchors its code phases to the absolute sample index, so sample_seq must
        // ride along.
        out_buf->allocate_new_metadata_object(out_id);
        get_gnss_chan_metadata(out_buf, out_id)->sample_seq =
            get_gnss_chan_metadata(in_buf, in_id)->sample_seq;

        in_buf->mark_frame_empty(unique_name, in_id++);
        out_buf->mark_frame_full(unique_name, out_id++);
    }
}
