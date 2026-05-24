#include "SimpleCrosscorr.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp" // for make_fengine_desc, make_power_corr_desc
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG

#include <functional> // for bind
#include <stdint.h>   // for uint32_t
#include <stdlib.h>   // for calloc, free
#include <string.h>   // for memset

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(SimpleCrosscorr);

SimpleCrosscorr::SimpleCrosscorr(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&SimpleCrosscorr::main_thread, this)) {

    buf_inA = get_buffer("in_bufA");
    buf_inA->register_consumer(unique_name);
    buf_inB = get_buffer("in_bufB");
    buf_inB->register_consumer(unique_name);
    buf_out = get_buffer("out_buf");
    buf_out->register_producer(unique_name);

    _spectrum_length = config.get_default<uint32_t>(unique_name, "spectrum_length", 1024);
    _integration_length = config.get_default<uint32_t>(unique_name, "integration_length", 1024);

    spectrum_out = (float*)calloc(_spectrum_length * 4, sizeof(float));

    // Inputs: cfloat32 1-D fengine spectra. Output (power_corr) descriptor
    // is *not* set here: rawFileWrite refuses NDArray-tagged buffers (it
    // can only round-trip raw bytes, not the layout metadata), and the
    // crosscorr unit test captures buf_out via rawFileWrite. networkPowerStream
    // asserts the expected power_corr layout on its consumer side, so the
    // contract is still documented in code.
    buf_inA->set_frame_desc(
        kotekan_airspy::make_fengine_desc(buf_inA->frame_size / (2 * sizeof(float))));
    buf_inB->set_frame_desc(
        kotekan_airspy::make_fengine_desc(buf_inB->frame_size / (2 * sizeof(float))));
}

SimpleCrosscorr::~SimpleCrosscorr() {
    free(spectrum_out);
}

void SimpleCrosscorr::main_thread() {
    float* out_local = nullptr;

    frame_inA = 0;
    frame_inB = 0;
    frame_out = 0;
    uint32_t integration_ct = 0;
    int out_loc = 0;

    const int samples_per_frame = buf_inA->frame_size / (2 * sizeof(float));

    // Assumes the two input buffers are aligned frame-for-frame.
    while (!stop_thread) {
        float* inA_local = (float*)buf_inA->wait_for_full_frame(unique_name, frame_inA);
        float* inB_local = (float*)buf_inB->wait_for_full_frame(unique_name, frame_inB);
        if (inA_local == nullptr || inB_local == nullptr)
            break;

        for (int j = 0; j < samples_per_frame; j += _spectrum_length) {
            for (uint32_t i = 0; i < _spectrum_length; i++) {
                float Ar = inA_local[(i + j) * 2];
                float Ai = inA_local[(i + j) * 2 + 1];
                float Br = inB_local[(i + j) * 2];
                float Bi = inB_local[(i + j) * 2 + 1];

                spectrum_out[i + _spectrum_length * 0] += (Ar * Ar + Ai * Ai) / _integration_length;
                spectrum_out[i + _spectrum_length * 1] += (Br * Br + Bi * Bi) / _integration_length;
                spectrum_out[i + _spectrum_length * 2] += (Ar * Br + Ai * Bi) / _integration_length;
                spectrum_out[i + _spectrum_length * 3] += (Ai * Br - Bi * Ar) / _integration_length;
            }
            integration_ct++;

            if (integration_ct >= _integration_length) {
                if (out_loc == 0)
                    out_local = (float*)buf_out->wait_for_empty_frame(unique_name, frame_out);
                if (out_local == nullptr)
                    return;

                // Per-element layout: ``[freqs floats, 1 uint count]`` repeated
                // once per visibility stream (AA, BB, Re{AB*}, Im{AB*}). That's
                // the format ``networkPowerStream`` expects to see when
                // ``num_elements == 4``; without the per-element count slot
                // it indexes past the end of the frame and crashes on the
                // first integration.
                for (uint32_t e = 0; e < 4; e++) {
                    for (uint32_t i = 0; i < _spectrum_length; i++)
                        out_local[out_loc++] = spectrum_out[i + _spectrum_length * e];
                    ((uint32_t*)out_local)[out_loc++] = integration_ct;
                }

                if (out_loc * sizeof(uint32_t) == (uint32_t)buf_out->frame_size) {
                    buf_out->mark_frame_full(unique_name, frame_out);
                    frame_out = (frame_out + 1) % buf_out->num_frames;
                    out_loc = 0;
                    DEBUG("Finished integrating a frame!");
                }

                memset(spectrum_out, 0, _spectrum_length * sizeof(float) * 4);
                integration_ct = 0;
            }
        }

        buf_inA->mark_frame_empty(unique_name, frame_inA);
        buf_inB->mark_frame_empty(unique_name, frame_inB);
        frame_inA = (frame_inA + 1) % buf_inA->num_frames;
        frame_inB = (frame_inB + 1) % buf_inB->num_frames;
    }
}
