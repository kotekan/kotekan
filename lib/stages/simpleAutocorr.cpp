#include "simpleAutocorr.hpp"

#include <stdint.h>             // for uint32_t
#include <stdlib.h>             // for calloc, free
#include <string.h>             // for memset
#include <sys/types.h>          // for uint
#include <functional>           // for bind, function
#include <memory>               // for shared_ptr

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp"  // for make_fengine_desc
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "kotekanLogging.hpp"   // for DEBUG
#include "fmt.hpp"              // for compile_string_to_view
#include "NDArray.hpp"          // for GenericNDArray


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(simpleAutocorr);

simpleAutocorr::simpleAutocorr(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&simpleAutocorr::main_thread, this)) {

    buf_in = get_buffer("in_buf");
    buf_in->register_consumer(unique_name);
    buf_out = get_buffer("out_buf");
    buf_out->register_producer(unique_name);

    spectrum_length = config.get_default<int>(unique_name, "spectrum_length", 1024);
    spectrum_out = (float*)calloc(spectrum_length, sizeof(float));
    integration_length = config.get_default<int>(unique_name, "integration_length", 1024);

    // Input: cfloat32 1-D fengine spectra. Output (power_corr) descriptor
    // is *not* set here: rawFileWrite refuses NDArray-tagged buffers (it
    // can only round-trip raw bytes, not the layout metadata), and the
    // test pipeline captures buf_out through rawFileWrite. networkPowerStream
    // asserts the expected power_corr layout on its consumer side, so the
    // contract is still documented in code.
    buf_in->ensure_frame_desc(
        kotekan_airspy::make_fengine_desc(buf_in->frame_size / (2 * sizeof(float))));
}

simpleAutocorr::~simpleAutocorr() {
    free(spectrum_out);
}

void simpleAutocorr::main_thread() {
    float* in_local;
    float* out_local = nullptr;

    float re, im;
    frame_in = 0;
    frame_out = 0;
    int integration_ct = 0;
    int out_loc = 0;

    int samples_per_frame = buf_in->frame_size / (2 * sizeof(float));

    while (!stop_thread) {
        in_local = (float*)buf_in->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        for (int j = 0; j < samples_per_frame; j += spectrum_length) {
            for (int i = 0; i < spectrum_length; i++) {
                re = in_local[(i + j) * 2];
                im = in_local[(i + j) * 2 + 1];
                spectrum_out[i] += (re * re + im * im) / integration_length;
            }
            integration_ct++;

            if (integration_ct >= integration_length) {
                if (out_loc == 0)
                    out_local = (float*)buf_out->wait_for_empty_frame(unique_name, frame_out);
                for (int i = 0; i < spectrum_length; i++)
                    out_local[out_loc++] = spectrum_out[i];
                // Trailing slot is the integration count, packed as uint32 in the
                // same word-width slot as the floats. Matches the layout that
                // networkPowerStream pulls back out via ``((uint*)frame)[...]``.
                ((uint32_t*)out_local)[out_loc++] = integration_ct;

                if (out_loc * sizeof(float) == (uint32_t)buf_out->frame_size) {
                    buf_out->mark_frame_full(unique_name, frame_out);
                    frame_out = (frame_out + 1) % buf_out->num_frames;
                    out_loc = 0;
                    DEBUG("Finished integrating a frame!");
                }

                memset(spectrum_out, 0, spectrum_length * sizeof(float));
                integration_ct = 0;
            }
        }
        buf_in->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % buf_in->num_frames;
    }
}
