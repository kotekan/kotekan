#include "SimpleCrosscorr.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, mark_frame_full, register_consumer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdint.h>    // for uint32_t
#include <stdlib.h>    // for calloc, free
#include <string.h>    // for memset
#include <sys/types.h> // for uint
#include <vector>      // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(SimpleCrosscorr);

SimpleCrosscorr::SimpleCrosscorr(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&SimpleCrosscorr::main_thread, this)) {

    buf_inA = get_buffer("in_bufA");
    register_consumer(buf_inA, unique_name.c_str());
    buf_inB = get_buffer("in_bufB");
    register_consumer(buf_inB, unique_name.c_str());

    buf_out = get_buffer("out_buf");
    register_producer(buf_out, unique_name.c_str());

    spectrum_length = config.get_default<uint>(unique_name, "spectrum_length", 1024);
    spectrum_out = (float*)calloc(spectrum_length, sizeof(float) * 4);
    integration_length = config.get_default<uint>(unique_name, "integration_length", 1024);
}

SimpleCrosscorr::~SimpleCrosscorr() {
    free(spectrum_out);
}

void SimpleCrosscorr::main_thread() {
    float* inA_local;
    float* inB_local;
    float* out_local; //AA, BB, ABr, ABi

    float Ar,Ai,Br,Bi;
    frame_inA = 0;
    frame_inB = 0;
    frame_out = 0;
    uint integration_ct = 0;
    int out_loc = 0;

    int samples_per_frame = buf_inA->frame_size / (2 * sizeof(float));

    //assumes the arrays are aligned
    while (!stop_thread) {
        inA_local = (float*)wait_for_full_frame(buf_inA, unique_name.c_str(), frame_inA);
        inB_local = (float*)wait_for_full_frame(buf_inB, unique_name.c_str(), frame_inB);

        if ((inA_local == nullptr) || (inB_local == nullptr))
            break;

        for (int j = 0; j < samples_per_frame; j += spectrum_length) { //for each spectrum j

            for (uint i = 0; i < spectrum_length; i++) { //take each spectral sample i
                Ar = inA_local[(i + j) * 2];
                Ai = inA_local[(i + j) * 2 + 1];
                Br = inB_local[(i + j) * 2];
                Bi = inB_local[(i + j) * 2 + 1];

                spectrum_out[i+spectrum_length*0] += (Ar * Ar + Ai * Ai) / integration_length;
                spectrum_out[i+spectrum_length*1] += (Br * Br + Bi * Bi) / integration_length;
                spectrum_out[i+spectrum_length*2] += (Ar * Br + Ai * Bi) / integration_length;
                spectrum_out[i+spectrum_length*3] += (Ai * Br - Bi * Ar) / integration_length;
            }
            integration_ct++;

            if (integration_ct >= integration_length) {
                if (out_loc == 0)
                    out_local =
                        (float*)wait_for_empty_frame(buf_out, unique_name.c_str(), frame_out);
                for (uint i = 0; i < spectrum_length*4; i++)
                    out_local[out_loc++] = spectrum_out[i];
                ((uint*)out_local)[out_loc++] = integration_ct;

                if (out_loc * sizeof(uint) == (uint32_t)buf_out->frame_size) {
                    mark_frame_full(buf_out, unique_name.c_str(), frame_out);
                    frame_out = (frame_out + 1) % buf_out->num_frames;
                    out_loc = 0;
                    DEBUG("Finished integrating a frame!");
                }

                memset(spectrum_out, 0, spectrum_length * sizeof(float)*4);
                integration_ct = 0;
            }
        }

        mark_frame_empty(buf_inA, unique_name.c_str(), frame_inA);
        mark_frame_empty(buf_inB, unique_name.c_str(), frame_inB);
        frame_inA = (frame_inA + 1) % buf_inA->num_frames;
        frame_inB = (frame_inB + 1) % buf_inB->num_frames;
    }
}
