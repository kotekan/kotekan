#include "SampleAutocorr.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
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

REGISTER_KOTEKAN_STAGE(SampleAutocorr);

SampleAutocorr::SampleAutocorr(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&SampleAutocorr::main_thread, this)) {

    buf_in = get_buffer("in_buf");
    register_consumer(buf_in, unique_name.c_str());
//    buf_out = get_buffer("out_buf");
//    register_producer(buf_out, unique_name.c_str());

    spectrum_length = config.get_default<int>(unique_name, "spectrum_length", 1024);
    spectrum_out = (float*)calloc(spectrum_length, sizeof(float));
//    out_local = (float*)malloc(spectrum_length,sizeof(float));
    integration_length = config.get_default<int>(unique_name, "integration_length", 1024);

    endpoint = unique_name + "/spectrum";
}

SampleAutocorr::~SampleAutocorr() {
    kotekan::restServer::instance().remove_json_callback(endpoint);
    free(spectrum_out);
}

void SampleAutocorr::rest_callback(kotekan::connectionInstance& conn,
                                   nlohmann::json& json_request) {
    // need a lock here?
    try {
        integration_length = json_request["integration_length"];
        sample_active=true;
    } catch (...) {
        conn.send_error("Couldn't parse sample request parameters.", kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    INFO("Recording spectrum of length {:d}, integration length {:d}", spectrum_length, integration_length);

    while (!sample_ready) {usleep(1000);}
    nlohmann::json reply;
    reply["stokes_i"] = nlohmann::json::array({});
    for (int i=0; i<spectrum_length; i++)
        reply["stokes_i"].push_back(spectrum_out[i]);
    memset(spectrum_out, 0, spectrum_length * sizeof(float));
    sample_ready = false;

    conn.send_json_reply(reply);
}


void SampleAutocorr::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    rest_server.register_post_callback(endpoint,
                                      std::bind(&SampleAutocorr::rest_callback, this, _1, _2));

    float* in_local;

    float re, im;
    frame_in = 0;
    frame_out = 0;
    int integration_ct = 0;
//    int out_loc = 0;

    int samples_per_frame = buf_in->frame_size / (2 * sizeof(float));

    while (!stop_thread) {
        in_local = (float*)wait_for_full_frame(buf_in, unique_name.c_str(), frame_in);
        if (in_local == nullptr)
            break;
        if (sample_active) {
            while (sample_ready) {usleep(1000);}
            for (int j = 0; j < samples_per_frame; j += spectrum_length) {
                for (int i = 0; i < spectrum_length; i++) {
                    re = in_local[(i + j) * 2];
                    im = in_local[(i + j) * 2 + 1];
                    spectrum_out[i] += (re * re + im * im) / integration_length;
                }
                integration_ct++;

                if (integration_ct >= integration_length) {
//                    if (out_loc == 0)
//                        out_local =
//                            (uint*)wait_for_empty_frame(buf_out, unique_name.c_str(), frame_out);
//                    for (int i = 0; i < spectrum_length; i++)
//                        out_local[out_loc++] = spectrum_out[i];
//                    out_local[out_loc++] = integration_ct;

//                    if (out_loc * sizeof(uint) == (uint32_t)buf_out->frame_size) {
//                        mark_frame_full(buf_out, unique_name.c_str(), frame_out);
//                        frame_out = (frame_out + 1) % buf_out->num_frames;
//                        out_loc = 0;
//                        DEBUG("Finished integrating a frame!");
//                    }

//                    memset(spectrum_out, 0, spectrum_length * sizeof(float));
                    sample_ready = true;
                    integration_ct = 0;
                    sample_active = false;
                    break;
                }
            }
        }
        mark_frame_empty(buf_in, unique_name.c_str(), frame_in);
        frame_in = (frame_in + 1) % buf_in->num_frames;
    }
}
