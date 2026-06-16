#include "SampleAutocorr.hpp"

#include <stdlib.h>             // for calloc, free
#include <string.h>             // for memset
#include <json.hpp>             // for json, basic_json
#include <condition_variable>   // for condition_variable
#include <functional>           // for bind, function, _1, _2
#include <mutex>                // for mutex, unique_lock, lock_guard
#include <memory>               // for shared_ptr

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp"  // for make_fengine_desc
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "kotekanLogging.hpp"   // for INFO
#include "restServer.hpp"       // for HTTP_RESPONSE, connectionInstance, restServer
#include "NDArray.hpp"          // for GenericNDArray
#include "fmt.hpp"              // for compile_string_to_view

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(SampleAutocorr);

SampleAutocorr::SampleAutocorr(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&SampleAutocorr::main_thread, this)) {

    buf_in = get_buffer("in_buf");
    buf_in->register_consumer(unique_name);

    // Input: cfloat32 1-D fengine spectra (consumer-only; no out_buf).
    buf_in->set_frame_desc(
        kotekan_airspy::make_fengine_desc(buf_in->frame_size / (2 * sizeof(float))));

    _spectrum_length = config.get_default<int>(unique_name, "spectrum_length", 1024);
    _integration_length = config.get_default<int>(unique_name, "integration_length", 1024);

    spectrum_out = (float*)calloc(_spectrum_length, sizeof(float));

    endpoint = unique_name + "/spectrum";
}

SampleAutocorr::~SampleAutocorr() {
    kotekan::restServer::instance().remove_json_callback(endpoint);
    free(spectrum_out);
}

void SampleAutocorr::rest_callback(kotekan::connectionInstance& conn,
                                   nlohmann::json& json_request) {
    int requested_length;
    try {
        requested_length = json_request["integration_length"];
    } catch (...) {
        conn.send_error("Couldn't parse sample request parameters.",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }

    std::unique_lock<std::mutex> lock(sample_mutex);
    if (sample_active || sample_ready) {
        conn.send_error("Concurrent sample request in progress.",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    _integration_length = requested_length;
    INFO("Recording spectrum of length {:d}, integration length {:d}", _spectrum_length,
         _integration_length);

    sample_active = true;
    sample_cv.wait(lock, [this] { return sample_ready || stop_thread; });
    if (stop_thread) {
        sample_active = false;
        lock.unlock();
        conn.send_error("Stage shutting down.", kotekan::HTTP_RESPONSE::INTERNAL_ERROR);
        return;
    }

    nlohmann::json reply;
    reply["stokes_i"] = nlohmann::json::array({});
    for (int i = 0; i < _spectrum_length; i++)
        reply["stokes_i"].push_back(spectrum_out[i]);
    memset(spectrum_out, 0, _spectrum_length * sizeof(float));
    sample_ready = false;

    lock.unlock();
    conn.send_json_reply(reply);
}

void SampleAutocorr::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    rest_server.register_post_callback(endpoint,
                                       std::bind(&SampleAutocorr::rest_callback, this, _1, _2));

    frame_in = 0;
    int integration_ct = 0;

    int samples_per_frame = buf_in->frame_size / (2 * sizeof(float));

    while (!stop_thread) {
        float* in_local = (float*)buf_in->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;

        bool just_completed = false;
        {
            std::lock_guard<std::mutex> lock(sample_mutex);
            if (sample_active) {
                for (int j = 0; j < samples_per_frame; j += _spectrum_length) {
                    for (int i = 0; i < _spectrum_length; i++) {
                        float re = in_local[(i + j) * 2];
                        float im = in_local[(i + j) * 2 + 1];
                        spectrum_out[i] += (re * re + im * im) / _integration_length;
                    }
                    integration_ct++;
                    if (integration_ct >= _integration_length) {
                        integration_ct = 0;
                        sample_active = false;
                        sample_ready = true;
                        just_completed = true;
                        break;
                    }
                }
            }
        }
        if (just_completed)
            sample_cv.notify_one();

        buf_in->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % buf_in->num_frames;
    }
    // Wake any REST callback waiting on a pending sample so it doesn't deadlock on shutdown.
    sample_cv.notify_all();
}
