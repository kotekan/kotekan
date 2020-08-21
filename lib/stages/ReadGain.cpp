#include "ReadGain.hpp"

#include "Config.hpp"         // for Config
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"      // for Telescope, FREQ_ID_NOT_SET
#include "buffer.h"           // for mark_frame_full, register_producer, wait_for_empty_frame
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for WARN, INFO, DEBUG
#include "restServer.hpp"     // for HTTP_RESPONSE, connectionInstance, restServer
#include "visUtil.hpp"        // for current_time

#include <algorithm>   // for copy, copy_backward, equal, max
#include <atomic>      // for atomic_bool
#include <chrono>      // for seconds
#include <cstdint>     // for int32_t
#include <deque>       // for deque
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, _Placeholder, _1, function
#include <memory>      // for allocator_traits<>::value_type
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdio.h>     // for fclose, fopen, fread, snprintf, FILE
#include <sys/types.h> // for uint


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(ReadGain);

// clang-format off

// Request gain file re-parse with e.g.
// FRB
// curl localhost:12048/frb_gain -X POST -H 'Content-Type: appication/json' -d '{"frb_gain_dir":"the_new_path"}'
// Tracking Beamformer
// curl localhost:12048/updatable_config/tracking_gain -X POST -H 'Content-Type: application/json' -d
// '{"tracking_gain_dir":["path0","path1","path2","path3","path4","path5","path6","path7","path8","path9"]}'
//
// clang-format on

ReadGain::ReadGain(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ReadGain::main_thread, this)),
    gains_last_update_success_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gains_last_update_success", unique_name, {"type"})),
    gains_last_update_timestamp_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gains_last_update_timestamp", unique_name, {"type"})) {
    // Apply config.
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_beams = config.get<int32_t>(unique_name, "num_beams");
    scaling = config.get_default<float>(unique_name, "frb_scaling", 1.0);
    vector<float> dg = {0.0, 0.0}; // re,im
    default_gains = config.get_default<std::vector<float>>(unique_name, "frb_missing_gains", dg);

    metadata_buf = get_buffer("in_buf");
    register_consumer(metadata_buf, unique_name.c_str());
    metadata_buffer_id = 0;
    metadata_buffer_precondition_id = 0;
    freq_idx = FREQ_ID_NOT_SET;
    freq_MHz = -1;

    // Gain for FRB
    gain_frb_buf = get_buffer("gain_frb_buf");
    gain_frb_buf_id = 0;
    register_producer(gain_frb_buf, unique_name.c_str());
    update_gains_frb = false;

    // Gain for Tracking Beamformer
    gain_tracking_buf = get_buffer("gain_tracking_buf");
    gain_tracking_buf_id = 0;
    register_producer(gain_tracking_buf, unique_name.c_str());
    update_gains_tracking = false;

    using namespace std::placeholders;

    // listen for gain updates FRB
    std::string gainfrb =
        config.get_default<std::string>(unique_name, "updatable_config/gain_frb", "");
    if (gainfrb.length() > 0)
        configUpdater::instance().subscribe(
            gainfrb, std::bind(&ReadGain::update_gains_frb_callback, this, _1));

    // Listen for gain updates Tracking Beamformer
    using namespace std::placeholders;
    for (int beam_id = 0; beam_id < _num_beams; beam_id++) {
        configUpdater::instance().subscribe(
            config.get<std::string>(unique_name, "updatable_config/gain_tracking") + "/"
                + std::to_string(beam_id),
            [beam_id, this](nlohmann::json& json_msg) -> bool {
                return update_gains_tracking_callback(json_msg, beam_id);
            });
    }
}

bool ReadGain::update_gains_frb_callback(nlohmann::json& json) {
    if (update_gains_frb) {
        WARN("[FRB] cannot handle two back-to-back gain updates, rejecting the latter");
        return true;
    }
    try {
        _gain_dir_frb = json.at("frb_gain_dir");
    } catch (std::exception& e) {
        WARN("[FRB] Fail to read gain_dir {:s}", e.what());
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mux);
        update_gains_frb = true;
    }
    cond_var.notify_all();

    return true;
}

bool ReadGain::update_gains_tracking_callback(nlohmann::json& json, const uint8_t beam_id) {
    {
        std::lock_guard<std::mutex> lock(mux);
        try {
            _gain_dir_tracking.push(
                std::make_pair(beam_id, json.at("gain_dir").get<std::string>()));
            INFO("[Tracking Beamformer] Updating gains from {:s}",
                 _gain_dir_tracking.back().second);
        } catch (std::exception const& e) {
            WARN("[Tracking Beamformer] Fail to read gain_dir {:s}", e.what());
            return false;
        }
        update_gains_tracking = true;
    }

    return true;
}

void ReadGain::read_gain_frb() {
    float* out_frame_frb =
        (float*)wait_for_empty_frame(gain_frb_buf, unique_name.c_str(), gain_frb_buf_id);
    if (out_frame_frb == nullptr) {
        return;
    }
    double start_time = current_time();
    FILE* ptr_myfile;
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin", _gain_dir_frb.c_str(),
             freq_idx);
    INFO("FRB Loading gains from {:s}", filename);
    ptr_myfile = fopen(filename, "rb");
    if (ptr_myfile == nullptr) {
        WARN("GPU Cannot open gain file {:s}", filename);
        gains_last_update_success_metric.labels({"frb"}).set(0);
        for (uint i = 0; i < _num_elements; i++) {
            out_frame_frb[i * 2] = default_gains[0] * scaling;
            out_frame_frb[i * 2 + 1] = default_gains[1] * scaling;
        }
    } else {
        if (_num_elements != fread(out_frame_frb, sizeof(float) * 2, _num_elements, ptr_myfile)) {
            WARN("Gain file ({:s}) wasn't long enough! Something went wrong, using default "
                 "gains",
                 filename);
            gains_last_update_success_metric.labels({"frb"}).set(0);
            for (uint i = 0; i < _num_elements; i++) {
                out_frame_frb[i * 2] = default_gains[0] * scaling;
                out_frame_frb[i * 2 + 1] = default_gains[1] * scaling;
            }
        } else {
            gains_last_update_success_metric.labels({"frb"}).set(1);
        }
        fclose(ptr_myfile);
        for (uint i = 0; i < _num_elements; i++) {
            out_frame_frb[i * 2] = out_frame_frb[i * 2] * scaling;
            out_frame_frb[i * 2 + 1] = out_frame_frb[i * 2 + 1] * scaling;
        }
    }
    gains_last_update_timestamp_metric.labels({"frb"}).set(start_time);
    mark_frame_full(gain_frb_buf, unique_name.c_str(), gain_frb_buf_id);
    DEBUG("Maked gain_frb_buf frame {:d} full", gain_frb_buf_id);
    INFO("Time required to load FRB gains: {:f}", current_time() - start_time);
    DEBUG("Gain_frb_buf: {:.2f} {:.2f} {:.2f} ", out_frame_frb[0], out_frame_frb[1],
          out_frame_frb[2]);
    gain_frb_buf_id = (gain_frb_buf_id + 1) % gain_frb_buf->num_frames;
}

void ReadGain::read_gain_tracking() {
    float* out_frame_tracking =
        (float*)wait_for_empty_frame(gain_tracking_buf, unique_name.c_str(), gain_tracking_buf_id);
    if (out_frame_tracking == nullptr) {
        return;
    }
    double start_time = current_time();
    FILE* ptr_myfile;
    char filename[256];
    bool all_beams_successful_update = true;
    {
        std::lock_guard<std::mutex> lock(mux);
        while (_gain_dir_tracking.size() > 0) {
            std::pair<uint8_t, std::string> beam = _gain_dir_tracking.front();
            _gain_dir_tracking.pop();
            uint8_t beam_id = beam.first;
            snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin",
                     beam.second.c_str(), freq_idx);
            INFO("Tracking Beamformer Loading gains from {:s}", filename);
            ptr_myfile = fopen(filename, "rb");
            if (ptr_myfile == nullptr) {
                WARN("GPU Cannot open gain file {:s}", filename);
                all_beams_successful_update = false;
                for (uint i = 0; i < _num_elements; i++) {
                    out_frame_tracking[(beam_id * _num_elements + i) * 2] = default_gains[0];
                    out_frame_tracking[(beam_id * _num_elements + i) * 2 + 1] = default_gains[1];
                }
            } else {
                if (_num_elements
                    != fread(&out_frame_tracking[beam_id * _num_elements * 2], sizeof(float) * 2,
                             _num_elements, ptr_myfile)) {
                    WARN("Gain file ({:s}) wasn't long enough! Something went wrong, using default "
                         "gains",
                         filename);
                    all_beams_successful_update = false;
                    for (uint i = 0; i < _num_elements; i++) {
                        out_frame_tracking[(beam_id * _num_elements + i) * 2] = default_gains[0];
                        out_frame_tracking[(beam_id * _num_elements + i) * 2 + 1] =
                            default_gains[1];
                    }
                }
                fclose(ptr_myfile);
            }
        } // end beam
    }
    if (all_beams_successful_update) {
        gains_last_update_success_metric.labels({"tracking"}).set(1);
    } else {
        gains_last_update_success_metric.labels({"tracking"}).set(0);
    }
    gains_last_update_timestamp_metric.labels({"tracking"}).set(start_time);
    mark_frame_full(gain_tracking_buf, unique_name.c_str(), gain_tracking_buf_id);
    DEBUG("Maked gain_tracking_buf frame {:d} full", gain_tracking_buf_id);
    INFO("Time required to load tracking beamformer gains: {:f}", current_time() - start_time);
    DEBUG("Gain_tracking_buf: {:.2f} {:.2f} {:.2f} ", out_frame_tracking[0], out_frame_tracking[1],
          out_frame_tracking[2]);
    gain_tracking_buf_id = (gain_tracking_buf_id + 1) % gain_tracking_buf->num_frames;
}

void ReadGain::main_thread() {

    uint8_t* frame =
        wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == nullptr)
        return;

    auto& tel = Telescope::instance();
    freq_idx = tel.to_freq_id(metadata_buf, metadata_buffer_id);
    freq_MHz = tel.to_freq(freq_idx);
    metadata_buffer_precondition_id =
        (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;

    mark_frame_empty(metadata_buf, unique_name.c_str(), metadata_buffer_id);
    metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;
    unregister_consumer(metadata_buf, unique_name.c_str());

    while (!stop_thread) {
        {
            std::unique_lock<std::mutex> lock(mux);
            while (!update_gains_frb && !update_gains_tracking && !stop_thread) {
                cond_var.wait_for(lock, std::chrono::seconds(5));
            }
        }
        if (stop_thread)
            break;
        if (update_gains_frb) {
            read_gain_frb();
            update_gains_frb = false;
        }
        if (update_gains_tracking) {
            read_gain_tracking();
            update_gains_tracking = false;
        }
    }
}
