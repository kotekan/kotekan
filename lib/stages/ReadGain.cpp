#include "ReadGain.hpp"

#include "Config.hpp"              // for Config
#include "StageFactory.hpp"        // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"                // for mark_frame_full, register_producer, wait_for_empty_frame
#include "chimeMetadata.h"         // for get_stream_id_t
#include "configUpdater.hpp"       // for configUpdater
#include "fpga_header_functions.h" // for bin_number_chime, freq_from_bin, stream_id_t
#include "kotekanLogging.hpp"      // for WARN, INFO, DEBUG
#include "restServer.hpp"          // for HTTP_RESPONSE, connectionInstance, restServer
#include "visUtil.hpp"             // for current_time

#include <algorithm>   // for copy
#include <atomic>      // for atomic_bool
#include <chrono>      // for seconds
#include <cstdint>     // for int32_t
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
// PSR
// curl localhost:12048/updatable_config/pulsar_gain -X POST -H 'Content-Type: application/json' -d
// '{"pulsar_gain_dir":["path0","path1","path2","path3","path4","path5","path6","path7","path8","path9"]}'
//
// clang-format on

ReadGain::ReadGain(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ReadGain::main_thread, this)) {
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
    freq_idx = -1;
    freq_MHz = -1;

    // Gain for FRB
    gain_frb_buf = get_buffer("gain_frb_buf");
    gain_frb_buf_id = 0;
    register_producer(gain_frb_buf, unique_name.c_str());
    update_gains_frb = false;

    // Gain for PSR
    gain_psr_buf = get_buffer("gain_psr_buf");
    gain_psr_buf_id = 0;
    register_producer(gain_psr_buf, unique_name.c_str());
    update_gains_psr = false;

    using namespace std::placeholders;

    // listen for gain updates FRB
    std::string gainfrb =
        config.get_default<std::string>(unique_name, "updatable_config/gain_frb", "");
    if (gainfrb.length() > 0)
        configUpdater::instance().subscribe(
            gainfrb, std::bind(&ReadGain::update_gains_frb_callback, this, _1));

    // listen for gain updates PSR
    std::string gainpsr = config.get<std::string>(unique_name, "updatable_config/gain_psr");
    if (gainpsr.length() > 0)
        configUpdater::instance().subscribe(
            gainpsr, std::bind(&ReadGain::update_gains_psr_callback, this, _1));
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

bool ReadGain::update_gains_psr_callback(nlohmann::json& json) {
    if (update_gains_psr) {
        WARN("[PSR] cannot handle two back-to-back gain updates, rejecting the latter");
        return true;
    }
    try {
        _gain_dir_psr = json.at("pulsar_gain_dir").get<std::vector<std::string>>();
        std::string output_msg = "[PSR] Updating gains from ";
        for (int i = 0; i < _num_beams; i++) {
            output_msg += _gain_dir_psr[i];
            output_msg += " ";
        }
        INFO("{:s}", output_msg);
    } catch (std::exception const& e) {
        WARN("[PSR] Fail to read gain_dir {:s}", e.what());
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mux);
        update_gains_psr = true;
    }
    cond_var.notify_all();

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
        for (uint i = 0; i < _num_elements; i++) {
            out_frame_frb[i * 2] = default_gains[0] * scaling;
            out_frame_frb[i * 2 + 1] = default_gains[1] * scaling;
        }
    } else {
        if (_num_elements != fread(out_frame_frb, sizeof(float) * 2, _num_elements, ptr_myfile)) {
            WARN("Gain file ({:s}) wasn't long enough! Something went wrong, using default "
                 "gains",
                 filename);
            for (uint i = 0; i < _num_elements; i++) {
                out_frame_frb[i * 2] = default_gains[0] * scaling;
                out_frame_frb[i * 2 + 1] = default_gains[1] * scaling;
            }
        }
        fclose(ptr_myfile);
        for (uint i = 0; i < _num_elements; i++) {
            out_frame_frb[i * 2] = out_frame_frb[i * 2] * scaling;
            out_frame_frb[i * 2 + 1] = out_frame_frb[i * 2 + 1] * scaling;
        }
    }
    mark_frame_full(gain_frb_buf, unique_name.c_str(), gain_frb_buf_id);
    DEBUG("Maked gain_frb_buf frame {:d} full", gain_frb_buf_id);
    INFO("Time required to load FRB gains: {:f}", current_time() - start_time);
    DEBUG("Gain_frb_buf: {:.2f} {:.2f} {:.2f} ", out_frame_frb[0], out_frame_frb[1],
          out_frame_frb[2]);
    gain_frb_buf_id = (gain_frb_buf_id + 1) % gain_frb_buf->num_frames;
}

void ReadGain::read_gain_psr() {
    float* out_frame_psr =
        (float*)wait_for_empty_frame(gain_psr_buf, unique_name.c_str(), gain_psr_buf_id);
    if (out_frame_psr == nullptr) {
        return;
    }
    double start_time = current_time();
    FILE* ptr_myfile;
    char filename[256];
    for (int b = 0; b < _num_beams; b++) {
        snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin",
                 _gain_dir_psr[b].c_str(), freq_idx);
        INFO("PSR Loading gains from {:s}", filename);
        ptr_myfile = fopen(filename, "rb");
        if (ptr_myfile == nullptr) {
            WARN("GPU Cannot open gain file {:s}", filename);
            for (uint i = 0; i < _num_elements; i++) {
                out_frame_psr[(b * _num_elements + i) * 2] = default_gains[0];
                out_frame_psr[(b * _num_elements + i) * 2 + 1] = default_gains[1];
            }
        } else {
            if (_num_elements
                != fread(&out_frame_psr[b * _num_elements * 2], sizeof(float) * 2, _num_elements,
                         ptr_myfile)) {
                WARN("Gain file ({:s}) wasn't long enough! Something went wrong, using default "
                     "gains",
                     filename);
                for (uint i = 0; i < _num_elements; i++) {
                    out_frame_psr[(b * _num_elements + i) * 2] = default_gains[0];
                    out_frame_psr[(b * _num_elements + i) * 2 + 1] = default_gains[1];
                }
            }
            fclose(ptr_myfile);
        }
    } // end beam
    mark_frame_full(gain_psr_buf, unique_name.c_str(), gain_psr_buf_id);
    DEBUG("Maked gain_psr_buf frame {:d} full", gain_psr_buf_id);
    INFO("Time required to load PSR gains: {:f}", current_time() - start_time);
    DEBUG("Gain_psr_buf: {:.2f} {:.2f} {:.2f} ", out_frame_psr[0], out_frame_psr[1],
          out_frame_psr[2]);
    gain_psr_buf_id = (gain_psr_buf_id + 1) % gain_psr_buf->num_frames;
}

void ReadGain::main_thread() {

    uint8_t* frame =
        wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == nullptr)
        return;
    stream_id_t stream_id = get_stream_id_t(metadata_buf, metadata_buffer_id);
    freq_idx = bin_number_chime(&stream_id);
    freq_MHz = freq_from_bin(freq_idx);
    metadata_buffer_precondition_id =
        (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;

    mark_frame_empty(metadata_buf, unique_name.c_str(), metadata_buffer_id);
    metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;
    unregister_consumer(metadata_buf, unique_name.c_str());

    while (!stop_thread) {
        {
            std::unique_lock<std::mutex> lock(mux);
            while (!update_gains_frb && !update_gains_psr && !stop_thread) {
                cond_var.wait_for(lock, std::chrono::seconds(5));
            }
        }
        if (stop_thread)
            break;
        if (update_gains_frb) {
            read_gain_frb();
            update_gains_frb = false;
        }
        if (update_gains_psr) {
            read_gain_psr();
            update_gains_psr = false;
        }
    }
}
