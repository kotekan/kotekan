#include "hsaBeamformKernel.hpp"

#include "Config.hpp"              // for Config
#include "buffer.h"                // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp"     // for bufferContainer
#include "chimeMetadata.h"         // for get_stream_id_t
#include "fpga_header_functions.h" // for bin_number_chime, freq_from_bin, stream_id_t
#include "gpuCommand.hpp"          // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaBase.h"               // for hsa_host_free, hsa_host_malloc
#include "hsaDeviceInterface.hpp"  // for hsaDeviceInterface, Config
#include "restServer.hpp"          // for restServer, connectionInstance, HTTP_RESPONSE, HTTP_R...

#include "fmt.hpp" // for format, fmt

#include <cmath>      // for sin, asin, cos, floor
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, _2, pla...
#include <regex>      // for match_results<>::_Base_type
#include <string.h>   // for memcpy, memset

namespace kotekan {
class configUpdater;
} // namespace kotekan

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_HSA_COMMAND(hsaBeamformKernel);

// clang-format off

// Request gain file re-parse with e.g.
// curl localhost:12048/updatable_config/frb_gain -X POST -H 'Content-Type: appication/json' -d '{"frb_gain_dir":"the_new_path"}'
// Update NS beam
// curl localhost:12048/gpu/gpu_<gpu_id>/frb/update_NS_beam/<gpu_id> -X POST -H 'Content-Type: application/json' -d '{"northmost_beam":<value>}'
// Update EW beam
// curl localhost:12048/gpu/gpu_<gpu_id>/frb/update_EW_beam/<gpu_id> -X POST -H 'Content-Type: application/json' -d '{"ew_id":<value>,"ew_beam":<value>}'

// clang-format on

hsaBeamformKernel::hsaBeamformKernel(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "zero_padded_FFT512" KERNEL_EXT,
               "unpack_shift_beamform_flip.hsaco") {
    command_type = gpuCommandType::KERNEL;

    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");

    _northmost_beam = config.get<float>(unique_name, "northmost_beam");
    freq_ref = (LIGHT_SPEED * (128) / (sin(_northmost_beam * PI / 180.) * FEED_SEP * 256)) / 1.e6;

    _ew_spacing = config.get<std::vector<float>>(unique_name, "ew_spacing");
    _ew_spacing_c = (float*)hsa_host_malloc(4 * sizeof(float), device.get_gpu_numa_node());
    for (int i = 0; i < 4; i++) {
        _ew_spacing_c[i] = _ew_spacing[i];
    }

    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    output_frame_len = _num_elements * _samples_per_data_set * 2 * sizeof(float);


    map_len = 256 * sizeof(int);
    host_map = (uint32_t*)hsa_host_malloc(map_len, device.get_gpu_numa_node());

    coeff_len = 32 * sizeof(float);
    host_coeff = (float*)hsa_host_malloc(coeff_len, device.get_gpu_numa_node());

    gain_len = 2 * 2048 * sizeof(float);

    // Figure out which frequency, is there a better way that doesn't involve reading in the whole
    // thing? Check later
    metadata_buf = host_buffers.get_buffer("network_buf");
    register_consumer(metadata_buf, unique_name.c_str());
    metadata_buffer_id = 0;
    metadata_buffer_precondition_id = 0;
    freq_idx = -1;
    freq_MHz = -1;

    update_NS_beam = true;
    update_EW_beam = true;
    first_pass = true;

    config_base = fmt::format(fmt("/gpu/gpu_{:d}"), device.get_gpu_id());

    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    endpoint_NS_beam =
        fmt::format(fmt("{:s}/frb/update_NS_beam/{:d}"), config_base, device.get_gpu_id());
    rest_server.register_post_callback(
        endpoint_NS_beam, std::bind(&hsaBeamformKernel::update_NS_beam_callback, this, _1, _2));
    endpoint_EW_beam =
        fmt::format(fmt("{:s}/frb/update_EW_beam/{:d}"), config_base, device.get_gpu_id());

    rest_server.register_post_callback(
        endpoint_EW_beam, std::bind(&hsaBeamformKernel::update_EW_beam_callback, this, _1, _2));
}

hsaBeamformKernel::~hsaBeamformKernel() {
    restServer::instance().remove_json_callback(endpoint_NS_beam);
    restServer::instance().remove_json_callback(endpoint_EW_beam);
    hsa_host_free(host_map);
    hsa_host_free(host_coeff);
    hsa_host_free(_ew_spacing_c);
    // TODO Free device memory allocations.
}

void hsaBeamformKernel::update_EW_beam_callback(connectionInstance& conn,
                                                nlohmann::json& json_request) {
    int ew_id;
    try {
        ew_id = json_request["ew_id"];
    } catch (...) {
        conn.send_error("could not parse FRB E-W beam update", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    _ew_spacing_c[ew_id] = json_request["ew_beam"];
    update_EW_beam = true;
    config.update_value(config_base, fmt::format(fmt("ew_spacing/{:d}"), ew_id),
                        json_request["ew_beam"]);
    conn.send_empty_reply(HTTP_RESPONSE::OK);
}

void hsaBeamformKernel::update_NS_beam_callback(connectionInstance& conn,
                                                nlohmann::json& json_request) {
    try {
        _northmost_beam = json_request["northmost_beam"];
    } catch (...) {
        conn.send_error("could not parse FRB N-S beam update", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    freq_ref = (LIGHT_SPEED * (128) / (sin(_northmost_beam * PI / 180.) * FEED_SEP * 256)) / 1.e6;
    update_NS_beam = true;

    config.update_value(config_base, "northmost_beam", json_request["northmost_beam"]);
    conn.send_empty_reply(HTTP_RESPONSE::OK);
}

int hsaBeamformKernel::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    uint8_t* frame =
        wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == nullptr)
        return -1;
    metadata_buffer_precondition_id =
        (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;
    return 0;
}


void hsaBeamformKernel::calculate_cl_index(uint32_t* host_map, float freq_now, double freq_ref) {
    float t, delta_t, beam_ref;
    int cl_index;
    float D2R = PI / 180.;
    int pad = 2;

    for (int b = 0; b < 256; ++b) {
        beam_ref =
            asin(LIGHT_SPEED * (b - 256 / 2.) / (freq_ref * 1.e6) / (256) / FEED_SEP) * 180. / PI;
        t = 256 * pad * (freq_ref * 1.e6) * (FEED_SEP / LIGHT_SPEED * sin(beam_ref * D2R)) + 0.5;
        delta_t = 256 * pad * (freq_now * 1e6 - freq_ref * 1e6)
                  * (FEED_SEP / LIGHT_SPEED * sin(beam_ref * D2R));
        cl_index = (int)floor(t + delta_t) + 256 * pad / 2.;

        if (cl_index < 0)
            cl_index = 256 * pad + cl_index;
        else if (cl_index > 256 * pad)
            cl_index = cl_index - 256 * pad;

        cl_index = cl_index - 256;
        if (cl_index < 0) {
            cl_index = 256 * pad + cl_index;
        }
        host_map[b] = cl_index;
    }
}

void hsaBeamformKernel::calculate_ew_phase(float freq_now, float* host_coeff,
                                           float* _ew_spacing_c) {
    for (int angle_iter = 0; angle_iter < 4; angle_iter++) {
        float anglefrac = sin(_ew_spacing_c[angle_iter] * PI / 180.);
        for (int cylinder = 0; cylinder < 4; cylinder++) {
            host_coeff[angle_iter * 4 * 2 + cylinder * 2] =
                cos(2 * PI * anglefrac * cylinder * 22 * freq_now * 1.e6 / LIGHT_SPEED);
            host_coeff[angle_iter * 4 * 2 + cylinder * 2 + 1] =
                sin(2 * PI * anglefrac * cylinder * 22 * freq_now * 1.e6 / LIGHT_SPEED);
        }
    }
}

hsa_signal_t hsaBeamformKernel::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, HSA kernel packets don't have precede_signals.
    (void)precede_signal;

    if (first_pass) {
        first_pass = false;
        stream_id_t stream_id = get_stream_id_t(metadata_buf, metadata_buffer_id);
        freq_idx = bin_number_chime(&stream_id);
        freq_MHz = freq_from_bin(freq_idx);
    }
    mark_frame_empty(metadata_buf, unique_name.c_str(), metadata_buffer_id);
    metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;

    if (update_NS_beam) {
        calculate_cl_index(host_map, freq_MHz, freq_ref);
        void* device_map = device.get_gpu_memory("beamform_map", map_len);
        device.sync_copy_host_to_gpu(device_map, (void*)host_map, map_len);
        update_NS_beam = false;
    }
    if (update_EW_beam) {
        calculate_ew_phase(freq_MHz, host_coeff, _ew_spacing_c);
        void* device_coeff_map = device.get_gpu_memory("beamform_coeff_map", coeff_len);
        device.sync_copy_host_to_gpu(device_coeff_map, (void*)host_coeff, coeff_len);
        update_EW_beam = false;
    }

    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* map_buffer;
        void* coeff_buffer;
        void* output_buffer;
        void* gain_buffer;
    } args;
    memset(&args, 0, sizeof(args));

    args.input_buffer = device.get_gpu_memory("input_reordered", input_frame_len);
    args.map_buffer = device.get_gpu_memory("beamform_map", map_len);
    args.coeff_buffer = device.get_gpu_memory("beamform_coeff_map", coeff_len);
    args.output_buffer = device.get_gpu_memory("beamform_output", output_frame_len);
    args.gain_buffer = device.get_gpu_memory_array("beamform_gain", gpu_frame_id, gain_len);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));
    kernelParams params;
    params.workgroup_size_x = 256;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = 256;
    params.grid_size_y = 2;
    params.grid_size_z = _samples_per_data_set;
    params.num_dims = 3;

    params.private_segment_size = 0;
    params.group_segment_size = 16384;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
