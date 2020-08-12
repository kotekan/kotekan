#include "hsaTrackingBeamform.hpp"

#include "Config.hpp"             // for Config
#include "chimeMetadata.h"        // for MAX_NUM_BEAMS
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config

#include "fmt.hpp" // for format, fmt

#include <cstdint>   // for int32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaTrackingBeamform);

hsaTrackingBeamform::hsaTrackingBeamform(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers,
                                         hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "trackingbf_float" KERNEL_EXT,
               "tracking_beamformer_nbeam.hsaco") {
    command_type = gpuCommandType::KERNEL;

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_beams = config.get<int32_t>(unique_name, "num_beams");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _num_pol = config.get<int32_t>(unique_name, "num_pol");

    input_frame_len = _num_elements * _samples_per_data_set;
    output_frame_len = _samples_per_data_set * _num_beams * _num_pol * 2 * sizeof(float);

    phase_len = _num_elements * _num_beams * 2 * sizeof(float);

    if (_num_beams > MAX_NUM_BEAMS)
        throw std::runtime_error(
            fmt::format(fmt("Too many beams (_num_beams: {:d}). Max allowed is: {:d}"), _num_beams,
                        MAX_NUM_BEAMS));
}

hsaTrackingBeamform::~hsaTrackingBeamform() {}

hsa_signal_t hsaTrackingBeamform::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* phase_buffer;
        void* output_buffer;
    } args;
    memset(&args, 0, sizeof(args));
    args.input_buffer = device.get_gpu_memory("input_reordered", input_frame_len);
    args.phase_buffer = device.get_gpu_memory_array("beamform_phase", gpu_frame_id, phase_len);
    args.output_buffer =
        device.get_gpu_memory_array("bf_trk_output", gpu_frame_id, output_frame_len);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 64; // 256;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = 128; // 512;
    params.grid_size_y = _num_beams;
    params.grid_size_z = _samples_per_data_set / 64; // 32;
    params.num_dims = 3;

    params.private_segment_size = 0;
    params.group_segment_size = 2048;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
