#include "hsaBeamformTranspose.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config

#include <cstdint>   // for int32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector
#include "hsaBase.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaBeamformTranspose);

hsaBeamformTranspose::hsaBeamformTranspose(Config& config, const std::string& unique_name,
                                           bufferContainer& host_buffers,
                                           hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "transpose" KERNEL_EXT,
               "transpose.hsaco") {
    command_type = gpuCommandType::KERNEL;

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");

    // Kernel uses fp16 complex numbers which are the same size as sizeof(float)
    beamform_frame_len = _num_elements * _samples_per_data_set * sizeof(float);
    // The + 64 is to avoid bank conflicts in the transpose output
    output_frame_len = _num_elements * (_samples_per_data_set + 64) * sizeof(float);
}

hsaBeamformTranspose::~hsaBeamformTranspose() {}

hsa_signal_t hsaBeamformTranspose::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t {
        void* beamform_buffer;
        void* output_buffer;
    } args;
    memset(&args, 0, sizeof(args));
    args.beamform_buffer = device.get_gpu_memory("beamform_output", beamform_frame_len);
    args.output_buffer = device.get_gpu_memory("transposed_output", output_frame_len);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 32;
    params.workgroup_size_y = 8;
    params.grid_size_x = _num_elements;
    params.grid_size_y = _samples_per_data_set / 4;
    params.num_dims = 2;

    params.private_segment_size = 0;
    params.group_segment_size = 33*32*4;

    //KV
    params.workgroup_size_x = 32;
    params.workgroup_size_y = 2;
    params.grid_size_x = _num_elements;
    params.grid_size_y = _samples_per_data_set / 16;
    params.group_segment_size = 0;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
