#include "hsaBeamformUpchan.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config

#include <cstdint>   // for int32_t, uint32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaBeamformUpchan);

hsaBeamformUpchan::hsaBeamformUpchan(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "upchannelize" KERNEL_EXT,
               "upchannelize_flip.hsaco") {
    command_type = gpuCommandType::KERNEL;

    // Read parameters from config file.
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _downsample_time = config.get<int32_t>(unique_name, "downsample_time");
    _downsample_freq = config.get<int32_t>(unique_name, "downsample_freq");
    _num_frb_total_beams = config.get<int32_t>(unique_name, "num_frb_total_beams");

    input_frame_len = _num_elements * (_samples_per_data_set + 32) * 2 * sizeof(float);
    output_frame_len = _num_frb_total_beams
                       * (_samples_per_data_set / _downsample_time / _downsample_freq)
                       * sizeof(float);
}

hsaBeamformUpchan::~hsaBeamformUpchan() {}

hsa_signal_t hsaBeamformUpchan::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* output_buffer;
    } args;
    memset(&args, 0, sizeof(args));

    args.input_buffer = device.get_gpu_memory("transposed_output", input_frame_len);
    args.output_buffer = device.get_gpu_memory_array("bf_output", gpu_frame_id, _gpu_buffer_depth, output_frame_len);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.grid_size_x = _samples_per_data_set / 6;
    params.grid_size_y = 1024; // No. of FRB beams
    params.num_dims = 2;

    params.private_segment_size = 0;
    params.group_segment_size = 3072;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
