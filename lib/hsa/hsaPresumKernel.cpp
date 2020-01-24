#include "hsaPresumKernel.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaCommand.hpp"         // for kernelParams, KERNEL_EXT, REGISTER_HSA_COMMAND, _facto...
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config

#include "fmt.hpp" // for format, fmt

#include <cstdint>   // for int32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaPresumKernel);

hsaPresumKernel::hsaPresumKernel(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name, host_buffers, device, "CHIME_presum" KERNEL_EXT,
                       "presum_opencl.hsaco") {
    command_type = gpuCommandType::KERNEL;

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    presum_len = _num_elements * _num_local_freq * 2 * sizeof(int32_t);

    // pre-allocate GPU memory
    device.get_gpu_memory_array("input", 0, input_frame_len);
    device.get_gpu_memory_array(fmt::format(fmt("presum_{:d}"), _sub_frame_index), 0, presum_len);
}

hsaPresumKernel::~hsaPresumKernel() {}

hsa_signal_t hsaPresumKernel::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    // Set kernel args
    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* mystery;
        int constant;
        void* presum_buffer;
    } args;

    memset(&args, 0, sizeof(args));

    // Index past the start of the input for the required sub frame
    args.input_buffer =
        (void*)((uint8_t*)device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len)
                + _num_elements * _num_local_freq * _sub_frame_samples * _sub_frame_index);
    args.mystery = nullptr;
    args.constant = _num_elements / 4; // global_x size
    args.presum_buffer = device.get_gpu_memory_array(
        fmt::format(fmt("presum_{:d}"), _sub_frame_index), gpu_frame_id, presum_len);

    // Copy kernel args into correct location for GPU
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    // Set kernel dims
    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = _num_elements / 4;
    params.grid_size_y = _sub_frame_samples / 128; // should be /1024 for presum.hsaco!
    params.grid_size_z = 1;
    params.num_dims = 2;

    // Should this be zero?
    params.private_segment_size = 0;
    params.group_segment_size = 0;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
