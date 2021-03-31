#include "hsaRfiTimeSum.hpp"

#include "Config.hpp"             // for Config
#include "chimeMetadata.hpp"      // for get_rfi_num_bad_inputs
#include "configUpdater.hpp"      // for configUpdater
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "kotekanLogging.hpp"     // for DEBUG, INFO, WARN
#include "visUtil.hpp"            // for parse_reorder_default

#include <algorithm>  // for copy
#include <cstdint>    // for uint32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <string.h>   // for memcpy, memset
#include <tuple>      // for get

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaRfiTimeSum);

hsaRfiTimeSum::hsaRfiTimeSum(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "rfi_chime_time_sum" KERNEL_EXT,
               "rfi_chime_time_sum.hsaco") {
    command_type = gpuCommandType::KERNEL;
    // Retrieve parameters from kotekan config
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // RFI Config Parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    // Compute Buffer lengths
    input_frame_len = sizeof(uint8_t) * _num_elements * _num_local_freq * _samples_per_data_set;
    output_frame_len =
        sizeof(float) * _num_local_freq * _num_elements * _samples_per_data_set / _sk_step;
    output_var_frame_len =
        sizeof(float) * _num_elements * _num_local_freq * _samples_per_data_set / _sk_step;

    auto input_reorder = parse_reorder_default(config, unique_name);
    input_remap = std::get<0>(input_reorder);
}

hsaRfiTimeSum::~hsaRfiTimeSum() {}

hsa_signal_t hsaRfiTimeSum::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    // Structure for gpu arguments
    struct __attribute__((aligned(16))) args_t {
        void* input;
        void* output;
        void* output_var;
        uint32_t sk_step;
    } args;
    // Initialize arguments
    memset(&args, 0, sizeof(args));
    // Set argumnets to correct values
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.output = device.get_gpu_memory("time_sum", output_frame_len);
    args.output_var = device.get_gpu_memory("rfi_time_sum_var", output_var_frame_len);
    args.sk_step = _sk_step;
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));
    // Apply correct kernel parameters
    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = _num_elements / 4;
    params.grid_size_y = _samples_per_data_set / _sk_step;
    params.grid_size_z = 1;
    params.num_dims = 2;
    // Should this be zero?
    params.private_segment_size = 0;
    params.group_segment_size = 0;

    // Execute kernel
    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    // return signal
    return signals[gpu_frame_id];
}
