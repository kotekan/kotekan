#include "hsaRfiBadInput.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "restServer.hpp"         // for HTTP_RESPONSE, connectionInstance, restServer

#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_HSA_COMMAND(hsaRfiBadInput);

hsaRfiBadInput::hsaRfiBadInput(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "rfi_bad_input" KERNEL_EXT,
               "rfi_bad_input.hsaco") {
    command_type = gpuCommandType::KERNEL;
    // Retrieve parameters from kotekan config
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // RFI Config Parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    // Compute Buffer lengths
    input_frame_len =
        sizeof(float) * _num_local_freq * _num_elements * _samples_per_data_set / _sk_step;
    output_frame_len = sizeof(float) * _num_local_freq * _num_elements;
    // Local Parameters
}

hsaRfiBadInput::~hsaRfiBadInput() {
    // Free allocated memory
}

hsa_signal_t hsaRfiBadInput::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    // Structure for gpu arguments
    struct __attribute__((aligned(16))) args_t {
        void* input;
        void* output;
        uint32_t M;
        uint32_t num_sk;
    } args;
    // Initialize arguments
    memset(&args, 0, sizeof(args));
    // Set argumnets to correct values
    args.input = device.get_gpu_memory("time_sum", input_frame_len);
    args.output = device.get_gpu_memory_array("rfi_bad_input", gpu_frame_id, _gpu_buffer_depth, output_frame_len);
    args.M = _sk_step;
    args.num_sk = _samples_per_data_set / _sk_step;
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));
    // Apply correct kernel parameters
    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.grid_size_x = _num_elements;
    params.grid_size_y = _num_local_freq;
    params.num_dims = 2;
    // Should this be zero?
    params.private_segment_size = 0;
    params.group_segment_size = 0;

    // Execute kernel
    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);
    // return signal
    return signals[gpu_frame_id];
}
