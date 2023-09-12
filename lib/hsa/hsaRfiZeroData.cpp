#include "hsaRfiZeroData.hpp"

#include "Config.hpp"             // for Config
#include "buffer.hpp"             // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "chimeMetadata.hpp"      // for set_rfi_zeroed
#include "configUpdater.hpp"      // for configUpdater
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "kotekanLogging.hpp"     // for INFO, WARN

#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, placehol...
#include <mutex>      // for lock_guard, mutex
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <string.h>   // for memcpy, memset
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;

REGISTER_HSA_COMMAND(hsaRfiZeroData);

hsaRfiZeroData::hsaRfiZeroData(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "rfi_chime_zero" KERNEL_EXT,
               "rfi_chime_zero.hsaco") {
    command_type = gpuCommandType::KERNEL;
    // Retrieve parameters from kotekan config
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // RFI Config Parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    // Compute Buffer lengths
    input_frame_len = sizeof(uint8_t) * _num_elements * _num_local_freq * _samples_per_data_set;
    mask_len = sizeof(uint8_t) * _num_local_freq * _samples_per_data_set / _sk_step;
    using namespace std::placeholders;
    configUpdater::instance().subscribe(
        config.get<std::string>(unique_name, "updatable_config/rfi_zeroing_toggle"),
        std::bind(&hsaRfiZeroData::update_rfi_zero_flag, this, _1));
    network_buf = host_buffers.get_buffer("network_buf");
    network_buffer_id = 0;
}

hsaRfiZeroData::~hsaRfiZeroData() {}

bool hsaRfiZeroData::update_rfi_zero_flag(nlohmann::json& json) {
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    try {
        _rfi_zeroing = json["rfi_zeroing"].get<bool>();
    } catch (std::exception& e) {
        WARN("Failed to set RFI zeroing flag {:s}", e.what());
        return false;
    }
    INFO("Changing RFI zero flag to {:d}", _rfi_zeroing);
    return true;
}


hsa_signal_t hsaRfiZeroData::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused variable, suppress warning.
    (void)precede_signal;

    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    // Structure for gpu arguments
    struct __attribute__((aligned(16))) args_t {
        void* input;
        void* mask;
        uint32_t sk_step;
        uint32_t rfi_zero_flag;
    } args;
    // Initialize arguments
    memset(&args, 0, sizeof(args));
    // Set argumnets to correct values
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.mask = device.get_gpu_memory_array("rfi_mask_output", gpu_frame_id, mask_len);
    args.sk_step = _sk_step;
    args.rfi_zero_flag = (uint32_t)_rfi_zeroing;
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
    set_rfi_zeroed(network_buf, network_buffer_id, (uint32_t)_rfi_zeroing);
    network_buffer_id = (network_buffer_id + 1) % network_buf->num_frames;
    // return signal
    return signals[gpu_frame_id];
}
