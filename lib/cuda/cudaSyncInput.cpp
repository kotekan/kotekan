#include "cudaSyncInput.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaSyncInput);

cudaSyncInput::cudaSyncInput(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device, int inst) :
    // Call the "subclassers" version of the cudaSyncStream constructor
    cudaSyncStream(config, unique_name, host_buffers, device, inst, true) {
    set_source_cuda_streams({0});
    set_command_type(gpuCommandType::KERNEL);
    kernel_command = "sync_input";
}

cudaSyncInput::~cudaSyncInput() {}
