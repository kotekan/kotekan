#include "cudaSyncOutput.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaSyncOutput);

cudaSyncOutput::cudaSyncOutput(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    // Call the "subclassers" version of the cudaSyncStream constructor
    cudaSyncStream(config, unique_name, host_buffers, device, true) {
    std::vector<int32_t> streams;
    INFO("Number of streams: {:d}", device.get_num_streams());
    for (int32_t s = 2; s < device.get_num_streams(); s++)
        streams.push_back(s);
    set_source_cuda_streams(streams);
    set_command_type(gpuCommandType::COPY_OUT);
}

cudaSyncOutput::~cudaSyncOutput() {}
