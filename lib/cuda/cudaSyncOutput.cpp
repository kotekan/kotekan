#include "cudaSyncOutput.hpp"

#include "cudaCommand.hpp"    // for cudaCommand, REGISTER_CUDA_COMMAND, _factory_aliascudaCommand
#include "gpuCommand.hpp"     // for gpuCommandType
#include "kotekanLogging.hpp" // for DEBUG

#include "fmt.hpp" // for compile_string_to_view

#include <stdint.h> // for int32_t
#include <vector>   // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaSyncOutput);

cudaSyncOutput::cudaSyncOutput(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               int inst) :
    // Call the "subclassers" version of the cudaSyncStream constructor
    cudaSyncStream(config, unique_name, host_buffers, device, inst, true) {
    // Sync on all compute streams (all streams >= 2)
    DEBUG("Number of streams: {:d}", device.get_num_streams());
    std::vector<int32_t> streams;
    for (int32_t s = 2; s < device.get_num_streams(); s++)
        streams.push_back(s);
    set_source_cuda_streams(streams);
    set_command_type(gpuCommandType::COPY_OUT);
    kernel_command = "sync_output";
}

cudaSyncOutput::~cudaSyncOutput() {}
