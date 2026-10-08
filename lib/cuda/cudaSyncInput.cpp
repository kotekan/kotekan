#include "cudaSyncInput.hpp"

#include "cudaCommand.hpp"          // for cudaCommand, REGISTER_CUDA_COMMAND, _factory_alias...
#include "cudaStreamAssignment.hpp" // for resolve_cuda_stream
#include "gpuCommand.hpp"           // for gpuCommandType

#include <vector> // for allocator, vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaSyncInput);

cudaSyncInput::cudaSyncInput(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device, int inst) :
    // Call the "subclassers" version of the cudaSyncStream constructor
    cudaSyncStream(config, unique_name, host_buffers, device, inst, true) {
    // Wait on this pipeline's own copy-in stream, resolved by the same rule that assigns it to
    // the copy-in commands. A literal 0 names another pipeline's stream once this one is
    // shifted, and cudaSyncStream::execute skips a null pre_events entry silently, so the kernel
    // would run before its input copy landed.
    const int32_t base = config.get_default<int32_t>(unique_name, "cuda_stream_base", 0);
    set_source_cuda_streams({resolve_cuda_stream(gpuCommandType::COPY_IN, base)});
    set_command_type(gpuCommandType::KERNEL);
    kernel_command = "sync_input";
}

cudaSyncInput::~cudaSyncInput() {}
