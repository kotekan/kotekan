#include "cudaSyncOutput.hpp"

#include "cudaCommand.hpp"          // for cudaCommand, REGISTER_CUDA_COMMAND, _factory_alias...
#include "cudaStreamAssignment.hpp" // for resolve_cuda_stream
#include "gpuCommand.hpp"           // for gpuCommandType
#include "kotekanLogging.hpp"       // for DEBUG

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
    std::vector<int32_t> streams;
    // Sync on this pipeline's compute streams: its kernel stream, resolved by the same rule
    // that assigns it to the kernels, and every further stream this stage declared. Starting at
    // a literal 2 would sweep in a shifted pipeline's own copy-in and copy-out streams, which
    // the same pipeline at base 0 deliberately excludes. The list stops at this stage's own
    // num_cuda_streams: a stream it never declared can carry none of its events.
    const int32_t base = config.get_default<int32_t>(unique_name, "cuda_stream_base", 0);
    const int32_t num_cuda_streams =
        config.get_default<int32_t>(unique_name, "num_cuda_streams", 3);
    // This stage's count, not the device's: the device pool is grown to the largest any stage
    // on it asked for, so logging that would name streams this sweep deliberately excludes.
    DEBUG("Number of streams: {:d}", num_cuda_streams);
    for (int32_t s = resolve_cuda_stream(gpuCommandType::KERNEL, base); s < num_cuda_streams; s++)
        streams.push_back(s);
    set_source_cuda_streams(streams);
    set_command_type(gpuCommandType::COPY_OUT);
    kernel_command = "sync_output";
}

cudaSyncOutput::~cudaSyncOutput() {}
