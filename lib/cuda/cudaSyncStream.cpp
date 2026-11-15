#include "cudaSyncStream.hpp"

#include "cudaStreamAssignment.hpp" // for check_stream_within
#include "cudaUtils.hpp"            // for CHECK_CUDA_ERROR
#include "cuda_runtime_api.h"       // for cudaStreamWaitEvent
#include "gpuCommand.hpp"           // for gpuCommandType

#include "fmt.hpp" // for format

#include <algorithm> // for copy
#include <stdexcept> // for runtime_error

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaSyncStream);

cudaSyncStream::cudaSyncStream(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst) {
    set_name("sync");
    set_source_cuda_streams(config.get<std::vector<int32_t>>(unique_name, "source_cuda_streams"));
    set_command_type(gpuCommandType::BARRIER);
}

cudaSyncStream::cudaSyncStream(kotekan::Config& config, const std::string& unique_name,
                               kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                               int inst, bool called_by_subclasser) :
    cudaCommand(config, unique_name, host_buffers, device, inst) {
    (void)called_by_subclasser;
    set_name("sync");
    // Don't call set_command_type!, because that requires set_source_cuda_streams first,
    // which would be called by the subclasser.
    // set_command_type(gpuCommandType::BARRIER);
}

void cudaSyncStream::set_source_cuda_streams(const std::vector<int32_t>& source_cuda_streams) {
    // Bounded by the owning stage's num_cuda_streams, as the commands themselves are: a stream
    // this stage never declared can carry none of its events, so waiting on it is a silent no-op.
    // Commands are named <stage>/commands/<n> (gpuProcess::init), which gives the stage's name.
    const int32_t base = config.get_default<int32_t>(unique_name, "cuda_stream_base", 0);
    const int32_t num_cuda_streams = config.get_default<int32_t>(unique_name, "num_cuda_streams",
                                                                 default_num_cuda_streams(base));
    const auto cut = unique_name.rfind("/commands/");
    const std::string stage = cut == std::string::npos ? unique_name : unique_name.substr(0, cut);
    _source_cuda_streams = source_cuda_streams;
    for (auto cuda_stream_id : _source_cuda_streams) {
        if (cuda_stream_id < 0)
            throw std::runtime_error(fmt::format("{:s}: source_cuda_streams entry {:d} is negative",
                                                 unique_name, cuda_stream_id));
        check_stream_within(unique_name + " (source stream)", cuda_stream_id, stage,
                            num_cuda_streams);
    }
}

cudaSyncStream::~cudaSyncStream() {}

cudaEvent_t cudaSyncStream::execute(cudaPipelineState&,
                                    const std::vector<cudaEvent_t>& pre_events) {
    pre_execute();
    record_start_event();
    for (auto source_stream_id : _source_cuda_streams) {
        if (pre_events[source_stream_id]) {
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(cuda_stream_id),
                                                 pre_events[source_stream_id]));
        }
    }
    return record_end_event();
}

std::string cudaSyncStream::get_performance_metric_string() {
    // Since this class syncs between different compute streams,
    // there's not really a firm notion of how long it takes!
    return "";
}
