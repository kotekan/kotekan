#include "cudaSyncStream.hpp"

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
    _source_cuda_streams = source_cuda_streams;
    for (auto cuda_stream_id : _source_cuda_streams) {
        if (cuda_stream_id >= device.get_num_streams()) {
            throw std::runtime_error(
                "Asked to sync on a CUDA stream greater than the maximum number available");
        }
    }
}

cudaSyncStream::~cudaSyncStream() {}

int cudaSyncStream::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    return 0;
}

cudaEvent_t cudaSyncStream::execute(cudaPipelineState& pipestate,
                                    const std::vector<cudaEvent_t>& pre_events) {
    for (auto source_stream_id : _source_cuda_streams) {
        if (pre_events[source_stream_id]) {
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(cuda_stream_id),
                                                 pre_events[source_stream_id]));
        }
    }
    // Create an event for the StreamWaits, so other sync commands can sync on this one.
    return record_end_event(pipestate.gpu_frame_id);
}

void cudaSyncStream::finalize_frame(int frame_id) {
    (void)frame_id;
}

std::string cudaSyncStream::get_performance_metric_string() {
    // Since this class syncs between different compute streams,
    // there's not really a firm notion of how long it takes!
    return "";
}
