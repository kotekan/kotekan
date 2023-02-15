#include "cudaSyncStream.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaSyncStream);

cudaSyncStream::cudaSyncStream(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "sync", "") {

    _source_cuda_streams = config.get<std::vector<int32_t>>(unique_name, "source_cuda_streams");
    for (auto cuda_stream_id : _source_cuda_streams) {
        if (cuda_stream_id >= device.get_num_streams()) {
            throw std::runtime_error(
                "Asked for a CUDA stream grater than the maximum number available");
        }
    }
    set_command_type(gpuCommandType::BARRIER);
}

cudaSyncStream::~cudaSyncStream() {}

int cudaSyncStream::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    return 0;
}

cudaEvent_t cudaSyncStream::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events) {
    for (auto source_stream_id : _source_cuda_streams) {
        if (pre_events[source_stream_id]) {
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(cuda_stream_id),
                                                 pre_events[source_stream_id]));
        }
    }
    // Create an event for the StreamWaits, so other sync commands can sync on this one.
    CHECK_CUDA_ERROR(cudaEventCreate(&end_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(end_events[gpu_frame_id], device.getStream(cuda_stream_id)));

    return end_events[gpu_frame_id];
}

void cudaSyncStream::finalize_frame(int frame_id) {
    (void)frame_id;
}