#include "cudaOutputDataZero.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaOutputDataZero);

cudaOutputDataZero::cudaOutputDataZero(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "", "") {

    output_len = config.get<int>(unique_name, "data_length");
    output_zeros = malloc(output_len);
    memset(output_zeros, 0, output_len);
    CHECK_CUDA_ERROR(cudaHostRegister(output_zeros, output_len, 0));

    set_command_type(gpuCommandType::COPY_IN);
}

cudaOutputDataZero::~cudaOutputDataZero() {
    free(output_zeros);
}

cudaEvent_t cudaOutputDataZero::execute(int gpu_frame_id,
                                        const std::vector<cudaEvent_t>& pre_events) {
    pre_execute(gpu_frame_id);

    void* gpu_memory_frame = device.get_gpu_memory_array("output", gpu_frame_id, output_len);

    if (pre_events[cuda_stream_id])
        CHECK_CUDA_ERROR(
            cudaStreamWaitEvent(device.getStream(cuda_stream_id), pre_events[cuda_stream_id], 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&start_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(start_events[gpu_frame_id], device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(gpu_memory_frame, output_zeros, output_len,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(end_events[gpu_frame_id], device.getStream(cuda_stream_id)));

    return end_events[gpu_frame_id];
}
