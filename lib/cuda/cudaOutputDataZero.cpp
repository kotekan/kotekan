#include "cudaOutputDataZero.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaOutputDataZero);

cudaOutputDataZero::cudaOutputDataZero(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, cudaDeviceInterface& device,
                                       int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst) {

    output_len = config.get<int>(unique_name, "data_length");
    output_zeros = malloc(output_len);
    memset(output_zeros, 0, output_len);
    CHECK_CUDA_ERROR(cudaHostRegister(output_zeros, output_len, 0));

    set_command_type(gpuCommandType::COPY_IN);
}

cudaOutputDataZero::~cudaOutputDataZero() {
    free(output_zeros);
}

cudaEvent_t cudaOutputDataZero::execute(cudaPipelineState& pipestate,
                                        const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    void* gpu_memory_frame =
        device.get_gpu_memory_array("output", pipestate.gpu_frame_id, output_len);

    record_start_event(pipestate.gpu_frame_id);

    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaMemcpyAsync(gpu_memory_frame, output_zeros, output_len,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));

    return record_end_event(pipestate.gpu_frame_id);
}
