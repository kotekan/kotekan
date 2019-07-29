#include "cudaInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaInputData);

cudaInputData::cudaInputData(Config& config, const string& unique_name, bufferContainer& host_buffers,
                         cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "", "") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");

    network_buf = host_buffers.get_buffer("network_buf");
    register_consumer(network_buf, unique_name.c_str());
    network_buffer_id = 0;
    network_buffer_precondition_id = 0;
    network_buffer_finalize_id = 0;

    command_type = gpuCommandType::COPY_IN;
}

cudaInputData::~cudaInputData() {}

int cudaInputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame =
        wait_for_full_frame(network_buf, unique_name.c_str(), network_buffer_precondition_id);
    if (frame == NULL)
        return -1;
    // INFO("Got full buffer %s[%d], gpu[%d][%d]", network_buf->buffer_name,
    // network_buffer_precondition_id,
    //        device.get_gpu_id(), gpu_frame_id);

    network_buffer_precondition_id = (network_buffer_precondition_id + 1) % network_buf->num_frames;
    return 0;
}

cudaEvent_t cudaInputData::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t input_frame_len = network_buf->frame_size;

    void *gpu_memory_frame = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    void *host_memory_frame = (void*)network_buf->frames[network_buffer_id];

    if (pre_event) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_INPUT_STREAM), pre_event, 0));
    // Data transfer to GPU
    cudaEventCreate(&pre_events[gpu_frame_id]);
    cudaEventRecord(pre_events[gpu_frame_id], device.getStream(CUDA_INPUT_STREAM));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(gpu_memory_frame, host_memory_frame,
                                     input_frame_len, cudaMemcpyHostToDevice , device.getStream(CUDA_INPUT_STREAM)));
    cudaEventCreate(&post_events[gpu_frame_id]);
    cudaEventRecord(post_events[gpu_frame_id], device.getStream(CUDA_INPUT_STREAM));

    network_buffer_id = (network_buffer_id + 1) % network_buf->num_frames;
    return post_events[gpu_frame_id];
}

void cudaInputData::finalize_frame(int frame_id) {
    cudaCommand::finalize_frame(frame_id);
    mark_frame_empty(network_buf, unique_name.c_str(), network_buffer_finalize_id);
    network_buffer_finalize_id = (network_buffer_finalize_id + 1) % network_buf->num_frames;
}
