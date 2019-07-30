#include "cudaOutputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaOutputData);

cudaOutputData::cudaOutputData(Config& config, const string& unique_name, bufferContainer& host_buffers,
                           cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "", "") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _num_blocks = config.get<int>(unique_name, "num_blocks");

    network_buffer = host_buffers.get_buffer("network_buf");
    register_consumer(network_buffer, unique_name.c_str());

    output_buffer = host_buffers.get_buffer("output_buf");
    register_producer(output_buffer, unique_name.c_str());

    for (int i=0; i<output_buffer->num_frames; i++){
        uint flags;
        //only register the memory if it isn't already...
        if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, output_buffer->frames[i])) {
            CHECK_CUDA_ERROR(cudaHostRegister(output_buffer->frames[i], output_buffer->frame_size, 0));
        }
    }

    output_buffer_execute_id = 0;
    output_buffer_precondition_id = 0;

    output_buffer_id = 0;
    network_buffer_id = 0;

    command_type = gpuCommandType::COPY_OUT;
}

cudaOutputData::~cudaOutputData() {
    for (int i=0; i<output_buffer->num_frames; i++){
        uint flags;
        //only register the memory if it isn't already...
        if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, output_buffer->frames[i])) {
            CHECK_CUDA_ERROR(cudaHostUnregister(output_buffer->frames[i]));
        }
    }
}

int cudaOutputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    // Wait for there to be data in the input (output) buffer.
    uint8_t* frame =
        wait_for_empty_frame(output_buffer, unique_name.c_str(), output_buffer_precondition_id);
    if (frame == NULL)
        return -1;
    // INFO("Got full buffer %s[%d], gpu[%d][%d]", output_buffer->buffer_name,
    // output_buffer_precondition_id,
    //        device.get_gpu_id(), gpu_frame_id);

    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buffer->num_frames;
    return 0;
}


cudaEvent_t cudaOutputData::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t output_len = output_buffer->frame_size;

    void *gpu_output_frame = device.get_gpu_memory_array("output", gpu_frame_id, output_len);
    void *host_output_frame = (void*)output_buffer->frames[output_buffer_execute_id];

    if (pre_event) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_OUTPUT_STREAM), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&pre_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(pre_events[gpu_frame_id], device.getStream(CUDA_OUTPUT_STREAM)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_output_frame, gpu_output_frame,
                                     output_len, cudaMemcpyDeviceToHost, device.getStream(CUDA_OUTPUT_STREAM)));
    CHECK_CUDA_ERROR(cudaEventCreate(&post_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(post_events[gpu_frame_id], device.getStream(CUDA_OUTPUT_STREAM)));

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;
    return post_events[gpu_frame_id];
}

void cudaOutputData::finalize_frame(int frame_id) {
    cudaCommand::finalize_frame(frame_id);

    pass_metadata(network_buffer, network_buffer_id, output_buffer, output_buffer_id);

    mark_frame_empty(network_buffer, unique_name.c_str(), network_buffer_id);
    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;

    mark_frame_full(output_buffer, unique_name.c_str(), output_buffer_id);
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}
