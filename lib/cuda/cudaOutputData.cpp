/**
 * @file
 * @brief CUDA command to copy data from the GPU to the host
 *  - cudaOutputData : public cudaCommand
 */

#include "cudaOutputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaOutputData);

cudaOutputData::cudaOutputData(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "", "") {

    in_buffer = host_buffers.get_buffer("in_buf");
    register_consumer(in_buffer, unique_name.c_str());

    output_buffer = host_buffers.get_buffer("output_buf");
    register_producer(output_buffer, unique_name.c_str());

    for (int i = 0; i < output_buffer->num_frames; i++) {
        uint flags;
        // only register the memory if it isn't already...
        if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, output_buffer->frames[i])) {
            CHECK_CUDA_ERROR(
                cudaHostRegister(output_buffer->frames[i], output_buffer->frame_size, 0));
        }
    }

    output_buffer_execute_id = 0;
    output_buffer_precondition_id = 0;

    output_buffer_id = 0;
    in_buffer_id = 0;

    command_type = gpuCommandType::COPY_OUT;
}

cudaOutputData::~cudaOutputData() {
    for (int i = 0; i < output_buffer->num_frames; i++) {
        uint flags;
        // only register the memory if it isn't already...
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
    if (frame == nullptr)
        return -1;

    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buffer->num_frames;
    return 0;
}


cudaEvent_t cudaOutputData::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t output_len = output_buffer->frame_size;

    void* gpu_output_frame = device.get_gpu_memory_array("output", gpu_frame_id, output_len);
    void* host_output_frame = (void*)output_buffer->frames[output_buffer_execute_id];

    device.async_copy_gpu_to_host(host_output_frame, gpu_output_frame, output_len, pre_event,
                                  pre_events[gpu_frame_id], post_events[gpu_frame_id]);

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;
    return post_events[gpu_frame_id];
}

void cudaOutputData::finalize_frame(int frame_id) {
    cudaCommand::finalize_frame(frame_id);

    pass_metadata(in_buffer, in_buffer_id, output_buffer, output_buffer_id);

    mark_frame_empty(in_buffer, unique_name.c_str(), in_buffer_id);
    in_buffer_id = (in_buffer_id + 1) % in_buffer->num_frames;

    mark_frame_full(output_buffer, unique_name.c_str(), output_buffer_id);
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}
