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

    in_buffer = host_buffers.get_buffer(config.get<std::string>(unique_name, "in_buf"));
    register_consumer(in_buffer, unique_name.c_str());

    output_buffer = host_buffers.get_buffer(config.get<std::string>(unique_name, "out_buf"));
    register_producer(output_buffer, unique_name.c_str());

    for (int i = 0; i < output_buffer->num_frames; i++) {
        uint flags;
        // only register the memory if it isn't already...
        if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, output_buffer->frames[i])) {
            CHECK_CUDA_ERROR(
                cudaHostRegister(output_buffer->frames[i], output_buffer->frame_size, 0));
        }
    }

    _gpu_mem = config.get<std::string>(unique_name, "gpu_mem");

    output_buffer_execute_id = 0;
    output_buffer_precondition_id = 0;

    output_buffer_id = 0;
    in_buffer_id = 0;

    skipped.resize(output_buffer->num_frames);

    set_command_type(gpuCommandType::COPY_OUT);

    kernel_command = "cudaOutputData: " + _gpu_mem;
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
    if (frame == nullptr) {
        DEBUG("FAILED to wait_for_empty_frame on output_buffer {:s}[:d]", unique_name.c_str(),
              output_buffer_precondition_id);
        return -1;
    }

    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buffer->num_frames;
    return 0;
}


cudaEvent_t cudaOutputData::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events, bool* quit) {
    pre_execute(gpu_frame_id);

    skipped[gpu_frame_id] = false;

    size_t output_len = output_buffer->frame_size;

    void* gpu_output_frame = device.get_gpu_memory_array(_gpu_mem, gpu_frame_id, output_len);
    void* host_output_frame = (void*)output_buffer->frames[output_buffer_execute_id];

    device.async_copy_gpu_to_host(host_output_frame, gpu_output_frame, output_len, cuda_stream_id,
                                  pre_events[cuda_stream_id], start_events[gpu_frame_id],
                                  end_events[gpu_frame_id]);

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;
    return end_events[gpu_frame_id];
}

void cudaOutputData::skipped_execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events) {
    skipped[gpu_frame_id] = true;
}

void cudaOutputData::finalize_frame(int frame_id) {
    cudaCommand::finalize_frame(frame_id);

    if (skipped[frame_id]) {
        DEBUG("Skipping Cuda output for gpu frame {:d}; input is {:s}[{:d}], output is {:s}[{:d}]", frame_id, in_buffer->buffer_name, in_buffer_id,
              output_buffer->buffer_name, output_buffer_id);
        mark_frame_empty(in_buffer, unique_name.c_str(), in_buffer_id);
        in_buffer_id = (in_buffer_id + 1) % in_buffer->num_frames;
        return;
    }
    DEBUG("Passing metadata from input {:s}[{:d}] to {:s}[{:d}]", in_buffer->buffer_name,
          in_buffer_id, output_buffer->buffer_name, output_buffer_id);
    pass_metadata(in_buffer, in_buffer_id, output_buffer, output_buffer_id);

    mark_frame_empty(in_buffer, unique_name.c_str(), in_buffer_id);
    in_buffer_id = (in_buffer_id + 1) % in_buffer->num_frames;

    mark_frame_full(output_buffer, unique_name.c_str(), output_buffer_id);
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}

std::string cudaOutputData::get_performance_metric_string() {
    double t = (double)get_last_gpu_execution_time();
    double transfer_speed = (double)output_buffer->frame_size / t * 1e-9;
    return fmt::format("Time: {:.6f} seconds, Speed: {:.2f} GB/s ({:.2f} Gb/s)", t, transfer_speed,
                       transfer_speed * 8);
}
