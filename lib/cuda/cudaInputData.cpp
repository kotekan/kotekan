#include "cudaInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaInputData);

cudaInputData::cudaInputData(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "", "") {

    in_buf = host_buffers.get_buffer(config.get<std::string>(unique_name, "in_buf"));
    register_consumer(in_buf, unique_name.c_str());

    for (int i = 0; i < in_buf->num_frames; i++) {
        uint flags;
        // only register the memory if it isn't already...
        if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, in_buf->frames[i])) {
            CHECK_CUDA_ERROR(cudaHostRegister(in_buf->frames[i], in_buf->frame_size, 0));
        }
    }

    _gpu_mem = config.get<std::string>(unique_name, "gpu_mem");

    in_buffer_id = 0;
    in_buffer_precondition_id = 0;
    in_buffer_finalize_id = 0;

    set_command_type(gpuCommandType::COPY_IN);

    kernel_command = "cudaInputData: " + _gpu_mem;
}

cudaInputData::~cudaInputData() {
    for (int i = 0; i < in_buf->num_frames; i++) {
        uint flags;
        // only unregister if it's already been registered
        if (cudaSuccess == cudaHostGetFlags(&flags, in_buf->frames[i])) {
            CHECK_CUDA_ERROR(cudaHostUnregister(in_buf->frames[i]));
        }
    }
}

int cudaInputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_precondition_id);
    if (frame == nullptr)
        return -1;

    in_buffer_precondition_id = (in_buffer_precondition_id + 1) % in_buf->num_frames;
    return 0;
}

cudaEvent_t cudaInputData::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                                   bool* quit) {
    (void)quit;
    pre_execute(gpu_frame_id);

    size_t input_frame_len = in_buf->frame_size;

    void* gpu_memory_frame = device.get_gpu_memory_array(_gpu_mem, gpu_frame_id, input_frame_len);
    void* host_memory_frame = (void*)in_buf->frames[in_buffer_id];

    device.async_copy_host_to_gpu(gpu_memory_frame, host_memory_frame, input_frame_len,
                                  cuda_stream_id, pre_events[cuda_stream_id],
                                  start_events[gpu_frame_id], end_events[gpu_frame_id]);

    in_buffer_id = (in_buffer_id + 1) % in_buf->num_frames;
    return end_events[gpu_frame_id];
}

void cudaInputData::finalize_frame(int frame_id) {
    cudaCommand::finalize_frame(frame_id);
    mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_finalize_id);
    in_buffer_finalize_id = (in_buffer_finalize_id + 1) % in_buf->num_frames;
}

std::string cudaInputData::get_performance_metric_string() {
    double t = (double)get_last_gpu_execution_time();
    double transfer_speed = (double)in_buf->frame_size / t * 1e-9;
    return fmt::format("Time: {:.3f} ms, Speed: {:.2f} GB/s ({:.2f} Gb/s)", t * 1e3, transfer_speed,
                       transfer_speed * 8);
}
