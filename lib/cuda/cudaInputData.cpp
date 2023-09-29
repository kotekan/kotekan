#include "cudaInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaInputData);

cudaInputData::cudaInputData(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device,
                             int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num) {

    in_buf = host_buffers.get_buffer(config.get<std::string>(unique_name, "in_buf"));
    if (instance_num == 0) {
        register_consumer(in_buf, unique_name.c_str());
        for (int i = 0; i < in_buf->num_frames; i++) {
            uint flags;
            // only register the memory if it isn't already...
            if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, in_buf->frames[i])) {
                CHECK_CUDA_ERROR(cudaHostRegister(in_buf->frames[i], in_buf->frame_size, 0));
            }
        }
    }

    _gpu_mem = config.get<std::string>(unique_name, "gpu_mem");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem, true, false, true));

    set_command_type(gpuCommandType::COPY_IN);
    set_name("input: " + _gpu_mem);
}

cudaInputData::~cudaInputData() {
    if (instance_num == 0) {
        for (int i = 0; i < in_buf->num_frames; i++) {
            uint flags;
            // only unregister if it's already been registered
            if (cudaSuccess == cudaHostGetFlags(&flags, in_buf->frames[i])) {
                CHECK_CUDA_ERROR(cudaHostUnregister(in_buf->frames[i]));
            }
        }
    }
}

int cudaInputData::wait_on_precondition() {
    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame =
        wait_for_full_frame(in_buf, unique_name.c_str(), gpu_frame_id % in_buf->num_frames);
    if (frame == nullptr)
        return -1;
    return 0;
}

cudaEvent_t cudaInputData::execute(cudaPipelineState&, const std::vector<cudaEvent_t>& pre_events) {
    pre_execute();

    size_t input_frame_len = in_buf->frame_size;

    void* gpu_memory_frame = device.get_gpu_memory_array(_gpu_mem, gpu_frame_id, input_frame_len);
    void* host_memory_frame = (void*)in_buf->frames[gpu_frame_id % in_buf->num_frames];

    device.async_copy_host_to_gpu(gpu_memory_frame, host_memory_frame, input_frame_len,
                                  cuda_stream_id, pre_events[cuda_stream_id], start_event,
                                  end_event);

    // Copy (reference to) metadata also
    metadataContainer* meta = in_buf->metadata[gpu_frame_id % in_buf->num_frames];
    if (meta)
        device.claim_gpu_memory_array_metadata(_gpu_mem, gpu_frame_id, meta);

    return end_event;
}

void cudaInputData::finalize_frame() {
    // Release reference to metadata, if we grabbed it
    device.release_gpu_memory_array_metadata(_gpu_mem, gpu_frame_id);

    cudaCommand::finalize_frame();
    mark_frame_empty(in_buf, unique_name.c_str(), gpu_frame_id % in_buf->num_frames);
}

std::string cudaInputData::get_performance_metric_string() {
    double t = (double)get_last_gpu_execution_time();
    double transfer_speed = (double)in_buf->frame_size / t * 1e-9;
    return fmt::format("Time: {:.3f} ms, Speed: {:.2f} GB/s ({:.2f} Gb/s)", t * 1e3, transfer_speed,
                       transfer_speed * 8);
}
