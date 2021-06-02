#include "hipInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using std::string;

REGISTER_HIP_COMMAND(hipInputData);

hipInputData::hipInputData(Config& config, const string& unique_name,
                             bufferContainer& host_buffers, hipDeviceInterface& device) :
    hipCommand(config, unique_name, host_buffers, device, "", "") {

    in_buf = host_buffers.get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    for (int i = 0; i < in_buf->num_frames; i++) {
        uint flags;
        // only register the memory if it isn't already...
        if (hipErrorInvalidValue == hipHostGetFlags(&flags, in_buf->frames[i])) {
            CHECK_HIP_ERROR(hipHostRegister(in_buf->frames[i], in_buf->frame_size, 0));
        }
    }

    in_buffer_id = 0;
    in_buffer_precondition_id = 0;
    in_buffer_finalize_id = 0;

    command_type = gpuCommandType::COPY_IN;
}

hipInputData::~hipInputData() {
    for (int i = 0; i < in_buf->num_frames; i++) {
        uint flags;
        // only unregister if it's already been registered
        if (hipSuccess == hipHostGetFlags(&flags, in_buf->frames[i])) {
            CHECK_HIP_ERROR(hipHostUnregister(in_buf->frames[i]));
        }
    }
}

int hipInputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_precondition_id);
    if (frame == NULL)
        return -1;

    // only register the memory if it isn't already...
    uint flags;
    if (hipErrorInvalidValue == hipHostGetFlags(&flags, frame)) {
        CHECK_HIP_ERROR(hipHostRegister(frame, in_buf->frame_size, 0));
    }

    in_buffer_precondition_id = (in_buffer_precondition_id + 1) % in_buf->num_frames;
    return 0;
}

hipEvent_t hipInputData::execute(int gpu_frame_id, hipEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t input_frame_len = in_buf->frame_size;

    void* gpu_memory_frame = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    void* host_memory_frame = (void*)in_buf->frames[in_buffer_id];

    device.async_copy_host_to_gpu(gpu_memory_frame, host_memory_frame, input_frame_len, pre_event,
                                  pre_events[gpu_frame_id], post_events[gpu_frame_id]);

    in_buffer_id = (in_buffer_id + 1) % in_buf->num_frames;
    return post_events[gpu_frame_id];
}

void hipInputData::finalize_frame(int frame_id) {
    hipCommand::finalize_frame(frame_id);
    mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_finalize_id);
    in_buffer_finalize_id = (in_buffer_finalize_id + 1) % in_buf->num_frames;
}

std::string hipInputData::get_performance_metric_string() {
    double transfer_speed = (double)in_buf->frame_size
                            / (double)get_last_gpu_execution_time() / 1000000000;
    return "Speed: " + std::to_string(transfer_speed) + " GB/s";
}