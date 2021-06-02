#include "clInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using std::string;

REGISTER_CL_COMMAND(clInputData);

clInputData::clInputData(Config& config, const string& unique_name, bufferContainer& host_buffers,
                         clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "clInputData", ""),
    in_buf(host_buffers.get_buffer(config.get<std::string>(unique_name, "in_buf"))),
    in_buf_id(in_buf),
    in_buf_precondition_id(in_buf),
    in_buf_finalize_id(in_buf),
    _gpu_memory_name(config.get<std::string>(unique_name, "gpu_memory_name")) {

    command_type = gpuCommandType::COPY_IN;

    register_consumer(in_buf, unique_name.c_str());

}

clInputData::~clInputData() {

}

int clInputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame =
        wait_for_full_frame(in_buf, unique_name.c_str(), in_buf_precondition_id);
    if (frame == nullptr)
        return -1;

    in_buf_precondition_id++;
    return 0;
}

cl_event clInputData::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    cl_mem gpu_memory_frame = device.get_gpu_memory_array(_gpu_memory_name,
                                                          gpu_frame_id, in_buf->frame_size);
    void* host_memory_frame = (void*)in_buf->frames[in_buf_id];

    // Data transfer to GPU
    CHECK_CL_ERROR(clEnqueueWriteBuffer(
        device.getQueue(0), gpu_memory_frame, CL_FALSE,
        0, // offset
        in_buf->frame_size, host_memory_frame, (pre_event == nullptr) ? 0 : 1,
        (pre_event == nullptr) ? nullptr : &pre_event, &post_events[gpu_frame_id]));

    in_buf_id++;
    return post_events[gpu_frame_id];
}

void clInputData::finalize_frame(int frame_id) {
    clCommand::finalize_frame(frame_id);
    mark_frame_empty(in_buf, unique_name.c_str(), in_buf_finalize_id);
    in_buf_finalize_id++;
}

std::string clInputData::get_performance_metric_string() {
    double transfer_speed = (double)in_buf->frame_size
                            / (double)get_last_gpu_execution_time() / 1000000000;
    return fmt::format("Speed: {:.2f} GB/s ({:.2f} Gb/s)", transfer_speed, transfer_speed * 8);
}