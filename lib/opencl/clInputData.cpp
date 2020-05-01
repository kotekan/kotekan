#include "clInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using std::string;

REGISTER_CL_COMMAND(clInputData);

clInputData::clInputData(Config& config, const string& unique_name, bufferContainer& host_buffers,
                         clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "clInputData", "") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;

    network_buf = host_buffers.get_buffer("network_buf");
    register_consumer(network_buf, unique_name.c_str());
    network_buffer_id = 0;
    network_buffer_precondition_id = 0;
    network_buffer_finalize_id = 0;

    command_type = gpuCommandType::COPY_IN;
}

clInputData::~clInputData() {}

int clInputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame =
        wait_for_full_frame(network_buf, unique_name.c_str(), network_buffer_precondition_id);
    if (frame == nullptr)
        return -1;
    // INFO("Got full buffer {:s}[{:d}], gpu[{:d}][{:d}]", network_buf->buffer_name,
    // network_buffer_precondition_id,
    //        device.get_gpu_id(), gpu_frame_id);

    network_buffer_precondition_id = (network_buffer_precondition_id + 1) % network_buf->num_frames;
    return 0;
}

cl_event clInputData::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    cl_mem gpu_memory_frame = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    void* host_memory_frame = (void*)network_buf->frames[network_buffer_id];

    // Data transfer to GPU
    CHECK_CL_ERROR(clEnqueueWriteBuffer(
        device.getQueue(0), gpu_memory_frame, CL_FALSE,
        0, // offset
        input_frame_len, host_memory_frame, (pre_event == nullptr) ? 0 : 1,
        (pre_event == nullptr) ? nullptr : &pre_event, &post_events[gpu_frame_id]));

    network_buffer_id = (network_buffer_id + 1) % network_buf->num_frames;
    return post_events[gpu_frame_id];
}

void clInputData::finalize_frame(int frame_id) {
    clCommand::finalize_frame(frame_id);
    mark_frame_empty(network_buf, unique_name.c_str(), network_buffer_finalize_id);
    network_buffer_finalize_id = (network_buffer_finalize_id + 1) % network_buf->num_frames;
}

std::string clInputData::get_performance_metric_string() {
    double transfer_speed = (double)network_buf->frame_size
                            / (double)get_last_gpu_execution_time() / 1000000000;
    return "Speed: " + std::to_string(transfer_speed) + " GB/s";
}