#include "clInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clInputData);

clInputData::clInputData(Config& config, const std::string& unique_name,
                         bufferContainer& host_buffers, clDeviceInterface& device, int inst) :
    clCommand(config, unique_name, host_buffers, device, inst, no_cl_command_state, "", "") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;

    if (inst == 0) {
        network_buf = host_buffers.get_buffer("network_buf");
        register_consumer(network_buf, unique_name.c_str());
    }
    command_type = gpuCommandType::COPY_IN;
}

clInputData::~clInputData() {}

int clInputData::wait_on_precondition() {
    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame = wait_for_full_frame(network_buf, unique_name.c_str(),
                                         gpu_frame_id % network_buf->num_frames);
    if (frame == nullptr)
        return -1;
    return 0;
}

cl_event clInputData::execute(cl_event pre_event) {
    pre_execute();

    int buf_index = gpu_frame_id % network_buf->num_frames;
    cl_mem gpu_memory_frame = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    void* host_memory_frame = (void*)network_buf->frames[buf_index];

    // Data transfer to GPU
    CHECK_CL_ERROR(
        clEnqueueWriteBuffer(device.getQueue(0), gpu_memory_frame, CL_FALSE,
                             0, // offset
                             input_frame_len, host_memory_frame, (pre_event == nullptr) ? 0 : 1,
                             (pre_event == nullptr) ? nullptr : &pre_event, &post_event));

    return post_event;
}

void clInputData::finalize_frame() {
    clCommand::finalize_frame();
    int buf_index = gpu_frame_id % network_buf->num_frames;
    mark_frame_empty(network_buf, unique_name.c_str(), buf_index);
}
