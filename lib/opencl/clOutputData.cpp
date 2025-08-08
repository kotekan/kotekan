#include "clOutputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clOutputData);

clOutputData::clOutputData(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, clDeviceInterface& device, int inst) :
    clCommand(config, unique_name, host_buffers, device, inst, no_cl_command_state, "", "") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _num_blocks = config.get<int>(unique_name, "num_blocks");

    network_buffer = host_buffers.get_buffer("network_buf");
    output_buffer = host_buffers.get_buffer("output_buf");
    if (inst == 0) {
        network_buffer->register_consumer(unique_name);
        output_buffer->register_producer(unique_name);
    }

    command_type = gpuCommandType::COPY_OUT;
}

clOutputData::~clOutputData() {}

int clOutputData::wait_on_precondition() {
    // Wait for there to be data in the input (output) buffer.
    int buf_index = gpu_frame_id % output_buffer->num_frames;
    uint8_t* frame = output_buffer->wait_for_empty_frame(unique_name, buf_index);
    if (frame == nullptr)
        return -1;
    return 0;
}


cl_event clOutputData::execute(cl_event pre_event) {
    pre_execute();

    int buf_index = gpu_frame_id % output_buffer->num_frames;
    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                          * _num_data_sets * sizeof(int32_t);

    cl_mem gpu_output_frame =
        device.get_gpu_memory_array("output", gpu_frame_id, _gpu_buffer_depth, output_len);
    void* host_output_frame = (void*)output_buffer->frames[buf_index];

    // Read the results
    CHECK_CL_ERROR(clEnqueueReadBuffer(device.getQueue(2), gpu_output_frame, CL_FALSE, 0,
                                       output_len, host_output_frame, 1, &pre_event, &post_event));
    return post_event;
}

void clOutputData::finalize_frame() {
    clCommand::finalize_frame();
    int net_index = gpu_frame_id % network_buffer->num_frames;
    int out_index = gpu_frame_id % output_buffer->num_frames;
    network_buffer->pass_metadata(net_index, output_buffer, out_index);
    network_buffer->mark_frame_empty(unique_name, net_index);
    output_buffer->mark_frame_full(unique_name, out_index);
}
