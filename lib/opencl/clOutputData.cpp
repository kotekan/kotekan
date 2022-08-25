#include "clOutputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clOutputData);

clOutputData::clOutputData(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "clOutputData", "") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _num_blocks = config.get<int>(unique_name, "num_blocks");

    network_buffer = host_buffers.get_buffer("network_buf");
    register_consumer(network_buffer, unique_name.c_str());

    output_buffer = host_buffers.get_buffer("output_buf");
    register_producer(output_buffer, unique_name.c_str());

    output_buffer_execute_id = 0;
    output_buffer_precondition_id = 0;

    output_buffer_id = 0;
    network_buffer_id = 0;

    command_type = gpuCommandType::COPY_OUT;
}

clOutputData::~clOutputData() {}

int clOutputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    // Wait for there to be data in the input (output) buffer.
    uint8_t* frame =
        wait_for_empty_frame(output_buffer, unique_name.c_str(), output_buffer_precondition_id);
    if (frame == nullptr)
        return -1;

    uint8_t* network_frame = wait_for_full_frame(network_buffer, unique_name.c_str(),
                                network_buffer_id);
    if (network_frame == nullptr)
        return -1;

    // INFO("Got full buffer {:s}[{:d}], gpu[{:d}][{:d}]", output_buffer->buffer_name,
    // output_buffer_precondition_id,
    //        device.get_gpu_id(), gpu_frame_id);

    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buffer->num_frames;
    return 0;
}


cl_event clOutputData::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                          * _num_data_sets * sizeof(int32_t);

    cl_mem gpu_output_frame = device.get_gpu_memory_array("output", gpu_frame_id, output_len);
    void* host_output_frame = (void*)output_buffer->frames[output_buffer_execute_id];

    // Read the results
    CHECK_CL_ERROR(clEnqueueReadBuffer(device.getQueue(2), gpu_output_frame, CL_FALSE, 0,
                                       output_len, host_output_frame, 1, &pre_event,
                                       &post_events[gpu_frame_id]));

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;
    return post_events[gpu_frame_id];
}

void clOutputData::finalize_frame(int frame_id) {
    clCommand::finalize_frame(frame_id);

    pass_metadata(network_buffer, network_buffer_id, output_buffer, output_buffer_id);

    mark_frame_empty(network_buffer, unique_name.c_str(), network_buffer_id);
    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;

    mark_frame_full(output_buffer, unique_name.c_str(), output_buffer_id);
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}
