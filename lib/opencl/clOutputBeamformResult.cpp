#include "clOutputBeamformResult.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clOutputBeamformResult);

clOutputBeamformResult::clOutputBeamformResult(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "", "") {
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    network_buffer = host_buffers.get_buffer("network_buf");
    output_buffer = host_buffers.get_buffer("beam_out_buf");
    output_buffer_execute_id = 0;
    output_buffer_precondition_id = 0;

    output_buffer_id = 0;
    network_buffer_id = 0;
}

clOutputBeamformResult::~clOutputBeamformResult() {}


int clOutputBeamformResult::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    // Wait for there to be data in the input (output) buffer.
    uint8_t* frame =
        wait_for_empty_frame(output_buffer, unique_name.c_str(), output_buffer_precondition_id);
    if (frame == nullptr)
        return -1;

    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buffer->num_frames;
    return 0;
}

cl_event clOutputBeamformResult::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t output_len = _samples_per_data_set * _num_data_sets * _num_local_freq * 2;
    cl_mem output_memory_frame =
        device.get_gpu_memory_array("beamform_output_buf", gpu_frame_id, output_len);

    void* host_output_frame = (void*)output_buffer->frames[output_buffer_execute_id];

    CHECK_CL_ERROR(clEnqueueReadBuffer(device.getQueue(2), output_memory_frame, CL_FALSE, 0,
                                       output_len, host_output_frame, 1, &pre_event,
                                       &post_events[gpu_frame_id]));

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;
    return post_events[gpu_frame_id];
}


void clOutputBeamformResult::finalize_frame(int frame_id) {
    clCommand::finalize_frame(frame_id);

    pass_metadata(network_buffer, network_buffer_id, output_buffer, output_buffer_id);

    //    ALREADY DONE BY clOutputData
    //    mark_frame_empty(network_buffer, unique_name.c_str(), network_buffer_id);
    //    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;

    mark_frame_full(output_buffer, unique_name.c_str(), output_buffer_id);
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}
