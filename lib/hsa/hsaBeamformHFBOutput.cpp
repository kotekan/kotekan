#include "hsaBeamformHFBOutput.hpp"

#include "gpuCommand.hpp" // for gpuCommandType, gpuCommandType::COPY_OUT

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaBeamformHFBOutputData);

hsaBeamformHFBOutputData::hsaBeamformHFBOutputData(Config& config, const std::string& unique_name,
                                                   bufferContainer& host_buffers,
                                                   hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "hsaBeamformHFBOutputData", "") {
    command_type = gpuCommandType::COPY_OUT;

    network_buffer = host_buffers.get_buffer("network_buf");
    network_buffer->register_consumer(unique_name);
    output_buffer = host_buffers.get_buffer("beamform_hfb_output_buf");
    output_buffer->register_producer(unique_name);

    network_buffer_id = 0;
    output_buffer_id = 0;
    output_buffer_execute_id = 0;
    output_buffer_precondition_id = 0;
    network_buffer_precondition_id = 0;
}

hsaBeamformHFBOutputData::~hsaBeamformHFBOutputData() {}

int hsaBeamformHFBOutputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    uint8_t* frame =
        output_buffer->wait_for_empty_frame(unique_name, output_buffer_precondition_id);
    if (frame == nullptr)
        return -1;
    uint8_t* network_frame =
        network_buffer->wait_for_full_frame(unique_name, network_buffer_precondition_id);
    if (network_frame == nullptr)
        return -1;
    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buffer->num_frames;
    network_buffer_precondition_id =
        (network_buffer_precondition_id + 1) % network_buffer->num_frames;
    return 0;
}

hsa_signal_t hsaBeamformHFBOutputData::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    void* gpu_output_ptr = device.get_gpu_memory_array(
        "hfb_sum_output", gpu_frame_id, _gpu_buffer_depth, output_buffer->frame_size);
    void* host_output_ptr = (void*)output_buffer->frames[output_buffer_execute_id];

    device.async_copy_gpu_to_host(host_output_ptr, gpu_output_ptr, output_buffer->frame_size,
                                  precede_signal, signals[gpu_frame_id]);

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;

    return signals[gpu_frame_id];
}

void hsaBeamformHFBOutputData::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    network_buffer->pass_metadata(network_buffer_id, output_buffer, output_buffer_id);

    network_buffer->mark_frame_empty(unique_name, network_buffer_id);
    output_buffer->mark_frame_full(unique_name, output_buffer_id);
    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}
