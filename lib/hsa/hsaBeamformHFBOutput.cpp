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
    register_consumer(network_buffer, unique_name.c_str());
    output_buffer = host_buffers.get_buffer("beamform_hfb_output_buf");
    register_producer(output_buffer, unique_name.c_str());

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
        wait_for_empty_frame(output_buffer, unique_name.c_str(), output_buffer_precondition_id);
    if (frame == nullptr)
        return -1;
    uint8_t* network_frame =
        wait_for_full_frame(network_buffer, unique_name.c_str(), network_buffer_precondition_id);
    if (network_frame == nullptr)
        return -1;
    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buffer->num_frames;
    network_buffer_precondition_id =
        (network_buffer_precondition_id + 1) % network_buffer->num_frames;
    return 0;
}

hsa_signal_t hsaBeamformHFBOutputData::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    void* gpu_output_ptr =
        device.get_gpu_memory_array("hfb_sum_output", gpu_frame_id, output_buffer->frame_size);
    void* host_output_ptr = (void*)output_buffer->frames[output_buffer_execute_id];

    device.async_copy_gpu_to_host(host_output_ptr, gpu_output_ptr, output_buffer->frame_size,
                                  precede_signal, signals[gpu_frame_id]);

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;

    return signals[gpu_frame_id];
}

void hsaBeamformHFBOutputData::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    pass_metadata(network_buffer, network_buffer_id, output_buffer, output_buffer_id);

    mark_frame_empty(network_buffer, unique_name.c_str(), network_buffer_id);
    mark_frame_full(output_buffer, unique_name.c_str(), output_buffer_id);
    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}
