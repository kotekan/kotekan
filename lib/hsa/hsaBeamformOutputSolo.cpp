#include "hsaBeamformOutputSolo.hpp"

REGISTER_HSA_COMMAND(hsaBeamformOutputSolo);

hsaBeamformOutputSolo::hsaBeamformOutputSolo(Config& config, const string &unique_name,
        bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand("", "", config, unique_name, host_buffers, device) {
    command_type = CommandType::COPY_OUT;

    network_buffer = host_buffers.get_buffer("network_buf");
    output_buffer = host_buffers.get_buffer("beamform_output_buf");

    network_buffer_id = 0;
    output_buffer_id = 0;
    output_buffer_execute_id = 0;
    output_buffer_precondition_id = 0;
}

hsaBeamformOutputSolo::~hsaBeamformOutputSolo() {

}

int hsaBeamformOutputSolo::wait_on_precondition(int gpu_frame_id) {
    uint8_t * frame = wait_for_empty_frame(output_buffer,
                          unique_name.c_str(), output_buffer_precondition_id);
    if (frame == NULL) return -1;
    DEBUG("Got empty buffer for output %s[%d], for GPU[%d][%d]", output_buffer->buffer_name,
            output_buffer_precondition_id, device.get_gpu_id(), gpu_frame_id);
    output_buffer_precondition_id = (output_buffer_precondition_id + 1) %
                                     output_buffer->num_frames;
    return 0;
}

hsa_signal_t hsaBeamformOutputSolo::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    void * gpu_output_ptr = device.get_gpu_memory_array("bf_output", gpu_frame_id, output_buffer->frame_size);

    void * host_output_ptr = (void *)output_buffer->frames[output_buffer_execute_id];

    device.async_copy_gpu_to_host(host_output_ptr,
            gpu_output_ptr, output_buffer->frame_size,
            precede_signal, signals[gpu_frame_id]);

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;

    return signals[gpu_frame_id];
}

void hsaBeamformOutputSolo::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    pass_metadata(network_buffer, network_buffer_id,
                  output_buffer,  output_buffer_id);

// NOTE: WILL NOT WORK ALONGSIDE N2! INDEPENDENTLY ONLY!
    mark_frame_empty(network_buffer, unique_name.c_str(), network_buffer_id);
    mark_frame_full(  output_buffer, unique_name.c_str(), output_buffer_id);
    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;
    output_buffer_id  =  (output_buffer_id + 1) %  output_buffer->num_frames;
}
