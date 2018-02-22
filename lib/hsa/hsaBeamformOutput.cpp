#include "hsaBeamformOutput.hpp"

hsaBeamformOutputData::hsaBeamformOutputData(const string& kernel_name,
        const string& kernel_file_name, hsaDeviceInterface& device,
        Config& config, bufferContainer& host_buffers,
        const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name) {
    command_type = CommandType::COPY_OUT;

    apply_config(0);

    network_buffer = host_buffers.get_buffer("network_buf");
    output_buffer = host_buffers.get_buffer("beamform_output_buf");

    network_buffer_id = 0;
    output_buffer_id = 0;
    output_buffer_excute_id = 0;
    output_buffer_precondition_id = 0;
}

hsaBeamformOutputData::~hsaBeamformOutputData() {

}

int hsaBeamformOutputData::wait_on_precondition(int gpu_frame_id) {
    uint8_t * frame = wait_for_empty_frame(output_buffer,
                          unique_name.c_str(), output_buffer_precondition_id);
    if (frame == NULL) return -1;
    INFO("Got empty buffer for output %s[%d], for GPU[%d][%d]", output_buffer->buffer_name,
            output_buffer_precondition_id, device.get_gpu_id(), gpu_frame_id);
    output_buffer_precondition_id = (output_buffer_precondition_id + 1) %
                                     output_buffer->num_frames;
    return 0;
}

hsa_signal_t hsaBeamformOutputData::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    void * gpu_output_ptr = device.get_gpu_memory_array("bf_output", gpu_frame_id, output_buffer->frame_size);

    void * host_output_ptr = (void *)output_buffer->frames[output_buffer_excute_id];

    device.async_copy_gpu_to_host(host_output_ptr,
            gpu_output_ptr, output_buffer->frame_size,
            precede_signal, signals[gpu_frame_id]);

    output_buffer_excute_id = (output_buffer_excute_id + 1) % output_buffer->num_frames;

    return signals[gpu_frame_id];
}

void hsaBeamformOutputData::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    pass_metadata(network_buffer, network_buffer_id,
                  output_buffer, output_buffer_id);

// NOTE: HACK TO ALLOW RUN ALONGSIDE N2! WILL NOT WORK INDEPENDENTLY!
//    mark_frame_empty(network_buffer, unique_name.c_str(), network_buffer_id);
    mark_frame_full(output_buffer, unique_name.c_str(), output_buffer_id);
    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}
