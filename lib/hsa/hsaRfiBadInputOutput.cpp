#include "hsaRfiBadInputOutput.hpp"

REGISTER_HSA_COMMAND(hsaRfiBadInputOutput);

hsaRfiBadInputOutput::hsaRfiBadInputOutput(Config& config, const string &unique_name,
                           bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "",""){
    command_type = CommandType::COPY_OUT;
    //Get buffers
    _network_buf = host_buffers.get_buffer("network_buf");
    _rfi_output_buf = host_buffers.get_buffer("rfi_bad_input_buf");
    //Initialize ID's
    _network_buf_id = 0;
    _rfi_output_buf_id = 0;
    _rfi_output_buf_precondition_id = 0;
    _rfi_output_buf_execute_id = 0;
}

hsaRfiBadInputOutput::~hsaRfiBadInputOutput() {
}

int hsaRfiBadInputOutput::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    // We want to make sure we have some space to put our results.
    uint8_t * frame = wait_for_empty_frame(_rfi_output_buf,
                          unique_name.c_str(), _rfi_output_buf_precondition_id);
    if (frame == NULL) return -1;
    //Update precondition ID
    _rfi_output_buf_precondition_id = (_rfi_output_buf_precondition_id + 1) %
                                    _rfi_output_buf->num_frames;
    return 0;
}

hsa_signal_t hsaRfiBadInputOutput::execute(int gpu_frame_id,
                                           hsa_signal_t precede_signal) {
    //Get GPU memory
    void * gpu_output_ptr = device.get_gpu_memory_array("rfi_bad_input", gpu_frame_id, _rfi_output_buf->frame_size);
    //Copy GPU memory to host
    void * host_output_ptr = (void *)_rfi_output_buf->frames[_rfi_output_buf_execute_id];
    device.async_copy_gpu_to_host(host_output_ptr,
            gpu_output_ptr, _rfi_output_buf->frame_size,
            precede_signal, signals[gpu_frame_id]);
    //Update execution ID
    _rfi_output_buf_execute_id = (_rfi_output_buf_execute_id + 1) % _rfi_output_buf->num_frames;
    //Return signal
    return signals[gpu_frame_id];
}

void hsaRfiBadInputOutput::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
    // Copy the information contained in the input buffer
    pass_metadata(_network_buf, _network_buf_id,
                  _rfi_output_buf, _rfi_output_buf_id);
    // Mark the output buffer as full, so it can be processed.
    mark_frame_full(_rfi_output_buf, unique_name.c_str(), _rfi_output_buf_id);
    // Note this will change once we do accumulation in the GPU
    _network_buf_id = (_network_buf_id + 1) % _network_buf->num_frames;
    _rfi_output_buf_id = (_rfi_output_buf_id + 1) % _rfi_output_buf->num_frames;
}
