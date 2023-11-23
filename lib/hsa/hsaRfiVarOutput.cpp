#include "hsaRfiVarOutput.hpp"

#include "buffer.hpp"             // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp"    // for bufferContainer
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::COPY_OUT
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaRfiVarOutput);

hsaRfiVarOutput::hsaRfiVarOutput(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "hsaRfiVarOutput", "") {
    command_type = gpuCommandType::COPY_OUT;
    // Get buffers
    _network_buf = host_buffers.get_buffer("network_buf");
    _network_buf->register_consumer(unique_name);
    _rfi_output_var_buf = host_buffers.get_buffer("rfi_var_output_buf");
    _rfi_output_var_buf->register_producer(unique_name);
    // Initialize ID's
    _rfi_output_var_buf_id = 0;
    _rfi_output_var_buf_precondition_id = 0;
    _rfi_output_var_buf_execute_id = 0;
    _network_buf_id = 0;
    _network_buf_precondition_id = 0;
}

hsaRfiVarOutput::~hsaRfiVarOutput() {}

int hsaRfiVarOutput::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    // We want to make sure we have some space to put our results.
    uint8_t* frame =
        _rfi_output_var_buf->wait_for_empty_frame(unique_name, _rfi_output_var_buf_precondition_id);
    if (frame == nullptr)
        return -1;

    frame = _network_buf->wait_for_full_frame(unique_name, _network_buf_precondition_id);
    if (frame == nullptr)
        return -1;

    // Update precondition ID
    _rfi_output_var_buf_precondition_id =
        (_rfi_output_var_buf_precondition_id + 1) % _rfi_output_var_buf->num_frames;
    _network_buf_precondition_id = (_network_buf_precondition_id + 1) % _network_buf->num_frames;
    return 0;
}

hsa_signal_t hsaRfiVarOutput::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    // Get GPU memory
    void* gpu_output_ptr = device.get_gpu_memory_array(
        "rfi_output_var", gpu_frame_id, _gpu_buffer_depth, _rfi_output_var_buf->frame_size);
    // Copy GPU memory to host
    void* host_output_ptr = (void*)_rfi_output_var_buf->frames[_rfi_output_var_buf_execute_id];
    device.async_copy_gpu_to_host(host_output_ptr, gpu_output_ptr, _rfi_output_var_buf->frame_size,
                                  precede_signal, signals[gpu_frame_id]);

    // Update execution ID
    _rfi_output_var_buf_execute_id =
        (_rfi_output_var_buf_execute_id + 1) % _rfi_output_var_buf->num_frames;
    // Return signal
    return signals[gpu_frame_id];
}

void hsaRfiVarOutput::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    pass_metadata(_network_buf, _network_buf_id, _rfi_output_var_buf, _rfi_output_var_buf_id);
    _rfi_output_var_buf->mark_frame_full(unique_name, _rfi_output_var_buf_id);

    _network_buf->mark_frame_empty(unique_name, _network_buf_id);
    _network_buf_id = (_network_buf_id + 1) % _network_buf->num_frames;
    _rfi_output_var_buf_id = (_rfi_output_var_buf_id + 1) % _rfi_output_var_buf->num_frames;
}
