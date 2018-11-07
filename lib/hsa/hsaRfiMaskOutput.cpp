#include "hsaRfiMaskOutput.hpp"
#include "chimeMetadata.h"

REGISTER_HSA_COMMAND(hsaRfiMaskOutput);

hsaRfiMaskOutput::hsaRfiMaskOutput(Config& config, const string &unique_name,
                           bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand("hsaRfiMaskOutput","", config, unique_name, host_buffers, device){
    command_type = CommandType::COPY_OUT;
    //Get buffers
    _network_buf = host_buffers.get_buffer("network_buf");
    _rfi_mask_output_buf = host_buffers.get_buffer("rfi_mask_output_buf");
    //Config parameters
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    //Rfi paramters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    //Initialize ID's
    _network_buf_id = 0;
    _rfi_mask_output_buf_id = 0;
    _rfi_mask_output_buf_precondition_id = 0;
    _rfi_mask_output_buf_execute_id = 0;
}

hsaRfiMaskOutput::~hsaRfiMaskOutput() {
}

int hsaRfiMaskOutput::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    // We want to make sure we have some space to put our results.
    uint8_t * frame = wait_for_empty_frame(_rfi_mask_output_buf,
                          unique_name.c_str(), _rfi_mask_output_buf_precondition_id);
    if (frame == NULL) return -1;
    //Update precondition ID
    _rfi_mask_output_buf_precondition_id = (_rfi_mask_output_buf_precondition_id + 1) %
                                    _rfi_mask_output_buf->num_frames;
    return 0;
}

hsa_signal_t hsaRfiMaskOutput::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
    //Get GPU memory
    void * gpu_output_ptr = device.get_gpu_memory_array("rfi_mask_output", gpu_frame_id, _rfi_mask_output_buf->frame_size);
    //Copy GPU memory to host
    void * host_output_ptr = (void *)_rfi_mask_output_buf->frames[_rfi_mask_output_buf_execute_id];
    device.async_copy_gpu_to_host(host_output_ptr,
            gpu_output_ptr, _rfi_mask_output_buf->frame_size,
            precede_signal, signals[gpu_frame_id]);
    //Update execution ID
    _rfi_mask_output_buf_execute_id = (_rfi_mask_output_buf_execute_id + 1) % _rfi_mask_output_buf->num_frames;
    //Return signal
    return signals[gpu_frame_id];
}

void hsaRfiMaskOutput::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    if(get_rfi_zeroed(_network_buf,_network_buf_id)){
        uint8_t * frame_mask = _rfi_mask_output_buf->frames[_rfi_mask_output_buf_id];
        uint32_t total_lost = 0;
        //Copy RFI mask to array
        for(int32_t i = 0; i < _rfi_mask_output_buf->frame_size; i++){
            if(frame_mask[i] == 1) total_lost += _sk_step;
        }
        atomic_add_lost_timesamples(_network_buf, _network_buf_id, total_lost);
    }
    // Copy the information contained in the input buffer
    pass_metadata(_network_buf, _network_buf_id,
                  _rfi_mask_output_buf, _rfi_mask_output_buf_id);
    //Un-comment the following during testing when the gpu command hsaOutputData is not in use.
    // Mark the input buffer as "empty" so that it can be reused.
    //mark_frame_empty(_network_buf, unique_name.c_str(), _network_buf_id);
    // Mark the output buffer as full, so it can be processed.
    mark_frame_full(_rfi_mask_output_buf, unique_name.c_str(), _rfi_mask_output_buf_id);
    // Note this will change once we do accumulation in the GPU
    _network_buf_id = (_network_buf_id + 1) % _network_buf->num_frames;
    _rfi_mask_output_buf_id = (_rfi_mask_output_buf_id + 1) % _rfi_mask_output_buf->num_frames;
}
