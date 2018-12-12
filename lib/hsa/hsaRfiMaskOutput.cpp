#include "hsaRfiMaskOutput.hpp"
#include "chimeMetadata.h"

REGISTER_HSA_COMMAND(hsaRfiMaskOutput);

hsaRfiMaskOutput::hsaRfiMaskOutput(Config& config, const string &unique_name,
                           bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name,
                                 host_buffers, device, "hsaRfiMaskOutput",""){
    command_type = gpuCommandType::COPY_OUT;
    //Get buffers
    _network_buf = host_buffers.get_buffer("network_buf");
    register_consumer(_network_buf, unique_name.c_str());
    _rfi_mask_output_buf = host_buffers.get_buffer("rfi_mask_output_buf");
    register_producer(_rfi_mask_output_buf, unique_name.c_str());
    _output_buf = host_buffers.get_buffer("output_buf");
    register_producer(_output_buf, unique_name.c_str());

    //Config parameters
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    //Rfi paramters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    //Initialize ID's
    _network_buf_id = 0;
    _output_buf_id = 0;
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

hsa_signal_t hsaRfiMaskOutput::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
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

    if (get_rfi_zeroed(_network_buf,_network_buf_id)){
        uint8_t * frame_mask = _rfi_mask_output_buf->frames[_rfi_mask_output_buf_id];
        uint32_t total_lost = 0;
        //Copy RFI mask to array
        for (uint32_t subframe = 0; subframe < _num_sub_frames; ++subframe) {
            uint32_t lost_in_subframe = 0;
            for (uint32_t i = subframe * _sub_frame_samples / _sk_step;
                i < (subframe + 1) * _sub_frame_samples / _sk_step;
                i++){
                assert(i < (uint32_t)_rfi_mask_output_buf->frame_size);
                if(frame_mask[i] == 1) lost_in_subframe += _sk_step;
            }
            atomic_add_lost_timesamples(_output_buf, _output_buf_id, lost_in_subframe);
            mark_frame_full(_output_buf, unique_name.c_str(), _output_buf_id);
            _output_buf_id = (_output_buf_id + 1) % _output_buf->num_frames;

            total_lost += lost_in_subframe;
        }
        atomic_add_lost_timesamples(_network_buf, _network_buf_id, total_lost);
    } else {
        // Since we are a producer we always need to sign off on the frame
        for (uint32_t subframe = 0; subframe < _num_sub_frames; ++subframe) {
            mark_frame_full(_output_buf, unique_name.c_str(), _output_buf_id);
            _output_buf_id = (_output_buf_id + 1) % _output_buf->num_frames;
        }
    }
    // Pass the information contained in the input buffer
    pass_metadata(_network_buf, _network_buf_id,
                  _rfi_mask_output_buf, _rfi_mask_output_buf_id);

    // Mark the input buffer as "empty" so that it can be reused.
    mark_frame_empty(_network_buf, unique_name.c_str(), _network_buf_id);
    // Mark the output buffer as full, so it can be processed.
    mark_frame_full(_rfi_mask_output_buf, unique_name.c_str(), _rfi_mask_output_buf_id);
    // Note this will change once we do accumulation in the GPU
    _network_buf_id = (_network_buf_id + 1) % _network_buf->num_frames;
    _rfi_mask_output_buf_id = (_rfi_mask_output_buf_id + 1) % _rfi_mask_output_buf->num_frames;
}
