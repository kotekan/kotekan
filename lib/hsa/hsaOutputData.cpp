#include "hsaOutputData.hpp"
#include <unistd.h>

hsaOutputData::hsaOutputData(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers, const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name){
    apply_config(0);

    network_buffer = host_buffers.get_buffer("network_buf");
    output_buffer = host_buffers.get_buffer("output_buf");

    network_buffer_id = 0;
    output_buffer_id = 0;
    output_buffer_precondition_id = 0;
    output_buffer_excute_id = 0;
}

hsaOutputData::~hsaOutputData() {
}

int hsaOutputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id; // Not used for this;
    // We want to make sure we have some space to put our results.
    uint8_t * frame = wait_for_empty_frame(output_buffer,
                          unique_name.c_str(), output_buffer_precondition_id);
    if (frame == NULL) return -1;
    output_buffer_precondition_id = (output_buffer_precondition_id + 1) %
                                    output_buffer->num_frames;
    return 0;
}

hsa_signal_t hsaOutputData::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    void * gpu_output_ptr = device.get_gpu_memory_array("corr", gpu_frame_id, output_buffer->frame_size);

    void * host_output_ptr = (void *)output_buffer->frames[output_buffer_excute_id];

    device.async_copy_gpu_to_host(host_output_ptr,
            gpu_output_ptr, output_buffer->frame_size,
            precede_signal, signals[gpu_frame_id]);

    output_buffer_excute_id = (output_buffer_excute_id + 1) % output_buffer->num_frames;

    return signals[gpu_frame_id];
}


void hsaOutputData::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    // Copy the information contained in the input buffer
    pass_metadata(network_buffer, network_buffer_id,
                  output_buffer, output_buffer_id);

    // Mark the input buffer as "empty" so that it can be reused.
    mark_frame_empty(network_buffer, unique_name.c_str(), network_buffer_id);

    // Mark the output buffer as full, so it can be processed.
    mark_frame_full(output_buffer, unique_name.c_str(), output_buffer_id);

    // Note this will change once we do accumulation in the GPU
    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}
