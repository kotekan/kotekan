#include "hsaOutputData.hpp"
#include <unistd.h>

hsaOutputData::hsaOutputData(const string& kernel_name, const string& kernel_file_name,
                            gpuHSADeviceInterface& device, Config& config,
                            bufferContainer& host_buffers) :
    gpuHSAcommand(kernel_name, kernel_file_name, device, config, host_buffers){
    apply_config(0);

    network_buffer = host_buffers.get_buffer("network_buf");
    output_buffer = host_buffers.get_buffer("output_buf");

    network_buffer_id = 0;
    output_buffer_id = 0;
    output_buffer_precondition_id = 0;
}

hsaOutputData::~hsaOutputData() {
}

void hsaOutputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id; // Not used for this;
    // We want to make sure we have some space to put our results.
    wait_for_empty_buffer(output_buffer, output_buffer_precondition_id);
    output_buffer_precondition_id = (output_buffer_precondition_id + 1) %
                                    output_buffer->num_buffers;
}

hsa_signal_t hsaOutputData::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    void * gpu_output_ptr = device.get_gpu_memory_array("corr", gpu_frame_id, output_buffer->buffer_size);

    void * host_output_ptr = (void *)output_buffer->data[output_buffer_id];

    signals[gpu_frame_id] = device.async_copy_gpu_to_host(host_output_ptr,
            gpu_output_ptr, output_buffer->buffer_size, precede_signal);

    return signals[gpu_frame_id];
}


void hsaOutputData::finalize_frame(int frame_id) {
    gpuHSAcommand::finalize_frame(frame_id);

    // Copy the information contained in the input buffer
    copy_buffer_info(network_buffer, network_buffer_id,
                     output_buffer, output_buffer_id);

    // Mark the input buffer as "empty" so that it can be reused.
    mark_buffer_empty(network_buffer, network_buffer_id);

    // Mark the output buffer as full, so it can be processed.
    mark_buffer_full(output_buffer, output_buffer_id);

    // Note this will change once we do accumulation in the GPU
    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_buffers;
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_buffers;
}
