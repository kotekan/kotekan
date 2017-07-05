#include "hsaInputData.hpp"
#include "buffers.h"
#include "bufferContainer.hpp"
#include "hsaBase.h"

hsaInputData::hsaInputData(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers, const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name){
    apply_config(0);
    network_buf = host_buffers.get_buffer("network_buf");
    network_buffer_id = 0;
    network_buffer_precondition_id = 0;
    network_buffer_finalize_id = 0;
}


hsaInputData::~hsaInputData() {
    hsa_host_free(presum_zeros);
}

void hsaInputData::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    input_frame_len =  _num_elements * _num_local_freq * _samples_per_data_set;
}

void hsaInputData::wait_on_precondition(int gpu_frame_id)
{
    // Wait for there to be data in the input (network) buffer.

    wait_for_full_buffer(network_buf, unique_name.c_str(), network_buffer_precondition_id);
    INFO("Got full buffer %s[%d], gpu[%d][%d]", network_buf->buffer_name, network_buffer_precondition_id,
            device.get_gpu_id(), gpu_frame_id);
    network_buffer_precondition_id = (network_buffer_precondition_id + 1) % network_buf->num_buffers;
}

hsa_signal_t hsaInputData::execute(int gpu_frame_id, const uint64_t& fpga_seq,
                                   hsa_signal_t precede_signal) {

    // Get the gpu and cpu memory pointers.
    void * gpu_memory_frame = device.get_gpu_memory_array("input",
                                                gpu_frame_id, input_frame_len);
    void * host_memory_frame = (void *)network_buf->data[network_buffer_id];

    // Do the input data copy.
    device.async_copy_host_to_gpu(gpu_memory_frame,
                                        host_memory_frame, input_frame_len,
                                        precede_signal, signals[gpu_frame_id]);

    network_buffer_id = (network_buffer_id + 1) % network_buf->num_buffers;

    return signals[gpu_frame_id];
}

void hsaInputData::finalize_frame(int frame_id)
{
    hsaCommand::finalize_frame(frame_id);
    // This is currently done in output data because we need to move the
    // info object first, this should be fixed at the buffer level somehow.
    //release_info_object(network_buf, network_buffer_id);
    //mark_buffer_empty(network_buf, unique_name.c_str(), network_buffer_finalize_id);
    //network_buffer_finalize_id = (network_buffer_finalize_id + 1) % network_buf->num_buffers;
}






