#include "hsaPresumZero.hpp"
#include "hsaBase.h"

hsaPresumZero::hsaPresumZero(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers) {

    apply_config(0);
    presum_zeros = hsa_host_malloc(presum_len);
    memset(presum_zeros, 0, presum_len);
}

hsaPresumZero::~hsaPresumZero() {
    hsa_host_free(presum_zeros);
}

void hsaPresumZero::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);
    _num_elements = config.get_int("/processing/num_elements");
    _num_local_freq = config.get_int("/processing/num_local_freq");
    presum_len = _num_elements * _num_local_freq * 2 * sizeof (int32_t);
}

hsa_signal_t hsaPresumZero::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    void * gpu_memory_frame = device.get_gpu_memory_array("presum",
                                                gpu_frame_id, presum_len);

    signals[gpu_frame_id] = device.async_copy_host_to_gpu(gpu_memory_frame,
                                        presum_zeros, presum_len, precede_signal);

    return signals[gpu_frame_id];
}
