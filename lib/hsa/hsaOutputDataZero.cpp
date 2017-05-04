#include "hsaOutputDataZero.hpp"
#include "hsaBase.h"

hsaOutputDataZero::hsaOutputDataZero(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers) {

    apply_config(0);

    output_zeros = hsa_host_malloc(output_len);
    INFO("hsaOutputDataZero gpu[%d], Creating the output zero buffer: %p, len: %d",
            device.get_gpu_id(), output_zeros, output_len);

    memset(output_zeros, 0, output_len);
}

hsaOutputDataZero::~hsaOutputDataZero() {
    hsa_host_free(output_zeros);
}

void hsaOutputDataZero::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);
    _num_blocks = config.get_int("/gpu", "num_blocks");
    output_len = _num_blocks * 32 * 32 * 2 * sizeof(int32_t);
}

hsa_signal_t hsaOutputDataZero::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    void * gpu_output_ptr = device.get_gpu_memory_array("corr", gpu_frame_id, output_len);

    signals[gpu_frame_id] = device.async_copy_host_to_gpu(gpu_output_ptr,
                                    output_zeros, output_len, precede_signal);

    return signals[gpu_frame_id];
}
