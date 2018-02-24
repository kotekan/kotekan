#include "hsaOutputDataZero.hpp"

REGISTER_HSA_COMMAND(hsaOutputDataZero);

hsaOutputDataZero::hsaOutputDataZero(Config& config, const string &unique_name,
                            bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand("","", config, unique_name, host_buffers, device) {
    command_type = CommandType::COPY_IN;

    int block_size = config.get_int(unique_name, "block_size");
    int num_elements = config.get_int(unique_name, "num_elements");
    _num_blocks = (int32_t)(num_elements / block_size) *
                    (num_elements / block_size + 1) / 2.;
    output_len = _num_blocks * block_size * block_size * 2 * sizeof(int32_t);

    output_zeros = hsa_host_malloc(output_len);
    INFO("hsaOutputDataZero gpu[%d], Creating the output zero buffer: %p, len: %d",
            device.get_gpu_id(), output_zeros, output_len);

    memset(output_zeros, 0, output_len);
}

hsaOutputDataZero::~hsaOutputDataZero() {
    hsa_host_free(output_zeros);
}

hsa_signal_t hsaOutputDataZero::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    void * gpu_output_ptr = device.get_gpu_memory_array("corr", gpu_frame_id, output_len);

    device.async_copy_host_to_gpu(gpu_output_ptr,
                                    output_zeros, output_len,
                                    precede_signal, signals[gpu_frame_id]);

    return signals[gpu_frame_id];
}
