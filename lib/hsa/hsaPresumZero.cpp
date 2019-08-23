#include "hsaPresumZero.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaPresumZero);

hsaPresumZero::hsaPresumZero(Config& config, const string& unique_name,
                             bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name, host_buffers, device, "hsaPresumZero", "") {
    command_type = gpuCommandType::COPY_IN;
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    presum_len = _num_elements * _num_local_freq * 2 * sizeof(int32_t);
    presum_zeros = hsa_host_malloc(presum_len, device.get_gpu_numa_node());
    memset(presum_zeros, 0, presum_len);
}

hsaPresumZero::~hsaPresumZero() {
    hsa_host_free(presum_zeros);
}

hsa_signal_t hsaPresumZero::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    void* gpu_memory_frame = device.get_gpu_memory_array(
        "presum_" + std::to_string(_sub_frame_index), gpu_frame_id, presum_len);

    device.async_copy_host_to_gpu(gpu_memory_frame, presum_zeros, presum_len, precede_signal,
                                  signals[gpu_frame_id]);

    return signals[gpu_frame_id];
}
