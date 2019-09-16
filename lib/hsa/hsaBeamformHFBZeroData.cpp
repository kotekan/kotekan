#include "hsaBeamformHFBZeroData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaBeamformHFBZeroData);

hsaBeamformHFBZeroData::hsaBeamformHFBZeroData(Config& config, const string& unique_name,
                                               bufferContainer& host_buffers,
                                               hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name, host_buffers, device, "hsaBeamformHFBZeroData", "") {
    command_type = gpuCommandType::COPY_IN;

    int block_size = config.get<int>(unique_name, "block_size");
    int num_elements = config.get<int>(unique_name, "num_elements");
    _num_blocks = (int32_t)(num_elements / block_size) * (num_elements / block_size + 1) / 2.;
    output_len = 1024
                 * 128 // No. of frequencies
                 //* (_samples_per_data_set / _downsample_time)
                 * sizeof(float);

    output_zeros = hsa_host_malloc(output_len, device.get_gpu_numa_node());
    INFO("hsaBeamformHFBZeroData gpu[%d], Creating the output zero buffer: %p, len: %d",
         device.get_gpu_id(), output_zeros, output_len);

    memset(output_zeros, 0, output_len);
}

hsaBeamformHFBZeroData::~hsaBeamformHFBZeroData() {
    hsa_host_free(output_zeros);
}

hsa_signal_t hsaBeamformHFBZeroData::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    void* gpu_output_ptr = device.get_gpu_memory_array("hfb_output", gpu_frame_id, output_len);

    device.async_copy_host_to_gpu(gpu_output_ptr, output_zeros, output_len, precede_signal,
                                  signals[gpu_frame_id]);

    return signals[gpu_frame_id];
}
