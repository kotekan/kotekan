#include "hsaBeamformHFBSum.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaBeamformHFBSum);

hsaBeamformHFBSum::hsaBeamformHFBSum(Config& config, const string& unique_name,
                                     bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "sum_hfb" KERNEL_EXT, "sum_hfb.hsaco") {
    command_type = gpuCommandType::KERNEL;

    // Read parameters from config file.
    _num_frb_total_beams = config.get<int32_t>(unique_name, "num_frb_total_beams");
    _num_sub_freqs = config.get<uint32_t>(unique_name, "num_sub_freqs");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    uint32_t _factor_upchan = config.get<uint32_t>(unique_name, "factor_upchan");
    uint32_t _downsample_time = config.get<uint32_t>(unique_name, "downsample_time");
    _num_samples = _samples_per_data_set / _factor_upchan / _downsample_time;

    input_frame_len = _num_frb_total_beams * _num_sub_freqs
                      * _num_samples // No. of samples per beam
                      * sizeof(float);
    output_frame_len = _num_frb_total_beams * _num_sub_freqs * sizeof(float);
    lost_samples_frame_len = sizeof(uint8_t) * _samples_per_data_set / _num_sub_freqs;
}

hsaBeamformHFBSum::~hsaBeamformHFBSum() {}

hsa_signal_t hsaBeamformHFBSum::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* lost_samples_buffer;
        void* output_buffer;
        uint32_t num_samples;
    } args;
    memset(&args, 0, sizeof(args));

    args.input_buffer = device.get_gpu_memory("hfb_output", input_frame_len);
    args.output_buffer =
        device.get_gpu_memory_array("hfb_sum_output", gpu_frame_id, output_frame_len);
    args.lost_samples_buffer =
        device.get_gpu_memory_array("lost_samples_buffer", gpu_frame_id, lost_samples_frame_len);
    args.num_samples = _num_samples;

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.grid_size_x = 64;
    params.grid_size_y = 1024; // No. of FRB beams
    params.num_dims = 2;

    params.private_segment_size = 0;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
