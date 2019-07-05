#include "hsaBeamformHFBSum.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaBeamformHFBSum);

hsaBeamformHFBSum::hsaBeamformHFBSum(Config& config, const string& unique_name,
                                     bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "sum_hfb" KERNEL_EXT,
               "sum_hfb.hsaco") {
    command_type = gpuCommandType::KERNEL;

    // Read parameters from config file.
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _num_frb_total_beams = config.get<int32_t>(unique_name, "num_frb_total_beams");

    input_frame_len = _num_frb_total_beams
                          * 128 // No. of frequencies
                          * 10 // No. of samples per beam
                          * sizeof(float);
    output_frame_len = _num_frb_total_beams
                       * 128 // No. of frequencies
                       * sizeof(float);
}

hsaBeamformHFBSum::~hsaBeamformHFBSum() {}

hsa_signal_t hsaBeamformHFBSum::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* output_buffer;
    } args;
    memset(&args, 0, sizeof(args));

    INFO("\nHFB frame length: %d", output_frame_len);

    args.input_buffer = device.get_gpu_memory("hfb_output", input_frame_len);
    args.output_buffer = device.get_gpu_memory_array("hfb_sum_output", gpu_frame_id, output_frame_len);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.grid_size_x = 64;
    params.grid_size_y = 1024; // No. of FRB beams
    params.num_dims = 2;

    params.private_segment_size = 0;
    params.group_segment_size = 128*10*sizeof(float);

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
