#include "hsaBeamformTranspose.hpp"
#include "hsaBase.h"

hsaBeamformTranspose::hsaBeamformTranspose(const string& kernel_name, const string& kernel_file_name,
			    hsaDeviceInterface& device, Config& config,
			    bufferContainer& host_buffers,
			    const string &unique_name ) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name) {
    command_type = CommandType::KERNEL;
    apply_config(0);
}

hsaBeamformTranspose::~hsaBeamformTranspose() {

}

void hsaBeamformTranspose::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);

    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    beamform_frame_len  = _num_elements * _samples_per_data_set * 2 * sizeof(float);
    output_frame_len = _num_elements * (_samples_per_data_set+32) * 2 * sizeof(float);

}

hsa_signal_t hsaBeamformTranspose::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
    struct __attribute__ ((aligned(16))) args_t {
        void *beamform_buffer;
        void *output_buffer;
    } args;
    memset(&args, 0, sizeof(args));
/*
    args.beamform_buffer = device.get_gpu_memory("beamform_output", beamform_frame_len);
    args.output_buffer = device.get_gpu_memory("transposed_output", output_frame_len);
*/
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 32;
    params.workgroup_size_y = 8;
    params.grid_size_x = _num_elements;
    params.grid_size_y = _samples_per_data_set/4;
    params.num_dims = 2;

    params.private_segment_size = 0;
    params.group_segment_size = 8192;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}

