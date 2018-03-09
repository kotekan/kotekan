#include "hsaRfiInputSum.hpp"
#include "hsaBase.h"
#include <math.h>

hsaRfiInputSum::hsaRfiInputSum(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers,
                            const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name) {

    command_type = CommandType::KERNEL;
    //Retrieve parameters from kotekan config
    apply_config(0); 
}

hsaRfiInputSum::~hsaRfiInputSum() {
}

void hsaRfiInputSum::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);

    //Data Parameters
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    //RFI Config Parameters
    _sk_step = config.get_int(unique_name, "sk_step");

    //Compute Buffer lengths
    input_frame_len = sizeof(float)*_num_elements*_num_local_freq*_samples_per_data_set/_sk_step;
    output_frame_len = sizeof(float)*_num_local_freq*_samples_per_data_set/_sk_step;
    _num_bad_inputs = 0;
    _M = (_num_elements - _num_bad_inputs)*_sk_step; 
}

hsa_signal_t hsaRfiInputSum::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    struct __attribute__ ((aligned(16))) args_t { 
	void *input; //Input Data
	void *output; 
	uint32_t num_elements; 
	uint32_t M; 
    } args;

    memset(&args, 0, sizeof(args));
    args.input = device.get_gpu_memory("timesum", input_frame_len);
    args.output = device.get_gpu_memory_array("rfi_output",gpu_frame_id, output_frame_len);
    args.num_elements = _num_elements;
    args.M = _M;
    //INFO("NUM ELEMENTS: %d M: %d",_num_elements, _M);
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 256;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = 256;
    params.grid_size_y = _num_local_freq;
    params.grid_size_z = _samples_per_data_set/_sk_step;
    params.num_dims = 3;

    params.private_segment_size = 0;
    params.group_segment_size = 16384;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}

