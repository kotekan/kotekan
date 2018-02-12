#include "hsaRfiTimeSum.hpp"
#include "hsaBase.h"
#include <math.h>

hsaRfiTimeSum::hsaRfiTimeSum(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers,
                            const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name) {

    command_type = CommandType::KERNEL;
    //Retrieve parameters from kotekan config
    apply_config(0); 

    InputMask = (uint8_t *)hsa_host_malloc(mask_len); //Allocate memory
    for(uint32_t i = 0; i < mask_len; i++){
        InputMask[i] = (uint8_t)0;
    }

    _num_bad_inputs = 0;
    
    //Initialize GPU memory and sopy over
    void * input_mask_map = device.get_gpu_memory("input_mask", mask_len);
    device.sync_copy_host_to_gpu(input_mask_map, (void *)InputMask, mask_len);
}

hsaRfiTimeSum::~hsaRfiTimeSum() {
}

void hsaRfiTimeSum::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);

    //Data Parameters
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    //RFI Config Parameters
    _sk_step = config.get_int(unique_name, "sk_step");

    //Compute Buffer lengths
    input_frame_len = sizeof(uint8_t)*_num_elements*_num_local_freq*_samples_per_data_set;
    output_frame_len = sizeof(float)*_num_local_freq*_num_elements*_samples_per_data_set/_sk_step;
    mask_len = sizeof(uint8_t)*_num_elements;
}

hsa_signal_t hsaRfiTimeSum::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    struct __attribute__ ((aligned(16))) args_t { 
	void *input;
	void *output; 
	void *InputMask; 
	uint32_t num_bad_inputs; 
	uint32_t sk_step; 
    } args;

    memset(&args, 0, sizeof(args));
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.output = device.get_gpu_memory("timesum", output_frame_len);
    args.InputMask = device.get_gpu_memory("input_mask", mask_len); 
    args.num_bad_inputs = _num_bad_inputs;
    args.sk_step = _sk_step;

//    INFO("%d %d %d %d %d",input_frame_len, output_frame_len, mask_len, _num_bad_inputs, _sk_step);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 1;
    params.workgroup_size_y = 256;
    params.workgroup_size_z = 1;
    params.grid_size_x = _num_local_freq*_num_elements;
    params.grid_size_y = 256;
    params.grid_size_z = _samples_per_data_set/_sk_step;
    params.num_dims = 3;

    params.private_segment_size = 0;
    params.group_segment_size = 16384;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}

