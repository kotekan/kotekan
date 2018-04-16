#include "hsaRfiTimeSum.hpp"
#include "hsaBase.h"
#include <math.h>
#include <unistd.h>

REGISTER_HSA_COMMAND(hsaRfiTimeSum);

hsaRfiTimeSum::hsaRfiTimeSum(Config& config,const string &unique_name,
                         bufferContainer& host_buffers, 
                         hsaDeviceInterface& device):
    hsaCommand("rfi_chime_timesum", "rfi_chime_timesum.hsaco", config, unique_name, host_buffers, device){
    command_type = CommandType::KERNEL;

    //Retrieve parameters from kotekan confint(unique_name, "num_elements");
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    //RFI Config Parameters
    _sk_step = config.get_int(unique_name, "sk_step");
    
    //Compute Buffer lengths
    input_frame_len = sizeof(uint8_t)*_num_elements*_num_local_freq*_samples_per_data_set;
    output_frame_len = sizeof(float)*_num_local_freq*_num_elements*_samples_per_data_set/_sk_step;
    mask_len = sizeof(uint8_t)*_num_elements;
    
    //Local Parameters
    _num_bad_inputs = 0;
    first_pass=true;
}

hsaRfiTimeSum::~hsaRfiTimeSum() {
}

hsa_signal_t hsaRfiTimeSum::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    if (first_pass) {
        first_pass = false;
        InputMask = (uint8_t *)hsa_host_malloc(mask_len); //Allocate memory
        for(uint32_t i = 0; i < mask_len; i++){
            InputMask[i] = (uint8_t)0;
        }
        void * input_mask_map = device.get_gpu_memory("input_mask", mask_len);
        device.sync_copy_host_to_gpu(input_mask_map, (void *)InputMask, mask_len);
    }

    struct __attribute__ ((aligned(16))) args_t { 
	void *input;
	void *output; 
	void *InputMask;
	uint32_t sk_step;
        uint32_t num_elements; 
    } args;

    memset(&args, 0, sizeof(args));
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.output = device.get_gpu_memory("timesum", output_frame_len);
    args.InputMask = device.get_gpu_memory("input_mask", mask_len); 
    args.sk_step = _sk_step;
    args.num_elements = _num_elements;

    //INFO("%d %d %d %d %d",input_frame_len, output_frame_len, mask_len, _num_bad_inputs, _sk_step);
    //INFO("GX %d GY %d GZ %d", _num_elements*_num_local_freq/4, 256, _samples_per_data_set/_sk_step);
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 1;
    params.workgroup_size_y = 256;
    params.workgroup_size_z = 1;
    params.grid_size_x = _num_elements*_num_local_freq/4;
    params.grid_size_y = 256;
    params.grid_size_z = _samples_per_data_set/_sk_step;
    params.num_dims = 3;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}

