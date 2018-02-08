#include "hsaRfi.hpp"
#include "hsaBase.h"
#include <math.h>

REGISTER_HSA_COMMAND(hsaRfi);

hsaRfi::hsaRfi(Config& config, const string &unique_name,
                bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand("rfi_chime","rfi_chime.hsaco", config, unique_name, host_buffers, device) {
    command_type = CommandType::KERNEL;

    //Retrieves relevant information regarding kotekan parameters
    _num_elements = config.get_int(unique_name, "num_elements"); //Data parameters
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    _sk_step = config.get_int(unique_name, "sk_step"); //RFI paramters
    rfi_sensitivity = config.get_int(unique_name, "rfi_sensitivity");
    rfi_zero = config.get_bool(unique_name, "rfi_zero");

    input_frame_len = _num_elements*_num_local_freq*_samples_per_data_set; //Buffer sizes
    mean_len = _num_elements*_num_local_freq*sizeof(float);
}

hsaRfi::~hsaRfi() {
}

void hsaRfi::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);

    //Data Parameters
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    //RFI Config Parameters
    _sk_step = config.get_int(unique_name, "sk_step");

    //Compute Buffer lengths
    input_frame_len = sizeof(uint8_t)*_num_elements*_num_local_freq*_samples_per_data_set;
    output_frame_len = sizeof(float)*_num_local_freq*_samples_per_data_set/_sk_step;
    swap_len = output_frame_len*_num_elements/256;
    mask_len = sizeof(uint8_t)*_num_elements;
}

hsa_signal_t hsaRfi::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    struct __attribute__ ((aligned(16))) args_t { 
	void *input; //Input Data
	void *output; 
	void *swap; 
	void *InputMask; 
	uint32_t num_bad_inputs; 
	uint32_t sk_step; 
    } args;

    memset(&args, 0, sizeof(args));
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.output = device.get_gpu_memory_array("rfi_output",gpu_frame_id, output_frame_len);
    args.swap = device.get_gpu_memory("swap", swap_len);
    args.InputMask = device.get_gpu_memory("input_mask", mask_len); 
    args.num_bad_inputs = _num_bad_inputs;
    args.sk_step = _sk_step;


//    INFO("%d %d %d %d %d %d",input_frame_len, output_frame_len, swap_len, mask_len, _num_bad_inputs, _sk_step);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    hsa_status_t hsa_status = hsa_signal_create(1, 0, NULL, &signals[gpu_frame_id]);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Obtain the current queue write index.
    uint64_t index = hsa_queue_load_write_index_acquire(device.get_queue());
    hsa_kernel_dispatch_packet_t* dispatch_packet = (hsa_kernel_dispatch_packet_t*)device.get_queue()->base_address +
                                                            (index % device.get_queue()->size);
    dispatch_packet->setup  |= 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS; //Dimensions
    dispatch_packet->workgroup_size_x = (uint32_t)256;//local Group Sizes
    dispatch_packet->workgroup_size_y = (uint16_t)1;
    dispatch_packet->workgroup_size_z = (uint16_t)1;
    dispatch_packet->grid_size_x = (uint32_t)_num_elements;//Global work sizes
    dispatch_packet->grid_size_y = (uint16_t)_num_local_freq;
    dispatch_packet->grid_size_z = (uint16_t)(_samples_per_data_set/_sk_step);
    dispatch_packet->completion_signal = signals[gpu_frame_id];
    dispatch_packet->kernel_object = kernel_object;
    dispatch_packet->kernarg_address = (void*) kernel_args[gpu_frame_id];
    dispatch_packet->private_segment_size = 0;
    dispatch_packet-> header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    hsa_queue_add_write_index_acquire(device.get_queue(), 1);
    hsa_signal_store_relaxed(device.get_queue()->doorbell_signal, index);

    return signals[gpu_frame_id];
}

