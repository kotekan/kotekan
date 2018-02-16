/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: August 2017
File Purpose: Handles the kotekan GPU process for RFI removal in CHIME data.
Details:
	-Constructor: Applies config and sets up the Mean array
	-apply_config: Gets values from config file and calculates buffer sizes
	-execute: Sets up kernel arguments, specifies HSA parameters, queues rfi kernel
Notes:
	This process was designed to run on CHIME data.

**********************************************************************************/

#include "hsaRfi.hpp"

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


    Mean_Array = (float *)hsa_host_malloc(mean_len); //Allocate memory

    for (uint32_t b = 0; b < mean_len/sizeof(float); b++){
        Mean_Array[b] = 0; //Initialize with 0's
    }

    //Initialize GPU memory and sopy over
    void * device_map = device.get_gpu_memory("in_means", mean_len);
    device.sync_copy_host_to_gpu(device_map, (void *)Mean_Array, mean_len);
}

hsaRfi::~hsaRfi() {
    // TODO Free device memory allocations.
}

hsa_signal_t hsaRfi::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    struct __attribute__ ((aligned(16))) args_t { //Kernel Arguments
	void *input; //Input Data
	void *count; //How many data points contain RFI
	void *in_means; //Input Mean values
	float sqrtM; //There is no sqrt function in HSA kernels
	int sensitivity; //How many deviations to place threshold
	int time_samples; //Number of time sample in data
	int zero; //Flag to determine whether or not to zero data
    } args;

    memset(&args, 0, sizeof(args));//Intialize arguments
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);//GPU buffer arguments
    args.count = device.get_gpu_memory("count", _num_local_freq*sizeof(unsigned int));
    args.in_means = device.get_gpu_memory("in_means", mean_len);
    args.sqrtM = sqrt(_num_elements*_sk_step); //CPU arguments
    args.sensitivity = rfi_sensitivity;
    args.time_samples = _samples_per_data_set;
    if(rfi_zero) args.zero = 1;
    else args.zero = 0;

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

