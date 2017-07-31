#include "hsaRfiVdif.hpp"
#include "hsaBase.h"
#include "vdif_functions.h"
#include <math.h>
hsaRfiVdif::hsaRfiVdif(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers,
                            const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name) {

    apply_config(0);

    Median_Array = (float *)hsa_host_malloc(median_len);

    for (int b = 0; b < median_len/sizeof(float); b++){
        Median_Array[b] = 0;
    }

    //Initialize Medians
    void * device_map = device.get_gpu_memory("in_means", median_len);
    device.sync_copy_host_to_gpu(device_map, (void *)Median_Array, median_len);
}

hsaRfiVdif::~hsaRfiVdif() {
    // TODO Free device memory allocations.
}

void hsaRfiVdif::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);

    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _sk_step = config.get_int(unique_name, "sk_step");
    rfi_sensitivity = config.get_int(unique_name, "rfi_sensitivity");
    input_frame_len = (_num_elements*_num_local_freq  + 64) * _samples_per_data_set;
    output_len =(_num_elements*_num_local_freq * _samples_per_data_set/_sk_step)*sizeof(float);
    median_len = _num_elements*_num_local_freq*sizeof(float);
}

hsa_signal_t hsaRfiVdif::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
    struct __attribute__ ((aligned(16))) args_t {
	void *input;
        void *output;
	void *in_means;
	float sqrtM;
	int sensitivity;
	int time_samples;
	int header_len;
    } args;
    memset(&args, 0, sizeof(args));
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.output = device.get_gpu_memory_array("rfi_output", gpu_frame_id, input_frame_len);
    args.in_means = device.get_gpu_memory("in_means", median_len);
    args.sqrtM = sqrt(_num_elements*_sk_step);
    args.sensitivity = rfi_sensitivity;
    args.time_samples = _samples_per_data_set;
    args.header_len = sizeof(VDIFHeader);
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    hsa_status_t hsa_status = hsa_signal_create(1, 0, NULL, &signals[gpu_frame_id]);
    assert(hsa_status == HSA_STATUS_SUCCESS);
   
    // Obtain the current queue write index.
    uint64_t index = hsa_queue_load_write_index_acquire(device.get_queue());
    hsa_kernel_dispatch_packet_t* dispatch_packet = (hsa_kernel_dispatch_packet_t*)device.get_queue()->base_address +
                                                            (index % device.get_queue()->size);
    dispatch_packet->setup  |= 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS; //Dimensions
    dispatch_packet->workgroup_size_x = (uint32_t)1;//local Group Sizes
    dispatch_packet->workgroup_size_y = (uint16_t)_num_elements;
    dispatch_packet->workgroup_size_z = (uint16_t)1;
    dispatch_packet->grid_size_x = (uint32_t)_num_local_freq;//Global work sizes
    dispatch_packet->grid_size_y = (uint16_t)_num_elements;
    dispatch_packet->grid_size_z = (uint16_t)(_samples_per_data_set/_sk_step);
    dispatch_packet->completion_signal = signals[gpu_frame_id];
    dispatch_packet->kernel_object = kernel_object;
    dispatch_packet->kernarg_address = (void*) kernel_args[gpu_frame_id];
    dispatch_packet->private_segment_size = 0;
    //dispatch_packet->group_segment_size = (uint32_t)(256*2 + 1)*sizeof(unsigned int); //Not sure if I need that
    dispatch_packet-> header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    hsa_queue_add_write_index_acquire(device.get_queue(), 1);
    hsa_signal_store_relaxed(device.get_queue()->doorbell_signal, index);

    return signals[gpu_frame_id];
}

