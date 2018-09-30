#include "hsaRfiZeroData.hpp"
#include "hsaBase.h"
#include <math.h>
#include <unistd.h>
#include <mutex>

REGISTER_HSA_COMMAND(hsaRfiZeroData);

hsaRfiZeroData::hsaRfiZeroData(Config& config,const string &unique_name,
                         bufferContainer& host_buffers,
                         hsaDeviceInterface& device):
    hsaCommand("rfi_chime_zero", "rfi_chime_zero.hsaco", config, unique_name, host_buffers, device){
    command_type = CommandType::KERNEL;
    //Retrieve parameters from kotekan config
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    //RFI Config Parameters
    _sk_step = config.get_int_default(unique_name, "sk_step", 256);
    //Compute Buffer lengths
    input_frame_len = sizeof(uint8_t)*_num_elements*_num_local_freq*_samples_per_data_set;
    mask_len = sizeof(uint8_t)*_num_local_freq*_samples_per_data_set/_sk_step;
}

hsaRfiZeroData::~hsaRfiZeroData() {
}

hsa_signal_t hsaRfiZeroData::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
    //Structure for gpu arguments
    struct __attribute__ ((aligned(16))) args_t {
        void *input;
        void *mask;
        uint32_t sk_step;
    } args;
    //Initialize arguments
    memset(&args, 0, sizeof(args));
    //Set argumnets to correct values
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.mask = device.get_gpu_memory_array("rfi_mask_output", gpu_frame_id, mask_len);
    args.sk_step = _sk_step;
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));
    // Apply correct kernel parameters
    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = _num_elements/4;
    params.grid_size_y = _samples_per_data_set/_sk_step;
    params.grid_size_z = 1;
    params.num_dims = 2;
    // Should this be zero?
    params.private_segment_size = 0;
    params.group_segment_size = 0;

    //Execute kernel
    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);
    //return signal
    return signals[gpu_frame_id];
}
