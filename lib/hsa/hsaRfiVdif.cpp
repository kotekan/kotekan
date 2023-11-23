/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: August 2017
File Purpose: Handles the kotekan GPU process for RFI removal in VDIF data.
Details:
    -Constructor: Applies config and sets up the Mean array
    -execute: Sets up kernel arguments, specifies HSA parameters, queues rfi kernel
Notes:
    This stage was designed to run on VDIF data.

**********************************************************************************/

#include "hsaRfiVdif.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaBase.h"              // for hsa_host_malloc, HSA_CHECK
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "vdif_functions.h"       // for VDIFHeader

#include <cmath>     // for sqrt
#include <cstdint>   // for int32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaRfiVdif);

hsaRfiVdif::hsaRfiVdif(Config& config, const std::string& unique_name,
                       bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "rfi_vdif" KERNEL_EXT, "rfi_vdif.hsaco") {
    command_type = gpuCommandType::KERNEL;

    // Grab values from config and calculates buffer size
    _num_elements = config.get<int32_t>(unique_name, "num_elements"); // Data parameters
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");

    _sk_step = config.get<int32_t>(unique_name, "sk_step"); // RFI parameters
    rfi_sensitivity = config.get<int32_t>(unique_name, "rfi_sensitivity");

    input_frame_len = (_num_elements * _num_local_freq + 64) * _samples_per_data_set; // Buffer
                                                                                      // sizes
    output_len =
        (_num_elements * _num_local_freq * _samples_per_data_set / _sk_step) * sizeof(float);
    mean_len = _num_elements * _num_local_freq * sizeof(float);

    // Allocates memory for Mean Array
    Mean_Array = (float*)hsa_host_malloc(mean_len, device.get_gpu_numa_node());

    for (uint32_t b = 0; b < mean_len / sizeof(float); b++) {
        Mean_Array[b] = 0; /// Initialize
    }

    // Initialize Mean memory on GPU
    void* device_map = device.get_gpu_memory("in_means", mean_len);
    device.sync_copy_host_to_gpu(device_map, (void*)Mean_Array, mean_len);
}

hsaRfiVdif::~hsaRfiVdif() {
    // TODO Free device memory allocations.
}

hsa_signal_t hsaRfiVdif::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t { // Kernel Arguments
        void* input;
        void* output;
        void* in_means;
        float sqrtM;
        int sensitivity;
        int time_samples;
        int header_len;
    } args;

    memset(&args, 0, sizeof(args)); // Intialize
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, _gpu_buffer_depth,
                                             input_frame_len); // Grab GPU memory
    args.output =
        device.get_gpu_memory_array("rfi_output", gpu_frame_id, _gpu_buffer_depth, input_frame_len);
    args.in_means = device.get_gpu_memory("in_means", mean_len);
    args.sqrtM = sqrt(_num_elements * _sk_step);
    args.sensitivity = rfi_sensitivity;
    args.time_samples = _samples_per_data_set;
    args.header_len = sizeof(VDIFHeader);
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    hsa_status_t hsa_status = hsa_signal_create(1, 0, nullptr, &signals[gpu_frame_id]);
    HSA_CHECK(hsa_status);

    // Obtain the current queue write index.
    uint64_t index = hsa_queue_load_write_index_acquire(device.get_queue());
    hsa_kernel_dispatch_packet_t* dispatch_packet =
        (hsa_kernel_dispatch_packet_t*)device.get_queue()->base_address
        + (index % device.get_queue()->size);
    dispatch_packet->setup |= 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS; // Dimensions
    dispatch_packet->workgroup_size_x = (uint32_t)1;                            // local Group Sizes
    dispatch_packet->workgroup_size_y = (uint16_t)_num_elements;
    dispatch_packet->workgroup_size_z = (uint16_t)1;
    dispatch_packet->grid_size_x = (uint32_t)_num_local_freq; // Global work sizes
    dispatch_packet->grid_size_y = (uint16_t)_num_elements;
    dispatch_packet->grid_size_z = (uint16_t)(_samples_per_data_set / _sk_step);
    dispatch_packet->completion_signal = signals[gpu_frame_id];
    dispatch_packet->kernel_object = kernel_object;
    dispatch_packet->kernarg_address = (void*)kernel_args[gpu_frame_id];
    dispatch_packet->private_segment_size = 0;
    dispatch_packet->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
                              | (1 << HSA_PACKET_HEADER_BARRIER)
                              | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE)
                              | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    // Queue Kernel
    hsa_queue_add_write_index_acquire(device.get_queue(), 1);
    hsa_signal_store_relaxed(device.get_queue()->doorbell_signal, index);

    return signals[gpu_frame_id];
}
