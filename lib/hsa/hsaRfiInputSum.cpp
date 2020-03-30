#include "hsaRfiInputSum.hpp"

#include "Config.hpp"             // for Config
#include "buffer.h"               // for Buffer, mark_frame_empty, register_consumer, wait_for_...
#include "bufferContainer.hpp"    // for bufferContainer
#include "chimeMetadata.h"        // for get_rfi_num_bad_inputs
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "kotekanLogging.hpp"     // for DEBUG
#include "restServer.hpp"         // for HTTP_RESPONSE, connectionInstance, restServer

#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_HSA_COMMAND(hsaRfiInputSum);

hsaRfiInputSum::hsaRfiInputSum(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, hsaDeviceInterface& device) :
    // Note, the rfi_chime_inputsum_private.hsaco kernel may be used in the future.
    hsaCommand(config, unique_name, host_buffers, device, "rfi_chime_inputsum" KERNEL_EXT,
               "rfi_chime_inputsum.hsaco") {
    command_type = gpuCommandType::KERNEL;
    // Retrieve parameters from kotekan config
    // Data Parameters
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // RFI Config Parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _rfi_sigma_cut = config.get_default<uint32_t>(unique_name, "rfi_sigma_cut", 5);
    // Compute Buffer lengths
    input_frame_len =
        sizeof(float) * _num_elements * _num_local_freq * _samples_per_data_set / _sk_step;
    output_frame_len = sizeof(float) * _num_local_freq * _samples_per_data_set / _sk_step;
    input_mask_len = sizeof(uint8_t) * _num_elements;
    output_mask_len = sizeof(uint8_t) * _num_local_freq * _samples_per_data_set / _sk_step;
    correction_frame_len = sizeof(uint32_t) * _samples_per_data_set / _sk_step;

    // Get buffers (for metadata)
    _network_buf = host_buffers.get_buffer("network_buf");
    register_consumer(_network_buf, unique_name.c_str());

    _network_buf_precondition_id = 0;
    _network_buf_execute_id = 0;
    _network_buf_finalize_id = 0;
}

hsaRfiInputSum::~hsaRfiInputSum() {}

int hsaRfiInputSum::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    uint8_t* frame =
        wait_for_full_frame(_network_buf, unique_name.c_str(), _network_buf_precondition_id);
    if (frame == nullptr)
        return -1;

    _network_buf_precondition_id = (_network_buf_precondition_id + 1) % _network_buf->num_frames;
    return 0;
}

hsa_signal_t hsaRfiInputSum::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    // Get the number of bad inputs from the metadata
    uint32_t num_bad_inputs = get_rfi_num_bad_inputs(_network_buf, _network_buf_execute_id);
    DEBUG("Number of bad inputs at execute in hsaRfiInputSum is: {:d}", num_bad_inputs);

    // Struct for hsa arguments
    struct __attribute__((aligned(16))) args_t {
        void* input;
        void* output;
        void* input_mask;
        void* output_mask;
        void* lost_sample_correction;
        uint32_t num_elements;
        uint32_t num_bad_inputs;
        uint32_t sk_step;
        uint32_t rfi_sigma_cut;
    } args;
    // Initialize arguments
    memset(&args, 0, sizeof(args));
    // Set arguments
    args.input = device.get_gpu_memory("timesum", input_frame_len);
    args.output = device.get_gpu_memory_array("rfi_output", gpu_frame_id, output_frame_len);
    args.input_mask = device.get_gpu_memory_array("input_mask", gpu_frame_id, input_mask_len);
    args.output_mask =
        device.get_gpu_memory_array("rfi_mask_output", gpu_frame_id, output_mask_len);
    args.lost_sample_correction = device.get_gpu_memory_array("compressed_lost_samples",
                                                              gpu_frame_id, correction_frame_len);
    args.num_elements = _num_elements;
    args.num_bad_inputs = num_bad_inputs;
    args.sk_step = _sk_step;
    args.rfi_sigma_cut = _rfi_sigma_cut;
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));
    // Set kernel execution parameters
    kernelParams params;
    params.workgroup_size_x = 256;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = 256;
    params.grid_size_y = _num_local_freq;
    params.grid_size_z = _samples_per_data_set / _sk_step;
    params.num_dims = 3;
    params.private_segment_size = 0;
    params.group_segment_size = 16384;
    // Parameters for rfi_chime_inputsum_private.hsaco, for easy switching if needed in future
    /*    params.workgroup_size_x = _num_local_freq;
        params.workgroup_size_y = 1;
        params.workgroup_size_z = 1;
        params.grid_size_x = _num_local_freq;
        params.grid_size_y = (_samples_per_data_set/_sk_step)/24;
        params.grid_size_z = 24;
        params.num_dims = 3;
        params.private_segment_size = 0;
        params.group_segment_size = 16384;*/
    // Execute kernel
    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    _network_buf_execute_id = (_network_buf_execute_id + 1) % _network_buf->num_frames;

    // Return signal
    return signals[gpu_frame_id];
}

void hsaRfiInputSum::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
    mark_frame_empty(_network_buf, unique_name.c_str(), _network_buf_finalize_id);
    _network_buf_finalize_id = (_network_buf_finalize_id + 1) % _network_buf->num_frames;
}
