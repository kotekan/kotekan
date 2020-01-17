#include "hsaRfiTimeSum.hpp"

#include "configUpdater.hpp"
#include "hsaBase.h"
#include "visUtil.hpp"

#include <math.h>
#include <mutex>
#include <unistd.h>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaRfiTimeSum);

hsaRfiTimeSum::hsaRfiTimeSum(Config& config, const string& unique_name,
                             bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "rfi_chime_timesum" KERNEL_EXT,
               "rfi_chime_timesum_private.hsaco") {
    command_type = gpuCommandType::KERNEL;
    // Retrieve parameters from kotekan config
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // RFI Config Parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    // Compute Buffer lengths
    input_frame_len = sizeof(uint8_t) * _num_elements * _num_local_freq * _samples_per_data_set;
    output_frame_len =
        sizeof(float) * _num_local_freq * _num_elements * _samples_per_data_set / _sk_step;
    output_var_frame_len = sizeof(float) * _num_local_freq * _samples_per_data_set / _sk_step;
    lost_samples_frame_len = sizeof(uint8_t) * _samples_per_data_set;
    lost_samples_correction_len = sizeof(uint32_t) * _samples_per_data_set / _sk_step;

    auto input_reorder = parse_reorder_default(config, unique_name);
    input_remap = std::get<0>(input_reorder);

    kotekan::configUpdater::instance().subscribe(
        config.get<std::string>(unique_name, "updatable_config/rfi_var_element_index"),
        std::bind(&hsaRfiTimeSum::update_element_index, this, std::placeholders::_1));
}

hsaRfiTimeSum::~hsaRfiTimeSum() {}

bool hsaRfiTimeSum::update_element_index(nlohmann::json& json) {
    uint32_t element_index_cylinder_order = 0;
    DEBUG("Current JSON: {:s}", json.dump());
    try {
        element_index_cylinder_order = json["element_index"].get<uint32_t>();
    } catch (std::exception& e) {
        WARN("Failed to set element index {:s}, json {:s}", e.what(), json.dump());
        return false;
    }
    _element_index = input_remap[element_index_cylinder_order];
    INFO("Updating element index for variance extract;"
         " cylinder_order: {:d}, correlator_order: {:d}",
         element_index_cylinder_order, _element_index);
    return true;
}

hsa_signal_t hsaRfiTimeSum::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    // Structure for gpu arguments
    struct __attribute__((aligned(16))) args_t {
        void* input;
        void* output;
        // void* output_var;
        //        void *LostSamples;
        //        void *LostSamplesCorrection;
        uint32_t sk_step;
        uint32_t num_elements;
        // uint32_t element_index;
    } args;
    // Initialize arguments
    memset(&args, 0, sizeof(args));
    // Set argumnets to correct values
    args.input = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.output = device.get_gpu_memory("timesum", output_frame_len);
    // args.output_var =
    //    device.get_gpu_memory_array("rfi_output_var", gpu_frame_id, output_var_frame_len);
    //    args.LostSamples = device.get_gpu_memory_array("lost_samples", gpu_frame_id,
    //    lost_samples_frame_len); args.LostSamplesCorrection =
    //    device.get_gpu_memory("lost_sample_correction", lost_samples_correction_len);
    args.sk_step = _sk_step;
    args.num_elements = _num_elements;
    // args.element_index = _element_index;
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));
    // Apply correct kernel parameters
    kernelParams params;
    params.workgroup_size_x = 64;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = _num_elements / 4;
    params.grid_size_y = _samples_per_data_set / _sk_step;
    params.grid_size_z = 1;
    params.num_dims = 2;
    // Should this be zero?
    params.private_segment_size = 0;
    params.group_segment_size = 0;

    // Execute kernel
    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);
    // return signal
    return signals[gpu_frame_id];
}
