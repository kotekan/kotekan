#include "clPresumKernel.hpp"

#include "math.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clPresumKernel);

clPresumKernel::clPresumKernel(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "offsetAccumulateElements",
              "offset_accumulator.cl") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    command_type = gpuCommandType::KERNEL;
}

clPresumKernel::~clPresumKernel() {}

void clPresumKernel::build() {
    clCommand::build();
    cl_int err;

    cl_device_id dev_id = device.get_id();

    std::string cl_options = "";
    cl_options += " -D ACTUAL_NUM_ELEMENTS=" + std::to_string(_num_elements);
    cl_options += " -D ACTUAL_NUM_FREQUENCIES=" + std::to_string(_num_local_freq);
    CHECK_CL_ERROR(clBuildProgram(program, 1, &dev_id, cl_options.c_str(), nullptr, nullptr));

    kernel = clCreateKernel(program, kernel_command.c_str(), &err);
    CHECK_CL_ERROR(err);

    // Accumulation kernel global and local work space sizes.
    gws[0] = 64 * _num_data_sets;
    gws[1] = (int)ceil(_num_elements * _num_local_freq / 256.0);
    gws[2] = _samples_per_data_set / 1024;

    lws[0] = 64;
    lws[1] = 1;
    lws[2] = 1;
}

cl_event clPresumKernel::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t presum_len = _num_elements * _num_local_freq * 2 * sizeof(int32_t);
    uint32_t input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;

    cl_mem input_memory =
        device.get_gpu_memory_array("input", gpu_frame_id, _gpu_buffer_depth, input_frame_len);
    cl_mem presum_memory =
        device.get_gpu_memory_array("presum", gpu_frame_id, _gpu_buffer_depth, presum_len);

    setKernelArg(0, input_memory);
    setKernelArg(1, presum_memory);

    CHECK_CL_ERROR(clEnqueueNDRangeKernel(device.getQueue(1), kernel, 3, nullptr, gws, lws, 1,
                                          &pre_event, &post_events[gpu_frame_id]));

    return post_events[gpu_frame_id];
}
