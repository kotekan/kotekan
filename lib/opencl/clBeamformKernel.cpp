#include "clBeamformKernel.hpp"

#include "Telescope.hpp"
#include "chimeMetadata.h"

#include <string>

using std::string;

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clBeamformKernel);

clBeamformKernel::clBeamformKernel(Config& config, const std::string& unique_name,
                                   bufferContainer& host_buffers, clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "gpu_beamforming",
              "beamform_tree_scale.cl") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    network_buf = host_buffers.get_buffer("network_buf");

    _element_mask = config.get<std::vector<int>>(unique_name, "element_mask");
    _product_remap = config.get<std::vector<int>>(unique_name, "product_remap");
    int remap_size = _product_remap.size();

    if (remap_size != _num_elements) {
        ERROR("The remap array must have the same size as the number of elements. array size {:d}, "
              "num_elements {:d}",
              remap_size, _num_elements);
    }
    _inverse_product_remap.reserve(remap_size);
    // Given a channel ID, where is it in FPGA order.
    for (int i = 0; i < remap_size; ++i) {
        _inverse_product_remap[_product_remap[i]] = i;
    }
    _scale_factor = config.get<int>(unique_name, "scale_factor");

    num_local_freq = Telescope::instance().num_freq_per_stream();
}

clBeamformKernel::~clBeamformKernel() {
    clReleaseMemObject(device_mask);
}

void clBeamformKernel::build() {
    clCommand::build();

    cl_int err;

    cl_device_id dev_id = device.get_id();

    std::string cl_options = "";
    cl_options += " -D NUM_ELEMENTS=" + std::to_string(_num_elements);
    cl_options += " -D NUM_TIMESAMPLES=" + std::to_string(_samples_per_data_set);

    CHECK_CL_ERROR(clBuildProgram(program, 1, &dev_id, cl_options.c_str(), nullptr, nullptr));

    kernel = clCreateKernel(program, kernel_command.c_str(), &err);
    CHECK_CL_ERROR(err);

    unsigned char mask[_num_elements];

    for (int i = 0; i < _num_elements; ++i) {
        mask[i] = 1;
    }
    for (uint32_t i = 0; i < _element_mask.size(); ++i) {
        int mask_position = _element_mask[i];
        mask_position = _inverse_product_remap[mask_position];
        mask[mask_position] = 0;
    }

    device_mask = clCreateBuffer(device.get_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 _num_elements * sizeof(unsigned char), mask, &err);

    CHECK_CL_ERROR(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&device_mask));

    float scale_factor = _scale_factor;
    INFO("setup_clBeamformKernel_worksize, setting scale factor to {:f}", scale_factor);
    CHECK_CL_ERROR(clSetKernelArg(kernel, 5, sizeof(float), &scale_factor));

    // Beamforming kernel global and local work space sizes.
    gws[0] = _num_elements / 4;
    gws[1] = num_local_freq;
    gws[2] = _samples_per_data_set / 32;

    lws[0] = 64;
    lws[1] = 1;
    lws[2] = 1;
}

cl_event clBeamformKernel::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    // TODO Make this a config file option
    // 390625 == 1 second.
    const uint64_t phase_update_period = 390625;

    int64_t current_seq = get_fpga_seq_num(network_buf, gpu_frame_id);
    int64_t bankID = (current_seq / phase_update_period) % 2;

    int32_t streamID = get_stream_id(network_buf, gpu_frame_id);

    uint32_t input_frame_len = _num_elements * num_local_freq * _samples_per_data_set;

    cl_mem input_memory = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    cl_mem phase_memory =
        device.get_gpu_memory_array("phases", bankID, _num_elements * sizeof(float));

    uint32_t output_len = _samples_per_data_set * _num_data_sets * num_local_freq * 2;
    cl_mem output_memory_frame =
        device.get_gpu_memory_array("beamform_output_buf", gpu_frame_id, output_len);

    setKernelArg(0, input_memory);
    setKernelArg(1, output_memory_frame);
    setKernelArg(2, get_freq_map(streamID));
    setKernelArg(3, phase_memory);

    CHECK_CL_ERROR(clEnqueueNDRangeKernel(device.getQueue(1), kernel, 3, nullptr, gws, lws, 1,
                                          &pre_event, &post_events[gpu_frame_id]));

    return post_events[gpu_frame_id];
}


cl_mem clBeamformKernel::get_freq_map(int32_t encoded_stream_id) {
    // CONVERT TO USE STANDARD MEM ALLOC!
    std::map<int32_t, cl_mem>::iterator it = device_freq_map.find(encoded_stream_id);

    if (it == device_freq_map.end()) {
        // Create the freq map for the first time.
        auto& tel = Telescope::instance();
        cl_int err;
        float freq[num_local_freq];

        for (int j = 0; j < num_local_freq; ++j) {
            freq[j] = tel.to_freq(encoded_stream_id, j) / 1000.0;
        }

        device_freq_map[encoded_stream_id] =
            clCreateBuffer(device.get_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           num_local_freq * sizeof(float), freq, &err);
        CHECK_CL_ERROR(err);
    }
    return device_freq_map.at(encoded_stream_id);
}
