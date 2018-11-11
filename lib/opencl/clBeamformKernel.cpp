#include "clBeamformKernel.hpp"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include <string>

using std::string;

REGISTER_CL_COMMAND(clBeamformKernel);

clBeamformKernel::clBeamformKernel(Config& config, const string &unique_name,
                            bufferContainer& host_buffers, clDeviceInterface& device) :
    clCommand("gpu_beamforming","beamform_tree_scale.cl", config, unique_name, host_buffers, device)
{
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    network_buf = host_buffers.get_buffer("network_buf");

}

clBeamformKernel::~clBeamformKernel()
{
    clReleaseMemObject(device_mask);
}

void clBeamformKernel::apply_config(const uint64_t& fpga_seq) {
    clCommand::apply_config(fpga_seq);

    _element_mask = config.get<std::vector<int>>(unique_name, "element_mask");
    _product_remap = config.get<std::vector<int>>(unique_name, "product_remap");
    int remap_size = _product_remap.size();

    if (remap_size != _num_elements) {
    ERROR("The remap array must have the same size as the number of elements. array size %d, num_elements %d",
        remap_size, _num_elements);
    }
    _inverse_product_remap.reserve(remap_size);
    // Given a channel ID, where is it in FPGA order.
    for(int i = 0; i < remap_size; ++i) {
        _inverse_product_remap[_product_remap[i]] = i;
    }
    _scale_factor = config.get<int>(unique_name, "scale_factor");
}

void clBeamformKernel::build()
{
    clCommand::build();

    cl_int err;

    cl_device_id dev_id = device.get_id();

    string cl_options = "";
    cl_options += " -D NUM_ELEMENTS=" + std::to_string(_num_elements);
    cl_options += " -D NUM_TIMESAMPLES=" + std::to_string(_samples_per_data_set);

    CHECK_CL_ERROR ( clBuildProgram( program, 1, &dev_id, cl_options.c_str(), NULL, NULL ) );

    kernel = clCreateKernel( program, kernel_command.c_str(), &err );
    CHECK_CL_ERROR(err);

////##OCCURS IN SETUP_OPEN_CL

    unsigned char mask[_num_elements];

    for (int i = 0; i < _num_elements; ++i) {
        mask[i] = 1;
    }
    for (uint32_t i = 0; i < _element_mask.size(); ++i) {
        int mask_position = _element_mask[i];
        mask_position = _inverse_product_remap[mask_position];
        mask[mask_position] = 0;
    }

    device_mask = clCreateBuffer(device.get_context(),
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        _num_elements * sizeof(unsigned char),
                                        mask,
                                        &err);

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                    4,
                                    sizeof(cl_mem),
                                    (void*) &device_mask) );

    float scale_factor = _scale_factor;
    INFO("setup_clBeamformKernel_worksize, setting scale factor to %f",
         scale_factor);
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   5,
                                   sizeof(float),
                                   &scale_factor) );

    // Beamforming kernel global and local work space sizes.
    gws[0] = _num_elements / 4;
    gws[1] = _num_local_freq;
    gws[2] = _samples_per_data_set / 32;

    lws[0] = 64;
    lws[1] = 1;
    lws[2] = 1;

}

cl_event clBeamformKernel::execute(int gpu_frame_id, const uint64_t& fpga_seq, cl_event pre_event)
{
    clCommand::execute(gpu_frame_id, 0, pre_event);

    // TODO Make this a config file option
    // 390625 == 1 second.
    const uint64_t phase_update_period = 390625;

    int64_t current_seq = get_fpga_seq_num(network_buf, gpu_frame_id);
    int64_t bankID = (current_seq / phase_update_period) % 2;

    int32_t streamID = get_stream_id(network_buf, gpu_frame_id);

    uint32_t input_frame_len =  _num_elements * _num_local_freq * _samples_per_data_set;

    cl_mem input_memory = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);

    cl_mem phase_memory = device.get_gpu_memory_array("phases", bankID, _num_elements * sizeof(float));

    uint32_t output_len = _samples_per_data_set * _num_data_sets * _num_local_freq * 2;
    cl_mem output_memory_frame = device.get_gpu_memory_array("beamform_output_buf",gpu_frame_id, output_len);

    setKernelArg(0, input_memory);
    setKernelArg(1, output_memory_frame);
    setKernelArg(2, device.get_device_freq_map(streamID));
    setKernelArg(3, phase_memory);

    CHECK_CL_ERROR( clEnqueueNDRangeKernel(device.getQueue(1),
                                    kernel,
                                    3,
                                    NULL,
                                    gws,
                                    lws,
                                    1,
                                    &pre_event,
                                    &post_event[gpu_frame_id]));

    return post_event[gpu_frame_id];
}