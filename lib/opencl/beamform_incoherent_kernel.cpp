
#include "beamform_incoherent_kernel.h"
#include "fpga_header_functions.h"

beamform_incoherent_kernel::beamform_incoherent_kernel(const char * param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name):
    gpu_command(param_gpuKernel, param_name, param_config, unique_name)
{

}

beamform_incoherent_kernel::~beamform_incoherent_kernel()
{
    clReleaseMemObject(device_mask);
}

void beamform_incoherent_kernel::build(class device_interface& param_Device)
{
    // Apply config.
    gpu_command::apply_config();

    _element_mask = config.get<std::vector<int32_t>>(
                "/beamforming", "element_mask");
    _inverse_product_remap = config.get<std::vector<int32_t>>(
                "/processing", "inverse_product_remap");
    _scale_factor = config.get<float>("/beamforming", "scale_factor");

    gpu_command::build(param_Device);

    cl_int err;

    cl_device_id valDeviceID;

    string cl_options = get_cl_options();

    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());

    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options.c_str(), NULL, NULL ) );

    kernel = clCreateKernel( program, "gpu_beamforming_incoherent", &err );
    CHECK_CL_ERROR(err);

    unsigned char mask[_num_adjusted_elements];

    for (int i = 0; i < _num_adjusted_elements; ++i) {
        mask[i] = 1;
    }
    for (uint32_t i = 0; i < _element_mask.size(); ++i) {
        int mask_position = _element_mask[i];
        mask_position = _inverse_product_remap[mask_position];
        mask[mask_position] = 0;
    }

    device_mask = clCreateBuffer(param_Device.getContext(),
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        _num_adjusted_elements * sizeof(unsigned char),
                                        mask,
                                        &err);

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                    4,
                                    sizeof(cl_mem),
                                    (void*) &device_mask) );

    INFO("setup_beamform_incoherent_kernel_worksize, setting scale factor to %f",
         _scale_factor);
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   5,
                                   sizeof(float),
                                   &_scale_factor) );

    // Beamforming kernel global and local work space sizes.
    gws[0] = _num_elements / 4;
    gws[1] = _num_local_freq;
    gws[2] = _samples_per_data_set / 32;

    lws[0] = 64;
    lws[1] = 1;
    lws[2] = 1;
}

cl_event beamform_incoherent_kernel::execute(int param_bufferID, const uint64_t& fpga_seq, class device_interface& param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);

    // TODO Make this a config file option
    // 390625 == 1 second.
    const uint64_t phase_update_period = 390625;

    int64_t current_seq = get_fpga_seq_num(param_Device.getInBuf(), param_bufferID);
    int64_t bankID = (current_seq / phase_update_period) % 2;

    int32_t streamID = get_streamID(param_Device.getInBuf(), param_bufferID);

    // TODO There is a bug here with the beamforming output buffer.
    setKernelArg(0, param_Device.getInputBuffer(param_bufferID));
    setKernelArg(1, param_Device.get_device_beamform_output_buffer(param_bufferID));
    setKernelArg(2, param_Device.get_device_freq_map(streamID));
    setKernelArg(3, param_Device.get_device_phases(bankID));

    CHECK_CL_ERROR( clEnqueueNDRangeKernel(param_Device.getQueue(1),
                                    kernel,
                                    3,
                                    NULL,
                                    gws,
                                    lws,
                                    1,
                                    &param_PrecedeEvent,
                                    &postEvent[param_bufferID]));

    return postEvent[param_bufferID];

}
