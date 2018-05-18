#include "offset_kernel.h"
#include "math.h"

offset_kernel::offset_kernel(const char * param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name):
    clCommand(param_gpuKernel, param_name, param_config, unique_name)
{
}

offset_kernel::~offset_kernel()
{
}

void offset_kernel::apply_config(const uint64_t& fpga_seq) {
    clCommand::apply_config(fpga_seq);
}


void offset_kernel::build(device_interface &param_Device)
{
    apply_config(0);
    clCommand::build(param_Device);
    cl_int err;
    cl_device_id valDeviceID;

    string cl_options = get_cl_options();

    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());

    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options.c_str(), NULL, NULL ) );

    kernel = clCreateKernel( program, "offsetAccumulateElements", &err );
    CHECK_CL_ERROR(err);

    // Accumulation kernel global and local work space sizes.
    gws[0] = 64*_num_data_sets;
    gws[1] = (int)ceil(_num_adjusted_elements * _num_adjusted_local_freq/256.0);
    gws[2] = _samples_per_data_set/1024;

    lws[0] = 64;
    lws[1] = 1;
    lws[2] = 1;
}

cl_event offset_kernel::execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent)
{
    clCommand::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);

    setKernelArg(0, param_Device.getInputBuffer(param_bufferID));
    setKernelArg(1, param_Device.getAccumulateBuffer(param_bufferID));

//    DEBUG("gws: %i, %i, %i. lws: %i, %i, %i", gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
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

