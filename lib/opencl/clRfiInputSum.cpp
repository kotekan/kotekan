#include "clRfiInputSum.hpp"
#include "math.h"
#include <string.h>
#include <mutex>


clRfiInputSum::clRfiInputSum(const char * param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name):
    gpu_command(param_gpuKernel, param_name, param_config, unique_name)
{
}

clRfiInputSum::~clRfiInputSum()
{
}


void clRfiInputSum::apply_config(const uint64_t& fpga_seq) {
    gpu_command::apply_config(fpga_seq);

    //RFI Config Parameters
    _sk_step = config.get_int(unique_name, "sk_step");

    //Compute Buffer lengths
    input_frame_len = sizeof(float)*_num_elements*_num_local_freq*_samples_per_data_set/_sk_step;
    output_frame_len = sizeof(float)*_num_local_freq*_samples_per_data_set/_sk_step;
//    _num_bad_inputs = 0;    
    _num_bad_inputs = config.get_int_array(unique_name, "bad_inputs").size();
    INFO("NUMBER OF BAD INPUTS %d",_num_bad_inputs);
    _M = (_num_elements - _num_bad_inputs)*_sk_step;    
}


void clRfiInputSum::build(device_interface &param_Device)
{
    apply_config(0);
    
    gpu_command::build(param_Device);
    cl_int err;
    cl_device_id valDeviceID;
    string cl_options = get_cl_options();

    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());
    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options.c_str(), NULL, NULL ) );
    kernel = clCreateKernel( program, "rfi_chime_inputsum", &err );
    CHECK_CL_ERROR(err);

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)2,
                                   sizeof(int32_t),
                                   &_num_elements) );

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(int32_t),
                                   &_M) );

    // Accumulation kernel global and local work space sizes.
    //gws[0] = 256;
    //gws[1] = _num_local_freq;
    //gws[2] = _samples_per_data_set/_sk_step;

    gws[0] = _num_local_freq;
    gws[1] = (_samples_per_data_set/_sk_step)/4;
    gws[2] = 4;

    lws[0] = _num_local_freq;
    lws[1] = 1;
    lws[2] = 1;
}
cl_event clRfiInputSum::execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);

    setKernelArg(0, param_Device.getRfiTimeSumBuffer(param_bufferID));
    setKernelArg(1, param_Device.getRfiOutputBuffer(param_bufferID));
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

