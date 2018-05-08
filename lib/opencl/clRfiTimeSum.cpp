#include "clRfiTimeSum.hpp"
#include "math.h"
#include <string.h>
#include <mutex>


clRfiTimeSum::clRfiTimeSum(const char * param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name):
    gpu_command(param_gpuKernel, param_name, param_config, unique_name)
{
}

clRfiTimeSum::~clRfiTimeSum()
{
}

void clRfiTimeSum::rest_callback(connectionInstance& conn, json& json_request) {
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    INFO("RFI Callbak Received... Changing Parameters")
    
    _sk_step =  json_request["sk_step"];
    gws[2] = (_samples_per_data_set/_sk_step);

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(int32_t),
                                   &_sk_step) );

    conn.send_empty_reply(STATUS_OK);
}

void clRfiTimeSum::apply_config(const uint64_t& fpga_seq) {
    gpu_command::apply_config(fpga_seq);
    
    _sk_step  = config.get_int(unique_name, "sk_step");
    mask_len = sizeof(uint8_t)*_num_elements;
}


void clRfiTimeSum::build(device_interface &param_Device)
{
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    apply_config(0);
    
    using namespace std::placeholders;
    restServer * rest_server = get_rest_server();
    string endpoint = "/rfi_callback/" + std::to_string(param_Device.getGpuID());
    rest_server->register_json_callback(endpoint,
            std::bind(&clRfiTimeSum::rest_callback, this, _1, _2));

    gpu_command::build(param_Device);
    cl_int err;
    cl_device_id valDeviceID;
    string cl_options = get_cl_options();

    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());
    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options.c_str(), NULL, NULL ) );
    kernel = clCreateKernel( program, "rfi_chime_timesum", &err );
    CHECK_CL_ERROR(err);

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(int32_t),
                                   &_sk_step) );

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)4,
                                   sizeof(int32_t),
                                   &_num_elements) );

    Input_Mask = (uint8_t *)malloc(mask_len); //Allocate memory
    for (uint32_t i = 0; i < mask_len/sizeof(uint8_t); i++){
        Input_Mask[i] = 0; //Initialize Input Mask
    }

    mem_input_mask = clCreateBuffer(param_Device.getContext(), 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mask_len, Input_Mask, &err);
    CHECK_CL_ERROR(err);

    // Accumulation kernel global and local work space sizes.
    gws[0] = _num_elements * _num_local_freq / 4;
    gws[1] = 256;
    gws[2] = _samples_per_data_set/_sk_step;

    lws[0] = 1;
    lws[1] = 256;
    lws[2] = 1;
}
cl_event clRfiTimeSum::execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);

    std::lock_guard<std::mutex> lock(rest_callback_mutex);

    setKernelArg(0, param_Device.getInputBuffer(param_bufferID));
    setKernelArg(1, param_Device.getRfiTimeSumBuffer(param_bufferID));
    CHECK_CL_ERROR( clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &mem_input_mask ));
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

