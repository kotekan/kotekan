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
    INFO("RFI Callback Received... Changing Parameters")

    bad_inputs.clear();
    for(uint32_t i = 0; i < json_request["bad_inputs"].size(); i++){
        bad_inputs.push_back(json_request["bad_inputs"][i]);
    }
    rebuildInputMask = true;

    conn.send_empty_reply(HTTP_RESPONSE::OK);
}

void clRfiTimeSum::apply_config(const uint64_t& fpga_seq) {
    gpu_command::apply_config(fpga_seq);

    _sk_step  = config.get_int(unique_name, "sk_step");
    bad_inputs = config.get_int_array(unique_name, "bad_inputs");
    mask_len = sizeof(uint8_t)*_num_elements;
}


void clRfiTimeSum::build(device_interface &param_Device)
{
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    apply_config(0);

    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    string endpoint = "/rfi_time_sum_callback/" + std::to_string(param_Device.getGpuID());
    rest_server.register_post_callback(endpoint,
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

    //Initialize Input Mask
    rebuildInputMask = false;
    Input_Mask = (uint8_t *)malloc(mask_len); //Allocate memory
    uint32_t j = 0;
    for(uint32_t i = 0; i < mask_len/sizeof(uint8_t); i++){
        Input_Mask[i] = (uint8_t)0;
        if(bad_inputs.size() > 0 && (int32_t)i == bad_inputs[j]){
                Input_Mask[i] = (uint8_t)1;
                j++;
        }
    }

    mem_input_mask = clCreateBuffer(param_Device.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mask_len, Input_Mask, &err);
    CHECK_CL_ERROR(err);

    // Accumulation kernel global and local work space sizes.
    gws[0] = 64;
    gws[1] = 8;
    gws[2] = _samples_per_data_set/_sk_step;

    lws[0] = 64;
    lws[1] = 1;
    lws[2] = 1;
}
cl_event clRfiTimeSum::execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);

    std::lock_guard<std::mutex> lock(rest_callback_mutex);

    setKernelArg(0, param_Device.getInputBuffer(param_bufferID));
    setKernelArg(1, param_Device.getRfiTimeSumBuffer(param_bufferID));

    if(rebuildInputMask){
        rebuildInputMask = false;
        Input_Mask = (uint8_t *)malloc(mask_len); //Allocate memory
        uint32_t j = 0;
        for(uint32_t i = 0; i < mask_len/sizeof(uint8_t); i++){
            Input_Mask[i] = (uint8_t)0;
            if(bad_inputs.size() > 0 && (int32_t)i == bad_inputs[j]){
                    Input_Mask[i] = (uint8_t)1;
                    j++;
            }
        }
        CHECK_CL_ERROR( clEnqueueWriteBuffer(param_Device.getQueue(1), mem_input_mask, CL_TRUE,
                                   0, mask_len, Input_Mask, 0, NULL, NULL));
    }

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

