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

void clRfiInputSum::rest_callback(connectionInstance& conn, json& json_request) {
    std::lock_guard<std::mutex> lock(rest_callback_mutex);

    INFO("RFI Input Sum Callback Received... Changing Parameters")

    _num_bad_inputs = json_request["num_bad_inputs"];

    _M = (_num_elements - _num_bad_inputs)*_sk_step;
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(int32_t),
                                   &_M) );

    conn.send_empty_reply(STATUS_OK);
}

void clRfiInputSum::apply_config(const uint64_t& fpga_seq) {
    gpu_command::apply_config(fpga_seq);

    //RFI Config Parameters
    _sk_step = config.get_int(unique_name, "sk_step");

    //Compute Buffer lengths
    _num_bad_inputs = config.get_int_array(unique_name, "bad_inputs").size();
    _M = (_num_elements - _num_bad_inputs)*_sk_step;
    INFO("Number of bad inputs computed: %d",_num_bad_inputs);
}


void clRfiInputSum::build(device_interface &param_Device)
{
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    apply_config(0);

    using namespace std::placeholders;
    restServer * rest_server = get_rest_server();
    string endpoint = "/rfi_input_sum_callback/" + std::to_string(param_Device.getGpuID());
    rest_server->register_json_callback(endpoint,
            std::bind(&clRfiInputSum::rest_callback, this, _1, _2));

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

