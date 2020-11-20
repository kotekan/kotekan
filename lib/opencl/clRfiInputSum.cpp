#include "clRfiInputSum.hpp"

#include "math.h"

#include <mutex>
#include <string.h>

using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

clRfiInputSum::clRfiInputSum(const char* param_gpuKernel, const char* param_name,
                             Config& param_config, const std::string& unique_name) :
    gpu_command(param_gpuKernel, param_name, param_config, unique_name) {}

clRfiInputSum::~clRfiInputSum() {
    restServer::instance().remove_json_callback(endpoint);
}

void clRfiInputSum::rest_callback(connectionInstance& conn, json& json_request) {
    // Lock rest server mutex
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    WARN("RFI Input Sum Callback Received... Changing Parameters")
    // Update parameters
    _num_bad_inputs = json_request["num_bad_inputs"].get<int>();
    WARN("RFI Input Sum Callback, num_bad_inputs {:d}", _num_bad_inputs)
    // Re-calculat integration length
    _M = (_num_elements - _num_bad_inputs) * _sk_step;
    // Set new kernel args
    CHECK_CL_ERROR(clSetKernelArg(kernel, (cl_uint)3, sizeof(int32_t), &_M));
    // Reply indicating success
    conn.send_empty_reply(HTTP_RESPONSE::OK);
}

void clRfiInputSum::build(device_interface& param_Device) {
    // Lock callback mutex
    std::lock_guard<std::mutex> lock(rest_callback_mutex);

    // Apply config
    gpu_command::apply_config();
    // RFI Config Parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _num_bad_inputs = config.get<std::vector<uint32_t>>(unique_name, "bad_inputs").size();
    _use_local_sum = config.get_default<bool>(unique_name, "local_sum", true);
    DEBUG("Number of bad inputs computed: {:d}", _num_bad_inputs);
    // Calculate integration length
    _M = (_num_elements - _num_bad_inputs) * _sk_step;

    // Register rest server endpoint
    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    endpoint = unique_name + "/rfi_input_sum_callback/" + std::to_string(param_Device.getGpuID());
    rest_server.register_post_callback(endpoint,
                                       std::bind(&clRfiInputSum::rest_callback, this, _1, _2));
    // General command build
    gpu_command::build(param_Device);
    cl_int err;
    cl_device_id valDeviceID;
    std::string cl_options = get_cl_options();
    // Build program
    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());
    CHECK_CL_ERROR(clBuildProgram(program, 1, &valDeviceID, cl_options.c_str(), nullptr, nullptr));
    // Create Kernel
    kernel = clCreateKernel(program, "rfi_chime_input_sum", &err);
    CHECK_CL_ERROR(err);
    // Set static kernel arguments
    CHECK_CL_ERROR(clSetKernelArg(kernel, (cl_uint)2, sizeof(int32_t), &_num_elements));
    CHECK_CL_ERROR(clSetKernelArg(kernel, (cl_uint)3, sizeof(int32_t), &_M));
    // Set kernel global and local work space sizes.
    if (_use_local_sum) {
        gws[0] = 256;
        gws[1] = _num_local_freq;
        gws[2] = _samples_per_data_set / _sk_step;
        lws[0] = 256;
    } else {
        gws[0] = _num_local_freq;
        gws[1] = (_samples_per_data_set / _sk_step) / 4;
        gws[2] = 4;
        lws[0] = _num_local_freq;
    }
    lws[1] = 1;
    lws[2] = 1;
}
cl_event clRfiInputSum::execute(int param_bufferID, device_interface& param_Device,
                                cl_event param_PrecedeEvent) {
    // General GPU command execute
    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);
    // Lock rest server mutex
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    // Set input/output buffer argumnets
    setKernelArg(0, param_Device.getRfiTimeSumBuffer(param_bufferID));
    setKernelArg(1, param_Device.getRfiOutputBuffer(param_bufferID));
    // Queue kernel for execution
    CHECK_CL_ERROR(clEnqueueNDRangeKernel(param_Device.getQueue(1), kernel, 3, nullptr, gws, lws, 1,
                                          &param_PrecedeEvent, &postEvent[param_bufferID]));
    // return post event
    return postEvent[param_bufferID];
}
