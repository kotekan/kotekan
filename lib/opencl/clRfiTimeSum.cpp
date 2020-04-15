#include "clRfiTimeSum.hpp"

#include "math.h"

#include <mutex>
#include <string.h>

using kotekan::bufferContainer;
using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

clRfiTimeSum::clRfiTimeSum(const char* param_gpuKernel, const char* param_name,
                           Config& param_config, const std::string& unique_name) :
    gpu_command(param_gpuKernel, param_name, param_config, unique_name) {}

clRfiTimeSum::~clRfiTimeSum() {
    restServer::instance().remove_json_callback(endpoint);
}

void clRfiTimeSum::rest_callback(connectionInstance& conn, json& json_request) {
    // Lock mutex
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    WARN("RFI Callback Received... Changing Parameters")
    // Update Paramters
    _bad_inputs.clear();
    for (uint32_t i = 0; i < json_request["bad_inputs"].size(); i++) {
        _bad_inputs.push_back(json_request["bad_inputs"][i].get<int>());
    }
    // Flag for rebuilding of Input Mask buffer
    rebuildInputMask = true;
    config.update_value(unique_name, "bad_inputs", _bad_inputs);
    // Send reply indicating success
    conn.send_empty_reply(HTTP_RESPONSE::OK);
}

void clRfiTimeSum::build(device_interface& param_Device) {
    // Lock callback mutex during build
    std::lock_guard<std::mutex> lock(rest_callback_mutex);

    // Apply general config
    gpu_command::apply_config();
    // RFI config parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _bad_inputs = config.get<std::vector<int32_t>>(unique_name, "bad_inputs");
    // Compute maske length
    mask_len = sizeof(uint8_t) * _num_elements;

    // Register Rest server endpoint
    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    endpoint = unique_name + "/rfi_time_sum_callback/" + std::to_string(param_Device.getGpuID());
    rest_server.register_post_callback(endpoint,
                                       std::bind(&clRfiTimeSum::rest_callback, this, _1, _2));
    // Build device
    gpu_command::build(param_Device);
    cl_int err;
    cl_device_id valDeviceID;
    std::string cl_options = get_cl_options();
    // Build program
    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());
    CHECK_CL_ERROR(clBuildProgram(program, 1, &valDeviceID, cl_options.c_str(), nullptr, nullptr));
    // Create the kernel
    kernel = clCreateKernel(program, "rfi_chime_time_sum", &err);
    CHECK_CL_ERROR(err);
    // Set some static arguments
    CHECK_CL_ERROR(clSetKernelArg(kernel, (cl_uint)3, sizeof(int32_t), &_sk_step));
    CHECK_CL_ERROR(clSetKernelArg(kernel, (cl_uint)4, sizeof(int32_t), &_num_elements));
    // Initialize Input Mask
    rebuildInputMask = false;
    Input_Mask = (uint8_t*)malloc(mask_len);
    uint32_t j = 0;
    for (uint32_t i = 0; i < mask_len / sizeof(uint8_t); i++) {
        Input_Mask[i] = (uint8_t)0;
        if (_bad_inputs.size() > 0 && (int32_t)i == _bad_inputs[j]) {
            Input_Mask[i] = (uint8_t)1;
            j++;
        }
    }
    // Create Input mask buffer
    mem_input_mask =
        clCreateBuffer(param_Device.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       mask_len, Input_Mask, &err);
    CHECK_CL_ERROR(err);
    // Set kernel global and local work space sizes.
    gws[0] = 64;
    gws[1] = 8;
    gws[2] = _samples_per_data_set / _sk_step;
    lws[0] = 64;
    lws[1] = 1;
    lws[2] = 1;
}
cl_event clRfiTimeSum::execute(int param_bufferID, device_interface& param_Device,
                               cl_event param_PrecedeEvent) {
    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);
    // Lock callback mutex
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    // Set input and output kernel arguments
    setKernelArg(0, param_Device.getInputBuffer(param_bufferID));
    setKernelArg(1, param_Device.getRfiTimeSumBuffer(param_bufferID));
    // Rebuild Input mask if necessary (only after rest-server callback)
    if (rebuildInputMask) {
        rebuildInputMask = false;
        Input_Mask = (uint8_t*)malloc(mask_len); // Allocate memory
        uint32_t j = 0;
        for (uint32_t i = 0; i < mask_len / sizeof(uint8_t); i++) {
            Input_Mask[i] = (uint8_t)0;
            if (_bad_inputs.size() > 0 && (int32_t)i == _bad_inputs[j]) {
                Input_Mask[i] = (uint8_t)1;
                j++;
            }
        }
        CHECK_CL_ERROR(clEnqueueWriteBuffer(param_Device.getQueue(1), mem_input_mask, CL_TRUE, 0,
                                            mask_len, Input_Mask, 0, nullptr, nullptr));
    }
    // Set input mask kernel argumnet
    CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mem_input_mask));
    // Queue kernel for execution
    CHECK_CL_ERROR(clEnqueueNDRangeKernel(param_Device.getQueue(1), kernel, 3, nullptr, gws, lws, 1,
                                          &param_PrecedeEvent, &postEvent[param_bufferID]));
    // Return post event
    return postEvent[param_bufferID];
}
