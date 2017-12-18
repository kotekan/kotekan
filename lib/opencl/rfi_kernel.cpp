#include "rfi_kernel.h"
#include "math.h"
#include <string.h>
#include <mutex>


rfi_kernel::rfi_kernel(const char * param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name):
    gpu_command(param_gpuKernel, param_name, param_config, unique_name)
{
    //INFO("Launched RFI Consturctor");
    config_local = param_config;
}

rfi_kernel::~rfi_kernel()
{
    for(int i = 0; i < num_links_per_gpu; i++){
        clReleaseMemObject(mem_Mean_Array[i]);
    }    
}

void rfi_kernel::rest_callback(connectionInstance& conn, json& json_request) {
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    INFO("RFI Callbak Received... Changing Parameters")
    _rfi_sensitivity = json_request["sensitivity"];
    _sk_step =  json_request["sk_step"];
    gws[2] = (_samples_per_data_set/_sk_step);
    sqrtM = sqrt(_num_elements*_sk_step);
    
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(float),
                                   &sqrtM) );

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)4,
                                   sizeof(int),
                                   &_rfi_sensitivity) );
    conn.send_empty_reply(STATUS_OK);
}

void rfi_kernel::apply_config(const uint64_t& fpga_seq) {
    gpu_command::apply_config(fpga_seq);
    //INFO("Applying RFI config");
    _sk_step  = config.get_int(unique_name, "sk_step");
    _rfi_sensitivity = config.get_int(unique_name, "rfi_sensitivity");
    _rfi_zero = config.get_bool(unique_name, "rfi_zero");
    link_id = 0;
    if(_rfi_zero) zero = 1;
    else zero = 0;

    sqrtM = sqrt(_num_elements*_sk_step);
}


void rfi_kernel::build(device_interface &param_Device)
{
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    apply_config(0);
    using namespace std::placeholders;
    restServer * rest_server = get_rest_server();
    string endpoint = "/rfi_callback/" + std::to_string(param_Device.getGpuID());
    rest_server->register_json_callback(endpoint,
            std::bind(&rfi_kernel::rest_callback, this, _1, _2));

    INFO("Starting RFI kernel build");
    gpu_command::build(param_Device);
    cl_int err;
    cl_device_id valDeviceID;
    num_links_per_gpu = config_local.num_links_per_gpu(param_Device.getGpuID());
    string cl_options = get_cl_options();

    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());
    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options.c_str(), NULL, NULL ) );
    kernel = clCreateKernel( program, "rfi_chime", &err );
    CHECK_CL_ERROR(err);
    INFO("RFI Kernel Created ... Setting Paramters")
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(float),
                                   &sqrtM) );

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)4,
                                   sizeof(int),
                                   &_rfi_sensitivity) );

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)5,
                                   sizeof(int),
                                   &_samples_per_data_set) );

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)6,
                                   sizeof(int),
                                   &zero) );

    Mean_Array = (float *)malloc(_num_elements*_num_local_freq*sizeof(float)); //Allocate memory
    for (int b = 0; b < (_num_elements*_num_local_freq); b++){
        Mean_Array[b] = 2.5; //Initialize with 0's
    }
    for(int i = 0; i < num_links_per_gpu; i++){
        mem_Mean_Array.push_back(clCreateBuffer(param_Device.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, _num_elements * _num_local_freq * sizeof(float), Mean_Array, &err));
	CHECK_CL_ERROR(err);
    }

    // Accumulation kernel global and local work space sizes.
    gws[0] = _num_elements;
    gws[1] = _num_local_freq;
    gws[2] = (_samples_per_data_set/_sk_step);

    lws[0] = 256;
    lws[1] = 1;
    lws[2] = 1;
    INFO("ENDING BUILD\n");
}

cl_event rfi_kernel::execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);

    std::lock_guard<std::mutex> lock(rest_callback_mutex);

    setKernelArg(0, param_Device.getInputBuffer(param_bufferID));
    setKernelArg(1, param_Device.getRfiCountBuffer(param_bufferID));
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                    2,
                                    sizeof(cl_mem),
                                    (void *) &mem_Mean_Array[link_id]) )

    CHECK_CL_ERROR( clEnqueueNDRangeKernel(param_Device.getQueue(1),
                                            kernel,
                                            3,
                                            NULL,
                                            gws,
                                            lws,
                                            1,
                                            &param_PrecedeEvent,
                                            &postEvent[param_bufferID]));

    link_id = (link_id + 1) % num_links_per_gpu;
    return postEvent[param_bufferID];
}
