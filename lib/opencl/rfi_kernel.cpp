#include "rfi_kernel.h"
#include "math.h"
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
    apply_config(0);
    //INFO("Starting RFI kernel build");
    gpu_command::build(param_Device);
    cl_int err;
    cl_device_id valDeviceID;
    num_links_per_gpu = config_local.num_links_per_gpu(param_Device.getGpuID());
    string cl_options = get_cl_options();

    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());
    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options.c_str(), NULL, NULL ) );
    kernel = clCreateKernel( program, "rfi_chime", &err );
    CHECK_CL_ERROR(err);
    //INFO("RFI Kernel Created")
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
        Mean_Array[b] = 0; //Initialize with 0's
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
}

cl_event rfi_kernel::execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);
    setKernelArg(0, param_Device.getInputBuffer(param_bufferID));
    setKernelArg(1, param_Device.getRfiCountBuffer(param_bufferID,link_id));
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

/*    unsigned int count_return_array[_num_local_freq*_samples_per_data_set/_sk_step];
    clEnqueueReadBuffer (param_Device.getQueue(1),
        param_Device.getRfiCountBuffer(param_bufferID,link_id),
        CL_TRUE,
        0,
	sizeof(count_return_array),
        (void *)count_return_array,
        0,
	NULL,
	NULL);
    FILE *f = fopen("../../../rfi_band.csv","a");
    for(int i = 0; i < _num_local_freq;i++){
	unsigned int counter = 0;
	for(int j = 0; j < _samples_per_data_set/_sk_step;j++){
		counter += count_return_array[i + _num_local_freq*j];
	}
	int freq_bin = 15 + link_id*16 + 128*i;
        float freq_mhz = 800 - freq_bin*((float)400/1024);
	fprintf(f,"%f,%f\n",freq_mhz,(float)counter/_samples_per_data_set);
        //INFO("Percentage Masked: %f Frequency %f\n", 100*(float)counter/_samples_per_data_set, freq_mhz);
    }
    fclose(f);*/
    link_id = (link_id + 1) % num_links_per_gpu;
    return postEvent[param_bufferID];
}
