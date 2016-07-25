
#include "beamform_kernel.h"
#include "fpga_header_functions.h"

beamform_kernel::beamform_kernel(char * param_gpuKernel, char* param_name):gpu_command(param_gpuKernel, param_name)
{

}

beamform_kernel::~beamform_kernel()
{
    clReleaseMemObject(device_mask);
}

void beamform_kernel::build(Config* param_Config, class device_interface& param_Device)
{

    gpu_command::build(param_Config, param_Device);
    
    cl_int err;
    //stream_id_t stream_id;
    
    cl_device_id valDeviceID;
      
    char * cl_options = gpu_command::get_cl_options(param_Config);
    
    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());

    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options, NULL, NULL ) );

    kernel = clCreateKernel( program, "gpu_beamforming", &err );
    CHECK_CL_ERROR(err);
    
////##OCCURS IN SETUP_OPEN_CL    

    unsigned char mask[param_Config->processing.num_adjusted_elements];

    for (int i = 0; i < param_Config->processing.num_adjusted_elements; ++i) {
        mask[i] = 1;
    }
    for (int i = 0; i < param_Config->beamforming.num_masked_elements; ++i) {
        int mask_position = param_Config->beamforming.element_mask[i];
        mask_position = param_Config->processing.inverse_product_remap[mask_position];
        mask[mask_position] = 0;
    }
    
    device_mask = clCreateBuffer(param_Device.getContext(),
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        param_Config->processing.num_elements * sizeof(unsigned char),
                                        mask,
                                        &err);

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                    4,
                                    sizeof(cl_mem),
                                    (void*) &device_mask) );
    
////##
    
////##OCCURS IN setup_beamform_kernel_worksize

//streamID NEEDS TO BE SET FOR ALL NUM_BUFFERS.
//DEVICE_FREQ_MAP NEEDS TO BE AN ARRAY OF DIMESION EQUAL TO NUM_BUFFERS. IT ALSO NEEDS TO LIKELY LIVE IN DEVICE_INTERFACE.


    INFO("setup_beamform_kernel_worksize, setting scale factor to %f",
         param_Config->beamforming.scale_factor);
    float scale_factor = param_Config->beamforming.scale_factor;
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   5,
                                   sizeof(float),
                                   &scale_factor) );

    // Beamforming kernel global and local work space sizes.
    gws[0] = param_Config->processing.num_elements / 4;
    gws[1] = param_Config->processing.num_local_freq;
    gws[2] = param_Config->processing.samples_per_data_set / 32;

    lws[0] = 64;
    lws[1] = 1;
    lws[2] = 1;
    
////##
}

cl_event beamform_kernel::execute(int param_bufferID, class device_interface& param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);
  
////##OCCURS IN add_queue_set
    // The beamforming kernel.

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
////##
    
}