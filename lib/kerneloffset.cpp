#include "kerneloffset.h"
#include "math.h"

kernelOffset::kernelOffset(char * param_gpuKernel): gpu_command(param_gpuKernel)
{

}

kernelOffset::~kernelOffset()
{

}


void kernelOffset::build(Config *param_Config, class device_interface &param_Device)
{
    gpu_command::build(param_Config, param_Device);

    cl_int err;
    char cl_options[1024];
    int num_blocks = param_Device.getNumBlocks();
    cl_device_id valDeviceID;

  //sprintf(cl_options, "-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du",
//	param_Config->processing.num_elements, param_Config->processing.num_local_freq,
//	param_Config->processing.num_adjusted_elements,
//	param_Config->processing.num_adjusted_local_freq,
//	param_Config->processing.num_blocks, param_Config->processing.samples_per_data_set);


//     num_blocks = (param_Config->processing.num_adjusted_elements / param_Config->gpu.block_size) *
//     (param_Config->processing.num_adjusted_elements / param_Config->gpu.block_size + 1) / 2.;

    //Peter has these defined differently in his code
    sprintf(cl_options,"-D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du -D NUM_TIME_ACCUM=%du -D BASE_ACCUM=%du -D SIZE_PER_SET=%du",
            param_Config->processing.num_elements, param_Config->processing.num_local_freq, num_blocks,
            param_Config->processing.samples_per_data_set, 256, 32u,num_blocks*32*32*2*param_Config->processing.num_adjusted_local_freq);
    //printf("Dynamic define statements for GPU OpenCL kernels\n");
    //printf("-D NUM_ELEMENTS=%du \n-D NUM_FREQUENCIES=%du \n-D NUM_BLOCKS=%du \n-D NUM_TIMESAMPLES=%du\n-D NUM_TIME_ACCUM=%du\n-D BASE_ACCUM=%du\n-D SIZE_PER_SET=%du\n", num_elem, num_freq,num_blocks, time_steps, time_accum, BASE_TIMESAMPLES_ACCUM, num_blocks*32*32*2*num_freq);


  valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());

  CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options, NULL, NULL ) );


  kernel = clCreateKernel( program, "offsetAccumulateElements", &err );
  CHECK_CL_ERROR(err);

  // Accumulation kernel global and local work space sizes.
  gws[0] = 64*param_Config->processing.num_data_sets;
  gws[1] = (int)ceil(param_Config->processing.num_adjusted_elements *
  param_Config->processing.num_adjusted_local_freq/256.0);
  gws[2] = param_Config->processing.samples_per_data_set/32;

  lws[0] = 64;
  lws[1] = 1;
  lws[2] = 1;

  //delete[] cl_options;
}

cl_event kernelOffset::execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent)
{
  //cl_event *postEvent;

  //postEvent = thisPostEvent[param_bufferID];
  //  cl_int err;

    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);

  CHECK_CL_ERROR( clEnqueueNDRangeKernel(param_Device.getQueue(1),
                                            kernel,
                                            3,
                                            NULL,
                                            gws,
                                            lws,
                                            1,
                                            //&precedeEvent[param_bufferID],
                                            &param_PrecedeEvent,
                                            &postEvent[param_bufferID]));

  //postEvent[param_bufferID]=clCreateUserEvent(param_Device.getContext(), &err);
  //CHECK_CL_ERROR(err);

  return postEvent[param_bufferID];
}
