#include "correlator_kernel.h"

correlator_kernel::correlator_kernel(char * param_gpuKernel, char* param_name):gpu_command(param_gpuKernel, param_name)
{

}

correlator_kernel::~correlator_kernel()
{
    free(zeros);

    clReleaseMemObject(device_block_lock);
    clReleaseMemObject(id_x_map);
    clReleaseMemObject(id_y_map);
}


void correlator_kernel::build(Config* param_Config, class device_interface& param_Device)
{
    gpu_command::build(param_Config, param_Device);

    cl_int err;
//    char cl_options[1024];
    int num_blocks = param_Device.getNumBlocks();
    unsigned int num_accumulations;
    int numFreq;
    cl_device_id valDeviceID;

    // Number of compressed accumulations.
    num_accumulations = param_Config->processing.samples_per_data_set/256;
    numFreq = param_Config->processing.num_local_freq;

  //I GENUINELY DON'T LIKE HOW THIS PORTION OF THE CODE IS DEFINED. WOULD BE NICE TO ENCAPSULATE THE USE OF NUM_BLOCKS AND THE CALL TO DEFINEOUTPUTDATAMAP SOMEWHERE ELSE.
//   TODO explain these numbers/formulas.
//   num_blocks = (param_Config->processing.num_adjusted_elements / param_Config->gpu.block_size) *
//   (param_Config->processing.num_adjusted_elements / param_Config->gpu.block_size + 1) / 2.;

    //Peter has these defined differently in his code
//    sprintf(cl_options,"-D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du -D NUM_TIME_ACCUM=%du -D BASE_ACCUM=%du -D SIZE_PER_SET=%du",
//            param_Config->processing.num_adjusted_elements, param_Config->processing.num_adjusted_local_freq, num_blocks,
//            param_Config->processing.samples_per_data_set, 256, 32u,num_blocks*32*32*2*param_Config->processing.num_adjusted_local_freq);

    //sprintf(cl_options, "-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du",
//	  param_Config->processing.num_elements, numFreq,
//	  param_Config->processing.num_adjusted_elements,
//	  param_Config->processing.num_adjusted_local_freq,
//	  param_Config->processing.num_blocks, param_Config->processing.samples_per_data_set);

    
    
/// THIS IS HOW ANDRE HAS HIS CURRENTLY DEFINED. FOLLOW THESE AND USE HIS KERNELS IN MASTER.
//        char cl_options[1024];
//    sprintf(cl_options, "-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du -D NUM_BUFFERS=%du",
//            cl_data->config->processing.num_elements, cl_data->config->processing.num_local_freq,
//            cl_data->config->processing.num_adjusted_elements,
//            cl_data->config->processing.num_adjusted_local_freq,
//            cl_data->config->processing.num_blocks,
//            cl_data->config->processing.samples_per_data_set,
//            cl_data->config->processing.buffer_depth);
//    

    char * cl_options = gpu_command::get_cl_options(param_Config);
    
    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());

    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options, NULL, NULL ) );


    kernel = clCreateKernel( program, "corr", &err );
    CHECK_CL_ERROR(err);
    defineOutputDataMap(param_Config, num_blocks, param_Device); //id_x_map and id_y_map depend on this call.

    //set other parameters that will be fixed for the kernels (changeable parameters will be set in run loops)
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)2,
                                   sizeof(id_x_map),
                                   (void*) &id_x_map) ); //this should maybe be sizeof(void *)?

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(id_y_map),
                                   (void*) &id_y_map) );

    zeros=(cl_int *)calloc(num_blocks*numFreq,sizeof(cl_int)); //for the output buffers

    device_block_lock = clCreateBuffer(param_Device.getContext(),
                                        CL_MEM_COPY_HOST_PTR,
                                        num_blocks*numFreq*sizeof(cl_int),
                                        zeros,
                                        &err);
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   4,
                                   sizeof(void *),
                                   (void*) &device_block_lock));


    // Correlation kernel global and local work space sizes.
    gws[0] = 8*param_Config->processing.num_data_sets;
    gws[1] = 8*param_Config->processing.num_adjusted_local_freq;
    gws[2] = num_blocks*num_accumulations;

    lws[0] = 8;
    lws[1] = 8;
    lws[2] = 1;

}

cl_event correlator_kernel::execute(int param_bufferID, class device_interface& param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);

//    DEBUG("gws: %i, %i, %i. lws: %i, %i, %i", gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
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
void correlator_kernel::defineOutputDataMap(Config* param_Config, int param_num_blocks, class device_interface& param_Device)
{
    cl_int err;
    // Create lookup tables

    //upper triangular address mapping --converting 1d addresses to 2d addresses
    unsigned int global_id_x_map[param_num_blocks];
    unsigned int global_id_y_map[param_num_blocks];

    //TODO: p260 OpenCL in Action has a clever while loop that changes 1 D addresses to X & Y indices for an upper triangle.
    // Time Test kernels using them compared to the lookup tables for NUM_ELEM = 256
    int largest_num_blocks_1D = param_Config->processing.num_adjusted_elements /param_Config->gpu.block_size;
    int index_1D = 0;
    for (int j = 0; j < largest_num_blocks_1D; j++){
        for (int i = j; i < largest_num_blocks_1D; i++){
            global_id_x_map[index_1D] = i;
            global_id_y_map[index_1D] = j;
            index_1D++;
        }
    }

    id_x_map = clCreateBuffer(param_Device.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    param_num_blocks * sizeof(cl_uint), global_id_x_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    id_y_map = clCreateBuffer(param_Device.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    param_num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }
}


