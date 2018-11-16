#include "correlator_kernel.h"

#include <string>
using std::string;

correlator_kernel::correlator_kernel(const char * param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name):
    gpu_command(param_gpuKernel, param_name, param_config, unique_name)
{
}

correlator_kernel::~correlator_kernel()
{
    free(zeros);

    clReleaseMemObject(device_block_lock);
    clReleaseMemObject(id_x_map);
    clReleaseMemObject(id_y_map);
}

void correlator_kernel::build(class device_interface& param_Device)
{
    gpu_command::apply_config();
    gpu_command::build(param_Device);

    cl_int err;

    unsigned int num_accumulations;
    cl_device_id valDeviceID;

    // Number of compressed accumulations.
    num_accumulations = _samples_per_data_set/256;

    string cl_options = get_cl_options();

    valDeviceID = param_Device.getDeviceID(param_Device.getGpuID());

    CHECK_CL_ERROR ( clBuildProgram( program, 1, &valDeviceID, cl_options.c_str(), NULL, NULL ) );

    kernel = clCreateKernel( program, "corr", &err );
    CHECK_CL_ERROR(err);
    defineOutputDataMap(param_Device); //id_x_map and id_y_map depend on this call.

    //set other parameters that will be fixed for the kernels (changeable parameters will be set in run loops)
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)2,
                                   sizeof(id_x_map),
                                   (void*) &id_x_map) ); //this should maybe be sizeof(void *)?

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(id_y_map),
                                   (void*) &id_y_map) );

    zeros=(cl_int *)calloc(_num_blocks*_num_local_freq,sizeof(cl_int)); //for the output buffers

    device_block_lock = clCreateBuffer(param_Device.getContext(),
                                        CL_MEM_COPY_HOST_PTR,
                                        _num_blocks*_num_local_freq*sizeof(cl_int),
                                        zeros,
                                        &err);
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   4,
                                   sizeof(void *),
                                   (void*) &device_block_lock));


    // Correlation kernel global and local work space sizes.
    gws[0] = 8*_num_data_sets;
    gws[1] = 8*_num_adjusted_local_freq;
    gws[2] = _num_blocks*num_accumulations;

    lws[0] = 8;
    lws[1] = 8;
    lws[2] = 1;
}

cl_event correlator_kernel::execute(int param_bufferID, const uint64_t& fpga_seq, class device_interface& param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);

    setKernelArg(0, param_Device.getInputBuffer(param_bufferID));
    setKernelArg(1, param_Device.getOutputBuffer(param_bufferID));

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
void correlator_kernel::defineOutputDataMap(device_interface& param_Device)
{
    cl_int err;
    // Create lookup tables

    //upper triangular address mapping --converting 1d addresses to 2d addresses
    unsigned int global_id_x_map[_num_blocks];
    unsigned int global_id_y_map[_num_blocks];

    //TODO: p260 OpenCL in Action has a clever while loop that changes 1 D addresses to X & Y indices for an upper triangle.
    // Time Test kernels using them compared to the lookup tables for NUM_ELEM = 256
    int largest_num_blocks_1D = _num_adjusted_elements /_block_size;
    int index_1D = 0;
    for (int j = 0; j < largest_num_blocks_1D; j++){
        for (int i = j; i < largest_num_blocks_1D; i++){
            global_id_x_map[index_1D] = i;
            global_id_y_map[index_1D] = j;
            index_1D++;
        }
    }

    id_x_map = clCreateBuffer(param_Device.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    _num_blocks * sizeof(cl_uint), global_id_x_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    id_y_map = clCreateBuffer(param_Device.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    _num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }
}
