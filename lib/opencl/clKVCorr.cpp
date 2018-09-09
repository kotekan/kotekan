#include "clKVCorr.hpp"

#include <string>
using std::string;

REGISTER_CL_COMMAND(clKVCorr);

clKVCorr::clKVCorr(Config& config, const string &unique_name,
                            bufferContainer& host_buffers, clDeviceInterface& device) :
    clCommand("corr","kv_corr.cl", config, unique_name, host_buffers, device)
{
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _block_size = config.get_int(unique_name, "block_size");
    _num_data_sets = config.get_int(unique_name, "num_data_sets");
    _num_blocks = config.get_int(unique_name,"num_blocks");
    _samples_per_data_set = config.get_int(unique_name,"samples_per_data_set");

    defineOutputDataMap(); //id_x_map and id_y_map depend on this call.

    command_type = clCommandType::KERNEL;
}

clKVCorr::~clKVCorr()
{
    free(zeros);

    clReleaseMemObject(device_block_lock);
    clReleaseMemObject(id_x_map);
    clReleaseMemObject(id_y_map);
}

void clKVCorr::apply_config(const uint64_t& fpga_seq) {
    clCommand::apply_config(fpga_seq);
}

void clKVCorr::build()
{
    apply_config(0);
    clCommand::build();

    cl_int err;

    string cl_options = "";
    cl_options += " -D NUM_ELEMENTS=" + std::to_string(_num_elements);
    cl_options += " -D NUM_FREQUENCIES=" + std::to_string(_num_local_freq);
    cl_options += " -D SAMPLES_PER_DATA_SET=" + std::to_string(_samples_per_data_set);

    cl_device_id dev_id = device.get_id();

    err = clBuildProgram( program, 1, &dev_id, cl_options.c_str(), NULL, NULL );
    if (err != CL_SUCCESS){
        size_t len = 0;
        CHECK_CL_ERROR(clGetProgramBuildInfo(program, device.get_id(), CL_PROGRAM_BUILD_LOG, 0, NULL, &len));
        char *buffer = (char*)calloc(len, sizeof(char));
        CHECK_CL_ERROR(clGetProgramBuildInfo(program, device.get_id(), CL_PROGRAM_BUILD_LOG, len, buffer, NULL));
        INFO("CL failed. Build log follows: \n %s",buffer);
        free(buffer);
    } CHECK_CL_ERROR(err); 
    kernel = clCreateKernel( program, "corr", &err );
    CHECK_CL_ERROR(err);

    //set other parameters that will be fixed for the kernels (changeable parameters will be set in run loops)
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)3,
                                   sizeof(id_x_map),
                                   (void*) &id_x_map) ); //this should maybe be sizeof(void *)?

    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   (cl_uint)4,
                                   sizeof(id_y_map),
                                   (void*) &id_y_map) );

    zeros=(cl_int *)calloc(_num_blocks*_num_local_freq,sizeof(cl_int)); //for the output buffers

    device_block_lock = clCreateBuffer(device.get_context(),
                                        CL_MEM_COPY_HOST_PTR,
                                        _num_blocks*_num_local_freq*sizeof(cl_int),
                                        zeros,
                                        &err);
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   5,
                                   sizeof(void *),
                                   (void*) &device_block_lock));


    // Correlation kernel global and local work space sizes.
    gws[0] = 8*_num_data_sets;
    gws[1] = 8*_num_local_freq;
    gws[2] = _num_blocks;//*num_accumulations;

    lws[0] = 8;
    lws[1] = 8;
    lws[2] = 1;
}

cl_event clKVCorr::execute(int gpu_frame_id, const uint64_t& fpga_seq, cl_event pre_event)
{
    clCommand::execute(gpu_frame_id, 0, pre_event);

    uint32_t input_frame_len =  _num_elements * _num_local_freq * _samples_per_data_set;
    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size*_block_size) * 2 * _num_data_sets  * sizeof(int32_t);
    uint32_t presum_len = _num_elements * _num_local_freq * 2 * sizeof (int32_t);

    cl_mem input_memory = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    cl_mem output_memory_frame = device.get_gpu_memory_array("output",gpu_frame_id, output_len);
    cl_mem presum_memory = device.get_gpu_memory_array("presum", gpu_frame_id, presum_len);

    setKernelArg(0, input_memory);
    setKernelArg(1, presum_memory);
    setKernelArg(2, output_memory_frame);

    CHECK_CL_ERROR( clEnqueueNDRangeKernel(device.getQueue(1),
                                            kernel,
                                            3,
                                            NULL,
                                            gws,
                                            lws,
                                            1,
                                            &pre_event,
                                            &post_event[gpu_frame_id]));

    return post_event[gpu_frame_id];
}

void clKVCorr::defineOutputDataMap()
{
    cl_int err;
    // Create lookup tables

    //upper triangular address mapping --converting 1d addresses to 2d addresses
    unsigned int global_id_x_map[_num_blocks];
    unsigned int global_id_y_map[_num_blocks];

    //TODO: p260 OpenCL in Action has a clever while loop that changes 1 D addresses to X & Y indices for an upper triangle.
    // Time Test kernels using them compared to the lookup tables for NUM_ELEM = 256
    int largest_num_blocks_1D = _num_elements /_block_size;
    int index_1D = 0;
    for (int j = 0; j < largest_num_blocks_1D; j++){
        for (int i = j; i < largest_num_blocks_1D; i++){
            global_id_x_map[index_1D] = i;
            global_id_y_map[index_1D] = j;
            index_1D++;
        }
    }

    id_x_map = clCreateBuffer(device.get_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    _num_blocks * sizeof(cl_uint), global_id_x_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    id_y_map = clCreateBuffer(device.get_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    _num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }
}
