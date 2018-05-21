#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "clDeviceInterface.hpp"
#include "clCommand.hpp"
#include "callbackdata.h"
#include "math.h"
#include <errno.h>

clDeviceInterface::clDeviceInterface(Config& config_, int32_t gpu_id_, int gpu_buffer_depth_) :
    config(config_), gpu_id(gpu_id_), gpu_buffer_depth(gpu_buffer_depth_) {

    cl_int err;

/*
    // Config variables
    num_links_per_gpu = config.num_links_per_gpu(gpu_id);
    enable_beamforming = config.get_bool(unique_name, "enable_beamforming");
    num_adjusted_elements = config.get_int(unique_name, "num_adjusted_elements");
    num_adjusted_local_freq = config.get_int(unique_name, "num_adjusted_local_freq");
    num_local_freq = config.get_int(unique_name, "num_local_freq");
    block_size = config.get_int(unique_name, "block_size");
    num_data_sets = config.get_int(unique_name, "num_data_sets");
    num_elements = config.get_int(unique_name, "num_elements");
    num_blocks = config.get_int(unique_name, "num_blocks");
    samples_per_data_set = config.get_int(unique_name,"samples_per_data_set");
    accumulate_len = num_adjusted_local_freq *
        num_adjusted_elements * 2 * num_data_sets * sizeof(cl_int);
    aligned_accumulate_len = PAGESIZE_MEM * (ceil((double)accumulate_len / (double)PAGESIZE_MEM));
    assert(aligned_accumulate_len >= accumulate_len);
*/

    // Get a platform.
    CHECK_CL_ERROR( clGetPlatformIDs( 1, &platform_id, NULL ) );
    INFO("GPU Id %i",gpu_id);

    // Find out how many GPUs can be probed.
    cl_uint max_num_gpus = 0;
    clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU,0,NULL,&max_num_gpus);
    INFO("Maximum number of GPUs: %d",max_num_gpus);

    // Find a GPU device..
    device_id = (cl_device_id *)malloc(max_num_gpus * sizeof(cl_device_id));
    CHECK_CL_ERROR( clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, max_num_gpus, device_id, NULL) );

    context = clCreateContext( NULL, 1, &device_id[gpu_id], NULL, NULL, &err);
    CHECK_CL_ERROR(err);
}

clDeviceInterface::~clDeviceInterface()
{
    cl_int err;

    for (int i = 0; i < NUM_QUEUES; ++i) {
        CHECK_CL_ERROR( clReleaseCommandQueue(queue[i]) );
    }

/*
    for (std::map<int32_t,cl_mem>::iterator it=device_freq_map.begin(); it!=device_freq_map.end(); ++it){
        CHECK_CL_ERROR( clReleaseMemObject(it->second) );
    }

    err = munlock((void *) accumulate_zeros, aligned_accumulate_len);
    if ( err == -1 ) {
        ERROR("Error unlocking memory");
        exit(errno);
    }
    free(accumulate_zeros);
*/
    CHECK_CL_ERROR( clReleaseContext(context) );
    free(device_id);
}

cl_mem clDeviceInterface::get_gpu_memory(const string& name, const uint32_t len) {
    cl_int err;
    // Check if the memory isn't yet allocated
    if (gpu_memory.count(name) == 0) {
        cl_mem ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, len, NULL, &err);
        CHECK_CL_ERROR(err);
        INFO("Allocating GPU[%d] OpenCL memory: %s, len: %d, ptr: %p", gpu_id, name.c_str(), len, ptr);
        assert(err == CL_SUCCESS);
        gpu_memory[name].len = len;
        gpu_memory[name].gpu_pointers.push_back(ptr);
    }
    // The size must match what has already been allocated.
    assert(len == gpu_memory[name].len);
    assert(gpu_memory[name].gpu_pointers.size() == 1);

    // Return the requested memory.
    return gpu_memory[name].gpu_pointers[0];
}

cl_mem clDeviceInterface::get_gpu_memory_array(const string& name, const uint32_t index, const uint32_t len) {
    cl_int err;
    // Check if the memory isn't yet allocated
    if (gpu_memory.count(name) == 0) {
        for (uint32_t i = 0; i < gpu_buffer_depth; ++i) {
            cl_mem ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, len, NULL, &err);
            CHECK_CL_ERROR(err);
            INFO("Allocating GPU[%d] OpenCL memory: %s, len: %d, ptr: %p", gpu_id, name.c_str(), len, ptr);
            assert(err == CL_SUCCESS);
            gpu_memory[name].len = len;
            gpu_memory[name].gpu_pointers.push_back(ptr);
        }
    }
    // The size must match what has already been allocated.
    assert(len == gpu_memory[name].len);
    // Make sure we aren't asking for an index past the end of the array.
    assert(index < gpu_memory[name].gpu_pointers.size());

    // Return the requested memory.
    return gpu_memory[name].gpu_pointers[index];
}


cl_context &clDeviceInterface::get_context(){
    return context;
}

size_t clDeviceInterface::get_opencl_resolution()
{
    //one tick per nanosecond of timing
    size_t time_res;

    CHECK_CL_ERROR(clGetDeviceInfo(device_id[gpu_id], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(time_res), &time_res, NULL));

    return time_res;
}

void clDeviceInterface::prepareCommandQueue(bool enable_profiling)
{
    cl_int err;

    // Create command queues
    for (int i = 0; i < NUM_QUEUES; ++i) {
        if (enable_profiling == true){
            queue[i] = clCreateCommandQueue( context, device_id[gpu_id], CL_QUEUE_PROFILING_ENABLE, &err );
            CHECK_CL_ERROR(err);
        } else{
            queue[i] = clCreateCommandQueue( context, device_id[gpu_id], 0, &err );
            CHECK_CL_ERROR(err);
        }

    }
}

void clDeviceInterface::allocateMemory()
{
/*
    cl_int err;
    // Array used to zero the output memory on the device.
    // TODO should this be in it's own function?
    err = posix_memalign((void **) &accumulate_zeros, PAGESIZE_MEM, aligned_accumulate_len);
    if ( err != 0 ) {
        ERROR("Error creating aligned memory for accumulate zeros");
        exit(err);
    }
    // Ask that all pages be kept in memory
    err = mlock((void *) accumulate_zeros, aligned_accumulate_len);
    if ( err == -1 ) {
        ERROR("Error locking memory - check ulimit -a to check memlock limits");
        exit(errno);
    }
    memset(accumulate_zeros, 0, aligned_accumulate_len );

    // We have two phase blanks
    const int num_phase_blanks = 2;
    device_phases = (cl_mem *) malloc(num_phase_blanks * sizeof(cl_mem));
    CHECK_MEM(device_phases);
    for (int i = 0; i < num_phase_blanks; ++i) {
        device_phases[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, num_elements * sizeof(float)
                , NULL, &err);
        CHECK_CL_ERROR(err);
    }
    */
}

cl_mem clDeviceInterface::get_device_freq_map(int32_t encoded_stream_id)
{
    std::map<int32_t, cl_mem>::iterator it = device_freq_map.find(encoded_stream_id);
/* 
    if(it == device_freq_map.end())
    {
        // Create the freq map for the first time.
        cl_int err;
        stream_id_t stream_id = extract_stream_id(encoded_stream_id);
        float freq[num_local_freq];

        for (int j = 0; j < num_local_freq; ++j) {
            freq[j] = freq_from_bin(bin_number(&stream_id, j))/1000.0;
        }

        device_freq_map[encoded_stream_id] = clCreateBuffer(context,
                                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            num_local_freq * sizeof(float), freq, &err);
        CHECK_CL_ERROR(err);
    }*/
    return device_freq_map[encoded_stream_id];
}

cl_command_queue clDeviceInterface::getQueue(int param_Dim)
{
    return queue[param_Dim];
}


clMemoryBlock::~clMemoryBlock()  {
    for (auto gpu_pointer : gpu_pointers) {
        clReleaseMemObject(gpu_pointer);
    }
}

