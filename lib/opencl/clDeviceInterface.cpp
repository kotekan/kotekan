#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "clDeviceInterface.hpp"
#include "clCommand.hpp"
#include "math.h"
#include <errno.h>

//from amd firepro demo:
char* oclGetOpenCLErrorCodeStr(cl_int input)
{
    int errorCode = (int)input;
    switch(errorCode)
    {
        case CL_SUCCESS:
            return (char*) "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return (char*) "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return (char*) "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return (char*) "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return (char*) "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return (char*) "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return (char*) "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return (char*) "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return (char*) "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return (char*) "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return (char*) "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return (char*) "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return (char*) "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return (char*) "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return (char*) "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:
            return (char*) "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:
            return (char*) "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:
            return (char*) "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:
            return (char*) "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return (char*) "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:
            return (char*) "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return (char*) "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return (char*) "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return (char*) "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return (char*) "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return (char*) "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return (char*) "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return (char*) "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return (char*) "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return (char*) "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return (char*) "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return (char*) "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return (char*) "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return (char*) "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return (char*) "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return (char*) "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return (char*) "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return (char*) "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return (char*) "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return (char*) "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return (char*) "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return (char*) "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return (char*) "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return (char*) "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return (char*) "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return (char*) "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return (char*) "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return (char*) "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return (char*) "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return (char*) "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return (char*) "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return (char*) "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return (char*) "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return (char*) "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return (char*) "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:
            return (char*) "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:
            return (char*) "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:
            return (char*) "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return (char*) "CL_INVALID_DEVICE_PARTITION_COUNT";
        default:
            return (char*) "unknown error code";
    }
    return (char*) "unknown error code";
}


clDeviceInterface::clDeviceInterface(Config& config_, int32_t gpu_id_, int gpu_buffer_depth_) :
    config(config_), gpu_id(gpu_id_), gpu_buffer_depth(gpu_buffer_depth_) {

    // Get a platform.
    CHECK_CL_ERROR( clGetPlatformIDs( 1, &platform_id, NULL ) );
    INFO("GPU Id %i",gpu_id);

    // Find out how many GPUs can be probed.
    cl_uint max_num_gpus;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU,0,NULL,&max_num_gpus);
    INFO("Maximum number of GPUs: %d",max_num_gpus);

    // Find a GPU device..
    cl_device_id device_ids[max_num_gpus];
    CHECK_CL_ERROR( clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, max_num_gpus, device_ids, NULL) );
    device_id = device_ids[gpu_id];

    int err;
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &err);
    CHECK_CL_ERROR(err);

    num_local_freq = config.get<int>("", "num_local_freq");
}

clDeviceInterface::~clDeviceInterface()
{
    for (int i = 0; i < NUM_QUEUES; ++i) {
        CHECK_CL_ERROR( clReleaseCommandQueue(queue[i]) );
    }

    CHECK_CL_ERROR( clReleaseContext(context) );
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

    CHECK_CL_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(time_res), &time_res, NULL));

    return time_res;
}

void clDeviceInterface::prepareCommandQueue(bool enable_profiling)
{
    cl_int err;

    // Create command queues
    for (int i = 0; i < NUM_QUEUES; ++i) {
        if (enable_profiling == true){
            queue[i] = clCreateCommandQueue( context, device_id, CL_QUEUE_PROFILING_ENABLE, &err );
            CHECK_CL_ERROR(err);
        } else{
            queue[i] = clCreateCommandQueue( context, device_id, 0, &err );
            CHECK_CL_ERROR(err);
        }

    }
}

cl_device_id clDeviceInterface::get_id() {
    return device_id;
}

cl_mem clDeviceInterface::get_device_freq_map(int32_t encoded_stream_id)
{
    //WHY ON EARTH IS THIS IN DEVICE?//
    std::map<int32_t, cl_mem>::iterator it = device_freq_map.find(encoded_stream_id);

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
    }
    return device_freq_map[encoded_stream_id];
}

cl_command_queue clDeviceInterface::getQueue(int param_Dim)
{
    return queue[param_Dim];
}


clMemoryBlock::~clMemoryBlock()  {
    for (auto &gpu_pointer : gpu_pointers) {
        clReleaseMemObject(gpu_pointer);
    }
}

