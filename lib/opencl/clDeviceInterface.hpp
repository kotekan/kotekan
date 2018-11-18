#ifndef CL_DEVICE_INTERFACE_H
#define CL_DEVICE_INTERFACE_H

#include <map>
#include <vector>
#ifdef __APPLE__
    #include <OpenCL/cl_platform.h>
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl_platform.h>
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif
#include "Config.hpp"
#include "buffer.h"
#include "kotekanLogging.hpp"
#include "clUtils.hpp"


// This adjusts the number of queues used by the OpenCL runtime
// One queue is for data transfers to the GPU, one is for kernels,
// and one is for data transfers from the GPU to host memory.
// Unless you really know what you are doing, don't change this.
#define NUM_QUEUES 3

// Store named set of gpu pointer(s) with uniform size
struct clMemoryBlock {
    vector<cl_mem> gpu_pointers;
    uint32_t len;

    // Need to be able to release the cl pointers
    ~clMemoryBlock() 
        { for (auto &gpu_pointer : gpu_pointers) clReleaseMemObject(gpu_pointer); };
};

class clDeviceInterface: public kotekanLogging
{
public:
    clDeviceInterface(Config& config_, int32_t gpu_id_, int gpu_buffer_depth_);
    ~clDeviceInterface();

    void prepareCommandQueue(bool enable_profiling);
    cl_command_queue getQueue(int param_Dim);
    cl_context &get_context();
    cl_device_id get_id();

    // Get one of the gpu memory pointers with the given name and size = len at the given index
    // The size of the set is equal to gpu_buffer_depth, so index < gpu_buffer_depth
    // If a region with this name exists then it will just return an existing pointer
    // at the give index, if the region doesn't exist, then it creates
    // it with gpu_buffer_depth pointers of size len
    // NOTE: if accessing an existing named region then len must match the existing
    // length or the system will throw an assert.
    cl_mem get_gpu_memory_array(const string &name, const uint32_t index, const uint32_t len);

    // Same as get_gpu_memory_array but gets just one gpu memory buffer
    // This can be used when internal memory is needed.
    // i.e. memory used for lookup tables that are the same between runs
    // or temporary buffers between kernels.
    // Should NOT be used for any memory that's copied between GPU and HOST memory.
    cl_mem get_gpu_memory(const string &name, const uint32_t len);


protected:

    // Extra data
    Config &config;

    int gpu_id;

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue[NUM_QUEUES];

    // Config variables
    uint32_t gpu_buffer_depth;

private:
    std::map<string, clMemoryBlock> gpu_memory;
};

#endif // CL_DEVICE_INTERFACE_H
