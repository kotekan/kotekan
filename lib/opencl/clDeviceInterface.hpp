#ifndef CL_DEVICE_INTERFACE_H
#define CL_DEVICE_INTERFACE_H

#include "gpuDeviceInterface.hpp"
#ifdef __APPLE__
#include "OpenCL/opencl.h"

#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>
#endif
#include "clUtils.hpp"

// This adjusts the number of queues used by the OpenCL runtime
// One queue is for data transfers to the GPU, one is for kernels,
// and one is for data transfers from the GPU to host memory.
// Unless you really know what you are doing, don't change this.
#define NUM_QUEUES 3

class clDeviceInterface final : public gpuDeviceInterface {
public:
    clDeviceInterface(kotekan::Config& config_, int32_t gpu_id_, int gpu_buffer_depth_);
    ~clDeviceInterface();

    void prepareCommandQueue(bool enable_profiling);
    cl_command_queue getQueue(int param_Dim);
    cl_context& get_context();
    cl_device_id get_id();

    // Function overrides to cast the generic gpu_memory retulsts appropriately.
    cl_mem get_gpu_memory_array(const std::string& name, const uint32_t index, const size_t len);
    cl_mem get_gpu_memory(const std::string& name, const size_t len);

protected:
    void* alloc_gpu_memory(size_t len) override;
    void free_gpu_memory(void*) override;

    // Extra data
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue[NUM_QUEUES];

private:
};

#endif // CL_DEVICE_INTERFACE_H
