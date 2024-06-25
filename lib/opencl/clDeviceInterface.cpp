#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "clDeviceInterface.hpp"

#include "clCommand.hpp"
#include "math.h"

#include <errno.h>

using kotekan::Config;

clDeviceInterface::clDeviceInterface(Config& config_, const std::string& unique_name,
                                     int32_t gpu_id_, int gpu_buffer_depth_) :
    gpuDeviceInterface(config_, unique_name, gpu_id_, gpu_buffer_depth_) {

    // Get a platform.
    CHECK_CL_ERROR(clGetPlatformIDs(1, &platform_id, nullptr));
    INFO("GPU Id {:d}", gpu_id);

    // Find out how many GPUs can be probed.
    cl_uint max_num_gpus;
    CHECK_CL_ERROR(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &max_num_gpus));
    INFO("Maximum number of GPUs: {:d}", max_num_gpus);

    // Find a GPU device..
    cl_device_id device_ids[max_num_gpus];
    CHECK_CL_ERROR(
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, max_num_gpus, device_ids, nullptr));
    device_id = device_ids[gpu_id];

    cl_int err;
    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    CHECK_CL_ERROR(err);
}

clDeviceInterface::~clDeviceInterface() {
    for (int i = 0; i < NUM_QUEUES; ++i) {
        CHECK_CL_ERROR(clReleaseCommandQueue(queue[i]));
    }
    CHECK_CL_ERROR(clReleaseContext(context));
    cleanup_memory();
}

void* clDeviceInterface::alloc_gpu_memory(size_t len) {
    cl_int err;
    cl_mem ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, len, nullptr, &err);
    CHECK_CL_ERROR(err);
    INFO("clCreateBuffer: {}, len: {}", fmt::ptr(ptr), len);
    return ptr;
}

void* clDeviceInterface::alloc_gpu_sub_memory(void* base_ptr, const size_t offset, const size_t len) {
   cl_int err;
   cl_buffer_region region;
   region.origin = offset;
   region.size = len;
   cl_mem ptr = clCreateSubBuffer((cl_mem)base_ptr, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
   CHECK_CL_ERROR(err);
   INFO("clCreateSubBuffer: {}, len: {}", fmt::ptr(ptr), len);
   return ptr;
}

void clDeviceInterface::free_gpu_memory(void* ptr) {
    CHECK_CL_ERROR(clReleaseMemObject((cl_mem)ptr));
}


cl_mem clDeviceInterface::get_gpu_memory(const std::string& name, const size_t len) {
    return (cl_mem)gpuDeviceInterface::get_gpu_memory(name, len);
}
cl_mem clDeviceInterface::get_gpu_memory_array(const std::string& name, const uint32_t index,
                                               const size_t len) {
    return (cl_mem)gpuDeviceInterface::get_gpu_memory_array(name, index, len);
}
cl_context& clDeviceInterface::get_context() {
    return context;
}
cl_device_id clDeviceInterface::get_id() {
    return device_id;
}
cl_command_queue clDeviceInterface::getQueue(int queue_id) {
    return queue[queue_id];
}

void clDeviceInterface::async_copy_host_to_gpu(const std::string& device_memory, const uint64_t& frame_id,
                                       void* host_memory, const size_t& frame_size,
                                       cl_event& pre_event, cl_event& post_event) {
    // We need to pass the base pointer to the GPU, not the offset pointer.  Instead we pass the
    // offset to the clEnqueueWriteBuffer function.
    cl_mem gpu_memory_frame = get_gpu_memory_array(device_memory, frame_id % gpu_buffer_depth, frame_size);
    CHECK_CL_ERROR(clEnqueueWriteBuffer(getQueue(0), gpu_memory_frame, CL_FALSE, 0,
                                        frame_size, host_memory, (pre_event == nullptr) ? 0 : 1,
                                        (pre_event == nullptr) ? nullptr : &pre_event,
                                        &post_event));
}

void clDeviceInterface::async_copy_gpu_to_host(const std::string& device_memory, const uint64_t& frame_id,
                                       void* host_memory, const size_t& frame_size,
                                       cl_event& pre_event, cl_event& post_event) {
    // We need to pass the base pointer to the GPU, not the offset pointer.  Instead we pass the
    // offset to the clEnqueueReadBuffer function.
    cl_mem gpu_memory_frame = get_gpu_memory_array(device_memory, frame_id % gpu_buffer_depth, frame_size);
    CHECK_CL_ERROR(clEnqueueReadBuffer(getQueue(2), gpu_memory_frame, CL_FALSE, 0,
                                       frame_size, host_memory, (pre_event == nullptr) ? 0 : 1,
                                       (pre_event == nullptr) ? nullptr : &pre_event,
                                       &post_event));
}

void clDeviceInterface::prepareCommandQueue(bool enable_profiling) {
    cl_int err;

    // Create command queues
    for (int i = 0; i < NUM_QUEUES; ++i) {
        if (enable_profiling == true) {
		
            // queue[i] = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	                // Get the maximum device queue size
            size_t queue_size;
            CHECK_CL_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, sizeof(size_t),
                                           &queue_size, nullptr));
            INFO("GPU id: {:d} Queue size: {:d}", (size_t)device_id, queue_size);
 
            cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0};
	    queue[i] = clCreateCommandQueueWithProperties(context, device_id, properties, &err);
	    CHECK_CL_ERROR(err);
        } else {
            queue[i] = clCreateCommandQueue(context, device_id, 0, &err);
            CHECK_CL_ERROR(err);
        }
    }
}

