#include "hipDeviceInterface.hpp"

#include "math.h"

#include <errno.h>

using kotekan::Config;
using std::string;

hipDeviceInterface::hipDeviceInterface(Config& config_, int32_t gpu_id_, int gpu_buffer_depth_) :
    gpuDeviceInterface(config_, gpu_id_, gpu_buffer_depth_) {

    // Find out how many GPUs can be probed.
    int max_num_gpus;
    CHECK_HIP_ERROR(hipGetDeviceCount(&max_num_gpus));
    INFO("Number of HIP GPUs: {:d}", max_num_gpus);
    INFO("Setting device ID to: {:d}", gpu_id);
    hipSetDevice(gpu_id);
}

hipDeviceInterface::~hipDeviceInterface() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_HIP_ERROR(hipStreamDestroy(stream[i]));
    }
    cleanup_memory();
}

void* hipDeviceInterface::alloc_gpu_memory(int len) {
    void* ret;
    CHECK_HIP_ERROR(hipMalloc(&ret, len));
    return ret;
}
void hipDeviceInterface::free_gpu_memory(void* ptr) {
    CHECK_HIP_ERROR(hipFree(ptr));
}


void* hipDeviceInterface::get_gpu_memory(const string& name, const size_t len) {
    return gpuDeviceInterface::get_gpu_memory(name, len);
}
void* hipDeviceInterface::get_gpu_memory_array(const string& name, const uint32_t index,
                                               const size_t len) {
    return gpuDeviceInterface::get_gpu_memory_array(name, index, len);
}
hipStream_t hipDeviceInterface::getStream(int stream_id) {
    return stream[stream_id];
}


void hipDeviceInterface::prepareStreams() {
    // Create command queues
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_HIP_ERROR(hipStreamCreate(&stream[i]));
    }
}
void hipDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, size_t len,
                                                hipEvent_t pre_event, hipEvent_t& copy_pre_event,
                                                hipEvent_t& copy_post_event) {
    if (pre_event)
        CHECK_HIP_ERROR(hipStreamWaitEvent(getStream(HIP_INPUT_STREAM), pre_event, 0));
    // Data transfer to GPU
    CHECK_HIP_ERROR(hipEventCreate(&copy_pre_event));
    CHECK_HIP_ERROR(hipEventRecord(copy_pre_event, getStream(HIP_INPUT_STREAM)));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(dst, src, len, hipMemcpyHostToDevice, getStream(HIP_INPUT_STREAM)));
    CHECK_HIP_ERROR(hipEventCreate(&copy_post_event));
    CHECK_HIP_ERROR(hipEventRecord(copy_post_event, getStream(HIP_INPUT_STREAM)));
}
void hipDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, size_t len,
                                                hipEvent_t pre_event, hipEvent_t& copy_pre_event,
                                                hipEvent_t& copy_post_event) {
    if (pre_event)
        CHECK_HIP_ERROR(hipStreamWaitEvent(getStream(HIP_OUTPUT_STREAM), pre_event, 0));
    // Data transfer to GPU
    CHECK_HIP_ERROR(hipEventCreate(&copy_pre_event));
    CHECK_HIP_ERROR(hipEventRecord(copy_pre_event, getStream(HIP_OUTPUT_STREAM)));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(dst, src, len, hipMemcpyDeviceToHost, getStream(HIP_OUTPUT_STREAM)));
    CHECK_HIP_ERROR(hipEventCreate(&copy_post_event));
    CHECK_HIP_ERROR(hipEventRecord(copy_post_event, getStream(HIP_OUTPUT_STREAM)));
}
