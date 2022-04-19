#include "cudaDeviceInterface.hpp"

#include "math.h"

#include <errno.h>

using kotekan::Config;

cudaDeviceInterface::cudaDeviceInterface(Config& config_, int32_t gpu_id_, int gpu_buffer_depth_) :
    gpuDeviceInterface(config_, gpu_id_, gpu_buffer_depth_) {

    // Find out how many GPUs can be probed.
    int max_num_gpus;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&max_num_gpus));
    INFO("Number of CUDA GPUs: {:d}", max_num_gpus);

    cudaSetDevice(gpu_id);
}

cudaDeviceInterface::~cudaDeviceInterface() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream[i]));
    }
    cleanup_memory();
}

void* cudaDeviceInterface::alloc_gpu_memory(int len) {
    void* ret;
    CHECK_CUDA_ERROR(cudaMalloc(&ret, len));
    return ret;
}
void cudaDeviceInterface::free_gpu_memory(void* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}


void* cudaDeviceInterface::get_gpu_memory(const std::string& name, const uint32_t len) {
    return gpuDeviceInterface::get_gpu_memory(name, len);
}
void* cudaDeviceInterface::get_gpu_memory_array(const std::string& name, const uint32_t index,
                                                const uint32_t len) {
    return gpuDeviceInterface::get_gpu_memory_array(name, index, len);
}
cudaStream_t cudaDeviceInterface::getStream(int stream_id) {
    return stream[stream_id];
}


void cudaDeviceInterface::prepareStreams() {
    // Create command queues
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream[i]));
    }
}
void cudaDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, size_t len,
                                                 cudaEvent_t pre_event, cudaEvent_t& copy_pre_event,
                                                 cudaEvent_t& copy_post_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(CUDA_INPUT_STREAM), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_pre_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_pre_event, getStream(CUDA_INPUT_STREAM)));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, getStream(CUDA_INPUT_STREAM)));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_post_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_post_event, getStream(CUDA_INPUT_STREAM)));
}
void cudaDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, size_t len,
                                                 cudaEvent_t pre_event, cudaEvent_t& copy_pre_event,
                                                 cudaEvent_t& copy_post_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(CUDA_OUTPUT_STREAM), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_pre_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_pre_event, getStream(CUDA_OUTPUT_STREAM)));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, getStream(CUDA_OUTPUT_STREAM)));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_post_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_post_event, getStream(CUDA_OUTPUT_STREAM)));
}
