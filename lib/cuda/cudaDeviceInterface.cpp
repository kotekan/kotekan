#include "cudaDeviceInterface.hpp"

#include "math.h"

#include <errno.h>

using kotekan::Config;

cudaDeviceInterface::cudaDeviceInterface(Config& config, const std::string& unique_name,
                                         int32_t gpu_id, int gpu_buffer_depth) :
    gpuDeviceInterface(config, unique_name, gpu_id, gpu_buffer_depth) {

    // Find out how many GPUs can be probed.
    int max_num_gpus;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&max_num_gpus));
    INFO("Number of CUDA GPUs: {:d}", max_num_gpus);

    if (gpu_id > max_num_gpus) {
        throw std::runtime_error(
            "Asked for a GPU ID which is higher than the maximum number of GPUs in the system");
    }

    set_thread_device();
}

cudaDeviceInterface::~cudaDeviceInterface() {
    for (auto& stream : streams) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }
    cleanup_memory();
}

void cudaDeviceInterface::set_thread_device() {
    CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
}

void* cudaDeviceInterface::alloc_gpu_memory(size_t len) {
    void* ret;
    CHECK_CUDA_ERROR(cudaMalloc(&ret, len));
    return ret;
}
void cudaDeviceInterface::free_gpu_memory(void* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

cudaStream_t cudaDeviceInterface::getStream(int32_t cuda_stream_id) {
    return streams[cuda_stream_id];
}

int32_t cudaDeviceInterface::get_num_streams() {
    return streams.size();
}

void cudaDeviceInterface::prepareStreams(uint32_t num_streams) {
    // Create command queues
    for (uint32_t i = 0; i < num_streams; ++i) {
        cudaStream_t stream = nullptr;
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
}
void cudaDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t& copy_start_event,
                                                 cudaEvent_t& copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_start_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_start_event, getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_end_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_end_event, getStream(cuda_stream_id)));
}
void cudaDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t& copy_start_event,
                                                 cudaEvent_t& copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_start_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_start_event, getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_end_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_end_event, getStream(cuda_stream_id)));
}
