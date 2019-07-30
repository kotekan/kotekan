#ifndef CUDA_DEVICE_INTERFACE_H
#define CUDA_DEVICE_INTERFACE_H

#include "gpuDeviceInterface.hpp"
#include "cuda_runtime_api.h"
#include "cudaUtils.hpp"

// This adjusts the number of queues used by the OpenCL runtime
// One queue is for data transfers to the GPU, one is for kernels,
// and one is for data transfers from the GPU to host memory.
// Unless you really know what you are doing, don't change this.
#define NUM_STREAMS 3

class cudaDeviceInterface final : public gpuDeviceInterface {
public:
    cudaDeviceInterface(kotekan::Config& config_, int32_t gpu_id_, int gpu_buffer_depth_);
    ~cudaDeviceInterface();

    void prepareStreams();
    cudaStream_t getStream(int param_Dim);

    // Function overrides to cast the generic gpu_memory retulsts appropriately.
    void* get_gpu_memory_array(const string& name, const uint32_t index, const uint32_t len);
    void* get_gpu_memory(const string& name, const uint32_t len);

protected:
    void* alloc_gpu_memory(int len) override;
    void free_gpu_memory(void*) override;

    // Extra data
    cudaStream_t stream[NUM_STREAMS];

private:
};

#endif // CUDA_DEVICE_INTERFACE_H
