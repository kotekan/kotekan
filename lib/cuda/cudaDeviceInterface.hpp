/**
 * @file
 * @brief Class to handle CUDA interactions with GPU hardware
 *  - cudaCommand
 */

#ifndef CUDA_DEVICE_INTERFACE_H
#define CUDA_DEVICE_INTERFACE_H

#include "cudaUtils.hpp"
#include "cuda_runtime_api.h"
#include "gpuDeviceInterface.hpp"

// These adjust the number of queues used by the CUDA runtime
// One queue is for data transfers to the GPU, one is for kernels,
// and one is for data transfers from the GPU to host memory.
// Unless you really know what you are doing, don't change these.
#define NUM_STREAMS 3
#define CUDA_INPUT_STREAM 0
#define CUDA_COMPUTE_STREAM 1
#define CUDA_OUTPUT_STREAM 2

/**
 * @class cudaDeviceInterface
 * @brief Class to handle CUDA interactions with GPU hardware.
 *
 * @par GPU Memory
 * @gpu_mem  bf_output       Output from the FRB pipeline, size 1024x128x16
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @todo   Add profiling flag.
 *
 * @author Keith Vanderlinde
 */
class cudaDeviceInterface final : public gpuDeviceInterface {
public:
    cudaDeviceInterface(kotekan::Config& config_, int32_t gpu_id_, int gpu_buffer_depth_);
    ~cudaDeviceInterface();

    void prepareStreams();
    cudaStream_t getStream(int param_Dim);

    /**
     * @brief Asynchronous copies memory from the host (CPU RAM) to the device GPU (global memory)
     *
     * @param dst The GPU memory pointer
     * @param src The CPU memory pointer
     * @param len The amount of data to copy in bytes
     * @param pre_event The event before this one to wait on, if NULL will not wait
     * @param copy_pre_event The profiling event at the start of this copy
     * @param copy_post_event The event at the end of the copy.
     */
    void async_copy_host_to_gpu(void* dst, void* src, size_t len, cudaEvent_t pre_event,
                                cudaEvent_t& copy_pre_event, cudaEvent_t& copy_post_event);

    /**
     * @brief Asynchronous Copies memory from the device GPU (global memory) to host (CPU RAM).
     *
     * @param dst The CPU memory pointer
     * @param src The GPU memory pointer
     * @param len The amount of data to copy in bytes
     * @param pre_event The event before this one to wait on, if NULL will not wait
     * @param copy_pre_event The profiling event at the start of this copy
     * @param copy_post_event The event at the end of the copy.
     */
    void async_copy_gpu_to_host(void* dst, void* src, size_t len, cudaEvent_t pre_event,
                                cudaEvent_t& copy_pre_event, cudaEvent_t& copy_post_event);

    // Function overrides to cast the generic gpu_memory retulsts appropriately.
    void* get_gpu_memory_array(const std::string& name, const uint32_t index, const uint32_t len);
    void* get_gpu_memory(const std::string& name, const uint32_t len);

protected:
    void* alloc_gpu_memory(int len) override;
    void free_gpu_memory(void*) override;

    // Extra data
    cudaStream_t stream[NUM_STREAMS];

private:
};

#endif // CUDA_DEVICE_INTERFACE_H
