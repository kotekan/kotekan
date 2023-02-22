/**
 * @file
 * @brief Class to handle HIP interactions with GPU hardware
 *  - hipCommand
 */

#ifndef HIP_DEVICE_INTERFACE_H
#define HIP_DEVICE_INTERFACE_H

#include "gpuDeviceInterface.hpp"
#include "hip/hip_runtime_api.h"
#include "hipUtils.hpp"

#include <string>

// These adjust the number of queues used by the HIP runtime
// One queue is for data transfers to the GPU, one is for kernels,
// and one is for data transfers from the GPU to host memory.
// Unless you really know what you are doing, don't change these.
#define NUM_STREAMS 3
#define HIP_INPUT_STREAM 0
#define HIP_COMPUTE_STREAM 1
#define HIP_OUTPUT_STREAM 2

/**
 * @class hipDeviceInterface
 * @brief Class to handle HIP interactions with GPU hardware.
 *
 * @par GPU Memory
 * @gpu_mem  bf_output       Output from the FRB pipeline, size 1024x128x16
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @todo   Add profiling flag.
 *
 * @author Andre Renard
 */
class hipDeviceInterface final : public gpuDeviceInterface {
public:
    hipDeviceInterface(kotekan::Config& config, const std::string& unique_name, int32_t gpu_id,
                       int gpu_buffer_depth);
    ~hipDeviceInterface();

    void prepareStreams();
    hipStream_t getStream(int param_Dim);

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
    void async_copy_host_to_gpu(void* dst, void* src, size_t len, hipEvent_t pre_event,
                                hipEvent_t& copy_pre_event, hipEvent_t& copy_post_event);

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
    void async_copy_gpu_to_host(void* dst, void* src, size_t len, hipEvent_t pre_event,
                                hipEvent_t& copy_pre_event, hipEvent_t& copy_post_event);

    // Function overrides to cast the generic gpu_memory retulsts appropriately.
    void* get_gpu_memory_array(const std::string& name, const uint32_t index, const size_t len);
    void* get_gpu_memory(const std::string& name, const size_t len);

protected:
    void* alloc_gpu_memory(int len) override;
    void free_gpu_memory(void*) override;

    // Extra data
    hipStream_t stream[NUM_STREAMS];

private:
};

#endif // HIP_DEVICE_INTERFACE_H
