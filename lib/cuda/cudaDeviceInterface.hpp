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
 * @author Keith Vanderlinde
 */
class cudaDeviceInterface final : public gpuDeviceInterface {
public:
    cudaDeviceInterface(kotekan::Config& config, const std::string& unique_name, int32_t gpu_id,
                        int gpu_buffer_depth);
    ~cudaDeviceInterface();

    void prepareStreams(uint32_t num_streams);
    cudaStream_t getStream(int32_t cuda_stream_id);

    /// Returns the number of streams available
    int32_t get_num_streams();

    /// This function calls cudaSetDevice and must be called from every thread operating with this
    /// gpuDeviceInterface, or making calls directly to one of the cuda streams
    void set_thread_device() override;

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
    void async_copy_host_to_gpu(void* dst, void* src, size_t len, uint32_t cuda_stream_id,
                                cudaEvent_t pre_event, cudaEvent_t& copy_pre_event,
                                cudaEvent_t& copy_post_event);

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
    void async_copy_gpu_to_host(void* dst, void* src, size_t len, uint32_t cuda_stream_id,
                                cudaEvent_t pre_event, cudaEvent_t& copy_pre_event,
                                cudaEvent_t& copy_post_event);

    // Function overrides to cast the generic gpu_memory retulsts appropriately.
    void* get_gpu_memory_array(const std::string& name, const uint32_t index, const uint32_t len);
    void* get_gpu_memory(const std::string& name, const uint32_t len);

protected:
    void* alloc_gpu_memory(int len) override;
    void free_gpu_memory(void*) override;

    // Extra data
    std::vector<cudaStream_t> streams;

private:
};

#endif // CUDA_DEVICE_INTERFACE_H
