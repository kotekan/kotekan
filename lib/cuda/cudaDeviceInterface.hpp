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

#include <cuda.h>

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
    /**
     * @brief Get/create a cudaDeviceInterface for the given gpu_id.
     */
    static std::shared_ptr<cudaDeviceInterface> get(int32_t gpu_id, const std::string& name,
                                                    kotekan::Config& config);

    cudaDeviceInterface(kotekan::Config& config, const std::string& unique_name, int32_t gpu_id);
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
     * @param cuda_stream_id The stream to run the copy on
     * @param pre_event The event before this one to wait on, if NULL will not wait
     * @param copy_start_event The profiling event at the start of this copy
     * @param copy_end_event The event at the end of the copy.
     */
    void async_copy_host_to_gpu(void* dst, void* src, size_t len, uint32_t cuda_stream_id,
                                cudaEvent_t pre_event, cudaEvent_t& copy_start_event,
                                cudaEvent_t& copy_end_event);

    /**
     * @brief Asynchronous Copies memory from the device GPU (global memory) to host (CPU RAM).
     *
     * @param dst The CPU memory pointer
     * @param src The GPU memory pointer
     * @param len The amount of data to copy in bytes
     * @param cuda_stream_id The stream to run the copy on
     * @param pre_event The event before this one to wait on, if NULL will not wait
     * @param copy_start_event The profiling event at the start of this copy
     * @param copy_end_event The event at the end of the copy.
     */
    void async_copy_gpu_to_host(void* dst, void* src, size_t len, uint32_t cuda_stream_id,
                                cudaEvent_t pre_event, cudaEvent_t& copy_start_event,
                                cudaEvent_t& copy_end_event);

    /**
     * @brief Builds a list of kernels from the file with name: @c kernel_file_name
     *
     * @param kernel_names Vector list of kernel names in the kernel file
     * @param opts         List of options to pass to nvrtc
     **/
    virtual void build(const std::string& kernel_filename,
                       const std::vector<std::string>& kernel_names,
                       const std::vector<std::string>& opts);

    virtual void build_ptx(const std::string& kernel_filename,
                           const std::vector<std::string>& kernel_names,
                           const std::vector<std::string>& opts);

    // Map containing the runtime kernels built with nvrtc from the kernel file (if needed)
    std::map<std::string, CUfunction> runtime_kernels;

    // Mutex for queuing GPU commands
    std::recursive_mutex gpu_command_mutex;

protected:
    void* alloc_gpu_memory(size_t len) override;
    void free_gpu_memory(void*) override;

    // Cuda Streams
    std::vector<cudaStream_t> streams;

    // Singleton dictionary
    static std::map<int, std::shared_ptr<cudaDeviceInterface>> inst_map;
};

#endif // CUDA_DEVICE_INTERFACE_H
