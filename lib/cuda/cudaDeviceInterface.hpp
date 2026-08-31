/**
 * @file
 * @brief Class to handle CUDA interactions with GPU hardware
 *  - cudaCommand
 */

#ifndef CUDA_DEVICE_INTERFACE_H
#define CUDA_DEVICE_INTERFACE_H

#include "Config.hpp"             // for Config
#include "driver_types.h"         // for cudaEvent_t, cudaStream_t
#include "gpuDeviceInterface.hpp" // for gpuDeviceInterface

#include <cuda.h>   // for CUfunction
#include <map>      // for map
#include <memory>   // for allocator, shared_ptr, weak_ptr
#include <mutex>    // for recursive_mutex
#include <stddef.h> // for size_t
#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string
#include <vector>   // for vector

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
                                cudaEvent_t pre_event, cudaEvent_t* copy_start_event,
                                cudaEvent_t* copy_end_event);

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
                                cudaEvent_t pre_event, cudaEvent_t* copy_start_event,
                                cudaEvent_t* copy_end_event);

    /**
     * @brief Builds a list of kernels from the file with name: @c kernel_file_name
     *
     * @param kernel_names Vector list of kernel names in the kernel file
     * @param opts         List of options to pass to nvrtc
     **/
    void build(const std::string& kernel_filename, const std::vector<std::string>& kernel_names,
               const std::vector<std::string>& opts);

    /**
     * @brief Builds a list of kernels from the PTX file with name: @c kernel_file_name
     *
     * Any @c --gpu-name option in @c opts is replaced with the compute capability of
     * the local GPU, so PTX generated for one GPU model runs on others as well.
     *
     * @param kernel_names       Vector list of kernel names in the kernel file
     * @param opts               List of options to pass to the PTX compiler
     * @param kernel_name_prefix Prefix to add to the kernel names in @c runtime_kernels
     **/
    void build_ptx(const std::string& kernel_filename, const std::vector<std::string>& kernel_names,
                   const std::vector<std::string>& opts,
                   const std::string& kernel_name_prefix = "");

    // Map containing the runtime kernels built with nvrtc from the kernel file (if needed)
    std::map<std::string, CUfunction> runtime_kernels;

    /// The most CUDA streams one device will ever have. Fixed so the mutex array below
    /// never reallocates: `prepareStreams` can be called from several stage constructors,
    /// and a reallocating container of live mutexes is not something to be clever about.
    static constexpr int32_t MAX_CUDA_STREAMS = 64;

    /// ⚠️ ONE MUTEX PER STREAM, NOT ONE PER DEVICE (2026-08-31). This used to be a single
    /// `gpu_command_mutex` covering the whole device, and that made every cudaProcess on a
    /// GPU serialize its command queuing against every other one -- across blocking CUDA
    /// driver calls. With 10 pipelines per GPU (7 GNSS chains + N2 + RFI + the copies) one
    /// thread stuck in `cuEventRecord` inside that lock stopped the entire GPU half, and the
    /// back-pressure propagated all the way to the NIC: the voltage ring filled, the
    /// cudaCopyToRingbuffer producer blocked, `host_voltage_buffer_N` pinned (peek_hold frees
    /// a frame only when EVERY consumer releases), transpose blocked, and the dpdk
    /// distributor dropped. Half the aperture went dark with nothing logging an error
    /// (proven by backtrace on cx19; see chord-gpu-command-mutex-wedge).
    ///
    /// Locking per STREAM instead is safe because intra-frame ordering never depended on
    /// exclusive stream access: each pipeline chains its own commands with explicit
    /// `cudaStreamWaitEvent` calls on events held in a vector LOCAL to its own
    /// `queue_commands` (see cudaSyncStream::execute). What the lock actually protects is
    /// shared `cudaCommandState` (cudaRechunkState is the documented case), which is shared
    /// only WITHIN one pipeline -- so a pipeline that locks the streams it uses keeps that
    /// invariant exactly.
    ///
    /// Pipelines with DISJOINT stream sets therefore never contend, which is what makes the
    /// wedge impossible rather than merely unlikely. Pipelines that do share streams (all
    /// the production processes still default to 0/1/2) still mutually exclude, byte-for-byte
    /// the old behaviour.
    ///
    /// ⚠️ ALWAYS LOCK IN ASCENDING STREAM ORDER -- that global order is the whole reason
    /// locking several of these cannot deadlock.
    std::recursive_mutex& stream_mutex(int32_t stream_id);

protected:
    void* alloc_gpu_memory(size_t len) override;
    void free_gpu_memory(void*) override;

    // Cuda Streams
    std::vector<cudaStream_t> streams;

    /// Per-stream command-queuing mutexes; see stream_mutex(). Fixed-size on purpose.
    std::recursive_mutex stream_mutexes[MAX_CUDA_STREAMS];

    // Cache of device instances (weak to avoid lifetime extension)
    static std::map<int32_t, std::weak_ptr<cudaDeviceInterface>> inst_map;
};

#endif // CUDA_DEVICE_INTERFACE_H
