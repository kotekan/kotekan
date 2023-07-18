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
//#include "library_types.h" // for cudaDataType_t; /usr/local/cuda/include/library_types.h

struct cudaArrayMetadata {
    cudaArrayMetadata() :
        //type(CUDA_R_8U),
        type(""),
        dims(0)
    {}

    std::string get_dimensions_string();

    //cudaDataType_t type;
    std::string type;
    int dims;
    int dim[10];
    // array strides / layouts?

    // These vectors all have the same length.  (Should it be a vector of structs instead?)
    // frequencies -- integer (0-2047) identifier for FPGA coarse frequencies
    std::vector<int> coarse_freq;
    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    std::vector<int> freq_upchan_factor;
    // Time sampling -- for each coarse frequency channel, 2x the FPGA
    // sample number of the first sample.  The 2x is there to handle
    // the upchannelization case, where 2 or more samples may get
    // averaged, producing a new sample that is effectively halfway in
    // between them, ie, at a half-FPGAsample time.
    std::vector<int64> half_fpga_sample0;
    // Time sampling -- for each coarse frequency channel, the factor
    // by which the time samples have been downsampled relative to
    // FPGA samples.
    std::vector<int> time_downsampling_fpga;
};

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

    cudaArrayMetadata* get_gpu_memory_array_metadata(const std::string& name, const uint32_t index, bool create=true);

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

protected:
    void* alloc_gpu_memory(size_t len) override;
    void free_gpu_memory(void*) override;

    void* alloc_gpu_metadata() override;
    void free_gpu_metadata(void*) override;

    // Extra data
    std::vector<cudaStream_t> streams;

private:
};

#endif // CUDA_DEVICE_INTERFACE_H
