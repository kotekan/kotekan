#ifndef KOTEKAN_CUDA_RECHUNK_HPP
#define KOTEKAN_CUDA_RECHUNK_HPP

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

/**
 * @class cudaRechunk
 * @brief cudaCommand for combining GPU frames before further processing.
 *
 * @author Dustin Lang
 *
 * @par GPU Memory
 * @gpu_mem  gpu_mem_input  Input time-stream (matrix) of float16 values
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c float16
 * @gpu_mem  gpu_mem_meanstd  Output array of float16 means and standard deviations (one mean,std
 * per 256-element "chunk")
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c float16
 * @gpu_mem  gpu_mem_output  Output array of int4 rechunkd values
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int4
 *
 * @conf   num_chunks          Int.  Number of 256-element "chunks" in the gpu_mem_input
 * time-stream.  Must be a factor of 32.
 */
class cudaRechunk : public cudaCommand {
public:
    cudaRechunk(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaRechunk();
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                        bool* quit) override;

protected:
private:
    size_t _len_inner_input;
    size_t _len_inner_output;
    size_t _len_outer;

    size_t num_accumulated;
    // void* leftover_memory;
    // size_t num_leftover;

    // GPU memory where we assemble the output
    // void* gpu_mem_accum;

    /// GPU side memory name for the time-stream input
    std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    std::string _gpu_mem_output;
};

#endif // KOTEKAN_CUDA_RECHUNK_HPP
