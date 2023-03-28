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
 * @conf len_inner_input  - Int
 * @conf len_inner_output - Int
 * @conf len_outer        - Int
 *
 * This code assumes 2-D input arrays that can be stacked along either
 * dimension.  This is, for input arrays A and B of size r x c, the
 * output will look like
 *    A1,1 A1,2 ... A1,c   B1,1 B1,2, ..., B1,c
 *    ...
 *    Ar,1 Ar,2 ... Ar,c   Br,1 Br,2, ..., Br,c
 *
 * if you set:
 *   len_inner_input = c
 *   len_inner_output = 2*c
 *   len_outer = r * sizeof(elements)
 *
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
