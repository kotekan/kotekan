#ifndef KOTEKAN_CUDA_QUANTIZE_HPP
#define KOTEKAN_CUDA_QUANTIZE_HPP

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

/**
 * @class cudaQuantize
 * @brief cudaCommand for quantizing float16 to int4, with Cuda code from Hans Hopkins.
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
 * @gpu_mem  gpu_mem_output  Output array of int4 quantized values
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int4
 *
 * @conf   num_chunks          Int.  Number of 256-element "chunks" in the gpu_mem_input
 * time-stream.  Must be a factor of 32.
 */
class cudaQuantize : public cudaCommand {
public:
    cudaQuantize(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device, int inst);
    ~cudaQuantize();
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    // These are the exact array sizes supported by the Cuda code
    static constexpr int CHUNK_SIZE = 256;
    static constexpr int FRAME_SIZE = 32;

private:
    int64_t _num_chunks;

    /// GPU side memory name for the time-stream input
    std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    std::string _gpu_mem_output;
    /// GPU side memory name for mean,stdev output
    std::string _gpu_mem_meanstd;
};

#endif // KOTEKAN_CUDA_QUANTIZE_HPP
