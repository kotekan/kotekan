#ifndef KOTEKAN_CUDA_QUANTIZE_HPP
#define KOTEKAN_CUDA_QUANTIZE_HPP

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for float16_t, uint4x2_t
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface

#include <cstdint>        // for int64_t
#include <driver_types.h> // for cudaEvent_t
#include <string>         // for string, basic_string
#include <vector>         // for vector

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
    const int _num_beams;
    const int _num_frequencies;
    const int _num_times;
    const int64_t _num_chunks;

    /// GPU side memory name for the time-stream input
    const std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    const std::string _gpu_mem_beams;
    /// GPU side memory name for mean,stdev output
    const std::string _gpu_mem_beams_meanstd;

    const NDArrayBuffer<float16_t, 3> input_buffer;
    NDArrayBuffer<kotekan::uint4x2_t, 3> beam_buffer;
    NDArrayBuffer<float16_t, 4> meanstd_buffer;
};

#endif // KOTEKAN_CUDA_QUANTIZE_HPP
