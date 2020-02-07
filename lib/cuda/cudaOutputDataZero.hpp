/**
 * @file
 * @brief CUDA command to copy a block of zeros onto the GPU.
 *  - cudaOutputDataZero : public cudaCommand
 */

#ifndef CUDA_OUTPUT_DATA_ZERO_H
#define CUDA_OUTPUT_DATA_ZERO_H

#include "cudaCommand.hpp"

#include <sys/mman.h>

/**
 * @class cudaOutputDataZero
 * @brief cudaCommand for copying zeros onto the GPU,
 * e.g. to initialize output buffers.
 *
 * This is a cudaCommand that async copies a buffer of zeros from CPU to GPU.
 *
 * @par GPU Memory
 * @gpu_mem output           Buffer to zero, arbitrary size
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Any
 *
 * @conf   data_length       Int, size of buffer to zero in Bytes.
 *
 * @author Keith Vanderlinde
 *
 */
class cudaOutputDataZero : public cudaCommand {
public:
    cudaOutputDataZero(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaOutputDataZero();
    cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) override;

private:
    int32_t output_len;
    void* output_zeros;
};

#endif // CUDA_OUTPUT_DATA_ZERO_H
