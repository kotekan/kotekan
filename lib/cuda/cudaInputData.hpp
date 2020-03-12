/**
 * @file
 * @brief CUDA command to copy a block of data onto the GPU.
 *  - cudaInputData : public cudaCommand
 */

#ifndef CUDA_INPUT_DATA_H
#define CUDA_INPUT_DATA_H

#include "cudaCommand.hpp"

/**
 * @class cudaInputData
 * @brief cudaCommand for copying data onto the GPU.
 *
 * This is a cudaCommand that async copies a buffer from CPU to GPU.
 * This code also passes metadata along.
 *
 * @par GPU Memory
 * @gpu_mem in_buf           Input buffer, arbitrary size
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Any
 *
 * @author Keith Vanderlinde
 *
 */
class cudaInputData : public cudaCommand {
public:
    cudaInputData(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaInputData();
    int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) override;
    void finalize_frame(int frame_id) override;


protected:
    cudaEvent_t* data_staged_event;

    int32_t in_buffer_id;
    int32_t in_buffer_precondition_id;
    int32_t in_buffer_finalize_id;
    Buffer* in_buf;
};

#endif // CUDA_INPUT_DATA_H
