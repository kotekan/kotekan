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
 *
 * @author Keith Vanderlinde and Andre Renard
 */
class cudaInputData : public cudaCommand {
public:
    cudaInputData(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaInputData();
    int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events, bool* quit) override;
    void finalize_frame(int frame_id) override;

    std::string get_performance_metric_string() override;

protected:
    cudaEvent_t* data_staged_event;

    /// Name of the GPU side memory to transfer data into.
    std::string _gpu_mem;

    int32_t in_buffer_id;
    int32_t in_buffer_precondition_id;
    int32_t in_buffer_finalize_id;
    Buffer* in_buf;
};

#endif // CUDA_INPUT_DATA_H
