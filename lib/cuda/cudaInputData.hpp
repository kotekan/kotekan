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
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                  int instance_num);
    ~cudaInputData();
    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

    std::string get_performance_metric_string() override;

protected:
    /// Name of the GPU side memory to transfer data into.
    std::string _gpu_mem;

    Buffer* in_buf;
};

#endif // CUDA_INPUT_DATA_H
