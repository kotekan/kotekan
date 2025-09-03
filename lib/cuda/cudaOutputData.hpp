#ifndef CUDA_OUTPUT_DATA_H
#define CUDA_OUTPUT_DATA_H

#include "cudaCommand.hpp"

/**
 * @class cudaOutputData
 * @brief cudaCommand for copying data onto the GPU.
 *
 * This is a cudaCommand that async copies a buffer from GPU to CPU.
 * This code also passes metadata along from another buffer.
 *
 * @author Keith Vanderlinde and Andre Renard
 */
class cudaOutputData : public cudaCommand {
public:
    cudaOutputData(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                   int instance_num);
    ~cudaOutputData();
    int wait_on_precondition() override;
    virtual cudaEvent_t execute_base(cudaPipelineState& pipestate,
                                     const std::vector<cudaEvent_t>& pre_events) override;
    virtual cudaEvent_t execute(cudaPipelineState& pipestate,
                                const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

    std::string get_performance_metric_string() override;

protected:
    /// Name of the GPU side memory to transfer data from.
    std::string _gpu_mem;

    Buffer* output_buffer;
    Buffer* in_buffer;

private:
    // Did this stage run for this input frame?
    // This is used internally to pass state between execute_base and finalize_frame.
    bool did_generate_output;
};

#endif // CUDA_OUTPUT_DATA_H
