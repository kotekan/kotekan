#ifndef HIP_OUTPUT_DATA_H
#define HIP_OUTPUT_DATA_H

#include "hipCommand.hpp"

#include <string>

/**
 * @class hipOutputData
 * @brief hipCommand for copying data onto the GPU.
 *
 * This is a hipCommand that async copies a buffer from GPU to CPU.
 * This code also passes metadata along from another buffer.
 *
 * @par GPU Memory
 * @gpu_mem out_buf          Output buffer, arbitrary size
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Any
 * @gpu_mem in_buf           Input buffer from which to copy metadata
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Any
 *
 * @author Andre Renard
 *
 */
class hipOutputData : public hipCommand {
public:
    hipOutputData(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, hipDeviceInterface& device);
    ~hipOutputData();
    int wait_on_precondition(int gpu_frame_id) override;
    virtual hipEvent_t execute(int buf_frame_id, hipEvent_t pre_event) override;
    void finalize_frame(int frame_id) override;
    std::string get_performance_metric_string();

protected:
    int32_t output_buffer_execute_id;
    int32_t output_buffer_precondition_id;

    /// Name of the GPU side memory to transfer data from.
    std::string _gpu_mem;

    Buffer* output_buffer;
    Buffer* in_buffer;

    int32_t output_buffer_id;
    int32_t in_buffer_id;

private:
    // Common configuration values (which do not change in a run)
};

#endif // HIP_OUTPUT_DATA_H
