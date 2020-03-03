/**
 * @file
 * @brief HIP command to copy a block of data onto the GPU.
 *  - hipInputData : public hipCommand
 */

#ifndef HIP_INPUT_DATA_H
#define HIP_INPUT_DATA_H

#include "hipCommand.hpp"

/**
 * @class hipInputData
 * @brief hipCommand for copying data onto the GPU.
 *
 * This is a hipCommand that async copies a buffer from CPU to GPU.
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
class hipInputData : public hipCommand {
public:
    hipInputData(kotekan::Config& config, const string& unique_name,
                  kotekan::bufferContainer& host_buffers, hipDeviceInterface& device);
    ~hipInputData();
    int wait_on_precondition(int gpu_frame_id) override;
    hipEvent_t execute(int gpu_frame_id, hipEvent_t pre_event) override;
    void finalize_frame(int frame_id) override;


protected:
    hipEvent_t* data_staged_event;

    int32_t in_buffer_id;
    int32_t in_buffer_precondition_id;
    int32_t in_buffer_finalize_id;
    Buffer* in_buf;
};

#endif // HIP_INPUT_DATA_H
