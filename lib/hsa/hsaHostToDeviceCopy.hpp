/*
 * @file
 * @brief Copy CPU buffer to GPU staging buffer
 *  - hsaHostToDeviceCopy : public hsaCommand
 */
#ifndef HSA_HOST_TO_DEVICE_COPY_H
#define HSA_HOST_TO_DEVICE_COPY_H

#include "Config.hpp"             // for Config
#include "buffer.hpp"             // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "visUtil.hpp"            // for frameID

#include <string> // for string

/*
 * @class hsaHostToDeviceCopy
 * @brief hsaCommand for transferring a CPU buffer to GPU staging buffer
 *
 * @par buffers
 * @buffer in_buf The host buffer to transfer data to the GPU from
 *      @buffer_format any
 *      @buffer_metadata any
 *
 * @config gpu_memory_name    String, the name of GPU side frames
 *
 * @par GPU Memory
 * @gpu_mem  Data from host side buffer after copy with name @c gpu_memory_name
 *     @gpu_mem_type         staging
 *
 * @author Andre Renard
 */
class hsaHostToDeviceCopy : public hsaCommand {
public:
    /// Constructor, applies config, initializes variables
    hsaHostToDeviceCopy(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaHostToDeviceCopy();
    /// Wait for full CPU frame
    int wait_on_precondition(int gpu_frame_id) override;
    /// Performs the async copy enqueue
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;
    /// Marks CPU buffer frame empty
    void finalize_frame(int frame_id) override;

private:
    /// Host buffer
    Buffer* in_buf;

    /// Frame ID
    frameID in_buf_id;
    /// Frame ID for precondition
    frameID in_buf_precondition_id;
    /// Frame ID for finilazation
    frameID in_buf_finalize_id;

    /// GPU memory name
    std::string _gpu_memory_name;
};

#endif /*HSA_HOST_TO_DEVICE_COPY_H*/
