/*
 * @file
 * @brief Copy compressed lost sample buffer from host to gpu
 *  - hsaInputCompressLostSamples : public hsaCommand
 */
#ifndef HSA_HOST_TO_DEVICE_COPY_H
#define HSA_HOST_TO_DEVICE_COPY_H

#include "Config.hpp"          // for Config
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "hsa/hsa.h"           // for hsa_signal_t
#include "hsaCommand.hpp"
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "visUtil.hpp"         // for FrameID

#include <stdint.h> // for uint32_t
#include <string>   // for string

/*
 * @class hsaInputCompressLostSamples
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
    // Destructor, cleans up local allocs
    virtual ~hsaHostToDeviceCopy();
    /// Wait for full metadata frame and keep track of precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Allocates kernel arguments, places kernel in queue
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;
    /// Marks frame empty for re-fill
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
