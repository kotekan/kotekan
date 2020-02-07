/*
 * @file
 * @brief Copy lost sample buffer from host to gpu
 *  - hsaInputLostSamples : public hsaCommand
 */
#ifndef HSA_INPUT_LOST_SAMPLES_H
#define HSA_INPUT_LOST_SAMPLES_H

#include "Config.hpp"             // for Config
#include "buffer.h"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for uint32_t
#include <string>   // for string

/*
 * @class hsaInputLostSamples
 * @brief hsaCommand for copying lost sample buffer from host to gpu
 *
 * A kotekan hsa command to copy the lost samples buffer from host memory to GPU memory. The command
 * waits for a full frame, async copies the frame contents to GPU memroy, and marks the frame empty.
 * Currently, only the RFI pipeline commands use the lost samples information to avoid marking
 * zeroed data as interference.
 *
 * @par GPU Memory
 * @gpu_mem  lost_samples     Array indicating if a given timestep was zeroed, size:
 * samples_per_data_set
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c uint8_t
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author Jacob Taylor
 */
class hsaInputLostSamples : public hsaCommand {
public:
    /// Constructor, applies config, initializes variables
    hsaInputLostSamples(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    // Destructor, cleans up local allocs
    virtual ~hsaInputLostSamples();
    /// Wait for full metadata frame and keep track of precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Allocates kernel arguments, places kernel in queue
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;
    /// Marks frame empty for re-fill
    void finalize_frame(int frame_id) override;

private:
    /// Frame ID
    uint32_t lost_samples_buffer_id;
    /// Frame ID for pre condition
    uint32_t lost_samples_buffer_precondition_id;
    /// Frame ID for finilazation
    uint32_t lost_samples_buffer_finalize_id;
    /// Lost samples buffer
    Buffer* lost_samples_buf;
    /// Length of lost samples frame
    uint32_t input_frame_len;
    /// Kotekan Config Variables
    /// Samples per data set, used for computing frame length
    uint32_t _samples_per_data_set;
};

#endif /*HSA_INPUT_LOST_SAMPLES_H*/
