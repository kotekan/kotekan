/*
 * @file
 * @brief Copy compressed lost sample buffer from host to gpu
 *  - hsaInputCompressLostSamples : public hsaCommand
 */
#ifndef HSA_INPUT_COMPRESS_LOST_SAMPLES_H
#define HSA_INPUT_COMPRESS_LOST_SAMPLES_H

#include "Config.hpp"          // for Config
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "hsa/hsa.h"           // for hsa_signal_t
#include "hsaCommand.hpp"
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for uint32_t
#include <string>   // for string

/*
 * @class hsaInputCompressLostSamples
 * @brief hsaCommand for copying compressed lost sample buffer from host to gpu
 *
 * A kotekan hsa command to copy the compressed lost samples buffer from host memory to GPU memory.
 * The command waits for a full frame, async copies the frame contents to GPU memroy, and marks the
 * frame empty. The compressed lost samples buffer is used in the HFB pipeline to exclude samples
 * when summing across time.
 *
 * @par GPU Memory
 * @gpu_mem  compressed_lost_samples Array indicating if a given timestep was zeroed, size:
 * samples_per_data_set / factor_upchan
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c uint8_t
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author James Willis
 */
class hsaInputCompressLostSamples : public hsaCommand {
public:
    /// Constructor, applies config, initializes variables
    hsaInputCompressLostSamples(kotekan::Config& config, const std::string& unique_name,
                                kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    // Destructor, cleans up local allocs
    virtual ~hsaInputCompressLostSamples();
    /// Wait for full metadata frame and keep track of precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Allocates kernel arguments, places kernel in queue
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;
    /// Marks frame empty for re-fill
    void finalize_frame(int frame_id) override;

private:
    /// Frame ID
    uint32_t compressed_lost_samples_buffer_id;
    /// Frame ID for pre condition
    uint32_t compressed_lost_samples_buffer_precondition_id;
    /// Frame ID for finilazation
    uint32_t compressed_lost_samples_buffer_finalize_id;
    /// Lost samples buffer
    Buffer* compressed_lost_samples_buf;

};

#endif /*HSA_INPUT_COMPRESS_LOST_SAMPLES_H*/
