/**
 * @file
 * @brief Copy HFB Output gpu to host (for N2 concurrent run)
 *  - hsaBeamformHFBOutput : public hsaCommand
 */

#ifndef HSA_BEAMFORM_HFB_OUTPUT_DATA_H
#define HSA_BEAMFORM_HFB_OUTPUT_DATA_H

#include "Config.hpp"          // for Config
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "hsa/hsa.h"           // for hsa_signal_t
#include "hsaCommand.hpp"
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string

/**
 * @class hsaBeamformHFBOutput
 * @brief hsaCommand for copying HFB output gpu to host.
 *
 * This is an hsaCommand that async copy HFB output buffer from GPU
 * to CPU. It marks the HFB output buffer to be full when done so that
 * it can be reused.
 *
 * @par GPU Memory
 * @gpu_mem  hfb_output      Output from the HFB pipeline, size 1024x128x4 (beams x sub-freq x
 * sizeof(float))
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *
 * @author James Willis
 *
 */

class hsaBeamformHFBOutputData : public hsaCommand {
public:
    /// Constructor
    hsaBeamformHFBOutputData(kotekan::Config& config, const std::string& unique_name,
                             kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor
    virtual ~hsaBeamformHFBOutputData();

    /// Wait for output buffer to be empty, keep track of output_buffer_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;

    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    /// Marks output full when done and passes metadata
    void finalize_frame(int frame_id) override;

private:
    /// The data from the dpdk network thread (used for metadata)
    Buffer* network_buffer;
    /// Output buffer from the FRB pipeline
    Buffer* output_buffer;

    /// ID for network_buffer
    int32_t network_buffer_id;
    /// ID for output_buffer
    int32_t output_buffer_id;

    /// ID for network_buffer_precondition
    int32_t network_buffer_precondition_id;
    /// ID for output_buffer_precondition
    int32_t output_buffer_precondition_id;
    /// ID for output_buffer_execute
    int32_t output_buffer_execute_id;
};

#endif
