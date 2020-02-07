/**
 * @file
 * @brief Copy FRB Output gpu to host (for N2 concurrent run)
 *  - hsaBeamformOutput : public hsaCommand
 */

#ifndef HSA_BEAMFORM_OUTPUT_DATA_H
#define HSA_BEAMFORM_OUTPUT_DATA_H

#include "Config.hpp"             // for Config
#include "buffer.h"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string
/**
 * @class hsaBeamformOutput
 * @brief hsaCommand for copying FRB output gpu to host (for run with N2).
 *
 * This is an hsaCommand that async copy FRB output buffer from GPU
 * to CPU. It marks the FRB output buffer to be full when done so that
 * it can be reused. This code also passes metadata along. The finalize_frame
 * function has been hacked by not marking network_buffer empty, for
 * concurrent run with N2, because the equivalent output code from the N2
 * side is already marking network buffer empty.
 *
 * @par GPU Memory
 * @gpu_mem  bf_output       Output from the FRB pipeline, size 1024x128x16
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @todo   tidy up hack in finalize_frame for concurrent run?
 *
 * @author Cherry Ng
 *
 */

class hsaBeamformOutputData : public hsaCommand {
public:
    /// Constructor
    hsaBeamformOutputData(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor
    virtual ~hsaBeamformOutputData();

    /// Wait for output buffer to be empty, keep track of output_buffer_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;

    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    /// Marks output full when done and passes metadata
    void finalize_frame(int frame_id) override;

private:
    /// The very first input data from dpdk
    Buffer* network_buffer;
    /// Output buffer from the FRB pipeline
    Buffer* output_buffer;

    /// ID for network_buffer
    int32_t network_buffer_id;
    /// ID for network_buffer_precondition
    int32_t network_buffer_precondition_id;

    /// ID for output_buffer
    int32_t output_buffer_id;
    /// ID for output_buffer_precondition
    int32_t output_buffer_precondition_id;
    /// ID for output_buffer_execute
    int32_t output_buffer_execute_id;
};

#endif
