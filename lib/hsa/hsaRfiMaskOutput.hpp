/*
 * @file
 * @brief Copy RFI Mask from gpu to host
 *  - hsaRfiMaskOutput : public hsaCommand
 */
#ifndef HSA_RFI_MASK_OUTPUT_H
#define HSA_RFI_MASK_OUTPUT_H

#include "Config.hpp"             // for Config
#include "buffer.h"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "hsaSubframeCommand.hpp" // for hsaSubframeCommand

#include <stdint.h> // for int32_t
#include <string>   // for string

/*
 * @class hsaRfiMaskOutput
 * @brief hsaCommand for copying RFI mask from gpu to host.
 *
 * This is an hsaCommand that async copies the RFI Mask buffer from GPU
 * to CPU. It marks the RFI Mask buffer to be full when done so that
 * it can be reused. This code also passes metadata along.
 *
 * Note: This commands MUST only be used before the hsaOutputData command. This command presumes
 * that the network buffer has not been marked empty yet.
 *
 * @par GPU Memory
 * @gpu_mem  rfi_mask_output The RFI mask, array of uint_t, 0 means not masked, 1 means masked
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author Jacob Taylor
 */
class hsaRfiMaskOutput : public hsaSubframeCommand {
public:
    /// Constructor
    hsaRfiMaskOutput(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor
    virtual ~hsaRfiMaskOutput();
    /// Wait for output buffer to be empty, keep track of _rfi_mask_output_buf_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;
    /// Marks output full when done and passes metadata
    void finalize_frame(int frame_id) override;

private:
    /// The raw time series data from the FPGAs
    Buffer* _network_buf;
    /// The array of missing sample flags for _network_buf
    Buffer* _rfi_mask_output_buf;
    /// ID for _network_buf
    int32_t _network_buf_id;

    int32_t _network_buf_precondition_id;
    /// ID for _rfi_mask_output_buf
    int32_t _rfi_mask_output_buf_id;
    /// ID for _rfi_mask_output_buf_precondition
    int32_t _rfi_mask_output_buf_precondition_id;
    /// ID for _rfi_mask_output_buf_execute
    int32_t _rfi_mask_output_buf_execute_id;
};

#endif
