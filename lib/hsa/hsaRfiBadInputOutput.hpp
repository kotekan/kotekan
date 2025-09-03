/*
 * @file
 * @brief Copy RFI bad input kernel output from gpu to host
 *  - hsaRfiBadInputOutput : public hsaCommand
 */
#ifndef HSA_RFI_BAD_INPUT_OUTPUT_H
#define HSA_RFI_BAD_INPUT_OUTPUT_H

#include "Config.hpp"             // for Config
#include "buffer.hpp"             // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string

/*
 * @class hsaRfiBadInputOutput
 * @brief hsaCommand for copying RFI bad input kernel output gpu to host.
 *
 * This is an hsaCommand that async copy RFI bad input kernel output buffer from GPU
 * to CPU. It marks the output buffer to be full when done so that it can be reused.
 * This code also passes metadata along.
 *
 * @par GPU Memory
 * @gpu_mem  rfi_bad_input   Output of averaged SK estimates for individual inputs.
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author Jacob Taylor
 */
class hsaRfiBadInputOutput : public hsaCommand {
public:
    /// Constructor
    hsaRfiBadInputOutput(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor
    virtual ~hsaRfiBadInputOutput();
    /// Wait for output buffer to be empty, keep track of _rfi_output_buf_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;
    /// Marks output full when done and passes metadata
    void finalize_frame(int frame_id) override;

private:
    /// The very first input data from dpdk
    Buffer* _network_buf;
    /// Output buffer from the FRB pipeline
    Buffer* _rfi_output_buf;
    /// ID for _network_buf
    int32_t _network_buf_id;

    int32_t _network_buf_precondition_id;
    /// ID for _rfi_output_buf
    int32_t _rfi_output_buf_id;
    /// ID for _rfi_output_buf_precondition
    int32_t _rfi_output_buf_precondition_id;
    /// ID for _rfi_output_buf_execute
    int32_t _rfi_output_buf_execute_id;
};

#endif
