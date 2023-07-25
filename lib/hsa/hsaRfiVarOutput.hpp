/*
 * @file
 * @brief Copy RFI variance Output gpu to host
 *  - hsaRfiVarOutput : public hsaCommand
 */
#ifndef HSA_RFI_VAR_OUTPUT_H
#define HSA_RFI_VAR_OUTPUT_H

#include "Config.hpp"             // for Config
#include "buffer.hpp"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string

/*
 * @class hsaRfiVarOutput
 * @brief hsaCommand for copying RFI variance output gpu to host.
 *
 * This is an hsaCommand that async copy RFI variance output buffer from GPU
 * to CPU. It marks the RFI output buffer to be full when done so that
 * it can be reused. This code also passes metadata along. The finalize_frame
 * function has been hacked by not marking _network_buf empty, for
 * concurrent run with N2, because the equivalent output code from the N2
 * side is already marking network buffer empty.
 *
 * @par GPU Memory
 * @gpu_mem  rfi_output_var  Output of SK variances estimates, size: sizeof_float x nfreqs x nsamps
 * / sk_step
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author James Willis
 */
class hsaRfiVarOutput : public hsaCommand {
public:
    /// Constructor
    hsaRfiVarOutput(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor
    virtual ~hsaRfiVarOutput();
    /// Wait for output buffer to be empty, keep track of _rfi_output_buf_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;
    /// Marks output full when done and passes metadata
    void finalize_frame(int frame_id) override;

private:
    /// Output buffer for the RFI variance
    Buffer* _rfi_output_var_buf;
    ///
    Buffer* _network_buf;
    int32_t _network_buf_id;
    int32_t _network_buf_precondition_id;
    /// ID for _rfi_output_var_buf
    int32_t _rfi_output_var_buf_id;
    /// ID for _rfi_output_var_buf_precondition
    int32_t _rfi_output_var_buf_precondition_id;
    /// ID for _rfi_output_var_buf_execute
    int32_t _rfi_output_var_buf_execute_id;
};

#endif
