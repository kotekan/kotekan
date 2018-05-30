/*
 * @file
 * @brief Copy RFI Output gpu to host
 *  - hsaRfiOutput : public hsaCommand
 */
#ifndef HSA_RFI_OUTPUT_H
#define HSA_RFI_OUTPUT_H

#include "hsaCommand.hpp"

/*
 * @class hsaRfiOutput
 * @brief hsaCommand for copying RFI output gpu to host.
 *
 * This is an hsaCommand that async copy RFI output buffer from GPU 
 * to CPU. It marks the RFI output buffer to be full when done so that 
 * it can be reused. This code also passes metadata along. The finalize_frame 
 * function has been hacked by not marking network_buffer empty, for 
 * concurrent run with N2, because the equivalent output code from the N2 
 * side is already marking network buffer empty.
 *
 * @par GPU Memory
 * @gpu_mem  rfi_output      Output of SK estimates, size: sizeof_float x nfreqs x nsamps / sk_step
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author Jacob Taylor
 */
class hsaRfiOutput: public hsaCommand
{
public:
    ///Constructor
    hsaRfiOutput(Config &config, const string &unique_name,
                 bufferContainer &host_buffers, hsaDeviceInterface &device);
    /// Destructor
    virtual ~hsaRfiOutput();
    /// Wait for output buffer to be empty, keep track of output_buffer_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) override;
    /// Marks output full when done and passes metadata
    void finalize_frame(int frame_id) override;
private:
    /// The very first input data from dpdk
    Buffer * network_buffer;
    /// Output buffer from the FRB pipeline
    Buffer * output_buffer;
    /// ID for network_buffer
    int32_t network_buffer_id;
    /// ID for output_buffer
    int32_t output_buffer_id;
    /// ID for output_buffer_precondition
    int32_t output_buffer_precondition_id;
    /// ID for output_buffer_execute
    int32_t output_buffer_execute_id;
};

#endif
