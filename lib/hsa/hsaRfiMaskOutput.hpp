/*
 * @file
 * @brief Copy RFI Output gpu to host
 *  - hsaRfiMaskOutput : public hsaCommand
 */
#ifndef HSA_RFI_MASK_OUTPUT_H
#define HSA_RFI_MASK_OUTPUT_H

#include "hsaCommand.hpp"

/*
 * @class hsaRfiMaskOutput
 * @brief hsaCommand for copying RFI output gpu to host.
 *
 * This is an hsaCommand that async copy RFI output buffer from GPU 
 * to CPU. It marks the RFI output buffer to be full when done so that 
 * it can be reused. This code also passes metadata along. The finalize_frame 
 * function has been hacked by not marking _network_buf empty, for 
 * concurrent run with N2, because the equivalent output code from the N2 
 * side is already marking network buffer empty.
 *
 * @par GPU Memory
 * @gpu_mem  rfi_mask_output      Output of SK estimates, size: sizeof_float x nfreqs x nsamps / sk_step
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author Jacob Taylor
 */
class hsaRfiMaskOutput: public hsaCommand
{
public:
    ///Constructor
    hsaRfiMaskOutput(Config &config, const string &unique_name,
                 bufferContainer &host_buffers, hsaDeviceInterface &device);
    /// Destructor
    virtual ~hsaRfiMaskOutput();
    /// Wait for output buffer to be empty, keep track of _rfi_mask_output_buf_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) override;
    /// Marks output full when done and passes metadata
    void finalize_frame(int frame_id) override;
private:
    /// The very first input data from dpdk
    Buffer * _network_buf;
    /// Output buffer from the FRB pipeline
    Buffer * _rfi_mask_output_buf;
    /// ID for _network_buf
    int32_t _network_buf_id;
    /// ID for _rfi_mask_output_buf
    int32_t _rfi_mask_output_buf_id;
    /// ID for _rfi_mask_output_buf_precondition
    int32_t _rfi_mask_output_buf_precondition_id;
    /// ID for _rfi_mask_output_buf_execute
    int32_t _rfi_mask_output_buf_execute_id;
    //General Config Parameters
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Total number of frequencies (1024)
    uint32_t _num_total_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    //RFI config parameters
    /// The kurtosis step (How many timesteps per kurtosis estimate)
    uint32_t  _sk_step;
};

#endif
