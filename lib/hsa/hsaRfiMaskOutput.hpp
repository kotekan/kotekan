/*
 * @file
 * @brief Copy RFI Mask from gpu to host
 *  - hsaRfiMaskOutput : public hsaCommand
 */
#ifndef HSA_RFI_MASK_OUTPUT_H
#define HSA_RFI_MASK_OUTPUT_H

#include "hsaSubframeCommand.hpp"

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
    hsaRfiMaskOutput(kotekan::Config& config, const string& unique_name,
                     kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor
    virtual ~hsaRfiMaskOutput();
    /// Wait for output buffer to be empty, keep track of _rfi_mask_output_buf_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;
    /// Function to handle updatble config rest server calls for rfi zeroing toggle
    bool update_rfi_add_lostsamples_flag(nlohmann::json& json);
    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;
    /// Marks output full when done and passes metadata
    void finalize_frame(int frame_id) override;

private:
    /// The raw time series data from the FPGAs
    Buffer* _network_buf;
    /// The array of missing sample flags for _network_buf
    Buffer* _lost_samples_buf;
    /// Output buffer from the FRB pipeline
    Buffer* _rfi_mask_output_buf;
    /// Output of the N2 correlation products
    Buffer* _output_buf;
    /// ID for _network_buf
    int32_t _network_buf_id;
    /// ID for _lost_samples_buf
    int32_t _lost_samples_buf_id;
    /// ID for _output_buf;
    int32_t _output_buf_id;
    /// ID for _rfi_mask_output_buf
    int32_t _rfi_mask_output_buf_id;
    /// ID for _rfi_mask_output_buf_precondition
    int32_t _rfi_mask_output_buf_precondition_id;
    /// ID for _rfi_mask_output_buf_execute
    int32_t _rfi_mask_output_buf_execute_id;
    // General Config Parameters
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Total number of frequencies (1024)
    uint32_t _num_total_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    // RFI config parameters
    /// The kurtosis step (How many timesteps per kurtosis estimate)
    uint32_t _sk_step;
    /// A Flag wich sets whether or not we compute and add the lost samples due to RFI flagging
    bool _rfi_add_lostsamples;
};

#endif
