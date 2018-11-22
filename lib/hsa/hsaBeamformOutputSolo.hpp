/**
 * @file
 * @brief Copy FRB Output gpu to host (for FRB solo run)
 *  - hsaBeamformOutputSolo : public hsaCommand
 */

#ifndef HSA_BEAMFORM_OUTPUT_SOLO_H
#define HSA_BEAMFORM_OUTPUT_SOLO_H

#include "hsaCommand.hpp"

/**
 * @class hsaBeamformOutputSolo
 * @brief hsaCommand for copying FRB output gpu to host (FRB solo run).
 *
 * This is an hsaCommand that async copies FRB output buffer from GPU
 * to CPU. It marks the FRB output buffer to be full and marks the network 
 * buffer to be empty when all done so that these buffers can be reused. 
 * This code also passes metadata along.
 *
 * @par GPU Memory
 * @gpu_mem  bf_output       Output from the FRB pipeline, size 1024x128x16
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author Cherry Ng
 *
 */

class hsaBeamformOutputSolo: public hsaCommand
{
public:
    /// Constructor
    hsaBeamformOutputSolo(Config &config, const string &unique_name,
                  bufferContainer &host_buffers, hsaDeviceInterface &device);

    /// Destructor
    virtual ~hsaBeamformOutputSolo();

    /// Wait for output buffer to be empty, keep track of output_buffer_precondition_id
    int wait_on_precondition(int gpu_frame_id) override;

    /// Async copy output form gpu to host
    hsa_signal_t execute(int gpu_frame_id,
                         hsa_signal_t precede_signal) override;

    /// Marks output full and network empty when done and passes metadata
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
