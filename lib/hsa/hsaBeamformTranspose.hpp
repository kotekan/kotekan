/**
 * @file
 * @brief Transpose FRB data from time-pol-beam to pol-beam-time
 *  - hsaBeamformTranspose : public hsaCommand
 */

#ifndef HSA_BEAMFORM_TRANSPOSE_H
#define HSA_BEAMFORM_TRANSPOSE_H

#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string
/**
 * @class hsaBeamformTranspose
 * @brief hsaCommand to transpose FRB data from time-pol-beam to pol-beam-time
 *
 * This is an hsaCommand that transposes FRB output from the beamform
 * kernel with format time-pol-beam to pol-beam-time, in order to
 * minimize striding for the next step of upchannelization, which
 * needs time to be fastest varying. 32 blank columns are padded to the output
 * of this transpose for GPU memory access optimization.
 *
 * @requires_kernel    transpose.hasco
 *
 * @par GPU Memory
 * @gpu_mem  beamform_output Input data of size beamform_frame_len
 *     @gpu_mem_type         static
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 * @gpu_mem  bf_output       Output data of size output_frame_len
 *     @gpu_mem_type         static
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   samples_per_data_set Int (default 49152). Number of time samples in a data set
 *
 * @author Cherry Ng
 *
 */

class hsaBeamformTranspose : public hsaCommand {
public:
    /// Constructor, initializes internal variables from config
    hsaBeamformTranspose(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor
    virtual ~hsaBeamformTranspose();

    /// Allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Input length, should be size of num_elem x nsamp x 2
    int32_t beamform_frame_len;
    /// Output length, should be num_elem x (nsamp+32) x 2
    int32_t output_frame_len;

    /// Number of elements, should be 2048
    int32_t _num_elements;
    /// Number of samples
    int32_t _samples_per_data_set;
};

#endif
