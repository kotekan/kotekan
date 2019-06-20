/**
 * @file
 * @brief Sum HFB data over each sample
 *  - hsaBeamformHFBSum : public hsaCommand
 */

#ifndef HSA_BEAMFORM_HFB_SUM_H
#define HSA_BEAMFORM_HFB_SUM_H

#include "hsaCommand.hpp"

/**
 * @class hsaBeamformHFBSum
 * @brief hsaCommand to upchannelize and downsample FRB data
 *
 * This is an hsaCommand that launches the kernel (sum_hfb) for
 * summing HFB data across all samples. Final output in float of
 * 1024 beams x 128 freq.
 *
 * @requires_kernel   sum_hfb.hasco
 *
 * @par GPU Memory
 * @gpu_mem  hfb_output  Input data of size input_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  hfb_sum_output     Output data of size output_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   samples_per_data_set Int (default 49152). Number of time samples in a data set
 * @conf   num_frb_total_beams  Int (default 1024). Number of total FRB formed beams
 *
 * @todo   Check that the 16 freq axis has the correct orientation
 *
 * @author James Willis
 *
 */

class hsaBeamformHFBSum : public hsaCommand {
public:
    /// Constructor, also initializes internal variables from config
    hsaBeamformHFBSum(kotekan::Config& config, const string& unique_name,
                      kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor
    virtual ~hsaBeamformHFBSum();

    /// Allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Input length, num_frb_total_beams x 128 x 10
    int32_t input_frame_len;
    /// Output length, num_frb_total_beams x 128
    int32_t output_frame_len;

    /// Number of elements, should be 2048
    int32_t _num_elements;
    /// Number of samples, needs to be a multiple of 128x3. currently set to 49152
    int32_t _samples_per_data_set;
    /// Total number of FRB formed beams, should be 1024
    int32_t _num_frb_total_beams;
};

#endif
