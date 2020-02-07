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
 * summing HFB data across all samples. Excludes samples from
 * the sum where there is >=1 lost samples. Final output in float of
 * 1024 beams x 128 freq.
 *
 * @requires_kernel   sum_hfb.hasco
 *
 * @par GPU Memory
 * @gpu_mem  hfb_output  Input data of size input_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 * @gpu_mem  compressed_lost_samples Array indicating if a given timestep was zeroed, size:
 *           samples_per_data_set / num_sub_freqs
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c uint8_t
 *     @gpu_mem_metadata     chimeMetadata
 * @gpu_mem  hfb_sum_output     Output data of size output_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *
 * @conf   num_frb_total_beams  Int (default 1024). Number of total FRB formed beams
 * @conf   num_sub_freqs  Int (default 128). Number of sub frequencies
 * @conf   samples_per_data_set Int. Number of time samples in a data set.
 *
 * @author James Willis
 *
 */

class hsaBeamformHFBSum : public hsaCommand {
public:
    /// Constructor, also initializes internal variables from config
    hsaBeamformHFBSum(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor
    virtual ~hsaBeamformHFBSum();

    /// Allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Input length, num_frb_total_beams x num_sub_freqs x 10
    uint32_t input_frame_len;
    /// Output length, num_frb_total_beams x num_sub_freqs
    uint32_t output_frame_len;
    /// Length of the compressed lost samples frame
    uint32_t compressed_lost_samples_frame_len;

    /// Total number of FRB formed beams, should be 1024
    uint32_t _num_frb_total_beams;
    /// Total number of sub frequencies, should be 128
    uint32_t _num_sub_freqs;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    /// Number of samples per frame
    uint32_t _num_samples;
};

#endif
