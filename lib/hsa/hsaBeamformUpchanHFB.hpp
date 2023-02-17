/**
 * @file
 * @brief Upchannelization of FRB data with HFB data extracted
 *  - hsaBeamformUpchanHFB : public hsaCommand
 */

#ifndef HSA_BEAMFORM_UPCHAN_HFB_H
#define HSA_BEAMFORM_UPCHAN_HFB_H

#include "Config.hpp"          // for Config
#include "bufferContainer.hpp" // for bufferContainer
#include "hsa/hsa.h"           // for hsa_signal_t
#include "hsaCommand.hpp"
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string

/**
 * @class hsaBeamformUpchanHFB
 * @brief hsaCommand to upchannelize and downsample FRB data
 *
 * This is an hsaCommand that launches the kernel (upchannelize_flip) for
 * upchannelizing and downsampling FRB data. First, 128 time samples
 * is traded for 128 upchannelized freq channels via FFT. The freq axis is
 * "flipped" (or more accurately, rolled) so that the higher power is in
 * the middle of the bandpass. Then, every 8 freqs and 3 time samples are
 * averaged in order to downsample to fit the output bandwidth. The 2
 * polarizations are summed to produce a final output in float of
 * 1024 beams x 128 times x 16 freq.
 *
 * Hyper fine beam data is also output at this stage. Before the FRB data is
 * downsampled, both polarizations are summed over all time samples for every
 * frequency. A bandpass filter is applied and the final output is in float of
 * 1024 beams x 128 freqs x 10 samples
 *
 * @requires_kernel    upchannelize_flip.hasco
 *
 * @par GPU Memory
 * @gpu_mem  transposed_output  Input data of size input_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  bf_output          Output data of size output_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  hfb_output         Output data of size output_hfb_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   samples_per_data_set Int (default 49152). Number of time samples in a data set
 * @conf   downsample_time      Int (default 3). Downsample factor in time
 * @conf   downsample_freq      Int (default 8). Downsample factor in freq
 * @conf   num_frb_total_beams  Int (default 1024). Number of total FRB formed beams
 * @conf   factor_upchan        Int (default 128). Upchannelise factor
 *
 * @author Cherry Ng and James Willis
 *
 */

class hsaBeamformUpchanHFB : public hsaCommand {
public:
    /// Constructor, also initializes internal variables from config
    hsaBeamformUpchanHFB(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor
    virtual ~hsaBeamformUpchanHFB();

    /// Allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Input length, (nsamp+32) x num_elem x 2
    size_t input_frame_len;
    /// Output length, num_frb_total_beams x (nsamp/downsample_time/downsample_freq)
    size_t output_frame_len;
    /// Output length, num_frb_total_beams x factor_upchan x nsamp
    size_t output_hfb_frame_len;

    /// Number of elements, should be 2048
    int32_t _num_elements;
    /// Number of samples, needs to be a multiple of 128x3. currently set to 49152
    int32_t _samples_per_data_set;
    /// Downsampling factor for the time axis, set to 3
    int32_t _downsample_time;
    /// Downsampling factor for the freq axis, set to 8
    int32_t _downsample_freq;
    /// Total number of FRB formed beams, should be 1024
    int32_t _num_frb_total_beams;
    /// Upchannelize factor, should be 128
    int32_t _factor_upchan;
};

#endif
