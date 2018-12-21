/*
 * @file rfiVDIF.hpp
 * @brief Contains a general VDIF kurtosis estimator kotekan process.
 * - rfiVDIF : public KotekanProcess
 */
#ifndef VDIF_RFI_H
#define VDIF_RFI_H

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "vdif_functions.h"

/*
 * @class rfiVDIF
 * @brief Producer and consumer ``KotekanProcess`` which consumes input VDIF data and
 * computes spectral kurtosis estimates.
 *
 * This process is a spectral kurtosis estimator that works on any general kotekan buffer containing
 * VDIF data. This process move block by block through the VDIF data while computing and integrating
 * power estimates. Once the desired integration length is over, the process does one of two things
 * (as specified by the user):
 *
 * 1) The process combines the sums across the element axis and kurtosis values are calculated on
 *the new sum
 *
 * 2) The process computes kurtosis values for each frequency-element pair
 *
 * There are advantages to both options, however the first is currently heavily favoured by other
 *processes.
 *
 * @par Buffers
 * @buffer vdif_in The kotekan buffer which conatins input VDIF data
 *	@buffer_format	Array of bytes (uint8_t) which conatin VDIF header and data
 *	@buffer_metadata chimeMetadata
 * @buffer rfi_out The kotekan buffer to be filled with kurtosis estimates
 *	@buffer_format Array of floats
 *	@buffer_metadata none
 *
 * @conf rfi_combined   Bool (default true). A flag indicating whether or not to combine data inputs
 * @conf sk_step        Int (default 256). The number of timesteps per kurtosis estimate
 *
 * @author Jacob Taylor
 */
class rfiVDIF : public KotekanProcess {
public:
    // Constructor, initializes class, sets up config
    rfiVDIF(Config& config, const string& unique_name, bufferContainer& buffer_containter);
    // Deconstructor, cleans up, does nothing
    ~rfiVDIF();

    // Main thread: reads vdif_in buffer, computes kurtosis values, fills rfi_out buffer
    void main_thread() override;

private:
    // Kotekan Buffer for VDIF input
    struct Buffer* buf_in;
    // Kotekan Buffer for kurtosis output
    struct Buffer* buf_out;
    // General config Paramaters
    // Number of elements in the input data
    uint32_t _num_elements;
    // Number of total frequencies in the input data
    uint32_t _num_local_freq;
    // Number of timesteps in a frame of input data
    uint32_t _samples_per_data_set;
    // RFI config parameters
    // Flag for whether or not to combine inputs in kurtosis estimates
    bool _rfi_combined;
    // Number of timesteps per kurtosis value
    uint32_t _sk_step;
};

#endif
