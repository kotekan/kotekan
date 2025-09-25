#ifndef GPU_SIMULATE_N2K_PLEXP_HPP
#define GPU_SIMULATE_N2K_PLEXP_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <cstdint>  // for int32_t
#include <string>   // for string

/**
 * @brief Perform on CPU the equivalent of the CudaCorrelator stage:
 * N2 PL 1 bit Correlator.
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_n2k.yaml`.
 *
 * This stage performs the 1-bit element-element correlation of the packet loss
 * (PL) mask and applies the RFI mask, resulting in the counts of good samples
 * contributing to each visibility matrix element.
 *
 * The PL mask is input in its expanded form, not downsampled in time or
 * frequency, but downsampled in element (really: dish) by a factor of 8. For
 * GPU reasons, the time axis is split into fast (64 time samples, each 1 bit)
 * and slow (length samples_per_data_set / 64) axes.
 *
 * The RFI mask is applied over all elements (it has no element axis) and its 
 * time axis is also split into fast (1024 time samples) and slow 
 * (samples_per_data_set / 1024) axes.
 *
 * In terms of 1-bit bools their shapes are:
 * PL:  bool1    [samples_per_data_set/64, num_local_freq, num_elements/8, 64]
 * RFI: bool1    [samples_per_data_set/1024, num_local_freq, 1024]
 *
 * This stage accesses these as u64's, giving them shapes:
 * PL:  u64     [samples_per_data_set/64, num_local_freq, num_elements/8]
 * RFI: u64     [samples_per_data_set/1024, num_local_freq, 16]
 *
 * The counts matrix is output in a form like the visibility matrix: it is
 * tiled into 8 x 8 blocks, and the lower diagonal blocks are returned as [B00,
 * B10, B11, B20, B21, ...].  There are lin_blocks = num_elements / 8 / 8
 * horizontally and vertically, resulting in num_blocks = lin_blocks
 *  * (lin_blocks + 1) / 2 total 8 x 8 blocks returned.  Each entry is an int32.
 *
 * Correlations are performed over sub_integration_ntime time samples, resulting
 * in num_integrations = samples_per_data_set / sub_integration_ntime outputs
 * in time.
 *
 * The output counts matrix will then have shape:
 *      [num_integrations, num_local_freq, num_blocks, 8, 8]
 *
 * @par Buffers
 * @buffer in_plmask_buf The input expanded packet loss mask.
 *      @buffer_format uint64 bitmask.
 *      @buffer_shape [samples_per_data_set / 64, num_local_freq,
 *          num_elements / 8]
 *      @buffer_metadata chordMetadata
 *
 * @buffer in_rfimask_buf The input RFI mask.
 *      @buffer_format uint64 bitmask.
 *      @buffer_shape [samples_per_data_set / 1024, num_local_freq, 16]
 *      @buffer_metadata chordMetadata
 * 
 * @buffer out_buf  The output counts matrix.
 *      @buffer_format int32
 *      @buffer_shape [num_integrations, num_local_freq, num_blocks, 8, 8]
 *      @buffer_metadata chordMetadata, time_downsample_fpga[] = sub_integration_ntime
 *
 * @conf    num_elements          Int.  Number of feeds or (antennas x
 *      polarizations).
 * @conf    num_local_freq        Int.  Number of frequencies.
 * @conf    samples_per_data_set  Int.  Number of samples per frame.
 * @conf    sub_integration_ntime Int.  Number of samples to correlate.
 */
class gpuSimulateN2kPL1bitCorr : public kotekan::Stage {
public:
    gpuSimulateN2kPL1bitCorr(kotekan::Config& config, const std::string& unique_name,
                             kotekan::bufferContainer& buffer_container);
    ~gpuSimulateN2kPL1bitCorr();
    void main_thread() override;

private:
    Buffer* input_plmask_buf;
    Buffer* input_rfimask_buf;
    Buffer* output_buf;

    int32_t _blocksize; // Always equal to 8.

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _sub_integration_ntime;
};

#endif
