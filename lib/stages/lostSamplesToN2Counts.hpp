#ifndef LOST_SAMPLES_TO_N2_COUNTS
#define LOST_SAMPLES_TO_N2_COUNTS

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string
#include <vector> // for vector

/**
 * @brief Converts CHIME lost samples and rfi masks to counts expected by N2Accumulate
 *
 * This stage will accumulate the logical AND of the lost samples and RFI masks, based
 * on the convention that a value of 1 correpsonds to a good sample and 0 to a flagged
 * samples in the RFI mask, and a value of 1 corresponds to a _missing_ sample in the
 * lost samples mask.
 *
 * This stage is specific to CHIME - others should likely use cudaPL1bitCorrelator and
 * do this operation on GPU. Because CHIME's lost sample (equivalent to packet loss) mask
 * is one-dimensions (i.e., all elements are flagged if any packet is lost for a given
 * frequency and time), only the 0th element index is actually set. Thus, this requires
 * that the N2Accumulate config value `_packet_loss_is_scalar` is True.
 *
 * @par Buffers
 * @buffer rfi_mask_buf Array of flags indicating samples affected by RFI
 *     @buffer_format uint1x8
 *     @buffer_shape     As uint8: [num_times / 8 / 128][freq][128]
 *                       As uint1: [num_times / 8 / 128][freq][8 * 128]
 *     @buffer_metadata chordMetadata
 * @buffer lost_samples_buf Array of flags which indicate if a sample in a given location is lost
 *     @buffer_format uint8
 *     @buffer_shape [num_times]
 *     @buffer_metadata chordMetadata
 * @buffer n2k_counts_buf Array counting the number of good samples per subintegration
 *     @buffer_format int32
 *     @buffer_shape [num_subintegrations][freq][tiles][blocksize][blocksize]
 *     @buffer_metadata chordMetadata
 *
 * @conf num_elements                        Int. Number of instrument elements. In CHIME, this
 *                                           is the number of inputs times polarisations.
 * @conf num_n2k_freq                        Int. The number of frequencies in each GPU frame.
 * @conf samples_per_data_set                Int. The number of FPGA samples per frame.
 * @conf sub_integration_ntime               Int. The number of FPGA samples to accumulate for.
 *                                           This must correspond to the number of samples
 *                                           accumulated by the correlator.
 *
 * @author Liam Gray
 */
class lostSamplesToN2Counts : public kotekan::Stage {
public:
    // Standard constructor
    lostSamplesToN2Counts(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);

    // Destructor
    ~lostSamplesToN2Counts();

    // Main thread accumulates lost_samples and rfi_mask
    void main_thread() override;

private:
    // RFI mask buffer
    Buffer* rfi_mask_buf;

    // PL mask buffer. One buffer per frequency
    std::vector<Buffer*> lost_samples_buffers;

    // Buffer with the reshaped counts array
    Buffer* n2k_counts_buf;

    // Set from config file
    size_t num_elements;          // number of instrument elements
    size_t num_n2k_freq;          // number of freqs per N2 frame
    size_t samples_per_data_set;  // number of fpga samples per input buffer
    size_t sub_integration_ntime; // number of fpga samples per subintegration

    // Private attrs for internal use
    size_t _num_freq_per_lost_samples_buffer;
    size_t _num_subintegrations; // tau_bar
    size_t _counts_ntiles;
    size_t _counts_stride;
};

#endif /* LOST_SAMPLES_TO_N2_COUNTS */
