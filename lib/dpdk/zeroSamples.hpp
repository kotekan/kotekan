#ifndef ZERO_SAMPLES_HPP
#define ZERO_SAMPLES_HPP

#include "KotekanProcess.hpp"

/**
 * @brief Zeros samples in the @c out_buf based on flags in the @lost_samples_buf
 *
 * Note the synchronization is a little non-standard here.  We wait for the buffer
 * which contains the flags to be full and register as a consumer on that buffer.
 * Because we know that will only happen once the data buffer is full, we can use
 * that as the synchronization on the data, and so can start zeroing data in the data
 * buffer (which we operate on as a producer).
 *
 * @par Buffers
 * @buffer out_buf Kotekan buffer with network data already filled
 *     @buffer_format Array with blocks of @C sample_size byte time samples
 *     @buffer_metadata chimeMetadata
 * @buffer lost_samples_buf Array of flags which indicate if a sample in a given location is lost
 *     @buffer_format Array of flags uint8_t flags which are either 0 (unset) or 1 (set)
 *     @buffer_metadata chimeMetadata
 *
 * @config  sample_size   Int. Default 2048.  The size of the time samples in @c out_buf
 *
 * @author Andre Renard
 */
class zeroSamples : public KotekanProcess {
public:

    /// Standard constructor
    zeroSamples(Config& config, const string& unique_name,
             bufferContainer &buffer_container);

    /// Destructor
    ~zeroSamples();

    /// Main thead which zeros the data from the lost_samples_buf
    void main_thread();

    /// To be removed later
    void apply_config(uint64_t fpga_seq) {};

private:

    /// The buffer with the network data
    struct Buffer * out_buf;

    /// The buffer with the array of flags indicating lost data.
    struct Buffer * lost_samples_buf;

    /// Current ID for out_buf
    int32_t out_buf_frame_id = 0;

    /// Current
    int32_t lost_samples_buf_frame_id = 0;

    /// The size of the time samples in @c out_buf
    uint32_t sample_size;
};

#endif /* ZERO_SAMPLES_HPP */