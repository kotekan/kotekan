#ifndef ZERO_SAMPLES_HPP
#define ZERO_SAMPLES_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t, uint32_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @brief Zeros samples in the @c out_buf based on flags in the @c lost_samples_buf
 *
 * Note the synchronization is a little non-standard here.  We wait for the buffer
 * which contains the flags to be full and register as a consumer on that buffer.
 * Because we know that will only happen once the data buffer is full, we can use
 * that as the synchronization on the data, and so can start zeroing data in the data
 * buffer (which we operate on as a producer).
 *
 * @par Buffers
 * @buffer out_buf Kotekan buffer with network data already filled
 *     @buffer_format Array with blocks of @c sample_size byte time samples
 *     @buffer_metadata chimeMetadata
 * @buffer lost_samples_buf Array of flags which indicate if a sample in a given location is lost
 *     @buffer_format Array of flags uint8_t flags which are either 0 (unset) or 1 (set)
 *     @buffer_metadata chimeMetadata
 *
 * @conf  sample_size               Int. Default 2048.  The size of the time samples in @c out_buf
 *
 * @conf  duplicate_ls_buffer       Bool. Default False. Whether or not to dupliate the lost
 * samples buf
 *
 * @conf  out_lost_sample_buffers   Buffers to hold the duplicated lost samples buffer. For
 * example: out_lost_sample_buffers:
 *                                        - lost_samples_buffer_0
 *                                        - lost_samples_buffer_1
 *                                        - lost_samples_buffer_2
 *                                        - lost_samples_buffer_3
 *
 * @conf  zero_value                Int Default 0x88  The 8-bit value to write overtop of bad data
 *                                    For offset encoded post PFB data this is 0x88
 *
 * @author Andre Renard
 */
class zeroSamples : public kotekan::Stage {
public:
    /// Standard constructor
    zeroSamples(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~zeroSamples();

    /// Main thead which zeros the data from the lost_samples_buf
    void main_thread() override;

private:
    /// The buffer with the network data
    struct Buffer* out_buf;

    /// The buffer with the array of flags indicating lost data.
    struct Buffer* lost_samples_buf;

    /// Current ID for out_buf
    int32_t out_buf_frame_id = 0;

    /// Current
    int32_t lost_samples_buf_frame_id = 0;

    /// The size of the time samples in @c out_buf
    uint32_t sample_size;

    /// Whether or not to duplicate the lost samples buffer
    bool _duplicate_ls_buffer;

    /// Vector to hold all duplicate lost sample buffers
    std::vector<Buffer*> out_lost_sample_bufs;

    /// The int8 "zero" value
    uint8_t zero_value;
};

#endif /* ZERO_SAMPLES_HPP */
