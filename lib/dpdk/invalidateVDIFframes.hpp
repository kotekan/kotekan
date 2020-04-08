/**
 * @file invalidateVDIFframes.hpp
 * @brief Stage which sets the invalid bit of VDIF frames based on the contents
 *        of the lost samples buffer.
 * - invalidateVDIFframes : public kotekan::Stage
 */

#ifndef INVALIDATE_VDIF_FRAMES_HPP
#define INVALIDATE_VDIF_FRAMES_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string

/**
 * @brief Invalidate VDIF frames in the @c out_buf based on flags in the @c lost_samples_buf
 *
 * Note the synchronization is a little non-standard here.  We wait for the buffer
 * which contains the flags to be full and register as a consumer on that buffer.
 * Because we know that will only happen once the data buffer is full, we can use
 * that as the synchronization on the data, and so can start zeroing data in the data
 * buffer (which we operate on as a producer).
 *
 * @par Buffers
 * @buffer out_buf Kotekan buffer with VDIF frame data already filled
 *     @buffer_format Array with blocks of @c sample_size byte time samples
 *     @buffer_metadata chimeMetadata
 * @buffer lost_samples_buf Array of flags which indicate if a sample in a given location is lost
 *     @buffer_format Array of flags uint8_t flags which are either 0 (unset) or 1 (set)
 *     @buffer_metadata chimeMetadata
 *
 * @author Andre Renard
 */
class invalidateVDIFframes : public kotekan::Stage {
public:
    /// Standard constructor
    invalidateVDIFframes(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~invalidateVDIFframes();

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

    /// The number of VDIF frames expected
    /// It might be possible to make this dynamic
    const uint32_t num_elements = 2;

    /// The size of each VDIF frame
    const uint32_t vdif_frame_size = 1056; // 32 header + 1024 data
};

#endif /* ZERO_SAMPLES_HPP */
