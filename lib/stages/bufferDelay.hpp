#ifndef BUFFER_DELAY_HPP
#define BUFFER_DELAY_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string


/**
 * @brief Delays release of frames until the buffer is sufficiently full.
 *
 *
 * @par Buffer
 * @buffer in_buf Buffer to delay frames from
 *        @buffer_format Matches the output buffer
 *        @buffer_metadata Matches the output buffer
 * @buffer out_buf Buffer to release frames to
 *        @buffer_format Matches the input buffer
 *        @buffer_metadata Matches the input buffer
 *
 * @conf   copy_frame Bool. Flag to copy or swap frames
 * @conf   num_frames_to_hold Int. No. of frames to hold in buffer before releasing one
 *
 * @author James Willis
 */
class bufferDelay : public kotekan::Stage {
public:
    /// Constructor
    bufferDelay(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~bufferDelay() = default;

    /// Thread for merging the frames.
    void main_thread() override;

protected:
    /// The input buffer to delay frames from
    Buffer* in_buf;

    /// The output buffer to release frames to
    Buffer* out_buf;

    /// Config variables
    /// Flag to copy or swap frames
    bool _copy_frame;

    /// No. of frames to hold in buffer before releasing one
    uint32_t _num_frames_to_hold;
};

#endif
