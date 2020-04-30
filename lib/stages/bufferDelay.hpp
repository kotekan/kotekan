#ifndef BUFFER_DELAY_HPP
#define BUFFER_DELAY_HPP

#include "Config.hpp" // for Config
#include "Stage.hpp"
#include "buffer.h"
#include "bufferContainer.hpp" // for bufferContainer
#include "visUtil.hpp"

#include <string> // for string
#include <tuple>  // for tuple
#include <vector> // for vector

/**
 * @brief Delays release of frames until the buffer is sufficiently full.
 *
 *
 * @par Buffer
 * @buffer in_buf Buffer to delay frames from
 *        @buffer_format Matches the output buffers
 *        @buffer_metadata Matches the output buffers
 * @buffer out_buf Buffer to release frames to
 *        @buffer_format Matches the input buffers
 *        @buffer_metadata Matches the input buffers
 *
 * * @author James Willis
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
    struct Buffer* in_buf;
    
    /// The output buffer to release frames to
    struct Buffer* out_buf;

    /// Config variables
    /// Flag to copy or swap frames
    bool _copy_frame;
    
    /// No. of frames to hold in buffer before releasing one
    uint32_t _num_frames_to_hold;

};

#endif
