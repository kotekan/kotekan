#ifndef BUFFER_QUIET_HPP
#define BUFFER_QUIET_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string


/**
 * @brief Waits for a quiet period without new input frames or imminent buffer
 * overflow before releasing of frames of the buffer.
 *
 *
 * @par Buffer
 * @buffer in_buf Buffer to release frames from
 *        @buffer_format Matches the output buffer
 *        @buffer_metadata Matches the output buffer
 * @buffer out_buf Buffer to release frames to
 *        @buffer_format Matches the input buffer
 *        @buffer_metadata Matches the input buffer
 *
 * @conf   copy_frame Bool. Flag to copy or swap frames
 * @conf   quiet_time Double. Seconds in of quiet before staring to relase
 *         frames.
 * @conf   num_reserve_frames Int. Optional. Start releasing frames if buffer
 *         has less than this many empty frames even when there are still
 *         incoming frames.
 *
 * @author Roland Haas
 */
class bufferQuiet : public kotekan::Stage {
public:
    /// Constructor
    bufferQuiet(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~bufferQuiet() = default;

    /// Thread for merging the frames.
    void main_thread() override;

protected:
    /// The input buffer to release frames from
    Buffer* in_buf;

    /// The output buffer to release frames to
    Buffer* out_buf;

    /// Config variables
    /// Flag to copy or swap frames
    bool _copy_frame;

    /// Seconds of quiet before releasing frames.
    double _quiet_time;

    /// Savety marging of empty frames
    int _num_reserve_frames;
};

#endif
