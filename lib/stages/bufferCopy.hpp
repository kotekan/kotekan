#ifndef BUFFER_COPY_HPP
#define BUFFER_COPY_HPP

#include "Stage.hpp"
#include "buffer.h"
#include "visUtil.hpp"

//#include <string>
//#include <tuple>
//#include <vector>

/**
 * @brief Copies frames from one buffer into many buffers.
 *
 *
 * @warning The sizes of the frames must be the same in all buffers, and the
 *       metadata types and underlying pools must also be the same.
 *
 *
 * @par Buffer
 * @buffer in_buf Buffer to copy frames from
 *        @buffer_format Matches the output buffers
 *        @buffer_metadata Matches the output buffers
 * @buffer out_bufs Array of output buffers to copy frames into.
 *                 This is a named array of output buffers in the format:
 *                 - internal_name_0: buffer_name_0
 *                 - internal_name_1: buffer_name_1
 *                 .
 *                 Or it can be provided without internal names:
 *                 - buffer_name_0
 *                 - buffer_name_1
 *                 .
 *                 The use of internal names is only needed if a subclass
 *                 of this function requires internal names to select frame.
 *        @buffer_format any, but all must be the same type.
 *        @buffer_metadata any, but all must be the same type.
 *
 * @author James Willis
 */
class bufferCopy : public kotekan::Stage {
public:
    /// Constructor
    bufferCopy(kotekan::Config& config, const string& unique_name,
                kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~bufferCopy() = default;

    /// Thread for merging the frames.
    void main_thread() override;

protected:
    /// The input buffer to copy frames from
    struct Buffer* in_buf;

    /// Config variables
    /// Flag to copy metadata or not
    bool _copy_metadata;

    /// Array of output buffers to copy frames into
    /// Items are "internal_name", "buffer", "frame_id", "use_memcpy"
    std::vector<std::tuple<std::string, Buffer*, frameID>> out_bufs;
};

#endif
