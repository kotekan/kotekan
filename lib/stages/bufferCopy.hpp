#ifndef BUFFER_COPY_HPP
#define BUFFER_COPY_HPP

#include "Config.hpp" // for Config
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp" // for bufferContainer
#include "visUtil.hpp"

#include <string> // for string
#include <tuple>  // for tuple
#include <vector> // for vector

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
 *                 - buffer_name_0
 *                 - buffer_name_1
 *                 .
 *        @buffer_format any, but all must be the same type.
 *        @buffer_metadata any, but all must be the same type.
 *
 * @author James Willis
 */
class bufferCopy : public kotekan::Stage {
public:
    /// Constructor
    bufferCopy(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~bufferCopy() = default;

    /// Thread for merging the frames.
    void main_thread() override;

protected:
    /// The input buffer to copy frames from
    Buffer* in_buf;

    /// Config variables
    /// Flag to copy metadata or not
    bool _copy_metadata;

    /// Array of output buffers to copy frames into
    /// Items are "internal_name", "buffer", "frame_id", "use_memcpy"
    std::vector<std::tuple<std::string, Buffer*, frameID>> out_bufs;
};

#endif
