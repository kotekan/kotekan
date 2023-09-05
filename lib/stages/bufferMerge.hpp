#ifndef BUFFER_MERGE_HPP
#define BUFFER_MERGE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "visUtil.hpp"         // for frameID

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <tuple>    // for tuple
#include <vector>   // for vector

/**
 * @brief Merges frames from many buffers into one buffer.
 *
 * Merges the frames in order in a round-robin pattern. This means
 * the frame arrival rate must be the same for all input buffers.
 *
 * If this stage is the only comsumer of the input buffers then
 * the operation is zero-copy, it just swaps the frames.  However
 * if there is more than one comsumer on the input buffer then it
 * does a full memcpy of the frame.
 *
 * @warning The sizes of the frames must be the same in all buffers, and the
 *       metadata types and underlying pools must also be the same.
 *
 *
 * @par Buffers
 * @buffer in_bufs Array of input buffers to merge frames from.
 *                 This is a named array of input buffers in the format:
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
 * @buffer out_buf Buffer to get all the frames from the @c in_bufs
 *        @buffer_format Matches the input buffers
 *        @buffer_metadata Matches the input buffers
 *
 * @conf timeout       Double. Default -1.0   Timeout in seconds for waiting
 *                       for a frame on any of the input buffers.
 *                       Set to a negative number for no timeout.
 * @conf force_copy    Bool. Default false.  Forces a copy of the frames into the @c out_buf
 *
 * @author Andre Renard
 */
class bufferMerge : public kotekan::Stage {
public:
    /// Constructor
    bufferMerge(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~bufferMerge() = default;

    /**
     * @brief This function can be overridden to implement different selection
     *        methods.  By default it just accepts everything.
     *
     * @param internal_name The name given in the config to the buffer
     *                      (not the buffer name)
     * @param in_buf        The buffer pointers
     * @param frame_id      The frame_id for the current frame to accept or reject.
     *
     * @return true if the frame should be swapped into the out_buf, false if
     *         it should be dropped.
     */
    virtual bool select_frame(const std::string& internal_name, Buffer* in_buf, uint32_t frame_id);

    /// Thread for merging the frames.
    void main_thread() override;

protected:
    /// Array of input buffers to get frames from
    /// Items are "internal_name", "buffer", "frame_id", "use_memcpy"
    std::vector<std::tuple<std::string, Buffer*, frameID>> in_bufs;

    /// The output buffer to put frames into
    struct Buffer* out_buf;

    /// The in seconds to wait for a new frame on one of the input buffers.
    double _timeout;

    /// For a copy of the frame
    bool _force_copy;
};

#endif
