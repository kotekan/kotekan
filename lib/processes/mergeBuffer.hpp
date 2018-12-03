#ifndef MERGE_BUFFER_HPP
#define MERGE_BUFFER_HPP

#include <vector>
#include <string>
#include <tuple>

#include "buffer.h"
#include "KotekanProcess.hpp"

/**
 * @brief Merges frames from many buffers into one buffer.
 *
 * Merges the frames in order in a round-robin pattern. This means
 * the frame arrival rate must be the same for all input buffers.
 *
 * This process uses a frame swapping model, which requires that it be
 * the only consumer of the input buffers, and the only producer of the
 * output buffer.  The upside of this is it is zero copy operation.
 *
 * @warn The sizes of the frames must be the same in all buffers, and the
 *       metadata types and underlying pools must also be the same.
 *
 * @todo Remove the restriction that all buffers have the same arrival rates.
 *       This involves being able to select the next available frame, which
 *       requires changes to the buffer API.
 *
 * @par Buffers
 * @buffer in_bufs Array of input buffers to merge frames from.
 *                 This is a named array of input buffers in the format:
 *                 - internal_name_0: buffer_name_0
 *                 - internal_name_1: buffer_name_1
 *        @buffer_format any, but all must be the same type.
 *        @buffer_metadata any, but all must be the same type.
 * @buffer out_buf Buffer to get all the frames from the @c in_bufs
 *        @buffer_format Matches the input buffers
 *        @buffer_metadata Matches the input buffers
 *
 * @config timeout    Double. Default -1.0   Timeout in seconds for waiting for a
 *                    frame on any of the input buffers.
 *                    Set to a negative number for no timeout.
 *
 * @author Andre Renard
 */
class mergeBuffer : public KotekanProcess {
public:

    /// Constructor
    mergeBuffer(Config& config,
                const string& unique_name,
                bufferContainer &buffer_container);

    /// Destructor
    ~mergeBuffer() = default;

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
    virtual bool select_frame(const std::string &internal_name,
                              Buffer * in_buf, uint32_t frame_id);

    /// Thread for merging the frames.
    void main_thread() override;
private:

    /// Array of input buffers to get frames from
    /// Items are "internal_name", "buffer", "frame_id"
    std::vector<std::tuple<std::string, Buffer *, uint32_t>> in_bufs;

    /// The output buffer to put frames into
    struct Buffer *out_buf;

    /// The in seconds to wait for a new frame on one of the input buffers.
    double _timeout;
};

#endif