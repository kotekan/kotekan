#ifndef MERGE_BUFFER_HPP
#define MERGE_BUFFER_HPP

#include <vector>
#include <string>

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
 * @todo Allow dynamic selection of frames (i.e. enable frame rejection)
 *
 * @par Buffers
 * @buffer in_bufs Array of input buffers to merge frames from
 *        @buffer_format any, but all must be the same type.
 *        @buffer_metadata any, but all must be the same type.
 * @buffer out_buf Buffer to get all the frames from the @c in_bufs
 *        @buffer_format Matches the input buffers
 *        @buffer_metadata Matches the input buffers
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
    ~mergeBuffer();

    /// Deprecated
    void apply_config(uint64_t fpga_seq) override;

    /// Thread for merging the frames.
    void main_thread() override;
private:

    /// Array of input buffers to get frames from
    std::vector<struct Buffer *> in_bufs;

    /// The output buffer to put frames into
    struct Buffer *out_buf;
};

#endif