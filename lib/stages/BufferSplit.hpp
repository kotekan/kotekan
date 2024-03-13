#ifndef BUFFER_SPLIT_HPP
#define BUFFER_SPLIT_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string
#include <vector> // for vector

/**
 * @brief Splits the input buffer into multiple output frames in a round robin fashion.
 *
 * Uses zero copy operations if this stage is the only consumer, if not it uses a deep copy.
 * This stage must be the only producer of the output buffers
 *
 * @par Buffers
 * @buffer in_buf The source buffer
 *        @buffer_format Matches the output buffers
 *        @buffer_metadata Matches the output buffers
 *
 * @buffer out_bufs Array of buffers to fill with frames from in_buf
 *        @buffer_format any, but all must be the same type and match in_buf
 *        @buffer_metadata any, but all must be the same type match in_buf
 *
 * @author Andre Renard
 */
class BufferSplit : public kotekan::Stage {
public:
    /// Constructor
    BufferSplit(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~BufferSplit();

    /// Thread for splitting the frames over the output buffers.
    void main_thread() override;

private:
    /// Input buffer
    Buffer* in_buf;

    /// The output buffers to put frames into
    std::vector<Buffer*> out_bufs;
};

#endif
