#ifndef BUFFER_DEDUP_H
#define BUFFER_DEDUP_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class bufferDedup
 * @brief Forward only frames whose contents differ from the last forwarded frame.
 *
 * The first frame after startup is always forwarded, so a record of the stream begins
 * with a known value. Metadata is passed through unchanged, so a forwarded frame keeps
 * e.g. the FPGA sequence number of the input frame that carried the change.
 *
 * @par Buffers
 * @buffer in_buf   Frames to compare; consumed at the full input rate.
 *      @buffer_format any
 *      @buffer_metadata any
 * @buffer out_buf  Frames that differed from the previously forwarded one.
 *      @buffer_format matches in_buf
 *      @buffer_metadata matches in_buf
 *
 * @conf  resend_after_frames  Int. Default 0 (disabled). If > 0, forward an unchanged
 *                             frame after this many were suppressed, so a downstream
 *                             consumer that can drop frames (e.g. a bufferSend link
 *                             that was down) recovers the current contents.
 *
 * @author James Mertens
 */
class bufferDedup : public kotekan::Stage {
public:
    bufferDedup(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~bufferDedup() = default;
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    const int resend_after_frames;
    /// Contents of the last forwarded frame; empty until one is forwarded.
    std::vector<uint8_t> last_sent;
};

#endif
