#ifndef DROP_ALL_FRAMES_H
#define DROP_ALL_FRAMES_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer


/**
 * @class dropAllFrames
 * @brief Immediately consumes all seen frames without processing.
 *
 * @par Buffers
 * @buffer in_buf         Buffer to consume
 *
 * @conf  max_frames      Int. If > 0, trigger Kotekan exit after dropping this many frames.
 *
 * @author Geoffrey Ryan
 */
class dropAllFrames : public kotekan::Stage {
public:
    dropAllFrames(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~dropAllFrames(){};
    void main_thread() override;

private:
    Buffer* in_buf;
    const int max_frames;
};

#endif
