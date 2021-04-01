#ifndef SAMPLE_STAGE_H
#define SAMPLE_STAGE_H

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

/**
 * @class SampleStage
 * @brief A skeleton consumer stage
 *
 * @par Buffers
 * @buffer in_buf The buffer to process the contents of.
 *      @buffer_format any
 *      @buffer_metadata any
 *
 * @conf    len     Default 128.    The number of bytes to process.
 * @conf    offset  Default 0.      The offset into the frame.
 */
class SampleStage : public kotekan::Stage {
public:
    SampleStage(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    virtual ~SampleStage();
    void main_thread() override;

private:
    struct Buffer* in_buf;
    int32_t _len;
    int32_t _offset;
};

#endif /* SAMPLE_STAGE_H */
