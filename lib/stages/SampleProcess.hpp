#ifndef SAMPLEPROCESS_H
#define SAMPLEPROCESS_H

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

/**
 * @class SampleProcess
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

class SampleProcess : public kotekan::Stage {
public:
    SampleProcess(kotekan::Config& config, const string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    virtual ~SampleProcess();
    void main_thread() override;

private:
    struct Buffer* in_buf;
    int32_t _len;
    int32_t _offset;
};

#endif /* SAMPLEPROCESS_H */
