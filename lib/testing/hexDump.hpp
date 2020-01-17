#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#include <unistd.h>

/**
 * @class hexDump
 * @brief Prints out contents of a buffer in hex in an xxd style format
 *
 * @par Buffers
 * @buffer in_buf The buffer to print the contents of.
 *     @buffer_format any
 *     @buffer_metadata any
 *
 * @conf    len     Default 128.  The number of bytes to print.
 * @conf    offset  Ddfault 0.    The offset into the frame.
 */
class hexDump : public kotekan::Stage {
public:
    hexDump(kotekan::Config& config, const string& unique_name,
            kotekan::bufferContainer& buffer_container);
    ~hexDump();
    void main_thread() override;

private:
    struct Buffer* in_buf;
    int32_t _len;
    int32_t _offset;
};

#endif
