#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string

/**
 * @class hexDump
 * @brief Prints out contents of a buffer in hex in an xxd style format
 *
 * @par Buffers
 * @buffer in_buf The buffer to print the contents of.
 *     @buffer_format any
 *     @buffer_metadata any
 *
 * @conf in_buf String. Input buffer to inspect.
 * @conf len    Int (default 128). Number of bytes to print.
 * @conf offset Int (default 0). Byte offset into the frame.
 *
 * @par Example
 * @code
 * hexDump:
 *   in_buf: raw_in
 *   len: 256
 *   offset: 0
 * @endcode
 */
class hexDump : public kotekan::Stage {
public:
    hexDump(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_container);
    ~hexDump();
    void main_thread() override;

private:
    Buffer* in_buf;
    int32_t _len;
    int32_t _offset;
};

#endif
