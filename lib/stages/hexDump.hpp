#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"
#include "visUtil.hpp"
#include "kotekanTrackers.hpp"

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
 * @conf    len     Default 128.  The number of bytes to print.
 * @conf    offset  Default 0.    The offset into the frame.
 */
class hexDump : public kotekan::Stage {
public:
    hexDump(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_container);
    ~hexDump();
    void main_thread() override;

private:
    struct Buffer* in_buf;
    int32_t _len;
    int32_t _offset;

    // kotekan trackers example
    std::shared_ptr<StatTracker> tracker_0;
    std::shared_ptr<StatTracker> tracker_1;
};

#endif
