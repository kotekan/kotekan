#ifndef RAW_FILE_WRITE_H
#define RAW_FILE_WRITE_H

#include "Stage.hpp"
#include "buffer.h"

#include <string>

/**
 * @class rawFileWrite
 * @brief Stream a buffer to disk.
 *
 * @par Buffers:
 * @buffer in_buf Buffer to write to disk.
 *     @buffer_format Any
 *     @buffer_metadata Any
 *
 * @conf base_dir  String. Directory to write into.
 * @conf file_name String. Base filename to write.
 * @conf file_ext  String. File extension.
 *
 * @par Metrics
 * @metric kotekan_rawfilewrite_write_time_seconds
 *         The write time to write out the last frame.
 *
 * @author Andre Renard
 **/
class rawFileWrite : public kotekan::Stage {
public:
    rawFileWrite(kotekan::Config& config, const string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    virtual ~rawFileWrite();
    void main_thread() override;

private:
    struct Buffer* buf;
    std::string base_dir;
    std::string file_name;
    std::string file_ext;
};

#endif
