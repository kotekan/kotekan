/**
 * @file
 * @brief foo bar
 *  - rawFileRead : public kotekan:Stage
 */

#ifndef RAW_FILE_READ_H
#define RAW_FILE_READ_H

#include "Stage.hpp"
#include "buffer.h"

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

/**
 * @class rawFileRead
 * @brief Read and stream a dumped buffer
 *
 * @par Buffers
 * @buffer buf The data read from the raw file.
 *         @buffer_format   Any
 *         @buffer_metadata Any
 *
 * @conf   base_dir         String. Directory to read from.
 * @conf   file_name        String. Base filename to read.
 * @conf   file_ext         String. File extension.
 * @conf   end_interrupt    Bool. Interrupt Kotekan if run out of files to read
 */
class rawFileRead : public kotekan::Stage {
public:
    rawFileRead(kotekan::Config& config, const string& unique_name,
                kotekan::bufferContainer& buffer_container);
    virtual ~rawFileRead();
    void main_thread() override;

private:
    struct Buffer* buf;
    std::string base_dir;
    std::string file_name;
    std::string file_ext;
    // Interrupt Kotekan if run out of files to read
    bool end_interrupt;
};

#endif
