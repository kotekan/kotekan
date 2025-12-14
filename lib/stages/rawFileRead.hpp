/**
 * @file
 * @brief Stage to read in frames from file and inject them into a pipeline buffer.
 *  - rawFileRead : public kotekan:Stage
 */

#ifndef RAW_FILE_READ_H
#define RAW_FILE_READ_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string, basic_string

/**
 * @class rawFileRead
 * @brief Read and stream a dumped buffer
 *
 * @par Buffers
 * @buffer buf The data read from the raw file.
 *         @buffer_format   Any
 *         @buffer_metadata Any
 *
 * @conf   buf              String. Output buffer name.
 * @conf   base_dir         String. Directory to read from.
 * @conf   file_name        String. Base filename to read (numeric index appended).
 * @conf   file_ext         String. File extension.
 * @conf   prefix_hostname  Bool (default false). Prefix filename with hostname.
 * @conf   end_interrupt    Bool (default false). Interrupt kotekan when files are exhausted.
 *
 * @par Example
 * @code
 * rawFileRead:
 *   buf: vis_in
 *   base_dir: /data/raw
 *   file_name: vis_dump
 *   file_ext: raw
 *   prefix_hostname: false
 *   end_interrupt: true
 * @endcode
 */
class rawFileRead : public kotekan::Stage {
public:
    rawFileRead(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    virtual ~rawFileRead();
    void main_thread() override;

private:
    Buffer* buf;
    std::string base_dir;
    std::string file_name;
    std::string file_ext;
    // Read file with a prefixed hostname or not
    bool prefix_hostname;
    // Interrupt Kotekan if run out of files to read
    bool end_interrupt;
};

#endif
