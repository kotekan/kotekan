#ifndef RAW_FILE_WRITE_H
#define RAW_FILE_WRITE_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t
#include <string>   // for string

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
 * @conf num_frames_per_file Int. No of frames to write into a single file.
 * @conf exit_after_n_files  Int. Stop writing after this many files, Default 0 = unlimited files.
 *
 * @par Metrics
 * @metric kotekan_rawfilewrite_write_time_seconds
 *         The write time to write out the last frame.
 *
 * @author Andre Renard
 **/
class rawFileWrite : public kotekan::Stage {
public:
    rawFileWrite(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    virtual ~rawFileWrite();
    void main_thread() override;

private:
    struct Buffer* buf;
    std::string _base_dir;
    std::string _file_name;
    std::string _file_ext;
    uint32_t _num_frames_per_file;
    uint32_t _exit_after_n_files;
    // Prefix file name with hostname or not
    bool _prefix_hostname;
};

#endif
