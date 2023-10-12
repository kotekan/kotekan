#ifndef HDF5_FILE_WRITE_H
#define HDF5_FILE_WRITE_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t
#include <string>   // for string

/**
 * @class hdf5FileWrite
 * @brief Stream a buffer to disk.
 *
 * This stage is intended to be used for code development and in tests, not in a production
 * pipeline. (In a production pipeline one would have to cycle file names.)
 *
 * @par Buffers:
 * @buffer in_buf Buffer to write to disk.
 *     @buffer_format Any
 *     @buffer_metadata Any
 *
 * @conf base_dir  String. Directory to write into.
 * @conf file_name String. Base filename to write.
 * @conf exit_after_n_frames  Int. Stop writing after this many frames, Default 0 = unlimited
 *       frames.
 * @conf exit_with_n_writers  Int. Exit after this many HDF5 writers finished writing, Default 0 =
 *       unlimited writers.
 *
 * @par Metrics
 * @metric kotekan_hdf5filewrite_write_time_seconds
 *         The write time to write out the last frame.
 *
 * @author Erik Schnetter
 **/
class hdf5FileWrite : public kotekan::Stage {
public:
    hdf5FileWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    virtual ~hdf5FileWrite();
    void main_thread() override;

private:
    struct Buffer* buf;
    std::string _base_dir;
    std::string _file_name;
    uint32_t _exit_after_n_frames;
    uint32_t _exit_with_n_writers;
    // Prefix file name with hostname or not
    bool _prefix_hostname;

    static std::atomic<uint32_t> n_finished;
};

#endif
