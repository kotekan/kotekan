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
 * @conf   base_dir         String. Directory to read from.
 * @conf   file_name        String. Base filename to read.
 * @conf   file_ext         String. File extension.
 * @conf   end_interrupt    Bool. Interrupt Kotekan if run out of files to read.
 *                          Default false.
 * @conf   prefix_hostname  Bool. Expect the local hostname in the filename, as written
 *                          by rawFileWrite. Default false -- note rawFileWrite defaults
 *                          the same key to TRUE, so a capture written with defaults
 *                          needs prefix_hostname: true here to be found.
 * @conf   frame_period_us  Uint64. Sleep this long after publishing each frame, so a
 *                          capture replays at roughly the rate it was acquired at.
 *                          Set it to one frame's worth of acquisition time: e.g. a frame
 *                          of 199680 samples taken at 20 MS/s is 9984 us.
 *                          Default 0 = unthrottled: frames are read as fast as the
 *                          downstream pipeline drains them, which is the behaviour you
 *                          want unless something in the pipeline is tied to the wall
 *                          clock. Note this is a floor on the frame period, not a rate
 *                          lock -- the read and per-file open overhead add to it, so a
 *                          paced replay runs somewhat slower than 1/frame_period_us
 *                          (0.80x measured on one replay bench). A consumer that
 *                          needs to know where in the file it is should derive that
 *                          from the frame metadata, not from elapsed wall time.
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
    // Realtime replay pacing: microseconds to sleep per frame (0 = unthrottled)
    uint64_t frame_period_us;
};

#endif
