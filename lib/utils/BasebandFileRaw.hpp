/*****************************************
@file
@brief Raw baseband dump output files
- BasebandFileRaw
*****************************************/
#ifndef BASEBAND_FILE_RAW_HPP
#define BASEBAND_FILE_RAW_HPP

#include "BasebandFrameView.hpp"
#include "kotekanLogging.hpp"

#include <string>   // for string
#include <unistd.h> // for ssize_t

/** @brief A CHIME baseband file in raw format
 *
 * The class creates and manages writes to a baseband dump file for a single frequency. It also
 * manages the lock file.
 *
 * The output has the following structure:
 *  - 1st byte is set to `1` if data is present (or is implicitly zero).
 *  - BasebandMetadata struct dump
 *  - baseband buffer frame contents
 *
 * @author Davor Cubranic
 */
class BasebandFileRaw : public kotekan::kotekanLogging {
public:
    BasebandFileRaw(const std::string& name);

    ~BasebandFileRaw();

    ssize_t write_frame(const BasebandFrameView& frame);

    // File name (used for debugging)
    const std::string name;

private:
    ssize_t write_raw(const void* data, size_t nb);

    // File descriptors and related
    int fd;
    std::string lock_filename;
};

#endif // BASEBAND_FILE_RAW_HPP
