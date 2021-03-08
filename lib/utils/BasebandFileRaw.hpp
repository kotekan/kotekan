/*****************************************
@file
@brief Raw baseband dump output files
- BasebandFileRaw
*****************************************/
#ifndef HFB_FILE_RAW_HPP
#define HFB_FILE_RAW_HPP

#include "BasebandFrameView.hpp"
#include "kotekanLogging.hpp"

#include <fstream> // for ofstream
#include <string>  // for string

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

private:
    ssize_t write_raw(const void* data, size_t nb);

    // File descriptors and related
    int fd;
    std::string lock_filename;

    // File name (used for debugging)
    std::string _name;
};

#endif // HFB_FILE_RAW_HPP
