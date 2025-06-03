/*****************************************
@file
@brief Raw baseband dump output files
- BasebandFileRaw
*****************************************/
#ifndef BASEBAND_FILE_RAW_HPP
#define BASEBAND_FILE_RAW_HPP

#include "BasebandFrameView.hpp" // for BasebandFrameView
#include "BasebandMetadata.hpp"  // for BasebandMetadata
#include "kotekanLogging.hpp"    // for kotekanLogging

#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string

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
    BasebandFileRaw(const std::string& name, const uint32_t frame_size);
    BasebandFileRaw() = delete;

    ~BasebandFileRaw();

    int32_t write_frame(const BasebandFrameView& frame);

    // File name (used for debugging)
    const std::string name;

private:
    // The size of each frame in the file (metadata + data).
    uint64_t frame_size;
    bool file_corrupt;

    // File descriptors and related
    int fd;
    std::string lock_filename;

    uint64_t write_index;
    uint32_t metadata_size;
};

#endif // BASEBAND_FILE_RAW_HPP
