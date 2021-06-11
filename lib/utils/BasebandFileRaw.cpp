#include "BasebandFileRaw.hpp"

#include "BasebandMetadata.hpp" // for BasebandMetadata
#include "visFile.hpp"          // for create_lockfile

#include "fmt.hpp" // for format, fmt

#include <cstdint>    // for uint32_t
#include <cstdio>     // for remove
#include <errno.h>    // for errno
#include <fcntl.h>    // for open, O_CREAT, O_WRONLY
#include <stdexcept>  // for runtime_error
#include <string.h>   // for strerror
#include <sys/stat.h> // for S_IRGRP, S_IROTH, S_IRUSR, S_IWGRP, S_IWUSR


BasebandFileRaw::BasebandFileRaw(const std::string& name) : name(name) {

    DEBUG("Creating new baseband file {}", name);

    // Create the lock file and then open other files
    lock_filename = create_lockfile(name);
    if ((fd = open((name + ".data").c_str(), O_CREAT | O_WRONLY,
                   S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH))
        == -1) {
        throw std::runtime_error(
            fmt::format(fmt("Failed to open file {:s}.data: {:s}."), name, strerror(errno)));
    }
}

BasebandFileRaw::~BasebandFileRaw() {
    DEBUG("Closing baseband file {}: {}", name, fd);
    // TODO: final sync of data file.
    close(fd);

    std::remove(lock_filename.c_str());
}


ssize_t BasebandFileRaw::write_frame(const BasebandFrameView& frame) {
    // Write the frame metadata
    const uint32_t metadata_size = sizeof(BasebandMetadata);
    write_raw(frame.metadata(), metadata_size);

    // Write the contents of the buffer frame to disk.
    return write_raw(frame.data(), frame.data_size());
}

ssize_t BasebandFileRaw::write_raw(const void* data, size_t nb) {

    // Write in a retry macro loop incase the write was interrupted by a signal
    ssize_t nbytes = TEMP_FAILURE_RETRY(write(fd, data, nb));

    if (nbytes < 0) {
        ERROR("Write error attempting to write {:d} bytes into file {:s}: {:s}", nb, name,
              strerror(errno));
        return false;
    }

    return nbytes;
}
