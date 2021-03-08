#include "BasebandFileRaw.hpp"

#include "visFile.hpp" // for create_lockfile

#include <errno.h>    // for errno
#include <fcntl.h>    // for O_CREAT, O_WRONLY
#include <sys/stat.h> // for mkdir

BasebandFileRaw::BasebandFileRaw(const std::string& name) : _name(name) {

    DEBUG("Creating new baseband file {}", name);

    // Create the lock file and then open other files
    lock_filename = create_lockfile(_name);
    if ((fd = open((_name + ".data").c_str(), O_CREAT | O_WRONLY,
                   S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH))
        == -1) {
        throw std::runtime_error(
            fmt::format(fmt("Failed to open file {:s}.data: {:s}."), _name, strerror(errno)));
    }
}

BasebandFileRaw::~BasebandFileRaw() {
    DEBUG("Closing baseband file {}: {}", _name, fd);
    // TODO: final sync of data file.
    close(fd);

    std::remove(lock_filename.c_str());
}


ssize_t BasebandFileRaw::write_frame(const BasebandFrameView& frame) {
    // Write the "1" marker to indicate that the frame is good
    const uint8_t ONE = 1;
    write_raw(&ONE, 1);

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
        ERROR("Write error attempting to write {:d} bytes into file {:s}: {:s}", nb, _name,
              strerror(errno));
        return false;
    }

    return nbytes;
}
