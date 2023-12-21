#include "BasebandFileRaw.hpp"

#include "visFile.hpp" // for create_lockfile

#include "fmt.hpp" // for format, fmt

#include <assert.h>   // for assert
#include <cstdio>     // for remove
#include <errno.h>    // for errno
#include <fcntl.h>    // for fallocate, open, posix_fadvise, sync_file_range, FALLOC_FL_KEEP_SIZE
#include <stdexcept>  // for runtime_error
#include <string.h>   // for strerror
#include <sys/stat.h> // for S_IRGRP, S_IROTH, S_IRUSR, S_IWGRP, S_IWUSR
#include <unistd.h>   // for pwrite, close, lseek, TEMP_FAILURE_RETRY, off_t, ssize_t


BasebandFileRaw::BasebandFileRaw(const std::string& name, const uint32_t frame_size) :
    name(name), frame_size(frame_size) {

    write_index = 0;

    // Create the lock file and then open other files
    DEBUG("Opening baseband file {:s}", name);
    lock_filename = create_lockfile(name);
    if ((fd = open((name + ".data").c_str(), O_CREAT | O_WRONLY,
                   S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH))
        == -1) {
        throw std::runtime_error(
            fmt::format(fmt("Failed to open file {:s}.data: {:s}."), name, strerror(errno)));
    }

    // Get the current file size
    off_t file_size = lseek(fd, 0, SEEK_END);

    if (file_size < 0 || file_size % frame_size != 0) {
        // TODO Better error handling here, e.g. open a new file
        ERROR("The current file is likely corrupt size {:d}, writes to it will be dropped.  "
              "Expected multiple of the frame size {:d}",
              file_size, frame_size);
        file_corrupt = true;
    } else {
        file_corrupt = false;
        write_index = file_size / frame_size;
    }
}

BasebandFileRaw::~BasebandFileRaw() {
    DEBUG("Closing baseband file {}: {}", name, fd);
    close(fd);

    std::remove(lock_filename.c_str());
}


int32_t BasebandFileRaw::write_frame(const BasebandFrameView& frame) {

    assert(metadata_size + frame.data_size() == frame_size);
    if (file_corrupt) {
        return -1;
    }

#ifdef __linux__
    fallocate(fd, FALLOC_FL_KEEP_SIZE, write_index * frame_size, frame_size);
#else
    ftruncate(fd, write_index * (frame_size + 1));
#endif

    // Write in a retry macro loop incase the write was interrupted by a signal
    // FIXME metadata serialization
    ssize_t nbytes = TEMP_FAILURE_RETRY(
                                        pwrite(fd, (void*)frame.metadata().get(), metadata_size, write_index * frame_size));

    if (nbytes < 0) {
        ERROR("Write error attempting to write metadata {:d} bytes into file {:s}: {:s}",
              metadata_size, name, strerror(errno));
        return 0;
    }

    nbytes = TEMP_FAILURE_RETRY(pwrite(fd, (void*)frame.data(), frame.data_size(),
                                       write_index * frame_size + metadata_size));

    if (nbytes < 0) {
        ERROR("Write error attempting to write data {:d} bytes into file {:s}: {:s}",
              frame.data_size(), name, strerror(errno));
        return 0;
    }

#ifdef __linux__
    sync_file_range(fd, write_index * frame_size, frame_size,
                    SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE
                        | SYNC_FILE_RANGE_WAIT_AFTER);
    posix_fadvise(fd, write_index * frame_size, frame_size, POSIX_FADV_DONTNEED);
#endif

    write_index++;
    return 1;
}
