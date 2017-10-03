#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "rawFileWrite.hpp"
#include "buffer.h"
#include "errors.h"

rawFileWrite::rawFileWrite(Config& config,
                 const string& unique_name,
                 bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                       std::bind(&rawFileWrite::main_thread, this)) {

    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    base_dir = config.get_string(unique_name, "base_dir");
    file_name = config.get_string(unique_name, "file_name");
    file_ext = config.get_string(unique_name, "file_ext");
}

rawFileWrite::~rawFileWrite() {
}

void rawFileWrite::apply_config(uint64_t fpga_seq) {
}

void rawFileWrite::main_thread() {

    int fd;
    int file_num = 0;
    int frame_id = 0;
    uint8_t * frame = NULL;
    char hostname[64];
    gethostname(hostname, 64);

    for (;;) {

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);

        const int full_path_len = 200;
        char full_path[full_path_len];

        snprintf(full_path, full_path_len, "%s/%s_%s_%07d.%s",
                base_dir.c_str(),
                hostname,
                file_name.c_str(),
                file_num,
                file_ext.c_str());

        fd = open(full_path, O_WRONLY | O_CREAT, 0666);

        if (fd == -1) {
            ERROR("Cannot open file");
            ERROR("File name was: %s", full_path);
            exit(errno);
        }

        ssize_t bytes_writen = write(fd, frame, buf->frame_size);

        if (bytes_writen != buf->frame_size) {
            ERROR("Failed to write buffer to disk for file %s", full_path);
            exit(-1);
        }

        INFO("Data file write done for %s", full_path);

        if (close(fd) == -1) {
            ERROR("Cannot close file %s", full_path);
        }

        mark_frame_empty(buf, unique_name.c_str(), frame_id);

        frame_id = ( frame_id + 1 ) % buf->num_frames;
        file_num++;
    }
}
