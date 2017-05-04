#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "rawFileWrite.hpp"
#include "buffers.h"
#include "errors.h"

rawFileWrite::rawFileWrite(Config& config,
                 const string& unique_name,
                 bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                       std::bind(&rawFileWrite::main_thread, this)) {

    buf = buffer_container.get_buffer("buf");
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
    int buffer_id = 0;

    for (;;) {

        // This call is blocking.
        buffer_id = get_full_buffer_from_list(buf, &buffer_id, 1);

        //INFO("Got buffer, id: %d", bufferID);

        // Check if the producer has finished, and we should exit.
        if (buffer_id == -1) {
            return;
        }

        const int full_path_len = 200;
        char full_path[full_path_len];

        snprintf(full_path, full_path_len, "%s/%s_%07d.%s",
                base_dir.c_str(),
                file_name.c_str(),
                file_num,
                file_ext.c_str());

        fd = open(full_path, O_WRONLY | O_CREAT, 0666);

        if (fd == -1) {
            ERROR("Cannot open file");
            ERROR("File name was: %s", full_path);
            exit(errno);
        }

        ssize_t bytes_writen = write(fd, buf->data[buffer_id], buf->buffer_size);

        if (bytes_writen != buf->buffer_size) {
            ERROR("Failed to write buffer to disk for file %s", full_path);
            exit(-1);
        }

        INFO("Data file write done for %s", full_path);

        //for (int i = 0; i < 10; ++i) {
        //    INFO("%s[%d][%d] = %d", buf.buffer_name, buffer_id, i, *((int *)&buf.data[buffer_id][i * sizeof(int)]));
        //}

        if (close(fd) == -1) {
            ERROR("Cannot close file %s", full_path);
        }

        // TODO make release_info_object work for nConsumers.
        //release_info_object(&buf, buffer_id);
        mark_buffer_empty(buf, buffer_id);

        buffer_id = ( buffer_id + 1 ) % buf->num_buffers;
        file_num++;
    }
}
