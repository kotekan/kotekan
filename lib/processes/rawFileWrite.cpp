#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "rawFileWrite.hpp"
#include "visUtil.hpp"
#include "prometheusMetrics.hpp"
#include "buffer.h"
#include "errors.h"

REGISTER_KOTEKAN_PROCESS(rawFileWrite);

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

    while (!stop_thread) {

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        // Start timing the write time
        double st = current_time();

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

        // Write the meta data to disk
        uint32_t metadata_size = 0;
        struct metadataContainer * mc = get_metadata_container(buf, frame_id);
        if (mc != NULL) {
            metadata_size = mc->metadata_size;
        }
        // Write metadata size to disk, if there is no metadata in the frame, then
        // just save 0 to the first word.
        if (write(fd, (void *)&metadata_size, sizeof(metadata_size)) != (int32_t)sizeof(metadata_size)) {
            ERROR("Failed to write metadata_size to disk for file %s", full_path);
            exit(-1);
        }
        if (mc !=NULL) {
            if (write(fd, mc->metadata, mc->metadata_size) != (int32_t)mc->metadata_size) {
                ERROR("Failed to write metadata_size to disk for file %s", full_path);
                exit(-1);
            }
        }

        // Write the contents of the buffer frame to disk.
        ssize_t bytes_writen = write(fd, frame, buf->frame_size);

        if (bytes_writen != buf->frame_size) {
            ERROR("Failed to write buffer to disk for file %s", full_path);
            exit(-1);
        }

        INFO("Data file write done for %s", full_path);

        if (close(fd) == -1) {
            ERROR("Cannot close file %s", full_path);
        }

        double elapsed = current_time() - st;
        prometheusMetrics::instance().add_process_metric(
            "kotekan_rawfilewrite_write_time_seconds",
            unique_name, elapsed
        );
        mark_frame_empty(buf, unique_name.c_str(), frame_id);

        frame_id = ( frame_id + 1 ) % buf->num_frames;
        file_num++;
    }
}
