#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "rawFileWrite.hpp"
#include "stream_raw_vdif.h"
#include "buffers.h"
#include "errors.h"

rawFileWrite::rawFileWrite(Config& config,
                    Buffer& buf_,
                    int disk_id_,
                    char* extension_,
                    int write_data_,
                    char* dataset_name_) :
        KotekanProcess(config, std::bind(&rawFileWrite::main_thread, this)),
        buf(buf_),
        disk_id(disk_id_),
        write_data(write_data_)
{
    strncpy(extension, extension_, RAW_FILE_STR_LEN);
    strncpy(dataset_name, dataset_name_, RAW_FILE_STR_LEN);
}

rawFileWrite::~rawFileWrite() {
}

void rawFileWrite::main_thread() {

    int fd;
    int file_num = disk_id;
    int useable_buffer_IDs[1] = {disk_id};
    int bufferID = disk_id;

    for (;;) {

        // This call is blocking.
        bufferID = get_full_buffer_from_list(&buf, useable_buffer_IDs, 1);

        //INFO("Got buffer, id: %d", bufferID);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            return;
        }

        const int file_name_len = 200;
        char file_name[file_name_len];

        snprintf(file_name, file_name_len, "%s/%s/%d/%s/%07d.%s",
                config.disk.disk_base,
                config.disk.disk_set,
                disk_id,
                dataset_name,
                file_num,
                extension);

        struct ErrorMatrix * error_matrix = get_error_matrix(&buf, bufferID);

        // Open the file to write
        if (write_data == 1) {

            fd = open(file_name, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: %s", file_name);
                exit(errno);
            }

            ssize_t bytes_writen = write(fd, buf.data[bufferID], buf.buffer_size);

            if (bytes_writen != buf.buffer_size) {
                ERROR("Failed to write buffer to disk!!!  Abort, Panic, etc.");
                exit(-1);
            } else {
                 //fprintf(stderr, "Data writen to file!");
            }

            if (close(fd) == -1) {
                ERROR("Cannot close file %s", file_name);
            }

            INFO("Data file write done for %s, lost_packets %d", file_name, error_matrix->bad_timesamples);
        } else {
            INFO("Lost Packets %d", error_matrix->bad_timesamples );
        }

        // Zero the buffer
        zero_buffer(&buf, bufferID);

        // TODO make release_info_object work for nConsumers.
        release_info_object(&buf, bufferID);
        mark_buffer_empty(&buf, bufferID);

        useable_buffer_IDs[0] = ( useable_buffer_IDs[0] + config.disk.num_disks ) % buf.num_buffers;
        file_num += config.disk.num_disks;
    }
}
