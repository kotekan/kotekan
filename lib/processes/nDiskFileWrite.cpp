#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "nDiskFileWrite.hpp"
#include "buffers.h"
#include "errors.h"

nDiskFileWrite::nDiskFileWrite(Config& config, const string& unique_name,
                                bufferContainer &buffer_containter,
                                int disk_id_, const string &dataset_name_) :
        KotekanProcess(config, unique_name, buffer_containter,
                       std::bind(&nDiskFileWrite::main_thread, this)),
        disk_id(disk_id_),
        dataset_name(dataset_name_)
{
    buf = buffer_containter.get_buffer("network_buffer");
    apply_config(0);
}

nDiskFileWrite::~nDiskFileWrite() {
}

void nDiskFileWrite::apply_config(uint64_t fpga_seq) {
    disk_base = config.get_string("/raw_capture/disk_base");
    disk_set = config.get_string("/raw_capture/disk_set");
    num_disks = config.get_int("/raw_capture/num_disks");
    write_to_disk = config.get_bool("/raw_capture/write_to_disk");
}

// TODO instead of there being N disks of this tread started, this thread should
// start N threads to write the data.
void nDiskFileWrite::main_thread() {

    int fd;
    int file_num = disk_id;
    int buffer_id = disk_id;

    for (;;) {

        // This call is blocking.
        buffer_id = get_full_buffer_from_list(buf, &buffer_id, 1);

        //INFO("Got buffer id: %d, disk id %d", buffer_id, disk_id);

        // Check if the producer has finished, and we should exit.
        if (buffer_id == -1) {
            break;
        }

        const int file_name_len = 100;
        char file_name[file_name_len];

        snprintf(file_name, file_name_len, "%s/%s/%d/%s/%07d.vdif",
                disk_base.c_str(),
                disk_set.c_str(),
                disk_id,
                dataset_name.c_str(),
                file_num);

        struct ErrorMatrix * error_matrix = get_error_matrix(buf, buffer_id);

        // Open the file to write
        if (write_to_disk) {

            fd = open(file_name, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: %s", file_name);
                exit(errno);
            }

            ssize_t bytes_writen = write(fd, buf->data[buffer_id], buf->buffer_size);

            if (bytes_writen != buf->buffer_size) {
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
            //usleep(0.070 * 1e6);
            INFO("Disk id %d, Lost Packets %d, buffer id %d", disk_id, error_matrix->bad_timesamples, buffer_id );
        }

        // Zero the buffer (needed for VDIF packet processing)
        zero_buffer(buf, buffer_id);

        // TODO make release_info_object work for nConsumers.
        release_info_object(buf, buffer_id);
        mark_buffer_empty(buf, buffer_id);

        buffer_id = ( buffer_id + num_disks ) % buf->num_buffers;
        file_num += num_disks;
    }
}
