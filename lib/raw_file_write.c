
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "raw_file_write.h"
#include "stream_raw_vdif.h"
#include "buffers.h"
#include "errors.h"

void raw_file_write_thread(void * arg)
{
    struct raw_file_write_thread_arg * args = (struct raw_file_write_thread_arg *) arg;

    int fd, file_num = args->diskID;
    int useableBufferIDs[1] = {args->diskID};
    int bufferID = args->diskID;

    for (;;) {

        // This call is blocking.
        bufferID = get_full_buffer_from_list(args->buf, useableBufferIDs, 1);

        //INFO("Got buffer, id: %d", bufferID);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            int ret;
            pthread_exit((void *) &ret);
        }

        const int file_name_len = 100;
        char file_name[file_name_len];

        snprintf(file_name, file_name_len, "%s/%s/%d/%s/%07d.vdif",
                args->disk_base,
                args->disk_set,
                args->diskID,
                args->dataset_name,
                file_num);

        struct ErrorMatrix * error_matrix = get_error_matrix(args->buf, bufferID);

        // Open the file to write
        if (args->write_packets == 1) {

            fd = open(file_name, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: %s", file_name);
                exit(errno);
            }

            ssize_t bytes_writen = write(fd, args->buf->data[bufferID], args->buf->buffer_size);

            if (bytes_writen != args->buf->buffer_size) {
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
        zero_buffer(args->buf, bufferID);

        // TODO make release_info_object work for nConsumers.
        release_info_object(args->buf, bufferID);
        mark_buffer_empty(args->buf, bufferID);

        useableBufferIDs[0] = ( useableBufferIDs[0] + args->numDisks ) % args->buf->num_buffers;
        file_num += args->numDisks;
    }
}
