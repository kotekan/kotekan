
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "file_write.h"
#include "buffers.h"

void file_write_thread(void * arg)
{
    struct file_write_thread_arg * args = (struct file_write_thread_arg *) arg;

    int fd, file_num;
    int useableBufferIDs[1] = {args->diskID};
    int bufferID = args->diskID;

    for (;;) {

        // This call is blocking.
        bufferID = getFullBufferFromList(args->buf, useableBufferIDs, 1);

        //printf("Got buffer, id: %d", bufferID);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            int ret;
            pthread_exit((void *) &ret);
        }

        file_num = getDataID(args->buf, bufferID);

        // Open the file to write

        const int file_name_len = 100;
        char file_name[file_name_len];

        snprintf(file_name, file_name_len, "%s/%d/%s/%07d.dat", args->disk_base, args->diskID, args->dataset_name, file_num);

        fd = open(file_name, O_WRONLY | O_CREAT, 0666);

        if (fd == -1) {
            perror("Cannot open file");
            fprintf(stderr, "File name was: %s", file_name);
            exit(errno);
        }

        ssize_t bytes_writen = write(fd, args->buf->data[bufferID], args->buf->buffer_size);

        if (bytes_writen != args->buf->buffer_size) {
            printf("Failed to write buffer to disk!!!  Abort, Panic, etc.");
            exit(-1);
        } else {
             //fprintf(stderr, "Data writen to file!");
        }

        if (close(fd) == -1) {
            fprintf(stderr, "Cannot close file");
        }

        markBufferEmpty_nConsumers(args->buf, bufferID);

        useableBufferIDs[0] = ( useableBufferIDs[0] + args->numDisks ) % ( args->bufferDepth * args->numDisks );
    }

}