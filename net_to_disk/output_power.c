
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "output_power.h"
#include "buffers.h"

#define PACKET_OFFSET 58

void output_power_thread(void * arg)
{
    struct output_power_thread_arg * args = (struct output_power_thread_arg *) arg;

    int fd, file_num;
    int useableBufferIDs[1] = {0};
    int bufferID = 0;

    // Open the file to write
    const int file_name_len = 100;
    char file_name[file_name_len];

    snprintf(file_name, file_name_len, "/mnt/ram_disk/power_data.dat");

    const int integration_time = 8*1024;

    int out_buf[args->num_freq*2];
    int * xx = out_buf;
    int * yy = out_buf + args->num_freq;

    const int packet_len = args->num_frames * args->num_freq * args->num_inputs + PACKET_OFFSET;
    const int frame_size = args->num_freq * args->num_inputs;

    int file_open = 0;

    for (;;) {

        if (file_open == 0) {
            fd = open(file_name, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                perror("Cannot open file");
                fprintf(stderr, "File name was: %s", file_name);
                exit(errno);
            }
            file_open = 1;
        }

        // This call is blocking.
        bufferID = getFullBufferFromList(args->buf, useableBufferIDs, 1);

        //printf("Got buffer, id: %d", bufferID);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            int ret;
            if (close(fd) == -1) {
                fprintf(stderr, "Cannot close file");
            }
            pthread_exit((void *) &ret);
        }

        file_num = getDataID(args->buf, bufferID);
        unsigned char * data = (unsigned char *) args->buf->data[bufferID];

        for (int packet = 0; packet < integration_time/args->num_frames; ++packet) {
            for (int frame = 0; frame < args->num_frames; ++frame) {
                for (int freq = 0; freq < args->num_freq; ++freq) {

                    const int index = packet * packet_len + PACKET_OFFSET + frame * frame_size + (freq + args->offset)*2;

                    int x_img = ((int)data[index] & 0x0f) - 8;
                    int x_real = ( ((int)data[index] & 0xf0) >> 4 ) - 8;
                    int y_img = ((int)data[index+1] & 0x0f) - 8;
                    int y_real = ( ((int)data[index+1] & 0xf0) >> 4 ) - 8;

                    if (packet == 0) {
                        xx[freq] = x_real * x_real + x_img * x_img;
                        yy[freq] = y_real * y_real + y_img * y_img;
                    } else {
                        xx[freq] += x_real * x_real + x_img * x_img;
                        yy[freq] += y_real * y_real + y_img * y_img;
                    }
                }
            }
        }

        ssize_t bytes_writen = write(fd, out_buf, args->num_freq*2*sizeof(int));

        if (bytes_writen != args->num_freq*2*sizeof(int)) {
            printf("Failed to write power data to ram disk!  Trying to open new file");
            file_open = 0;
            close(fd);
        }

        markBufferEmpty_nConsumers(args->buf, bufferID);

        useableBufferIDs[0] = ( useableBufferIDs[0] + 1 ) % ( args->bufferDepth * args->numDisks );
    }

}