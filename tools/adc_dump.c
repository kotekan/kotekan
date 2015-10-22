#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <unistd.h>
#include <memory.h>
#include <sched.h>
#include <getopt.h>
#include <time.h>
#include <stdio.h>
#include <pthread.h>

#include "buffers.h"
#include "file_write.h"
#include "simple_dna_cap.h"
#include "errors.h"

void make_dirs(char * disk_base, char * data_set, char * symlink_dir, int num_disks) {

    // Make the data location.
    int err = 0;
    char dir_name[100];
    if (num_disks == 1) {
        snprintf(dir_name, 100, "%s/%s", disk_base, data_set);
        err = mkdir(dir_name, 0777);

        if (err != -1) {
            return;
        }

        if (errno == EEXIST) {
            printf("The data set: %s, already exists.\nPlease delete the data set, or use another name.\n", data_set);
        } else {
            perror("Error creating data set directory.\n");
            printf("The directory was: %s/%s \n", disk_base, data_set);
        }
        exit(errno);
    } else {
        for (int i = 0; i < num_disks; ++i) {
            snprintf(dir_name, 100, "%s/%d/%s", disk_base, i, data_set);
            err = mkdir(dir_name, 0777);

            if (err != -1) {
                continue;
            }

            if (errno == EEXIST) {
                printf("The data set: %s, already exists.\nPlease delete the data set, or use another name.\n", data_set);
            } else {
                perror("Error creating data set directory.\n");
                printf("The directory was: %s/%d/%s \n", disk_base, i, data_set);
            }
            exit(errno);
        }
    }
}


int main(int argc, char ** argv) {

    const int packet_len = 2112;
    const int packets_per_frame = 100000;
    const int num_disks = 1;
    const int buffer_depth = 10;
    const int dna_numer = 0;
    const int num_buffers = num_disks * buffer_depth;
    const int buffer_len = packet_len * packets_per_frame;
    char disk_base[100];
    char symlink_dir[100];

    struct Buffer buf;
    struct InfoObjectPool pool;
    create_info_pool(&pool, num_buffers, 1024, 256);
    create_buffer(&buf, num_buffers, buffer_len, 1, 1, &pool, "adc_frames");


    // Compute the data set name.
    char data_set[150];
    char data_time[64];
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = gmtime(&rawtime);

    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(data_set, sizeof(data_set), "%s_adc_raw_burst", data_time);
    snprintf(disk_base, sizeof(disk_base), "/data/test/");

    // Make the data set directory
    make_dirs(disk_base, data_set, symlink_dir, num_disks);

    struct dnaCapArgs cap_args;
    cap_args.buf = &buf;
    cap_args.buffer_depth = buffer_depth;
    cap_args.dna_id = dna_numer;
    cap_args.close_on_block = 1;
    cap_args.packet_size = packet_len;

    pthread_t network_t;
    CHECK_ERROR( pthread_create(&network_t, NULL, (void *)&simple_dna_cap, (void *)&cap_args ) );

    fprintf(stderr, "started network thread\n");

    // Start file write threads.
    pthread_t file_write_t[num_disks];
    struct fileWriteThreadArg file_write_args[num_disks];
    for (int i = 0; i < num_disks; ++i) {
        file_write_args[i].buf = &buf;
        file_write_args[i].disk_ID = i;
        file_write_args[i].num_disks = num_disks;
        file_write_args[i].buffer_depth = buffer_depth;
        file_write_args[i].dataset_name = data_set;
        file_write_args[i].disk_base = disk_base;
        CHECK_ERROR(pthread_create(&file_write_t[i], NULL, (void *)&file_write_thread, (void*)&file_write_args[i]));
    }

    fprintf(stderr, "started file write threads\n");

    // Join threads
    int ret;

    for (int i = 0; i < num_disks; ++i) {
        CHECK_ERROR(pthread_join(file_write_t[i], (void **)&ret));
    }

    fprintf(stderr, "Finished writting all files...\n");

    CHECK_ERROR( pthread_join(network_t, (void **)&ret) );

}