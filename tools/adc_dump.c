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
#include <math.h>
#include <pthread.h>

#include "buffer.h"
#include "file_write.h"
#include "simple_dna_cap.h"
#include "errors.h"

void make_dirs(char * disk_base, char * data_set, int num_disks) {

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

    const int packet_len = 9536;
    const int packets_per_frame = 20000;
    const int num_links = 8;
    const int num_disks = 1;
    const int buffer_depth =
        (int)floor( (14.0*1024*1024*1024) / (packet_len * packets_per_frame * num_links) );
    const int buffer_len = packet_len * packets_per_frame;
    const int integration_edge = 16777216;
    char disk_base[100];
    snprintf(disk_base, sizeof(disk_base), "/data/glock/archive/");


    struct InfoObjectPool pool;
    create_info_pool(&pool, buffer_depth * num_links * 2, 1, 1);

    struct Buffer buf[num_links];
    for (int i = 0; i < num_links; ++i) {
        create_buffer(&buf[i], buffer_depth, buffer_len, 1, 1, &pool, "adc_frames");
    }

    // Compute the data set name.
    char data_set[150];
    char data_time[64];
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = gmtime(&rawtime);

    char hostname[256];
    hostname[255] = '\0';
    gethostname(hostname, 256);

    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(data_set, sizeof(data_set), "%s_%s_fft_burst", data_time, hostname);

    // Make the data set directory
    make_dirs(disk_base, data_set, num_disks);

    struct dnaCapArgs cap_args[num_links];
    pthread_t network_t[num_links];
    for (int i = 0; i < num_links; ++i) {
        cap_args[i].buf = &buf[i];
        cap_args[i].buffer_depth = buffer_depth;
        cap_args[i].dna_id = i;
        cap_args[i].close_on_block = 1;
        cap_args[i].packet_size = packet_len;
        cap_args[i].integration_edge = integration_edge;

        CHECK_ERROR( pthread_create(&network_t[i], NULL, (void *)&simple_dna_cap, (void *)&cap_args[i] ) );
    }

    fprintf(stderr, "started network thread\n");

    // Start file write threads.
    pthread_t file_write_t[num_links];
    struct fileWriteThreadArg file_write_args[num_links];
    for (int i = 0; i < num_links; ++i) {
        file_write_args[i].buf = &buf[i];
        file_write_args[i].disk_ID = 0;
        file_write_args[i].link_ID = i;
        file_write_args[i].num_disks = num_disks;
        file_write_args[i].num_links = num_links;
        file_write_args[i].buffer_depth = buffer_depth;
        file_write_args[i].dataset_name = data_set;
        file_write_args[i].disk_base = disk_base;
        CHECK_ERROR(pthread_create(&file_write_t[i], NULL, (void *)&file_write_thread, (void*)&file_write_args[i]));
    }

    fprintf(stderr, "started file write threads\n");

    // Join threads
    int ret;

    for (int i = 0; i < num_links; ++i) {
        CHECK_ERROR(pthread_join(file_write_t[i], (void **)&ret));
    }

    fprintf(stderr, "Finished writting all files...\n");

    for (int i = 0; i < num_links; ++i) {
        CHECK_ERROR(pthread_join(network_t[i], (void **)&ret));
    }

}