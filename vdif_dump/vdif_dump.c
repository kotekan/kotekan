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

#include "buffer.c"
#include "file_write.h"
#include "simple_udp_cap.h"
#include "errors.h"

void makeDirs(char * disk_base, char * data_set, char * symlink_dir, int num_disks) {

    // Make the data location.
    int err = 0;
    char dir_name[100];
    for (int i = 0; i < num_disks; ++i) {

        snprintf(dir_name, 100, "%s/%d/%s", disk_base, i, data_set);
        err = mkdir(dir_name, 0777);

        if (err != -1) {
            continue;
        }

        if (errno == EEXIST) {
            printf("The data set: %s, already exists.\nPlease delete the data set, or use another name.\n", data_set);
            printf("The current data set can be deleted with: rm -fr %s/*/%s && rm -fr %s/%s\n", disk_base, data_set, symlink_dir, data_set);
        } else {
            perror("Error creating data set directory.\n");
            printf("The directory was: %s/%d/%s \n", disk_base, i, data_set);
        }
        exit(errno);
    }

    // Make the symlink location.
    char symlink_path[100];

    snprintf(symlink_path, 100, "%s/%s", symlink_dir, data_set);
    err = mkdir(symlink_path, 0777);

    if (err == -1) {
        if (errno == EEXIST) {
            printf("The symlink output director: %s/%s, already exists.\n", symlink_dir, data_set);
            printf("Please delete the data set, or use another name.\n");
            printf("The current data set can be deleted with: sudo rm -fr %s/*/%s && sudo rm -fr %s/%s\n", disk_base, data_set, symlink_dir, data_set);
        } else {
            perror("Error creating symlink output director directory.\n");
            printf("The symlink output directory was: %s/%s \n", symlink_dir, data_set);
        }
        exit(errno);
    }
}

// Note this could be done in the file writing threads, but I put it here so there wouldn't
// be any delays caused by writing symlinks to the system disk.
void makeSymlinks(char * disk_base, char * symlink_dir, char * data_set, int num_disks, int num_files, int buffer_size) {

    int err = 0;
    char file_name[100];
    char link_name[100];
    int disk_id;

    for (int i = 0; i < num_files; ++i) {
        disk_id = i % num_disks;

        snprintf(file_name, 100, "%s/%d/%s/%07d.dat", disk_base, disk_id, data_set, i);
        snprintf(link_name, 100, "%s/%s/%07d.dat", symlink_dir, data_set, i);
        err = symlink(file_name, link_name);
        if (err == -1) {
            perror("Error creating a symlink.");
            exit(errno);
        }
    }
}

int main(int argc, char ** argv) {

    const int vdif_len = 5032;
    const int packets_per_frame = 20000;
    const int num_disks = 8;
    const int buffer_depth = 4;
    const int num_buffers = num_disks * buffer_depth;
    const int buffer_len = vdif_len * packets_per_frame;
    char disk_base[100];
    char symlink_dir[100];

    struct Buffer buf;
    struct InfoObjectPool pool;
    create_info_pool(&pool, num_buffers, 1024, 256);
    create_buffer(&buf, num_buffers, buffer_len, 1, 1, &pool, "vdif_frames");


    // Compute the data set name.
    char data_set[150];
    char data_time[64];
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = gmtime(&rawtime);

    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(data_set, sizeof(data_set), "%s_chime_beamformed", data_time);
    snprintf(disk_base, sizeof(disk_base), "/drives/");
    snprintf(symlink_dir, sizeof(disk_base), "/drives/0/baseband/");

    // Make the data set directory
    makeDirs(disk_base, data_set, symlink_dir, num_disks);
    // Create symlinks.
    printf("Creating symlinks in %s/%s\n", symlink_dir, data_set);
    makeSymlinks(disk_base, symlink_dir, data_set, num_disks, 10000, buffer_len);
    fprintf(stderr ,"Symlinks done.");

    struct udpCapArgs cap_args;
    char ip[100] = "0.0.0.0";
    cap_args.ip_address = ip;
    cap_args.buf = &buf;
    cap_args.port_number = 10251;
    cap_args.buffer_depth = buffer_depth;
    cap_args.data_limit = 10;
    cap_args.num_links = 1;
    cap_args.link_id = 0;

    pthread_t network_t;
    CHECK_ERROR( pthread_create(&network_t, NULL, (void *)&simple_udp_cap, (void *)&cap_args ) );

    fprintf(stderr, "started network thread");

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

    fprintf(stderr, "started file write threads");

    // Join threads
    int ret;
    CHECK_ERROR( pthread_join(network_t, (void **)&ret) );

    for (int i = 0; i < num_disks; ++i) {
        CHECK_ERROR(pthread_join(file_write_t[i], (void **)&ret));
    }
}