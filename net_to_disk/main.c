#define _GNU_SOURCE

#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <unistd.h>
#include <memory.h>
#include <pthread.h>
#include <sched.h>
#include <getopt.h>

#include "network.h"
#include "buffers.h"
#include "file_write.h"
#include "output_power.h"

// TODO Replace defines with command line options.

// The number of buffers to keep for each disk.
#define BUFFER_DEPTH 10

void print_help() {

    printf("Program: net_to_disk\n\n");
    printf("Records data from the network to disk.\n\n");

    printf("Required Options:\n\n");
    //printf("--interface -i [interface]    The interface to listen on.\n");
    printf("--data-set -d [dir name]      The name of the data set.\n");
    printf("--symlink-dir -s [dir name]   The directory to put the symlinks into.\n");
    printf("--data-limit -l [number]      The maximum number of GB to save.\n");

    printf("\nExtra Options:\n\n");
    printf("--num-disks -n [number]       The number of disks, default: 9\n");
    printf("--disk-base -b [dir name]     The base dir of the disks, default: /drives/ \n");
    printf("--disable-packet-dump -x      Don't write the packets to disk \n");
    printf("--num-freq -f [number]        The number of frequencies to record, default 1024\n");
    printf("--offset -o [number]          Offset of the frequencies to record, default 0\n ");
}

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
void makeSymlinks(char * disk_base, char * symlink_dir, char * data_set, int num_disks, int data_limit, int buffer_size) {

    int err = 0;
    char file_name[100];
    char link_name[100];
    int disk_id;

    int num_files = (data_limit * 1024) / (buffer_size/ (1024*1024));
    printf("Number of files: %d\n", num_files);

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

    int opt_val = 0;

    // Default values:

    char * interface = "*";
    char * data_set = "*";
    int num_disks = 9;
    int data_limit = -1;
    char * symlink_dir = "*";
    char * disk_base = "/drives";
    int num_links = 8;
    int write_packets = 1;
    int write_powers = 1;
    int num_consumers = 2;

    int num_timesamples = 16*1024;
    int header_len = 58;

    // Data format
    int num_frames = 4;
    int num_inputs = 2;
    int num_freq = 1024;
    int offset = 0;

    for (;;) {
        static struct option long_options[] = {
            {"ip-address", required_argument, 0, 'i'},
            {"data-set", required_argument, 0, 'd'},
            {"num-disks", required_argument, 0, 'n'},
            {"disk-base", required_argument, 0, 'b'},
            {"data-limit", required_argument, 0, 'l'},
            {"symlink-dir", required_argument, 0, 's'},
            {"disable-packet-dump", no_argument, 0, 'x'},
            {"num-freq", required_argument, 0, 'f'},
            {"offset", required_argument, 0, 'o'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        opt_val = getopt_long (argc, argv, "hi:d:l:n:b:s:xf:o:",
                            long_options, &option_index);

        // End of args
        if (opt_val == -1) {
            break;
        }

        switch (opt_val) {
            case 'h':
                print_help();
                return 0;
                break;
            case 'i':
                interface = optarg;
                break;
            case 'd':
                data_set = optarg;
                break;
            case 'n':
                num_disks = atoi(optarg);
                break;
            case 'b':
                disk_base = optarg;
                break;
            case 'l':
                data_limit = atoi(optarg);
                break;
            case 's':
                symlink_dir = optarg;
                break;
            case 'x':
                write_packets = 0;
                num_consumers = 1;
                break;
            case 'f':
                num_freq = atoi(optarg);
                break;
            case 'o':
                offset = atoi(optarg);
                break;
            default:
                print_help();
                break;
        }
    }

    if (data_limit <= 0) {
        printf("--data-limit needs to be set.\nUse -h for help.\n");
        return -1;
    }

    //if (interface[0] == '*') {
    //    printf("--ip-address needs to be set.\nUse -h for help.\n");
    //    return -1;
    //}

    if (data_set[0] == '*') {
        printf("--data-set needs to be set.\nUse -h for help.\n");
        return -1;
    }

    if (symlink_dir[0] == '*') {
        printf("--symlink-dir needs to be set.\nUse -h for help.\n");
        return -1;
    }

    if (write_packets == 1) {
        // Make the data set directory
        makeDirs(disk_base, data_set, symlink_dir, num_disks);
    }

    int packet_len = num_frames * num_inputs * num_freq + header_len;
    int buffer_len = (num_timesamples / num_frames) * packet_len;

    pthread_t network_t[num_links], file_write_t[num_disks], output_power_t;
    int * ret;
    struct Buffer buf;

    createBuffer(&buf, num_disks * BUFFER_DEPTH, buffer_len, num_links, num_consumers);

    if (write_packets == 1) {
        // Create symlinks.
        printf("Creating symlinks in %s/%s\n", symlink_dir, data_set);
        makeSymlinks(disk_base, symlink_dir, data_set, num_disks, data_limit, buffer_len);
    }

    // Let the disks flush
    sleep(5);

    // Create the network thread.
    struct network_thread_arg network_args[num_links];
    for (int i = 0; i < num_links; ++i) {
        char * ip_address = malloc(100*sizeof(char));
        snprintf(ip_address, 100, "dna%d", i);
        network_args[i].interface = ip_address;
        network_args[i].buf = &buf;
        network_args[i].bufferDepth = BUFFER_DEPTH;
        network_args[i].numLinks = num_links;
        network_args[i].data_limit = data_limit;
        network_args[i].link_id = i;
        network_args[i].num_frames = num_frames;
        network_args[i].num_inputs = num_inputs;
        network_args[i].num_freq = num_freq;
        network_args[i].offset = offset;
        HANDLE_ERROR( pthread_create(&network_t[i], NULL, (void *) &network_thread, (void *)&network_args[i] ) );
    }
    // Create the file writing threads.
    struct file_write_thread_arg file_write_args[num_disks];

    if (write_packets == 1) {
        for (int i = 0; i < num_disks; ++i) {
            file_write_args[i].buf = &buf;
            file_write_args[i].diskID = i;
            file_write_args[i].numDisks = num_disks;
            file_write_args[i].bufferDepth = BUFFER_DEPTH;
            file_write_args[i].dataset_name = data_set;
            file_write_args[i].disk_base = disk_base;
            HANDLE_ERROR( pthread_create(&file_write_t[i], NULL, (void *) &file_write_thread, (void *)&file_write_args[i] ) );
        }
    }

    struct output_power_thread_arg output_arg;
    if (write_powers == 1) {
        output_arg.buf = &buf;
        output_arg.bufferDepth = BUFFER_DEPTH;
        output_arg.disk_base = disk_base;
        output_arg.dataset_name = data_set;
        output_arg.diskID = 0;
        output_arg.numDisks = num_disks;
        if (num_freq == 1024) {
            output_arg.num_freq = 512;
            output_arg.offset = 512;
        } else {
            output_arg.num_freq = num_freq;
            output_arg.offset = 0;
        }
        output_arg.num_frames = num_frames;
        output_arg.num_inputs = num_inputs;
        HANDLE_ERROR( pthread_create(&output_power_t, NULL, (void *)&output_power_thread, (void *)&output_arg) );
    }

    // TODO Trap signals

    // clean up threads here.
    for (int i = 0; i < num_links; ++i) {
        HANDLE_ERROR( pthread_join(network_t[i], (void **) &ret) );
    }
    if (write_packets == 1) {
        for (int i = 0; i < num_disks; ++i) {
            HANDLE_ERROR( pthread_join(file_write_t[i], (void **) &ret) );
        }
    }

    if (write_powers == 1) {
        HANDLE_ERROR( pthread_join(output_power_t, (void **) &ret) );
    }

    deleteBuffer(&buf);

    return 0;
}

