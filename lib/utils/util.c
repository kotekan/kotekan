#include "util.h"

#include <errno.h>    // for errno, EEXIST, EINTR
#include <fcntl.h>    // for open, O_CREAT, O_EXCL, O_RDONLY, O_WRONLY
#include <stdio.h>    // for printf, perror, snprintf, NULL
#include <stdlib.h>   // for exit
#include <sys/stat.h> // for mkdir
#include <sys/time.h> // for gettimeofday, timeval
#include <unistd.h>   // for close, read, write, ssize_t

double e_time(void) {
    struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec + now.tv_usec / 1000000.0);
}

void make_rfi_dirs(int streamID, const char* write_to, const char* time_dir) {
    // Initialize variables
    int err = 0;
    char dir_name[100];
    // Create sub-directory for current stream
    snprintf(dir_name, 100, "%s/%d", write_to, streamID);
    err = mkdir(dir_name, 0777);
    // If there is an error
    if (err == -1) {
        // If that error is NOT that the directory already exists
        if (errno != EEXIST) {
            perror("Error creating data set directory.\n");
            printf("The directory was: %s/%d\n", write_to, streamID);
            exit(errno);
        }
    }
    // Create sub-sub-directory for current capture
    dir_name[0] = '\0';
    snprintf(dir_name, 100, "%s/%d/%s", write_to, streamID, time_dir);
    err = mkdir(dir_name, 0777);
    // If there is an error
    if (err == -1) {
        // If that error is that the directory already exists
        if (errno == EEXIST) {
            printf("The data set: %s, already exists.\nPlease delete the data set, or use another "
                   "name.\n",
                   time_dir);
        } else {
            // IF the error is more serious
            perror("Error creating data set directory.\n");
            printf("The directory was: %s/%d/%s\n", write_to, streamID, time_dir);
        }
        exit(errno);
    }
}


int make_dir(const char* dir_name) {
    // Make the data dir from the absolute path
    int err = 0;
    err = mkdir(dir_name, 0777);

    if (err != -1) {
        return err;
    }
    if (errno == EEXIST) {
    
    // printf("The data set: %s, already exists.\nPlease delete the data set, or use another
           // name.\n", data_set); printf("The current data set can be deleted with: rm -fr %s/*/%s
           // && rm -fr %s/%s\n", disk_base, data_set, symlink_dir, data_set);
    } else {
        perror("Error creating data set directory.\n");
        printf("The directory was: %s \n", dir_name);
    }
    exit(errno);
}

void make_dirs(char* disk_base, char* data_set, int num_disks) {

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
            // printf("The data set: %s, already exists.\nPlease delete the data set, or use another
            // name.\n", data_set); printf("The current data set can be deleted with: rm -fr %s/*/%s
            // && rm -fr %s/%s\n", disk_base, data_set, symlink_dir, data_set);
        } else {
            perror("Error creating data set directory.\n");
            printf("The directory was: %s/%d/%s \n", disk_base, i, data_set);
        }
        exit(errno);
    }
}

// TODO Merge this with the above function.
void make_raw_dirs(const char* disk_base, const char* disk_set, const char* data_set,
                   int num_disks) {

    // Make the data location.
    int err = 0;
    char dir_name[100];
    for (int i = 0; i < num_disks; ++i) {

        snprintf(dir_name, 100, "%s/%s/%d/%s", disk_base, disk_set, i, data_set);
        err = mkdir(dir_name, 0777);

        if (err != -1) {
            continue;
        }

        if (errno == EEXIST) {
            printf("The data set: %s, already exists.\nPlease delete the data set, or use another "
                   "name.\n",
                   data_set);
            printf("The current data set can be deleted with: rm -fr %s/%s/*/%s\n", disk_base,
                   disk_set, data_set);
        } else {
            perror("Error creating data set directory.\n");
            printf("The directory was: %s/%s/%d/%s \n", disk_base, disk_set, i, data_set);
        }
        exit(errno);
    }
}

int cp(const char* to, const char* from) {
    int fd_to, fd_from;
    char buf[4096];
    ssize_t nread;
    int saved_errno;

    fd_from = open(from, O_RDONLY);
    if (fd_from < 0)
        return -1;

    fd_to = open(to, O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd_to < 0)
        goto out_error;

    while (nread = read(fd_from, buf, sizeof buf), nread > 0) {
        char* out_ptr = buf;
        ssize_t nwritten;

        do {
            nwritten = write(fd_to, out_ptr, nread);

            if (nwritten >= 0) {
                nread -= nwritten;
                out_ptr += nwritten;
            } else if (errno != EINTR) {
                goto out_error;
            }
        } while (nread > 0);
    }

    if (nread == 0) {
        if (close(fd_to) < 0) {
            fd_to = -1;
            goto out_error;
        }
        close(fd_from);

        /* Success! */
        return 0;
    }

out_error:
    saved_errno = errno;

    close(fd_from);
    if (fd_to >= 0)
        close(fd_to);

    errno = saved_errno;
    return -1;
}

int64_t mod(int64_t a, int64_t b) {
    int ret = a % b;
    if (ret < 0)
        ret += b;
    return ret;
}

void hex_dump(const int rows, void* addr, int len) {
    int i;
    unsigned char* char_buf = (unsigned char*)addr;

    for (i = 0; i < len; i++) {
        if ((i % rows) == 0) {
            // Add a new line as needed.
            if (i != 0)
                printf("\n");

            // Print the offset.
            printf("  %04x ", i);
        }

        // Print the hex value
        printf(" %02x", char_buf[i]);
    }
    printf("\n");
}
