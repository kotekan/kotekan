#ifndef FILE_WRITE
#define FILE_WRITE

#include "buffers.h"

struct fileWriteThreadArg {
    struct Buffer * buf;
    int disk_ID;
    int num_disks;
    int link_ID;
    int num_links;
    int buffer_depth;
    char * dataset_name;
    char * disk_base;
};

void file_write_thread(void * arg);

#endif