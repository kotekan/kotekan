#ifndef FILE_WRITE
#define FILE_WRITE

#include "buffers.h"

struct file_write_thread_arg {
    struct Buffer * buf;
    int diskID;
    int numDisks;
    int bufferDepth;
    char * dataset_name;
    char * disk_base;
};

void file_write_thread(void * arg);

#endif