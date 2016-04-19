#ifndef FILE_WRITE
#define FILE_WRITE

#include "buffers.h"

struct raw_file_write_thread_arg {
    struct Buffer * buf;
    int diskID;
    int numDisks;
    int bufferDepth;
    char * dataset_name;
    char * disk_base;
    char * disk_set;
};

void raw_file_write_thread(void * arg);

#endif
