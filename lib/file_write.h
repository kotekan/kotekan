#ifndef FILE_WRITE
#define FILE_WRITE

#include "buffers.h"

struct file_write_thread_arg {
    struct Buffer * buf;
    int num_links;
    int bufferDepth;
    char * dataset_name;
    char * data_dir;
};

void file_write_thread(void * arg);

#endif