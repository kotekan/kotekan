#ifndef FILE_WRITE
#define FILE_WRITE

#include "buffers.h"

struct fileWriteThreadArg {
    struct Buffer * buf;
    int num_links;
    int buffer_depth;
    char * dataset_name;
    char * data_dir;

    int actual_num_freq;
    int actual_num_elements;
};

void file_write_thread(void * arg);

#endif