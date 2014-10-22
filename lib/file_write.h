#ifndef FILE_WRITE
#define FILE_WRITE

#include "buffers.h"

struct fileWriteThreadArg {
    struct Config * config;
    struct Buffer * buf;
    char * dataset_name;
    char * data_dir;
};

void file_write_thread(void * arg);

#endif