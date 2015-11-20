#ifndef FILE_WRITE
#define FILE_WRITE

#include "buffers.h"

#ifdef __cplusplus
extern "C" {
#endif

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

void* file_write_thread(void * arg);

#ifdef __cplusplus
}
#endif

#endif