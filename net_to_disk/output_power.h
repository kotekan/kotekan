#ifndef OUTPUT_POWER
#define OUTPUT_POWER

#include "buffers.h"

struct output_power_thread_arg {
    struct Buffer * buf;
    int diskID;
    int numDisks;
    int bufferDepth;
    char * dataset_name;
    char * disk_base;

    int num_frames;
    int num_inputs;
    int num_freq;
    int offset;
};

void output_power_thread(void * arg);

#endif