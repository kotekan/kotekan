#ifndef OUTPUT_POWER_H
#define OUTPUT_POWER_H

#include "buffers.h"

#ifdef __cplusplus
extern "C" {
#endif

struct output_power_thread_arg {
    struct Buffer * buf;
    char * ram_disk;
    int num_timesamples;
    int integration_samples;
    int legacy_output;
    int num_freq;
};

void *output_power_thread(void * arg);

#ifdef __cplusplus
}
#endif

#endif
