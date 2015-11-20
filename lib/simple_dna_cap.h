#ifndef SIMPLE_DNA_CAP
#define SIMPLE_DNA_CAP

#include "buffers.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dnaCapArgs {
    struct Buffer * buf;
    int buffer_depth;
    int packet_size;
    int dna_id;
    int close_on_block;
    int integration_edge;
};

void simple_dna_cap(void * arg);

#ifdef __cplusplus
}
#endif

#endif