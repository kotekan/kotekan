#ifndef SIMPLE_DNA_CAP
#define SIMPLE_DNA_CAP

#include "buffers.h"

struct dnaCapArgs {
    struct Buffer * buf;
    int buffer_depth;
    int packet_size;
    int dna_id;
    int close_on_block;
};

void simple_dna_cap(void * arg);

#endif