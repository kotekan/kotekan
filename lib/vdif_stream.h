#ifndef VDIF_STREAM
#define VDIF_STREAM

#include "config.h"
#include "buffers.h"

struct VDIFstreamArgs {
    struct Config * config;
    struct Buffer * buf;
};

void vdif_stream(void * arg);

#endif