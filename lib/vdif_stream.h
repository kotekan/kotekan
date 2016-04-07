#ifndef VDIF_STREAM
#define VDIF_STREAM

#include "config.h"
#include "buffers.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VDIFstreamArgs {
    struct Config * config;
    struct Buffer * buf;
};

void* vdif_stream(void * arg);

#ifdef __cplusplus
}
#endif

#endif