#ifndef RAW_STREAM_VDIF_H
#define RAW_STREAM_VDIF_H

#include "buffers.h"

#ifdef __cplusplus
extern "C" {
#endif

struct stream_raw_vdif_arg {
    struct Buffer * buf;
};

void *stream_raw_vdif(void * arg);

#ifdef __cplusplus
}
#endif

#endif
