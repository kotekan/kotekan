#ifndef BEAMFORMING_POST_PROCESS
#define BEAMFORMING_POST_PROCESS

#include "config.h"
#include "buffers.h"

#ifdef __cplusplus
extern "C" {
#endif

struct BeamformingPostProcessArgs {
    struct Config * config;
    struct Buffer * in_buf;
    struct Buffer * out_buf;
};

void* beamforming_post_process(void * arg);

#ifdef __cplusplus
}
#endif

#endif