#ifndef BEAMFORMING_POST_PROCESS
#define BEAMFORMING_POST_PROCESS

#include "config.h"
#include "buffers.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VDIFHeader {
    uint32_t seconds : 30;
    uint32_t legacy : 1;
    uint32_t invalid : 1;
    uint32_t data_frame : 24;
    uint32_t ref_epoch : 6;
    uint32_t unused : 2;
    uint32_t frame_len : 24;
    uint32_t log_num_chan : 5;
    uint32_t vdif_version : 3;
    uint32_t station_id : 16;
    uint32_t therad_id : 10;
    uint32_t bits_depth : 5;
    uint32_t data_type : 1;
    uint32_t eud1 : 24;
    uint32_t edv : 8;
    uint32_t eud2 : 32;
    uint32_t eud3 : 32;
    uint32_t eud4 : 32;
};

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