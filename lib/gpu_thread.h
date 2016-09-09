
#ifndef GPU_THREAD_H
#define GPU_THREAD_H


#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b) & 0x0F)

#define SDK_SUCCESS 0

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <pthread.h>
#include "fpga_header_functions.h"

struct gpuThreadArgs {
    struct Config * config;

    struct Buffer * in_buf;
    struct Buffer * out_buf;
    struct Buffer * beamforming_out_buf;
    struct Buffer * beamforming_out_incoh_buf;

    int started;
    int gpu_id;

    pthread_mutex_t lock;  // Lock for the is_ready function.
    pthread_cond_t cond;

    pthread_barrier_t * barrier;
};
void* gpu_thread(void * arg);
void wait_for_gpu_thread_ready(struct gpuThreadArgs * args);
void CL_CALLBACK read_complete(cl_event param_event, cl_int param_status, void *data);

#endif