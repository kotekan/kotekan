#ifndef GPU_THREAD_H
#define GPU_THREAD_H


#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b) & 0x0F)

#define SDK_SUCCESS 0

//check pagesize:
//getconf PAGESIZE
// result: 4096
#define PAGESIZE_MEM 4096

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "pthread.h"

// This adjusts the number of queues used by the OpenCL runtime
// One queue is for data transfers to the GPU, one is for kernels,
// and one is for data transfers from the GPU to host memory.
// Unless you really know what you are doing, don't change this.
#define NUM_QUEUES 3

// The maximum number of expected GPUs in a host.  Increase as needed.
#define MAX_GPUS 4


struct gpuThreadArgs {
    struct Config * config;

    struct Buffer * in_buf;
    struct Buffer * out_buf;
    struct Buffer * beamforming_out_buf;

    int started;
    int gpu_id;

    pthread_mutex_t lock;  // Lock for the is_ready function.
    pthread_cond_t cond;
};

struct StreamINFO {
    stream_id_t stream_id;
    // Add time tracking of some kind.
};
void* gpu_thread(void * arg);
void wait_for_gpu_thread_ready(struct gpuThreadArgs * args);
void CL_CALLBACK read_complete(cl_event param_event, cl_int param_status, void *data);

#endif