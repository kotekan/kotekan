#ifndef GPU_THREAD
#define GPU_THREAD

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

    int started;
    int gpu_id;

    pthread_mutex_t lock;  // Lock for the is_ready function.
    pthread_cond_t cond;
};

struct callBackData {
    int buffer_id;

    struct OpenCLData * cl_data;
};

struct OpenCLData {
    cl_context context;
    cl_program program;
    cl_kernel corr_kernel; /// Correlation Kernel
    cl_kernel offset_accumulate_kernel;
    cl_kernel preseed_kernel;
    cl_device_id device_id[MAX_GPUS];
    cl_platform_id platform_id;

    cl_command_queue queue[NUM_QUEUES];

    // Buffer of zeros to zero the accumulate buffer on the device.
    cl_int * accumulate_zeros;

    // Device Buffers
    cl_mem * device_input_buffer;
    cl_mem * device_output_buffer;
    cl_mem * device_accumulate_buffer;

    // User events.
    cl_event * host_buffer_ready;
    cl_event * input_data_written;
    cl_event * accumulate_data_zeroed;
    cl_event * offset_accumulate_finished;
    cl_event * preseed_finished;
    cl_event * corr_finished;
    cl_event * read_finished;

    // Call back data.
    struct callBackData * cb_data;

    // Extra data
    struct Config * config;

    int num_blocks;
    int output_len;
    int accumulate_len;
    int aligned_accumulate_len;
    int gpu_id; // Internal GPU ID.

    // Kernel values.
    unsigned int num_accumulations;
    size_t gws_corr[3]; // TODO Rename to something more meaningful - or comment.
    size_t lws_corr[3];

    size_t gws_accum[3];
    size_t lws_accum[3];

    size_t gws_preseed[3];
    size_t lws_preseed[3];

    // Buffer objects
    struct Buffer * in_buf;
    struct Buffer * out_buf;

    // Locks
    pthread_mutex_t queue_lock;
    pthread_mutex_t status_lock;
    pthread_cond_t status_cond;
};

void gpu_thread(void * arg); 

void wait_for_gpu_thread_ready(struct gpuThreadArgs * args);

void CL_CALLBACK read_complete(cl_event event, cl_int status, void *data);

#endif