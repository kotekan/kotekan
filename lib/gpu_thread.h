/*
 * Copyright (c) 2015 <copyright holder> <email>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef GPU_THREAD_H
#define GPU_THREAD_H

// pthread_mutex_t status_lock;
// pthread_cond_t status_cond;

#endif // GPU_THREAD_H
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
void* gpu_thread(void * arg); 
void wait_for_gpu_thread_ready(struct gpuThreadArgs * args);
void CL_CALLBACK read_complete(cl_event param_event, cl_int param_status, void *data);

#endif