#ifndef DEVICE_INTERFACE_H
#define DEVICE_INTERFACE_H

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h>
#endif


// The maximum number of expected GPUs in a host.  Increase as needed.
#define MAX_GPUS 4

// This adjusts the number of queues used by the OpenCL runtime
// One queue is for data transfers to the GPU, one is for kernels,
// and one is for data transfers from the GPU to host memory.
// Unless you really know what you are doing, don't change this.
#define NUM_QUEUES 3

#include "gpu_command.h"

class device_interface
{
public:
    device_interface();
    device_interface(struct Buffer* param_In_Buf, struct Buffer* param_Out_Buf, Config* param_Config, int param_GPU_ID);
    Buffer* getInBuf();
    Buffer* getOutBuf();
    cl_context getContext();
    int getGpuID();
    cl_device_id getDeviceID(int param_GPUID);
    //cl_mem getIDxMap();
    //cl_mem getIDyMap();
    cl_mem getInputBuffer(int param_BufferID);
    cl_mem getOutputBuffer(int param_BufferID);
    cl_mem getAccumulateBuffer(int param_BufferID);
    cl_command_queue getQueue(int param_Dim);
    cl_int* getAccumulateZeros();
    int getNumBlocks();
    int getAlignedAccumulateLen() const;
    void prepareCommandQueue();
    void allocateMemory();

    //defineOutputDataMap(Config *param_Config);


    void release_events_for_buffer(int param_BufferID);
    void deallocateResources();
protected:
    // Buffer objects
    struct Buffer * in_buf;
    struct Buffer * out_buf;
    // Extra data
    struct Config * config;

    int accumulate_len;
    int aligned_accumulate_len;
    int gpu_id; // Internal GPU ID.
    int num_blocks;

    cl_platform_id platform_id;
    cl_device_id device_id[MAX_GPUS];
    cl_context context;
    cl_command_queue queue[NUM_QUEUES];

    // Device Buffers
    cl_mem * device_input_buffer;
    cl_mem * device_accumulate_buffer;
    cl_mem * device_output_buffer;

    // Buffer of zeros to zero the accumulate buffer on the device.
    cl_int * accumulate_zeros;



    int output_len;
};

#endif // DEVICE_INTERFACE_H

#ifndef DEVICE_INTERFACE
#define DEVICE_INTERFACE

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "buffers.h"
#include "config.h"
#include "errors.h"
#include <assert.h>
#include <sys/mman.h>
#include <errno.h>
#include "string.h"
#include <stdio.h>

//check pagesize:
//getconf PAGESIZE
// result: 4096
#define PAGESIZE_MEM 4096





#endif // DEVICE_INTERFACE