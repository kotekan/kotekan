#ifndef GPU_COMMAND_H
#define GPU_COMMAND_H

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

#include "Config.hpp"
#include "errors.h"
#include <stdio.h>
#include "device_interface.h"
#include "assert.h"
#include "buffers.h"
#include <string>

class gpu_command
{
public:
    gpu_command(const char* param_name, Config &param_config);
    gpu_command(const char * param_gpuKernel, const char* param_name, Config &param_config);//, cl_device_id *param_DeviceID, cl_context param_Context);
    virtual ~gpu_command();
    cl_event getPreceedEvent();
    cl_event getPostEvent();
    char* get_name();
    string get_cl_options();
    virtual void build(device_interface& param_Device);

    void setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer);

    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface& param_Device, cl_event param_PrecedeEvent);
    virtual void cleanMe(int param_BufferID);
    virtual void freeMe();

    virtual void apply_config(const uint64_t &fpga_seq);
protected:
    cl_kernel kernel;
    cl_program program;

    Config &config;

    // Kernel values.
    size_t gws[3]; // TODO Rename to something more meaningful - or comment.
    size_t lws[3];

    // Kernel Events
    cl_event * precedeEvent;
    cl_event * postEvent;

    int gpuCommandState;//Default state to non-kernel executing command. 1 means kernel is defined with this command.

    char * gpuKernel;
    char* name;

    // Common configuration values (which do not change in a run)
    int32_t _num_adjusted_elements;
    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    int32_t _num_data_sets;
    int32_t _num_adjusted_local_freq;
    int32_t _num_blocks;
    int32_t _block_size;
    int32_t _buffer_depth;
};

#endif // GPU_COMMAND_H

