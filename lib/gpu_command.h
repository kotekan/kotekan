#ifndef GPU_COMMAND_H
#define GPU_COMMAND_H

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "config.h"
#include "errors.h"
#include <stdio.h>
#include "device_interface.h"
#include "assert.h"
#include "buffers.h"

class gpu_command
{
public:
    gpu_command(char* param_name);
    gpu_command(char * param_gpuKernel, char* param_name);//, cl_device_id *param_DeviceID, cl_context param_Context);
    virtual ~gpu_command();
    cl_event getPreceedEvent();
    cl_event getPostEvent();
    char* get_name();
    char* get_cl_options(Config * param_Config);
    virtual void build(Config* param_Config, class device_interface& param_Device);

    void setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer);

    virtual cl_event execute(int param_bufferID, device_interface& param_Device, cl_event param_PrecedeEvent);
    virtual void cleanMe(int param_BufferID);
    virtual void freeMe();
protected:
    cl_kernel kernel;
    cl_program program;

    // Kernel values.
    size_t gws[3]; // TODO Rename to something more meaningful - or comment.
    size_t lws[3];

    // Kernel Events
    cl_event * precedeEvent;
    cl_event * postEvent;

    int gpuCommandState = 0;//Default state to non-kernel executing command. 1 means kernel is defined with this command.

    char * gpuKernel = NULL;
    char* name;
};

#endif // GPU_COMMAND_H

