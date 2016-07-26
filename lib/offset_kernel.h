#ifndef OFFSET_KERNEL_H
#define OFFSET_KERNEL_H

#include "gpu_command.h"
#include "device_interface.h"

class offset_kernel: public gpu_command
{
public:
    offset_kernel(char* param_name);
    offset_kernel(char* param_gpuKernel, char* param_name);
    ~offset_kernel();
    virtual void build(Config* param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
};

#endif // KERNELOFFSET_H

