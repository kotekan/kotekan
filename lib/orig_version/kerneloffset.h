#ifndef KERNELOFFSET_H
#define KERNELOFFSET_H

#include "gpu_command.h"
#include "device_interface.h"

class kernelOffset: public gpu_command
{
public:
    kernelOffset();
    kernelOffset(char* param_gpuKernel);
    ~kernelOffset();
    virtual void build(Config* param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
};

#endif // KERNELOFFSET_H
