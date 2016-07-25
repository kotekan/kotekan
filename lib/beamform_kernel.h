#ifndef BEAMFORM_KERNEL_H
#define BEAMFORM_KERNEL_H

#include "gpu_command.h"
#include "device_interface.h"

class beamform_kernel: public gpu_command
{
public:
    beamform_kernel(char* param_name);
    beamform_kernel(char* param_gpuKernel, char* param_name);
    ~beamform_kernel();
    virtual void build(Config* param_Config, class device_interface& param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
protected:
    cl_mem device_mask;
        
};

#endif

