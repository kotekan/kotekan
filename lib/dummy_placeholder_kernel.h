#ifndef CORRELATOR_KERNEL_H
#define DUMMY_PLACEHOLDER_KERNEL_H

#include "gpu_command.h"
#include "device_interface.h"

class dummy_placeholder_kernel: public gpu_command
{
public:
    dummy_placeholder_kernel(char* param_name);
    ~dummy_placeholder_kernel();
    virtual void build(Config* param_Config, class device_interface& param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
};

#endif


