#ifndef PRESEED_KERNEL_H
#define PRESEED_KERNEL_H

#include "gpu_command.h"
#include "device_interface.h"

class preseed_kernel: public gpu_command
{
public:
    preseed_kernel(char* param_name);
    preseed_kernel(char* param_gpuKernel, char* param_name);
    ~preseed_kernel();
    virtual void build(Config* param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
protected:
    void defineOutputDataMap(Config* param_Config, int param_num_blocks, device_interface& param_Device);
    //Host Buffers
    cl_mem id_x_map;
    cl_mem id_y_map;
};

#endif 
