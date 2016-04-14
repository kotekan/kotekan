#ifndef KERNELPRESEED_H
#define KERNELPRESEED_H

#include "gpu_command.h"
#include "device_interface.h"

class kernelPreseed: public gpu_command
{
public:
    kernelPreseed();
    kernelPreseed(char* param_gpuKernel);
    ~kernelPreseed();
    virtual void build(Config* param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
protected:
    void defineOutputDataMap(Config* param_Config, int param_num_blocks, device_interface& param_Device);
    //Host Buffers
    cl_mem id_x_map;
    cl_mem id_y_map;
};

#endif // KERNELPRESEED_H
