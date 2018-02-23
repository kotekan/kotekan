#ifndef PRESEED_KERNEL_H
#define PRESEED_KERNEL_H

#include "gpu_command.h"
#include "device_interface.h"

class preseed_kernel: public gpu_command
{
public:
    preseed_kernel(const char* param_name, Config &config);
    preseed_kernel(const char* param_gpuKernel, const char* param_name, Config &config, const string &unique_name);
    ~preseed_kernel();
    virtual void build(device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:
    void defineOutputDataMap(device_interface& param_Device);
    //Host Buffers
    cl_mem id_x_map;
    cl_mem id_y_map;
};

#endif
