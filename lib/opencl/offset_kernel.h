#ifndef OFFSET_KERNEL_H
#define OFFSET_KERNEL_H

#include "clCommand.hpp"
#include "clDeviceInterface.hpp"

class offset_kernel: public clCommand
{
public:
    offset_kernel(const char* param_name, Config &config);
    offset_kernel(const char* param_gpuKernel, const char* param_name, Config &config, const string &unique_name);
    ~offset_kernel();
    virtual void build(device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:
    void apply_config(const uint64_t& fpga_seq) override;
};

#endif // KERNELOFFSET_H

