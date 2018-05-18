#ifndef CORRELATOR_KERNEL_H
#define CORRELATOR_KERNEL_H

#include "clCommand.hpp"
#include "clDeviceInterface.hpp"

class correlator_kernel: public clCommand
{
public:
    correlator_kernel(const char* param_name, Config &config);
    correlator_kernel(const char* param_gpuKernel, const char* param_name, Config &config, const string &unique_name);
    ~correlator_kernel();
    virtual void build(class device_interface& param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:
    void apply_config(const uint64_t& fpga_seq) override;
    void defineOutputDataMap(device_interface& param_Device);
    cl_mem device_block_lock;
    cl_int *zeros;

    //Host Buffers
    cl_mem id_x_map;
    cl_mem id_y_map;
};

#endif
