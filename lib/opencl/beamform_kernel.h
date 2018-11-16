#ifndef BEAMFORM_KERNEL_H
#define BEAMFORM_KERNEL_H

#include "gpu_command.h"
#include "device_interface.h"

#include <vector>

class beamform_kernel: public gpu_command
{
public:
    beamform_kernel(const char* param_name, Config &param_config);
    beamform_kernel(const char* param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name);
    ~beamform_kernel();
    virtual void build(class device_interface& param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, class device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:

    cl_mem device_mask;

    vector<int32_t> _element_mask;
    vector<int32_t> _product_remap;
    vector<int32_t> _inverse_product_remap;
    uint32_t _scale_factor;
};

#endif

