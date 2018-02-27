#ifndef BEAMFORM_INCOHERENT_KERNEL_H
#define BEAMFORM_INCOHERENT_KERNEL_H

#include <vector>

#include "gpu_command.h"
#include "device_interface.h"

class beamform_incoherent_kernel: public gpu_command
{
public:
    beamform_incoherent_kernel(const char* param_name, Config& param_Config);
    beamform_incoherent_kernel(const char* param_gpuKernel, const char* param_name, Config& param_Config, const string &unique_name);
    ~beamform_incoherent_kernel();
    virtual void build(device_interface& param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:

    void apply_config(const uint64_t& fpga_seq) override;

    cl_mem device_mask;

    vector<int32_t> _element_mask;
    vector<int32_t> _inverse_product_remap;
    float _scale_factor;
};



#endif /* BEAMFORM_INCOHERENT_KERNEL_H */

