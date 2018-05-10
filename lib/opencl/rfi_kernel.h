#ifndef RFI_KERNEL_H
#define RFI_KERNEL_H

#include "gpu_command.h"
#include "device_interface.h"
#include "restServer.hpp"
#include <mutex>

class rfi_kernel: public gpu_command
{
public:
    rfi_kernel(const char* param_name, Config &config);
    rfi_kernel(const char* param_gpuKernel, const char* param_name, Config &config, const string &unique_name);
    ~rfi_kernel();
    virtual void build(device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;
    void rest_callback(connectionInstance& conn, json& json_request);
protected:
    void apply_config(const uint64_t& fpga_seq) override;

private:

    int _sk_step;
    int _rfi_sensitivity;
    bool _rfi_zero;
    int zero;
    int link_id;
    int num_links_per_gpu;
    float sqrtM;
    float * Mean_Array;
    vector<cl_mem> mem_Mean_Array;
    Config config_local;
    std::mutex rest_callback_mutex;
    std::string endpoint;
};

#endif // RFI_KERNEL_H

