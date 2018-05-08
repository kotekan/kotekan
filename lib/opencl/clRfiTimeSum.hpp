#ifndef CL_RFI_TIME_SUM_H
#define CL_RFI_TIME_SUM_H

#include "gpu_command.h"
#include "device_interface.h"
#include "restServer.hpp"
#include <mutex>

class clRfiTimeSum: public gpu_command
{

public:

    clRfiTimeSum(const char* param_name, Config &config);

    clRfiTimeSum(const char* param_gpuKernel, const char* param_name, Config &config, const string &unique_name);

    ~clRfiTimeSum();

    virtual void build(device_interface &param_Device) override;

    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;

    void rest_callback(connectionInstance& conn, json& json_request);

protected:

    void apply_config(const uint64_t& fpga_seq) override;

private:

    Config config_local;
    std::mutex rest_callback_mutex;
    
    cl_mem mem_input_mask;
    
    uint32_t _sk_step;
    uint32_t mask_len;
    uint8_t * Input_Mask;


};

#endif

