#ifndef CL_RFI_INPUT_SUM_H
#define CL_RFI_INPUT_SUM_H

#include "gpu_command.h"
#include "device_interface.h"
#include "restServer.hpp"
#include <mutex>

class clRfiInputSum: public gpu_command
{

public:

    clRfiInputSum(const char* param_name, Config &config);

    clRfiInputSum(const char* param_gpuKernel, const char* param_name, Config &config, const string &unique_name);

    ~clRfiInputSum();

    virtual void build(device_interface &param_Device) override;

    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;

    void rest_callback(connectionInstance& conn, json& json_request);

protected:

    void apply_config(const uint64_t& fpga_seq) override;

private:

    /// Length of the input frame, should be sizeof_float x n_elem x n_freq x nsamp / sk_step
    uint32_t input_frame_len;
    /// Length of the input frame, should be sizeof_float x n_freq x nsamp / sk_step
    uint32_t output_frame_len;

    /// Integration length of spectral kurtosis estimate in time
    uint32_t _sk_step;
    /// The total number of faulty inputs
    uint32_t _num_bad_inputs;
    /// The total integration length of the spectral kurtosis estimate
    uint32_t _M;

};

#endif

