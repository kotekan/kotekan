#ifndef CL_RFI_OUTPUT_H
#define CL_RFI_OUTPUT_H

#include "gpu_command.h"
#include "callbackdata.h"

class clRfiOutput: public gpu_command
{
public:
    clRfiOutput(const char* param_name, Config &config, const string &unique_name);
    ~clRfiOutput();
    virtual void build(device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:

};

#endif
