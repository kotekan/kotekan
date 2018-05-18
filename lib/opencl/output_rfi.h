#ifndef OUTPUT_RFI_H
#define OUTPUT_RFI_H

#include "clCommand.hpp"
#include "callbackdata.h"

class output_rfi: public clCommand
{
public:
    output_rfi(const char* param_name, Config &config, const string &unique_name);
    ~output_rfi();
    virtual void build(device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:

};

#endif
