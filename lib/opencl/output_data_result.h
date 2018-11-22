#ifndef OUTPUT_DATA_RESULT_H
#define OUTPUT_DATA_RESULT_H

#include "gpu_command.h"
#include "callbackdata.h"

class output_data_result: public gpu_command
{
public:
    output_data_result(const char* param_name, Config &config, const string &unique_name);
    ~output_data_result();
    virtual void build(device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, device_interface &param_Device, cl_event param_PrecedeEvent) override;
protected:

};

#endif
