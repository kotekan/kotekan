
#ifndef OUTPUT_DATA_RESULT_H
#define OUTPUT_DATA_RESULT_H

#include "gpu_command.h"
#include "callbackdata.h"

class output_data_result: public gpu_command
{
public:
    output_data_result(char* param_name);
    ~output_data_result();
    virtual void build(Config *param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);

protected:

};

#endif

