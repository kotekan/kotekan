#ifndef BEAMFORM_DATA_STAGE_H
#define BEAMFORM_DATA_STAGE_H

#include "gpu_command.h"
#include "callbackdata.h"

class beamform_data_stage: public gpu_command
{
public:
    beamform_data_stage(char* param_name);
    ~beamform_data_stage();
    virtual void build(Config *param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);

protected:

};

#endif

