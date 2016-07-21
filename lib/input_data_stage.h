#ifndef INPUT_DATA_STAGE_H
#define INPUT_DATA_STAGE_H

#include "gpu_command.h"

class input_data_stage: public gpu_command
{
public:
    input_data_stage(char* param_name);
    ~input_data_stage();
    virtual void build(Config *param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface& param_Device, cl_event param_PrecedeEvent);
    virtual void cleanMe(int param_BufferID);
    virtual void freeMe();
protected:
    cl_event * data_staged_event;
};

#endif // INITQUEUESEQUENCE_COMMAND_H

