#ifndef INITQUEUESEQUENCE_COMMAND_H
#define INITQUEUESEQUENCE_COMMAND_H

#include "gpu_command.h"

class initQueueSequence_command: public gpu_command
{
public:
    initQueueSequence_command();
    ~initQueueSequence_command();
    virtual void build(Config *param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface& param_Device, cl_event param_PrecedeEvent);
    virtual void cleanMe(int param_BufferID);
    virtual void freeMe();
protected:
    //cl_event * input_data_written;
    cl_event * input_data_written;
};

#endif // INITQUEUESEQUENCE_COMMAND_H
