#ifndef FINALQUEUESEQUENCE_COMMAND_H
#define FINALQUEUESEQUENCE_COMMAND_H

#include "gpu_command.h"
#include "callbackdata.h"

class finalQueueSequence_Command: public gpu_command
{
public:
    finalQueueSequence_Command();
    ~finalQueueSequence_Command();
    //void setCBData(callBackData * param_CBData);
    virtual void build(Config *param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
    //virtual freeMe();
protected:
  //callBackData * cb_data;

};

#endif // FINALQUEUESEQUENCE_COMMAND_H
