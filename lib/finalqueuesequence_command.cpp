#include "finalqueuesequence_command.h"

finalQueueSequence_Command::finalQueueSequence_Command():gpu_command()
{

}

finalQueueSequence_Command::~finalQueueSequence_Command()
{

}

//void finalQueueSequence_Command::setCBData(callBackData* param_CBData)
//{
//  cb_data = param_CBData;
//}
void finalQueueSequence_Command::build(Config* param_Config, class device_interface &param_Device)
{
      gpu_command::build(param_Config, param_Device);
}

cl_event finalQueueSequence_Command::execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent)
{
    //cl_event curPostEvent;

    //curPostEvent = thisPostEvent[param_bufferID];

    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);

    // Read the results
    CHECK_CL_ERROR( clEnqueueReadBuffer(param_Device.getQueue(2),
                                            param_Device.getOutputBuffer(param_bufferID),
                                            CL_FALSE,
                                            0,
                                            param_Device.getOutBuf()->aligned_buffer_size,
                                            param_Device.getOutBuf()->data[param_bufferID],
                                            1,
                                            //&precedeEvent[param_bufferID],
                                            &param_PrecedeEvent,
					    &postEvent[param_bufferID]) );
                                            //&curPostEvent) );

    // Setup call back.
    //CHECK_CL_ERROR( clSetEventCallback(postEvent,
                                            //CL_COMPLETE,
                                            //param_Device.read_complete(),
                                            //&cb_data) );
    //return curPostEvent;

    return postEvent[param_bufferID];

}

//void finalQueueSequence_Command::freeMe()
//{
//  free(cb_data);
//}

