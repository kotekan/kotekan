/*
 * Copyright (c) 2015 <copyright holder> <email>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

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

