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

#include "initqueuesequence_command.h"
#include "device_interface.h"

void initQueueSequence_command::build(const Config &param_Config, const device_interface &param_Device)
{
    input_data_written = malloc(param_Device.getInBuf()->num_buffers * sizeof(cl_event));
    CHECK_MEM(input_data_written);
    
    gpu_command::createThisEvent(param_Device);
}

cl_event initQueueSequence_command::execute(int param_bufferID, const device_interface &param_Device)
{
  cl_int err;
    
    cl_event *postEvent;
    
    postEvent = thisPostEvent[param_bufferID];
    
    // Data transfer to GPU
    CHECK_CL_ERROR( clEnqueueWriteBuffer(param_Device.getQueue()[0],
                                            param_Device.getInputBuffer(param_bufferID),
                                            CL_FALSE,
                                            0, //offset
                                            param_Device.getInBuf()->aligned_buffer_size,
                                            param_Device.getInBuf()->data[param_bufferID],
                                            1,
                                            &preceedEvent, // Wait on this user event (network finished).
                                            &input_data_written[param_bufferID]) );

    CHECK_CL_ERROR( clEnqueueWriteBuffer(param_Device.getQueue()[0],
                                            param_Device.getAccumulateBuffer()[param_bufferID],
                                            CL_FALSE,
                                            0,
                                            param_Device.getAlignedAccumulateLen(),
                                            param_Device.getAccumulateZeros(), 
                                            1,
                                            &input_data_written[param_bufferID],
                                            &postEvent) ); 
    
    return postEvent;
}

void initQueueSequence_command::cleanMe(int param_BufferID)
{
  assert(input_data_written[param_BufferID] != NULL);
  
  clReleaseEvent(input_data_written[param_BufferID]);
  
  gpu_command.cleanMe(param_BufferID);

}

void initQueueSequence_command::freeMe()
{
  free(input_data_written);
  
  gpu_command.freeMe();
}
