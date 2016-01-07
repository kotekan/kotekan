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

#ifndef GPU_COMMAND_H
#define GPU_COMMAND_H

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "config.h"
#include "errors.h"
#include <stdio.h>
#include "device_interface.h"
#include "assert.h"
#include "buffers.h"

class gpu_command
{
public:
    gpu_command();
    gpu_command(char param_gpuKernel);//, cl_device_id *param_DeviceID, cl_context param_Context);
    //gpu_command getCommand() const;
    //size_t* getGWS() const;
    //size_t* getLWS() const;
    void setPreceedEvent(cl_event param_Event);
    cl_event getPreceedEvent();
    void setPostEvent(int param_BufferID);
    cl_event getPostEvent();
    virtual void build(Config* param_Config, class device_interface& param_Device);
    
    void setKernelArg(int param_ArgPos, cl_mem param_Buffer);
    
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device)=0;
    virtual void cleanMe(int param_BufferID);
    virtual void freeMe();
protected:
  //void createThisEvent(const class device_interface &param_device);
  cl_kernel kernel;
  cl_program program;
  
  // Kernel values.
  size_t gws[3]; // TODO Rename to something more meaningful - or comment.
  size_t lws[3];
  
  // Kernel Events
  cl_event preceedEvent;
  cl_event postEvent;
  //cl_event * postEventArray;
  
  char * gpuKernel;
};

#endif // GPU_COMMAND_H
