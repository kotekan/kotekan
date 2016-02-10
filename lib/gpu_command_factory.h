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

#ifndef GPU_COMMAND_FACTORY_H
#define GPU_COMMAND_FACTORY_H


#include "config.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "gpu_command.h"
#include "kernelcorrelator.h"
#include "kerneloffset.h"
#include "kernelpreseed.h"
#include "device_interface.h"
#include "initqueuesequence_command.h"
#include "finalqueuesequence_command.h"
#include "callbackdata.h"


class gpu_command_factory
{
public:
    gpu_command_factory();
    void initializeCommands(class device_interface & param_Device, Config* param_Config);
    gpu_command* getNextCommand(device_interface& param_Device, int param_BufferID);
    cl_uint getNumCommands() const;
    void deallocateResources();
protected:
    gpu_command ** listCommands;
    cl_uint numCommands = 0;
    cl_uint currentCommandCnt;
    
    // Call back data.
    //struct callBackData * cb_data;    
};

#endif // GPU_COMMAND_FACTORY_H
