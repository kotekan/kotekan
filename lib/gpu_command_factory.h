#ifndef GPU_COMMAND_FACTORY_H
#define GPU_COMMAND_FACTORY_H


#include "config.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "gpu_command.h"
#include "correlator_kernel.h"
#include "offset_kernel.h"
#include "preseed_kernel.h"
#include "device_interface.h"
#include "input_data_stage.h"
#include "output_data_result.h"
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
};

#endif // GPU_COMMAND_FACTORY_H
