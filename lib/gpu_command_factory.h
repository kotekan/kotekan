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
#include "beamform_data_stage.h"
#include "beamform_kernel.h"
#include "beamform_phase_data.h"
#include "output_beamform_result.h"
#include "dummy_placeholder_kernel.h"
#include "beamform_incoherent_kernel.h"
#include "output_beamform_incoh_result.h"

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
    cl_uint numCommands;
    cl_uint currentCommandCnt;
    
    int use_beamforming;
};

#endif // GPU_COMMAND_FACTORY_H
