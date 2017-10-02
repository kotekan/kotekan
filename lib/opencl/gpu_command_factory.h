#ifndef GPU_COMMAND_FACTORY_H
#define GPU_COMMAND_FACTORY_H

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

#include "Config.hpp"
#include "gpu_command.h"
#include "correlator_kernel.h"
#include "offset_kernel.h"
#include "preseed_kernel.h"
#include "device_interface.h"
#include "input_data_stage.h"
#include "output_data_result.h"
#include "callbackdata.h"
#include "beamform_kernel.h"
#include "beamform_phase_data.h"
#include "output_beamform_result.h"
#include "beamform_incoherent_kernel.h"
#include "output_beamform_incoh_result.h"
#include "rfi_kernel.h"
#include "output_rfi.h"

class gpu_command_factory
{
public:
    gpu_command_factory(class device_interface & param_Device, Config& param_Config, const string& unique_name);
    //void initializeCommands(class device_interface & param_Device, Config& param_Config);
    gpu_command* getNextCommand();//device_interface& param_Device, int param_BufferID);
    cl_uint getNumCommands() const;
    void deallocateResources();

protected:
    //gpu_command ** list_commands;
    cl_uint num_commands;
    cl_uint current_command_cnt;

    int use_beamforming;
    //int use_incoh_beamforming;

    Config &config;
    device_interface &device;
    string unique_name;

    vector<gpu_command *> list_commands;
};

#endif // GPU_COMMAND_FACTORY_H
