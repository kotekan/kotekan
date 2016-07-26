#include "beamform_data_stage.h"

beamform_data_stage::beamform_data_stage(char* param_name):gpu_command(param_name)
{

}

beamform_data_stage::~beamform_data_stage()
{

}

void beamform_data_stage::build(Config* param_Config, class device_interface &param_Device)
{
      gpu_command::build(param_Config, param_Device);
}

cl_event beamform_data_stage::execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);

    // Read the results
    CHECK_CL_ERROR( clEnqueueReadBuffer(param_Device.getQueue(2),
                                            param_Device.getOutputBuffer(param_bufferID),
                                            CL_FALSE,
                                            0,
                                            param_Device.getOutBuf()->aligned_buffer_size,
                                            param_Device.getOutBuf()->data[param_bufferID],
                                            1,
                                            &param_PrecedeEvent,
					    &postEvent[param_bufferID]) );

    return postEvent[param_bufferID];

}





