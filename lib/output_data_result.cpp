#include "output_data_result.h"

output_data_result::output_data_result(char* param_name):gpu_command(param_name)
{

}

output_data_result::~output_data_result()
{

}

void output_data_result::build(Config* param_Config, class device_interface &param_Device)
{
      gpu_command::build(param_Config, param_Device);
}

cl_event output_data_result::execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent)
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


