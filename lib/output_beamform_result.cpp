#include "output_beamform_result.h"

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
    
##OCCURS IN add_queue_set    
            // Read the results
        CHECK_CL_ERROR( clEnqueueReadBuffer(cl_data->queue[2],
                                            cl_data->device_beamform_output_buffer[buffer_id],
                                            CL_FALSE,
                                            0,
                                            cl_data->beamforming_out_buf->aligned_buffer_size,
                                            cl_data->beamforming_out_buf->data[buffer_id],
                                            1,
                                            &cl_data->beamform_finished[buffer_id],
                                            &cl_data->beamform_read_finished[buffer_id]) );
##
}



