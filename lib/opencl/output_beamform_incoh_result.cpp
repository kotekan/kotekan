#include "output_beamform_incoh_result.h"

output_beamform_incoh_result::output_beamform_incoh_result(const char* param_name, Config &param_config, const string &unique_name):
    gpu_command(param_name, param_config, unique_name)
{

}

output_beamform_incoh_result::~output_beamform_incoh_result()
{

}

void output_beamform_incoh_result::build(device_interface &param_Device)
{
    gpu_command::apply_config();
    gpu_command::build(param_Device);
}

cl_event output_beamform_incoh_result::execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);

    CHECK_CL_ERROR( clEnqueueReadBuffer(param_Device.getQueue(2),
                                        param_Device.get_device_beamform_output_incoh_buffer(param_bufferID),
                                        CL_FALSE,
                                        0,
                                        param_Device.get_beamforming_out_incoh_buf()->aligned_frame_size,
                                        param_Device.get_beamforming_out_incoh_buf()->frames[param_bufferID],
                                        1,
                                        &param_PrecedeEvent,
                                        &postEvent[param_bufferID]) );

    return postEvent[param_bufferID];

}




