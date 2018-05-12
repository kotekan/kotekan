#include "clRfiOutput.hpp"

clRfiOutput::clRfiOutput(const char* param_name, Config &param_config, const string &unique_name):
    gpu_command(param_name, param_config, unique_name)
{
}

clRfiOutput::~clRfiOutput()
{
}

void clRfiOutput::build(device_interface &param_Device)
{
    apply_config(0);
    gpu_command::build(param_Device);
}

cl_event clRfiOutput::execute(int param_bufferID, const uint64_t& fpga_seq, class device_interface &param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);

    // Read the results
    CHECK_CL_ERROR( clEnqueueReadBuffer(param_Device.getQueue(0),
                                            param_Device.getRfiOutputBuffer(param_bufferID),
                                            CL_FALSE,
                                            0,
                                            param_Device.getRfiBuf()->frame_size,
                                            param_Device.getRfiBuf()->frames[param_bufferID],
                                            1,
                                            &param_PrecedeEvent,
					    &postEvent[param_bufferID]) );
    DEBUG("RFI output copied to buffer id %d",param_bufferID);
    return postEvent[param_bufferID];
}
