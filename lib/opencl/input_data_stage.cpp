#include "input_data_stage.h"

input_data_stage::input_data_stage(const char* param_name, Config &param_config, const string &unique_name):
    gpu_command(param_name, param_config, unique_name)
{

}

input_data_stage::~input_data_stage()
{

}

void input_data_stage::build(device_interface &param_Device)
{
    gpu_command::apply_config();
    gpu_command::build(param_Device);
    data_staged_event = (cl_event *)malloc(param_Device.getInBuf()->num_frames * sizeof(cl_event));
    CHECK_MEM(data_staged_event);
}

cl_event input_data_stage::execute(int param_bufferID, device_interface& param_Device, cl_event param_PrecedeEvent)
{
    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);

    // Data transfer to GPU
    CHECK_CL_ERROR( clEnqueueWriteBuffer(param_Device.getQueue(0),
                                            param_Device.getInputBuffer(param_bufferID),
                                            CL_FALSE,
                                            0, //offset
                                            param_Device.getInBuf()->aligned_frame_size,
                                            param_Device.getInBuf()->frames[param_bufferID],
                                            0,
                                            NULL,
                                            &data_staged_event[param_bufferID]) );

    CHECK_CL_ERROR( clEnqueueWriteBuffer(param_Device.getQueue(0),
                                            param_Device.getAccumulateBuffer(param_bufferID),
                                            CL_FALSE,
                                            0,
                                            param_Device.getAlignedAccumulateLen(),
                                            param_Device.getAccumulateZeros(),
                                            1,
                                            &data_staged_event[param_bufferID],
                                            &postEvent[param_bufferID]) );

    return postEvent[param_bufferID];
}

void input_data_stage::cleanMe(int param_BufferID)
{
    gpu_command::cleanMe(param_BufferID);

    assert(data_staged_event[param_BufferID] != NULL);

    clReleaseEvent(data_staged_event[param_BufferID]);

}

void input_data_stage::freeMe()
{
    gpu_command::freeMe();
    free(data_staged_event);
}

