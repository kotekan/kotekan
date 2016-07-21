#include "input_data_stage.h"

input_data_stage::input_data_stage(char* param_name):gpu_command(param_name)
{

}

input_data_stage::~input_data_stage()
{

}


void input_data_stage::build(Config *param_Config, class device_interface &param_Device)
{
    gpu_command::build(param_Config, param_Device);
    data_staged_event = (cl_event *)malloc(param_Device.getInBuf()->num_buffers * sizeof(cl_event));
    //input_data_written = (cl_event)malloc(sizeof(cl_event));
    CHECK_MEM(data_staged_event);

    //gpu_command::createThisEvent(param_Device);
}

cl_event input_data_stage::execute(int param_bufferID, class device_interface& param_Device, cl_event param_PrecedeEvent)
{
    cl_int err;
    int numEvents;

    gpu_command::execute(param_bufferID, param_Device, param_PrecedeEvent);

    //cl_event postEvent;

    //postEvent = thisPostEvent[param_bufferID];

    if (param_PrecedeEvent==NULL){
        //precedeEvent[param_bufferID]=clCreateUserEvent(param_Device.getContext(), &err);
        //CHECK_CL_ERROR(err);
        numEvents = 0;
    }
    else
        numEvents = 1;

    // Data transfer to GPU
    CHECK_CL_ERROR( clEnqueueWriteBuffer(param_Device.getQueue(0),
                                            param_Device.getInputBuffer(param_bufferID),
                                            CL_FALSE,
                                            0, //offset
                                            param_Device.getInBuf()->aligned_buffer_size,
                                            param_Device.getInBuf()->data[param_bufferID],
                                            //numEvents,
                                            //&precedeEvent[param_bufferID], // Wait on this user event (network finished).
                                            0,NULL,
                                            //&precedeEvent[param_bufferID],
                                            //1, &param_PrecedeEvent,
					    &data_staged_event[param_bufferID]) );

    CHECK_CL_ERROR( clEnqueueWriteBuffer(param_Device.getQueue(0),
                                            param_Device.getAccumulateBuffer(param_bufferID),
                                            CL_FALSE,
                                            0,
                                            param_Device.getAlignedAccumulateLen(),
                                            param_Device.getAccumulateZeros(),
                                            1,
                                            //&input_data_written[param_bufferID],
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

