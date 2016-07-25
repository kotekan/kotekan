#include "gpu_thread.h"
#include "device_interface.h"
#include "gpu_command.h"
#include "gpu_command_factory.h"
#include "callbackdata.h"
#include "unistd.h"
#include "vdif_functions.h"
#include "fpga_header_functions.h"
#include <iostream>
#include "beamforming.h"

pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;

using namespace std;

void* gpu_thread(void* arg)
{
    struct gpuThreadArgs * args = (struct gpuThreadArgs *) arg;

    gpu_command * currentCommand;
    cl_event sequenceEvent;
    cl_int err;
    loopCounter * loopCnt = new loopCounter;

    device_interface device(args->in_buf, args->out_buf, args->config, args->gpu_id, args->beamforming_out_buf);

    gpu_command_factory factory;

    factory.initializeCommands(device, args->config);

    device.prepareCommandQueue();

    device.allocateMemory();
    DEBUG("Device Initialized\n");
    //device.defineOutputDataMap();//COPY INTO OFFSET KERNEL AND CORRELATOR KERNEL

    callBackData ** cb_data = new callBackData * [device.getInBuf()->num_buffers];

    for (int j=0;j<device.getInBuf()->num_buffers;j++)
        cb_data[j] = new callBackData(factory.getNumCommands());

    CHECK_MEM(cb_data);

    // Just wait on one buffer.
    int buffer_list[1] = {0};
    int bufferID = 0;

    CHECK_ERROR( pthread_mutex_lock(&args->lock) );
    args->started = 1;
    CHECK_ERROR( pthread_mutex_unlock(&args->lock) );

    // Signal consumer (main thread in this case).
    CHECK_ERROR( pthread_cond_broadcast(&args->cond) );
    // Main loop for collecting data from producers.
    for(;;) {
        // Wait for data, this call will block.
        bufferID = get_full_buffer_from_list(args->in_buf, buffer_list, 1);
        
        // If buffer id is -1, then all the producers are done.
        if (bufferID == -1) {
            break;
        }

        // Todo get/set time information here as well.

        CHECK_ERROR( pthread_mutex_lock(&loopCnt->lock));
        loopCnt->iteration++;
        CHECK_ERROR( pthread_mutex_unlock(&loopCnt->lock));

        // Set call back data
        cb_data[bufferID]->buffer_id = bufferID;
        cb_data[bufferID]->in_buf = device.getInBuf();
        cb_data[bufferID]->out_buf = device.getOutBuf();
        cb_data[bufferID]->numCommands = factory.getNumCommands();
        cb_data[bufferID]->cnt = loopCnt;
        if (args->config->gpu.use_beamforming == 1)
        {
            cb_data[bufferID]->use_beamforming = args->config->gpu.use_beamforming;
            cb_data[bufferID]->beamforming_out_buf = device.get_beamforming_out_buf();
        }

        sequenceEvent = NULL; //WILL THE INIT COMMAND WORK WITH A NULL PRECEEDING EVENT?

        DEBUG("cb_data initialized\n");

        for (int i = 0; i < factory.getNumCommands(); i++){
            currentCommand = factory.getNextCommand(device, bufferID);
            sequenceEvent = currentCommand->execute(bufferID, device, sequenceEvent);
            cb_data[bufferID]->listCommands[i] = currentCommand;
        }

        DEBUG("Commands Queued\n");
        // Setup call back.
        CHECK_CL_ERROR( clSetEventCallback(sequenceEvent,
                                            CL_COMPLETE,
                                            &read_complete,
                                            cb_data[bufferID]) );

        // Wait for the output buffer to be empty as well.
        // This should almost never block, since the output buffer should clear quickly.
        wait_for_empty_buffer(args->out_buf, bufferID);

        if (args->config->gpu.use_beamforming) {
            wait_for_empty_buffer(args->beamforming_out_buf, bufferID);
        }

        buffer_list[0] = (buffer_list[0] + 1) % args->in_buf->num_buffers;
    }

    //LOOP THROUGH THE LOCKING ROUTINE

    DEBUG("Closing\n");

    CHECK_ERROR( pthread_mutex_lock(&loopCnt->lock) );
    DEBUG("LockCnt\n");
    while ( loopCnt->iteration > 0) {
        pthread_cond_wait(&loopCnt->cond, &loopCnt->lock);
    }
    CHECK_ERROR( pthread_mutex_unlock(&loopCnt->lock) );

    DEBUG("LockConditionReleased\n");
    factory.deallocateResources();
    DEBUG("FactoryDone\n");
    device.deallocateResources();
    DEBUG("DeviceDone\n");

    mark_producer_done(args->out_buf, 0);

    delete loopCnt;
    delete[] cb_data;

    int ret;
    pthread_exit((void *) &ret);
}
void wait_for_gpu_thread_ready(struct gpuThreadArgs * args)
{
    CHECK_ERROR( pthread_mutex_lock(&args->lock) );

    while ( args->started == 0 ) {
        pthread_cond_wait(&args->cond, &args->lock);
    }

    CHECK_ERROR( pthread_mutex_unlock(&args->lock) );
}
void CL_CALLBACK read_complete(cl_event param_event, cl_int param_status, void* data)
{

    struct callBackData * cb_data = (struct callBackData *) data;
    
        // Copy the information contained in the input buffer
    if (cb_data->use_beamforming) {
        copy_buffer_info(cb_data->in_buf, cb_data->buffer_id,
                         cb_data->beamforming_out_buf, cb_data->buffer_id);
    //}
    
        // Mark the beamforming buffer as full.
    //if (cb_data->cl_data->config->gpu.use_beamforming) {
        mark_buffer_full(cb_data->beamforming_out_buf, cb_data->buffer_id);
    }

    // Copy the information contained in the input buffer
    move_buffer_info(cb_data->in_buf, cb_data->buffer_id,
                     cb_data->out_buf, cb_data->buffer_id);

    // Mark the input buffer as "empty" so that it can be reused.
    mark_buffer_empty(cb_data->in_buf, cb_data->buffer_id);

    // Mark the output buffer as full, so it can be processed.
    mark_buffer_full(cb_data->out_buf, cb_data->buffer_id);

    for (int i = 0; i < cb_data->numCommands; i++){
        cb_data->listCommands[i]->cleanMe(cb_data->buffer_id);
    }

    CHECK_ERROR( pthread_mutex_lock(&cb_data->cnt->lock));
    cb_data->cnt->iteration--;
    CHECK_ERROR( pthread_mutex_unlock(&cb_data->cnt->lock));

    CHECK_ERROR( pthread_cond_broadcast(&cb_data->cnt->cond) );
}

