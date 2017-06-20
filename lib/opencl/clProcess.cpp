#include "clProcess.hpp"
#include "device_interface.h"
#include "gpu_command.h"
#include "gpu_command_factory.h"
#include "callbackdata.h"
#include "unistd.h"
#include "vdif_functions.h"
#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "device_interface.h"

#include <iostream>
#include <sys/time.h>

using namespace std;

double e_time(void){
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

// TODO Remove the GPU_ID from this constructor
clProcess::clProcess(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container):
    KotekanProcess(config_, unique_name, buffer_container, std::bind(&clProcess::main_thread, this))
{
    // TODO Remove this and move it to the command objects (see hsaProcess).
    gpu_id = config.get_int(unique_name, "gpu_id");
    in_buf = get_buffer("network_buf");
    out_buf = get_buffer("corr_buf");
    beamforming_out_buf = get_buffer("beam_out_buf");
    beamforming_out_incoh_buf = get_buffer("beam_incoh_out_buf");
}

void clProcess::apply_config(uint64_t fpga_seq) {
    _use_beamforming = config.get_bool("/gpu", "enable_beamforming");
}

clProcess::~clProcess() {

}

void clProcess::main_thread()
{
    apply_config(0);

    gpu_command * currentCommand;
    cl_event sequenceEvent;

    loopCounter * loopCnt = new loopCounter;

    device_interface device(in_buf, out_buf, config, gpu_id,
                            beamforming_out_buf, beamforming_out_incoh_buf);

    gpu_command_factory factory;

    factory.initializeCommands(device, config);

    device.prepareCommandQueue();

    device.allocateMemory();
    DEBUG("Device Initialized\n");

    callBackData ** cb_data = new callBackData * [device.getInBuf()->num_buffers];
    CHECK_MEM(cb_data);

    buffer_id_lock ** buff_id_lock_list = new buffer_id_lock * [device.getInBuf()->num_buffers];
    CHECK_MEM(buff_id_lock_list);

    for (int j=0;j<device.getInBuf()->num_buffers;j++)
    {
        cb_data[j] = new callBackData(factory.getNumCommands());
        buff_id_lock_list[j] = new buffer_id_lock;
    }

    // Just wait on one buffer.
    int buffer_list[1] = {0};
    int bufferID = 0;

    double last_time = e_time();

    for(;;) {
        // Wait for data, this call will block.
        bufferID = get_full_buffer_from_list(in_buf, buffer_list, 1);
        double cur_time = e_time();
        INFO("Got full buffer after time: %f", cur_time - last_time );
        last_time = cur_time;

        //INFO("GPU_THREAD: got full buffer ID %d", bufferID);
        // If buffer id is -1, then all the producers are done.
        if (bufferID == -1) {
            break;
        }

        CHECK_ERROR( pthread_mutex_lock(&buff_id_lock_list[bufferID]->lock) );
        while (buff_id_lock_list[bufferID]->in_process == 1) {
            DEBUG("gpu_thread%d: waiting for in flight queue to finish(!)", gpu_id);
            pthread_cond_wait(&buff_id_lock_list[bufferID]->cond, &buff_id_lock_list[bufferID]->lock);
        }
        CHECK_ERROR( pthread_mutex_unlock(&buff_id_lock_list[bufferID]->lock) );

        buff_id_lock_list[bufferID]->in_process = 1;

        // Wait for the output buffer to be empty as well.
        // This should almost never block, since the output buffer should clear quickly.
        wait_for_empty_buffer(out_buf, bufferID);

        if (_use_beamforming) {
            wait_for_empty_buffer(beamforming_out_buf, bufferID);
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
        cb_data[bufferID]->buff_id_lock = buff_id_lock_list[bufferID];
        cb_data[bufferID]->use_beamforming = _use_beamforming;
        cb_data[bufferID]->start_time = e_time();
        if (_use_beamforming == 1)
        {
            cb_data[bufferID]->beamforming_out_buf = device.get_beamforming_out_buf();
        }

        sequenceEvent = NULL; //WILL THE INIT COMMAND WORK WITH A NULL PRECEEDING EVENT?

        //DEBUG("cb_data initialized\n");
        for (int i = 0; i < factory.getNumCommands(); i++){
            currentCommand = factory.getNextCommand(device, bufferID);
            sequenceEvent = currentCommand->execute(bufferID, 0, device, sequenceEvent);
            cb_data[bufferID]->listCommands[i] = currentCommand;
        }

        //DEBUG("Commands Queued\n");
        // Setup call back.
        CHECK_CL_ERROR( clSetEventCallback(sequenceEvent,
                                            CL_COMPLETE,
                                            &read_complete,
                                            cb_data[bufferID]) );

        buffer_list[0] = (buffer_list[0] + 1) % in_buf->num_buffers;

    }

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

    mark_producer_done(out_buf, 0);

    delete loopCnt;
    delete[] cb_data;

    int ret;
    pthread_exit((void *) &ret);
}

void CL_CALLBACK read_complete(cl_event param_event, cl_int param_status, void* data)
{
    callBackData * cb_data = (callBackData *) data;

    double end_time_1 = e_time();

    INFO("GPU_THREAD: Read Complete Buffer ID %d", cb_data->buffer_id);
    // Copy the information contained in the input buffer
    if (cb_data->use_beamforming == 1)
    {
        copy_buffer_info(cb_data->in_buf, cb_data->buffer_id,
            cb_data->beamforming_out_buf, cb_data->buffer_id);

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

    CHECK_ERROR( pthread_mutex_lock(&cb_data->buff_id_lock->lock));
    cb_data->buff_id_lock->in_process = 0;
    CHECK_ERROR( pthread_mutex_unlock(&cb_data->buff_id_lock->lock));

    CHECK_ERROR( pthread_cond_broadcast(&cb_data->buff_id_lock->cond) );

    double end_time_2 = e_time();
    INFO("running_time 1: %f, running_time 2: %f, function_time: %f", end_time_1 - cb_data->start_time, end_time_2 - cb_data->start_time, end_time_2 - end_time_1);
}

