#include "clProcess.hpp"
#include "gpu_command.h"
#include "callbackdata.h"
#include "unistd.h"
#include "vdif_functions.h"
#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "device_interface.h"

#include <iostream>
#include <sys/time.h>

using namespace std;

double e_time_1(void){
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
    in_buf = get_buffer("network_buffer");
    register_consumer(get_buffer("network_buffer"), unique_name.c_str());
    out_buf = get_buffer("output_buffer");
    register_producer(get_buffer("output_buffer"), unique_name.c_str());
    beamforming_out_buf = get_buffer("beam_out_buf");
    register_producer(get_buffer("beam_out_buf"), unique_name.c_str());
    //beamforming_out_incoh_buf = NULL;  //get_buffer("beam_incoh_out_buf");

    //device_interface device(in_buf, out_buf, config, gpu_id,
    //                        beamforming_out_buf, beamforming_out_incoh_buf);

    //device = new device_interface(config, gpu_id);
    device = new device_interface(in_buf, out_buf, config, gpu_id, beamforming_out_buf, unique_name);


    factory = new gpu_command_factory(*device, config, unique_name);

    //factory.initializeCommands(device, config);

}

void clProcess::apply_config(uint64_t fpga_seq) {
    _use_beamforming = config.get_bool(unique_name, "enable_beamforming");
}

clProcess::~clProcess() {
    delete factory;
    delete device;
}

void clProcess::main_thread()
{
    apply_config(0);

    gpu_command * currentCommand;
    cl_event sequenceEvent;

    loopCounter * loopCnt = new loopCounter;
    bool first_run = true;

    device->prepareCommandQueue(false);
    //    device->prepareCommandQueue(true);

    device->allocateMemory();
//    DEBUG("Device Initialized\n");

    //callBackData ** cb_data = new callBackData * [device->getInBuf()->num_buffers];
    //CHECK_MEM(cb_data);

    buffer_id_lock ** buff_id_lock_list = new buffer_id_lock * [device->getInBuf()->num_frames];
    CHECK_MEM(buff_id_lock_list);

    for (int j=0;j<device->getInBuf()->num_frames;j++)
    {
        //cb_data[j] = new callBackData(factory->getNumCommands());
        cb_data.push_back(new callBackData(factory->getNumCommands()));
        buff_id_lock_list[j] = new buffer_id_lock;
        cb_data[j]->buff_id_lock = buff_id_lock_list[j];
    }

    // Just wait on one buffer.
//    int buffer_list[1] = {0};
    int frame_id = 0;
    uint8_t * frame = NULL;

    double last_time = e_time_1();

    for(;;) {
        // Wait for data, this call will block.
        //Now will return a uint8_t* wait_for_empty_frame(...)!!!!
        frame = wait_for_full_frame(device->getInBuf(), unique_name.c_str(), frame_id);
        double cur_time = e_time_1();
        //INFO("Got full buffer after time: %f", cur_time - last_time );
        last_time = cur_time;

        //INFO("GPU_THREAD: got full buffer ID %d", bufferID);
        // If buffer id is -1, then all the producers are done.
        if (frame == NULL) {
            break;
        }

        CHECK_ERROR( pthread_mutex_lock(&buff_id_lock_list[frame_id]->lock) );
        while (buff_id_lock_list[frame_id]->mem_in_use == 1) {
            pthread_cond_wait(&buff_id_lock_list[frame_id]->mem_cond, &buff_id_lock_list[frame_id]->lock);
        }
        CHECK_ERROR( pthread_mutex_unlock(&buff_id_lock_list[frame_id]->lock) );

        CHECK_ERROR( pthread_mutex_lock(&buff_id_lock_list[frame_id]->lock));
            buff_id_lock_list[frame_id]->mem_in_use = 1;
        CHECK_ERROR( pthread_mutex_unlock(&buff_id_lock_list[frame_id]->lock));

        // Wait for the output buffer to be empty as well.
        // This should almost never block, since the output buffer should clear quickly.
        wait_for_empty_frame(device->getOutBuf(), unique_name.c_str(), frame_id);

        if (_use_beamforming) {
            wait_for_empty_frame(device->get_beamforming_out_buf(), unique_name.c_str(), frame_id);
        }

        // Todo get/set time information here as well.

        CHECK_ERROR( pthread_mutex_lock(&loopCnt->lock));
        loopCnt->iteration++;
        CHECK_ERROR( pthread_mutex_unlock(&loopCnt->lock));

        // Set call back data
        cb_data[frame_id]->buffer_id = frame_id;
        cb_data[frame_id]->in_buf = device->getInBuf();
        cb_data[frame_id]->out_buf = device->getOutBuf();
        cb_data[frame_id]->numCommands = factory->getNumCommands();
        cb_data[frame_id]->cnt = loopCnt;
        cb_data[frame_id]->use_beamforming = _use_beamforming;
        cb_data[frame_id]->start_time = e_time_1();
        cb_data[frame_id]->unique_name = unique_name;
        if (_use_beamforming == 1)
        {
            cb_data[frame_id]->beamforming_out_buf = device->get_beamforming_out_buf();
        }

        sequenceEvent = NULL; //WILL THE INIT COMMAND WORK WITH A NULL PRECEEDING EVENT?

        //DEBUG("cb_data initialized\n");
        for (int i = 0; i < factory->getNumCommands(); i++){
            currentCommand = factory->getNextCommand();
            sequenceEvent = currentCommand->execute(frame_id, 0, *device, sequenceEvent);
            cb_data[frame_id]->listCommands[i] = currentCommand;
        }

        if (first_run)
        {
            mem_reconcil_thread_handle = std::thread(&clProcess::mem_reconcil_thread, std::ref(*this));

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (int j = 4; j < 12; j++)
                CPU_SET(j, &cpuset);
            pthread_setaffinity_np(mem_reconcil_thread_handle.native_handle(),
                                    sizeof(cpu_set_t), &cpuset);

            first_run = false;
        }
        //DEBUG("Commands Queued\n");
        // Setup call back.
        CHECK_CL_ERROR( clSetEventCallback(sequenceEvent,
                                            CL_COMPLETE,
                                            &read_complete,
                                            cb_data[frame_id]) );

        //buffer_list[0] = (buffer_list[0] + 1)
        frame_id = (++frame_id) % device->getInBuf()->num_frames;
    }

    DEBUG("Closing\n");

    // TODO Make the exiting process actually work here.
    mem_reconcil_thread_handle.join();

    CHECK_ERROR( pthread_mutex_lock(&loopCnt->lock) );
    DEBUG("LockCnt\n");
    while ( loopCnt->iteration > 0) {
        pthread_cond_wait(&loopCnt->cond, &loopCnt->lock);
    }
    CHECK_ERROR( pthread_mutex_unlock(&loopCnt->lock) );


    DEBUG("LockConditionReleased\n");
    factory->deallocateResources();
    DEBUG("FactoryDone\n");
    device->deallocateResources();
    DEBUG("DeviceDone\n");

//    mark_producer_done(device->getOutBuf(), 0);
//    if (_use_beamforming == 1)
//    {
//        mark_producer_done(device->get_beamforming_out_buf(), 0);
//    }

    delete loopCnt;
    //delete[] cb_data;
    for (auto command : cb_data) {
        delete command;
    }
    int ret;
    pthread_exit((void *) &ret);
}

void clProcess::mem_reconcil_thread()
{
    //Based on assumption that buffer_ids are processed in order, so start with [0].]
    int frame_id_limit = cb_data[0]->in_buf->num_frames;
    std::clock_t    start;
    for(;;) {
        for (int j=0;j<frame_id_limit;j++)
        {
            start = std::clock();
            double end_time_1 = e_time_1();

            CHECK_ERROR( pthread_mutex_lock(&cb_data[j]->buff_id_lock->lock));

            while (cb_data[j]->buff_id_lock->clean == 0) {
                pthread_cond_wait(&cb_data[j]->buff_id_lock->clean_cond, &cb_data[j]->buff_id_lock->lock);
                //INFO("GPU_THREAD: Read Complete Buffer ID %d", cb_data[j]->buffer_id);
            }

            CHECK_ERROR( pthread_mutex_unlock(&cb_data[j]->buff_id_lock->lock));

            //INFO("GPU_THREAD: Read Complete Buffer ID %d", cb_data->buffer_id);
            // Copy the information contained in the input buffer

            if (cb_data[j]->use_beamforming == 1)
            {
                pass_metadata(cb_data[j]->in_buf, cb_data[j]->buffer_id,
                    cb_data[j]->beamforming_out_buf, cb_data[j]->buffer_id);

                mark_frame_full(cb_data[j]->beamforming_out_buf, cb_data[j]->unique_name.c_str(), cb_data[j]->buffer_id);
            }

            pass_metadata(cb_data[j]->in_buf, cb_data[j]->buffer_id,
                             cb_data[j]->out_buf, cb_data[j]->buffer_id);

            // Mark the input buffer as "empty" so that it can be reused.
            mark_frame_empty(cb_data[j]->in_buf, cb_data[j]->unique_name.c_str(), cb_data[j]->buffer_id);

            // Mark the output buffer as full, so it can be processed.
            mark_frame_full(cb_data[j]->out_buf, cb_data[j]->unique_name.c_str(), cb_data[j]->buffer_id);

            for (int i = 0; i < cb_data[j]->numCommands; i++){
                cb_data[j]->listCommands[i]->cleanMe(cb_data[j]->buffer_id);
            }

            CHECK_ERROR( pthread_mutex_lock(&cb_data[j]->buff_id_lock->lock));
            cb_data[j]->buff_id_lock->mem_in_use = 0;
            cb_data[j]->buff_id_lock->clean = 0;
            CHECK_ERROR( pthread_mutex_unlock(&cb_data[j]->buff_id_lock->lock));

            CHECK_ERROR( pthread_cond_broadcast(&cb_data[j]->buff_id_lock->mem_cond) );
            double end_time_2 = e_time_1();
            //INFO("running_time 1: %f, running_time 2: %f, function_time: %f", end_time_1 - cb_data[j]->start_time, end_time_2 - cb_data[j]->start_time, end_time_2 - end_time_1);
        }
    }
}

void CL_CALLBACK read_complete(cl_event param_event, cl_int param_status, void* data)
{
    struct callBackData * cur_cb_data = (struct callBackData *) data;
    std::clock_t    start;

    CHECK_ERROR( pthread_mutex_lock(&cur_cb_data->cnt->lock));
    cur_cb_data->cnt->iteration--;
    CHECK_ERROR( pthread_mutex_unlock(&cur_cb_data->cnt->lock));

    //CHECK_ERROR( pthread_cond_broadcast(&cb_data[j]->cnt->cond) );

    start = std::clock();

    CHECK_ERROR( pthread_mutex_lock(&cur_cb_data->buff_id_lock->lock));
    cur_cb_data->buff_id_lock->clean = 1;
    // your test
    //std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << " read_complete:BufferID_" << cur_cb_data->buffer_id << " iteration:" << cur_cb_data->cnt->iteration << std::endl;

    //std::cout << "Time: " << (std::clock()) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << " read_complete:BufferID_" << cur_cb_data->buffer_id << std::endl;

    CHECK_ERROR( pthread_mutex_unlock(&cur_cb_data->buff_id_lock->lock));

    CHECK_ERROR( pthread_cond_broadcast(&cur_cb_data->buff_id_lock->clean_cond) );

}
