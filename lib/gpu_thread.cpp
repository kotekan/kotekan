#include "gpu_thread.h"
#include "device_interface.h"
#include "gpu_command.h"
#include "gpu_command_factory.h"
#include "callbackdata.h"
#include "unistd.h"
#include "vdif_functions.h"
#include "fpga_header_functions.h"
#include <vector>
#include <iostream>

pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;

using namespace std;

void* gpu_thread(void* arg)
{
    struct gpuThreadArgs * args = (struct gpuThreadArgs *) arg;

    gpu_command * currentCommand;
    //cl_event * host_buffer_ready;
    cl_event precedeEvent;
    cl_int err;
    loopCounter * loopCnt = new loopCounter;

    device_interface device(args->in_buf, args->out_buf, args->config, args->gpu_id);

    gpu_command_factory factory;

    factory.initializeCommands(device, args->config);

    device.prepareCommandQueue();

    device.allocateMemory();
    DEBUG("Device Initialized\n");
    //device.defineOutputDataMap();//COPY INTO OFFSET KERNEL AND CORRELATOR KERNEL

    callBackData ** cb_data = new callBackData * [device.getInBuf()->num_buffers]; //(callBackData*)malloc(device.getInBuf()->num_buffers*sizeof(callBackData));
        //vector<callBackData> cb_data(device.getInBuf()->num_buffers);

    for (int j=0;j<device.getInBuf()->num_buffers;j++)
        cb_data[j] = new callBackData(factory.getNumCommands());

    CHECK_MEM(cb_data);

        //cb_data.listCommands = (gpu_command *)malloc(factory.getNumCommands() * sizeof(class gpu_command));

    //host_buffer_ready = malloc(args->in_buf->num_buffers * sizeof(cl_event));
    //CHECK_MEM(host_buffer_ready);

    // Just wait on one buffer.
    int buffer_list[1] = {0};
    int bufferID = 0;
    int first_time = 1;

    CHECK_ERROR( pthread_mutex_lock(&args->lock) );
    args->started = 1;
    CHECK_ERROR( pthread_mutex_unlock(&args->lock) );

    // Signal consumer (main thread in this case).
    CHECK_ERROR( pthread_cond_broadcast(&args->cond) );
    // Main loop for collecting data from producers.
    int iteration = 0;
    for(;;) {
    //for (int x=0;x<1;x++){
        DEBUG("Loop Started\n");
        // Wait for data, this call will block.
        bufferID = get_full_buffer_from_list(args->in_buf, buffer_list, 1);
	DEBUG("BufferID obtained\n");

        // If buffer id is -1, then all the producers are done.
        if (bufferID == -1) {
            break;
        }

        CHECK_ERROR( pthread_mutex_lock(&loopCnt->lock));
        loopCnt->iteration++;
        CHECK_ERROR( pthread_mutex_unlock(&loopCnt->lock));

	//THERE IS A PROBLEM WITH HOW THIS EVENT IS HANDLED. IT IS PASSED TO THE FIRST COMMAND OBJECT, BUT MUST BE DEALLOCATED IN THE DEVICE INTERFACE OBJECT. HOWEVER, IT IS ALSO NEEDED IN THE GPU_THREAD FOR THE LOGIC
	//BELOW WAIT_FOR_EMPTY_BUFFER
	//PUT THIS INTO THE INITIALIZE COMMAND OBJECT.
	//host_buffer_ready[bufferID] = clCreateUserEvent(device.getContext(), &err);

//        CHECK_CL_ERROR(err);

        //pthread_mutex_lock(&queue_lock);
	// Set call back data
        cb_data[bufferID]->buffer_id = bufferID;
        cb_data[bufferID]->in_buf = device.getInBuf();
        cb_data[bufferID]->out_buf = device.getOutBuf();
        cb_data[bufferID]->numCommands = factory.getNumCommands();
        cb_data[bufferID]->cnt = loopCnt;

        //cl_event storeEvent;
	precedeEvent = NULL; //WILL THE INIT COMMAND WORK WITH A NULL PRECEEDING EVENT?
	//precedeEvent = clCreateUserEvent(device.getContext(), &err);
        //CHECK_CL_ERROR(err);
        //storeEvent = precedeEvent;
	cb_data[bufferID]->buffer_id = bufferID;
        DEBUG("cb_data initialized\n");

        std::cout << "BufferID = " << bufferID << endl;
	for (int i = 0; i < factory.getNumCommands(); i++){
            std::cout << "i = " << i << endl;
          std::cout << "precedeEvent = " << precedeEvent << endl;
	  currentCommand = factory.getNextCommand(device, bufferID);
          precedeEvent = currentCommand->execute(bufferID, device, precedeEvent);
	  cb_data[bufferID]->listCommands[i] = currentCommand;

          /*if (i==1)//FOR DEBUGGING ONLY
          {
           storeEvent = precedeEvent;
          }*/
          std::cout << "i = " << i << endl;
          std::cout << "precedeEvent = " << precedeEvent << endl;
	}
	DEBUG("Commands Queued\n");
        std::cout << "precedeEvent = " << precedeEvent << endl;
	// Setup call back.
	CHECK_CL_ERROR( clSetEventCallback(precedeEvent,
                                            CL_COMPLETE,
                                            &read_complete,
                                            cb_data[bufferID]) );



        //pthread_mutex_unlock(&queue_lock);

        //CHECK_CL_ERROR( clSetUserEventStatus(storeEvent, CL_SUCCESS) );

        // Wait for the output buffer to be empty as well.
        // This should almost never block, since the output buffer should clear quickly.
        wait_for_empty_buffer(args->out_buf, bufferID);

        //INFO("GPU Kernel started on gpu %d in buffer (%d,%d)", args->gpu_id, args->gpu_id - 1, bufferID);

        //CHECK_CL_ERROR( clSetUserEventStatus(host_buffer_ready[bufferID], CL_SUCCESS) );

        buffer_list[0] = (buffer_list[0] + 1) % args->in_buf->num_buffers;

        iteration++;
    }
    //precedeEvent = NULL;

    //LOOP THROUGH THE LOCKING ROUTINE

    DEBUG("Closing\n");

    CHECK_ERROR( pthread_mutex_lock(&loopCnt->lock) );
    DEBUG("LockCnt\n");
        while ( loopCnt->iteration > 0) {
            pthread_cond_wait(&loopCnt->cond, &loopCnt->lock);
            DEBUG("LockEval\n");
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
    //free(precedeEvent);

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

    DEBUG("Read Complete Started\n");

    //INFO("GPU Kernel Finished on GPUID: %d", cb_data->cl_data->gpu_id);

    // Copy the information contained in the input buffer
    move_buffer_info(cb_data->in_buf, cb_data->buffer_id,
                     cb_data->out_buf, cb_data->buffer_id);
    DEBUG("Move done\n");

    // Mark the input buffer as "empty" so that it can be reused.
    mark_buffer_empty(cb_data->in_buf, cb_data->buffer_id);

    DEBUG("Empty done\n");

    // Mark the output buffer as full, so it can be processed.
    mark_buffer_full(cb_data->out_buf, cb_data->buffer_id);

    DEBUG("Mark done, buffer_id: %d\n", cb_data->buffer_id);

    std::cout << "BufferID = " << cb_data->buffer_id << endl;
    for (int i = 0; i < cb_data->numCommands; i++){
        std::cout << "i = " << i << endl;
        cb_data->listCommands[i]->cleanMe(cb_data->buffer_id);
        DEBUG("Cleaned\n");
    }

    DEBUG("CleanMe done\n");

    CHECK_ERROR( pthread_mutex_lock(&cb_data->cnt->lock));
    cb_data->cnt->iteration--;
    CHECK_ERROR( pthread_mutex_unlock(&cb_data->cnt->lock));
    DEBUG("Iteration done\n");

    CHECK_ERROR( pthread_cond_broadcast(&cb_data->cnt->cond) );

    DEBUG("Read Complete Done\n");
    //free(cb_data->listKernels);

}
