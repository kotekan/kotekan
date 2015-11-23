/*
 * Copyright (c) 2015 <copyright holder> <email>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "gpu_thread.h"
#include "device_interface.h"
#include "gpu_command.h"
#include "gpu_command_factory.h"
#include "callbackdata.h"
#include "unistd.h"

pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;

void gpu_thread(void* arg)
{
    struct gpuThreadArgs * args = (struct gpuThreadArgs *) arg;

    gpu_command currentCommand;
    cl_event * host_buffer_ready;
    cl_event * preseedEvent;
    cl_int err;
    
    device_interface device(args->in_buf, args->out_buf, args->config, args->gpu_id);
    
    gpu_command_factory factory();
            
    factory.initializeCommands(args->config, device);    
    
    device.prepareCommandQueue();
    
    device.allocateMemory();
    
    //device.defineOutputDataMap();//COPY INTO OFFSET KERNEL AND CORRELATOR KERNEL
    
    // Just wait on one buffer.
    int buffer_list[1] = {0};
    int bufferID = 0;
    int first_time = 1;
    callBackData * cb_data;
    
    cb_data = malloc(device.getInBuf()->num_buffers * sizeof(struct callBackData));
    CHECK_MEM(cb_data);
    
    //host_buffer_ready = malloc(args->in_buf->num_buffers * sizeof(cl_event));
    //CHECK_MEM(host_buffer_ready);

    // Main loop for collecting data from producers.
    for(;;) {

        // Wait for data, this call will block.
        bufferID = get_full_buffer_from_list(args->in_buf, buffer_list, 1);
	
	
	//THERE IS A PROBLEM WITH HOW THIS EVENT IS HANDLED. IT IS PASSED TO THE FIRST COMMAND OBJECT, BUT MUST BE DEALLOCATED IN THE DEVICE INTERFACE OBJECT. HOWEVER, IT IS ALSO NEEDED IN THE GPU_THREAD FOR THE LOGIC
	//BELOW WAIT_FOR_EMPTY_BUFFER
	//PUT THIS INTO THE INITIALIZE COMMAND OBJECT.
	//host_buffer_ready[bufferID] = clCreateUserEvent(device.getContext(), &err);
	
	CHECK_CL_ERROR(err);
	
	pthread_mutex_lock(&queue_lock);
	// Set call back data
	cb_data[bufferID].buffer_id = bufferID; //SHOULD CB_DATA BE MANAGED THROUGH FINALIZEQUEUESEQUENCE_COMMAND SINCE IT IS BEING FREED THERE?
	cb_data[bufferID].in_buf = device.getInBuf();
	cb_data[bufferID].out_buf = device.getOutBuf();
	cb_data[bufferID].numCommands = factory.getNumCommands() - 1;
	
	preseedEvent != nullptr; //WILL THE INIT COMMAND WORK WITH A NULL PRECEEDING EVENT?
	for (int i = 0; i < factory.getNumCommands(); i++){
	  currentCommand = factory.getNextCommand(device, bufferID, &preseedEvent);
	  preseedEvent = currentCommand.execute(device, bufferID);	  
	  cb_data[bufferID].listKernels[i] = currentCommand;
	}
	
	// Setup call back.
	CHECK_CL_ERROR( clSetEventCallback(preseedEvent,
                                            CL_COMPLETE,
                                            device.read_complete(),
                                            &cb_data) );
    
    
    
	pthread_mutex_unlock(&queue_lock);
    
    	CHECK_ERROR( pthread_mutex_lock(&args->lock) );
	args->started = 1;
	CHECK_ERROR( pthread_mutex_unlock(&args->lock) );

	// Signal consumer (main thread in this case).
	CHECK_ERROR( pthread_cond_broadcast(&args->cond) );

        if (args->gpu_id == 0 && first_time == 1) {
            usleep(100000);
            first_time = 0;
        }
        // If buffer id is -1, then all the producers are done.
        if (bufferID == -1) {
            break;
        }

        // Wait for the output buffer to be empty as well.
        // This should almost never block, since the output buffer should clear quickly.
        wait_for_empty_buffer(args->out_buf, bufferID);

        //INFO("GPU Kernel started on gpu %d in buffer (%d,%d)", args->gpu_id, args->gpu_id - 1, bufferID);

        //CHECK_CL_ERROR( clSetUserEventStatus(host_buffer_ready[bufferID], CL_SUCCESS) );

        buffer_list[0] = (buffer_list[0] + 1) % args->in_buf->num_buffers;

    }

    DEBUG("Closing\n");
    
    factory.deallocateResources();
    device.deallocateResources();

    mark_producer_done(args->out_buf, 0);

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

