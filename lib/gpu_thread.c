
#include <stdio.h>

#include <math.h>
#include <memory.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdlib.h>
#include <assert.h>

#include "gpu_thread.h"
#include "errors.h"
#include "buffers.h"
#include "config.h"

#include "pthread.h"
#include <unistd.h>

pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;

void setup_open_cl(struct OpenCLData * cl_data);

void close_open_cl(struct OpenCLData * cl_data);

void add_queue_set(struct OpenCLData * cl_data, int buffer_id);

void release_events_for_buffer(struct OpenCLData * cl_data, int buffer_id);

void gpu_thread(void* arg)
{
    struct gpuThreadArgs * args = (struct gpuThreadArgs *) arg;

    struct OpenCLData cl_data;

    cl_data.in_buf = args->in_buf;
    cl_data.out_buf = args->out_buf;

    cl_data.config = args->config;

    cl_data.accumulate_len = cl_data.config->processing.num_adjusted_local_freq *
    cl_data.config->processing.num_adjusted_elements * 2 * cl_data.config->processing.num_data_sets * sizeof(cl_int);
    cl_data.aligned_accumulate_len = PAGESIZE_MEM * (ceil((double)cl_data.accumulate_len / (double)PAGESIZE_MEM));
    assert(cl_data.aligned_accumulate_len >= cl_data.accumulate_len);

    cl_data.gpu_id = args->gpu_id;

    setup_open_cl(&cl_data);

    // Queue the initial commands in buffer order.
    for (int i = 0; i < args->in_buf->num_buffers; ++i) {
        add_queue_set(&cl_data, i);
    }

    CHECK_ERROR( pthread_mutex_lock(&args->lock) );
    args->started = 1;
    CHECK_ERROR( pthread_mutex_unlock(&args->lock) );

    // Signal consumer (main thread in this case).
    CHECK_ERROR( pthread_cond_broadcast(&args->cond) );

    // Just wait on one buffer.
    int buffer_list[1] = {0};
    int bufferID = 0;
    int first_time = 1;

    // Main loop for collecting data from producers.
    for(;;) {

        // Wait for data, this call will block.
        bufferID = get_full_buffer_from_list(args->in_buf, buffer_list, 1);

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

        CHECK_CL_ERROR( clSetUserEventStatus(cl_data.host_buffer_ready[bufferID], CL_SUCCESS) );

        buffer_list[0] = (buffer_list[0] + 1) % args->in_buf->num_buffers;

    }

    DEBUG("Closing OpenCL\n");

    close_open_cl(&cl_data);

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

void CL_CALLBACK read_complete(cl_event event, cl_int status, void *data) {

    struct callBackData * cb_data = (struct callBackData *) data;

    //INFO("GPU Kernel Finished on GPUID: %d", cb_data->cl_data->gpu_id);

    // Copy the information contained in the input buffer
    move_buffer_info(cb_data->cl_data->in_buf, cb_data->buffer_id,
                     cb_data->cl_data->out_buf, cb_data->buffer_id);

    // Mark the input buffer as "empty" so that it can be reused.
    mark_buffer_empty(cb_data->cl_data->in_buf, cb_data->buffer_id);

    // Mark the output buffer as full, so it can be processed.
    mark_buffer_full(cb_data->cl_data->out_buf, cb_data->buffer_id);

    // Relase the events from the last queue set run.
    release_events_for_buffer(cb_data->cl_data, cb_data->buffer_id);

    // TODO move this to the consumer thread for the output data.
    add_queue_set(cb_data->cl_data, cb_data->buffer_id);

}

void release_events_for_buffer(struct OpenCLData * cl_data, int buffer_id)
{
    assert(cl_data != NULL);
    assert(cl_data->host_buffer_ready[buffer_id] != NULL);
    assert(cl_data->input_data_written[buffer_id] != NULL);
    assert(cl_data->accumulate_data_zeroed[buffer_id] != NULL);
    assert(cl_data->offset_accumulate_finished[buffer_id] != NULL);
    assert(cl_data->preseed_finished[buffer_id] != NULL);
    assert(cl_data->corr_finished[buffer_id] != NULL);
    assert(cl_data->read_finished[buffer_id] != NULL);

    clReleaseEvent(cl_data->host_buffer_ready[buffer_id]);
    clReleaseEvent(cl_data->input_data_written[buffer_id]);
    clReleaseEvent(cl_data->accumulate_data_zeroed[buffer_id]);
    clReleaseEvent(cl_data->offset_accumulate_finished[buffer_id]);
    clReleaseEvent(cl_data->preseed_finished[buffer_id]);
    clReleaseEvent(cl_data->corr_finished[buffer_id]);
    clReleaseEvent(cl_data->read_finished[buffer_id]);
}

void add_queue_set(struct OpenCLData * cl_data, int buffer_id)
{
    cl_int err;

    // This function can be called from a call back, so it must be thread safe to avoid
    // having queues out of order.
    pthread_mutex_lock(&queue_lock);

    // Set call back data
    cl_data->cb_data[buffer_id].buffer_id = buffer_id;
    cl_data->cb_data[buffer_id].cl_data = cl_data;

    cl_data->host_buffer_ready[buffer_id] = clCreateUserEvent(cl_data->context, &err);

    CHECK_CL_ERROR(err);

    // Data transfer to GPU
    CHECK_CL_ERROR( clEnqueueWriteBuffer(cl_data->queue[0],
                                            cl_data->device_input_buffer[buffer_id],
                                            CL_FALSE,
                                            0, //offset
                                            cl_data->in_buf->aligned_buffer_size,
                                            cl_data->in_buf->data[buffer_id],
                                            1,
                                            &cl_data->host_buffer_ready[buffer_id], // Wait on this user event (network finished).
                                            &cl_data->input_data_written[buffer_id]) );

    CHECK_CL_ERROR( clEnqueueWriteBuffer(cl_data->queue[0],
                                            cl_data->device_accumulate_buffer[buffer_id],
                                            CL_FALSE,
                                            0,
                                            cl_data->aligned_accumulate_len,
                                            cl_data->accumulate_zeros,
                                            1,
                                            &cl_data->input_data_written[buffer_id],
                                            &cl_data->accumulate_data_zeroed[buffer_id]) );

    // The offset accumulate kernel args.
    // Set 2 arguments--input array and zeroed output array
    CHECK_CL_ERROR( clSetKernelArg(cl_data->offset_accumulate_kernel,
                                        0,
                                        sizeof(void*),
                                        (void*) &cl_data->device_input_buffer[buffer_id]) );

    CHECK_CL_ERROR( clSetKernelArg(cl_data->offset_accumulate_kernel,
                                        1,
                                        sizeof(void *),
                                        (void*) &cl_data->device_accumulate_buffer[buffer_id]) );

    CHECK_CL_ERROR( clEnqueueNDRangeKernel(cl_data->queue[1],
                                            cl_data->offset_accumulate_kernel,
                                            3,
                                            NULL,
                                            cl_data->gws_accum,
                                            cl_data->lws_accum,
                                            1,
                                            &cl_data->accumulate_data_zeroed[buffer_id],
                                            &cl_data->offset_accumulate_finished[buffer_id]) );

    // The perseed kernel
    // preseed_kernel--set 2 of the 6 arguments (the other 4 stay the same)
    CHECK_CL_ERROR( clSetKernelArg(cl_data->preseed_kernel,
                                        0,
                                        sizeof(void *),
                                        (void*) &cl_data->device_accumulate_buffer[buffer_id]) );

    CHECK_CL_ERROR( clSetKernelArg(cl_data->preseed_kernel,
                                        1,
                                        sizeof(void *),
                                        (void *) &cl_data->device_output_buffer[buffer_id]) );

    CHECK_CL_ERROR( clEnqueueNDRangeKernel(cl_data->queue[1],
                                            cl_data->preseed_kernel,
                                            3, //3d global dimension, also worksize
                                            NULL, //no offsets
                                            cl_data->gws_preseed,
                                            cl_data->lws_preseed,
                                            1,
                                            &cl_data->offset_accumulate_finished[buffer_id],
                                            &cl_data->preseed_finished[buffer_id]) );

    // The correlation kernel.
    CHECK_CL_ERROR( clSetKernelArg(cl_data->corr_kernel,
                                        0,
                                        sizeof(void *),
                                        (void*) &cl_data->device_input_buffer[buffer_id]) );

    CHECK_CL_ERROR( clSetKernelArg(cl_data->corr_kernel,
                                        1,
                                        sizeof(void *),
                                        (void*) &cl_data->device_output_buffer[buffer_id]) );

    // The correlation kernel.
    CHECK_CL_ERROR( clEnqueueNDRangeKernel(cl_data->queue[1],
                                            cl_data->corr_kernel,
                                            3, 
                                            NULL,
                                            cl_data->gws_corr,
                                            cl_data->lws_corr,
                                            1,
                                            &cl_data->preseed_finished[buffer_id],
                                            &cl_data->corr_finished[buffer_id]) );

    // Read the results
    CHECK_CL_ERROR( clEnqueueReadBuffer(cl_data->queue[2], 
                                            cl_data->device_output_buffer[buffer_id],
                                            CL_FALSE,
                                            0,
                                            cl_data->out_buf->aligned_buffer_size,
                                            cl_data->out_buf->data[buffer_id],
                                            1,
                                            &cl_data->corr_finished[buffer_id],
                                            &cl_data->read_finished[buffer_id]) );

    // Setup call back.
    CHECK_CL_ERROR( clSetEventCallback(cl_data->read_finished[buffer_id],
                                            CL_COMPLETE,
                                            &read_complete,
                                            &cl_data->cb_data[buffer_id]) );

    pthread_mutex_unlock(&queue_lock);
}


void setup_open_cl(struct OpenCLData * cl_data)
{
    cl_int err;

    // Get a platform.
    CHECK_CL_ERROR( clGetPlatformIDs( 1, &cl_data->platform_id, NULL ) );

    // Find a GPU device..
    CHECK_CL_ERROR( clGetDeviceIDs( cl_data->platform_id, CL_DEVICE_TYPE_GPU, MAX_GPUS, cl_data->device_id, NULL) );

    cl_data->context = clCreateContext( NULL, 1, &cl_data->device_id[cl_data->gpu_id], NULL, NULL, &err);
    CHECK_CL_ERROR(err);

    // TODO move this out of this function?
    // TODO explain these numbers/formulas.
    cl_data->num_blocks = (cl_data->config->processing.num_adjusted_elements / cl_data->config->gpu.block_size) *
    (cl_data->config->processing.num_adjusted_elements / cl_data->config->gpu.block_size + 1) / 2.;

    // TODO Move this into a function for just loading kernels.
    // Load kernels and compile them.
    char cl_options[1024];
    sprintf(cl_options, "-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du",
            cl_data->config->processing.num_elements, cl_data->config->processing.num_local_freq,
            cl_data->config->processing.num_adjusted_elements,
            cl_data->config->processing.num_adjusted_local_freq,
            cl_data->config->processing.num_blocks, cl_data->config->processing.samples_per_data_set);

    size_t cl_program_size[cl_data->config->gpu.num_kernels];
    FILE *fp;
    char *cl_program_buffer[cl_data->config->gpu.num_kernels];

    for (int i = 0; i < cl_data->config->gpu.num_kernels; i++){
        fp = fopen(cl_data->config->gpu.kernels[i], "r");
        if (fp == NULL){
            ERROR("error loading file: %s", cl_data->config->gpu.kernels[i]);
            exit(errno);
        }
        fseek(fp, 0, SEEK_END);
        cl_program_size[i] = ftell(fp);
        rewind(fp);
        cl_program_buffer[i] = (char*)malloc(cl_program_size[i]+1);
        cl_program_buffer[i][cl_program_size[i]] = '\0';
        int sizeRead = fread(cl_program_buffer[i], sizeof(char), cl_program_size[i], fp);
        if (sizeRead < cl_program_size[i])
            ERROR("Error reading the file: %s", cl_data->config->gpu.kernels[i]);
        fclose(fp);
    }

    cl_data->program = clCreateProgramWithSource( cl_data->context,
                                                  cl_data->config->gpu.num_kernels,
                                                  (const char**)cl_program_buffer,
                                                  cl_program_size, &err );
    CHECK_CL_ERROR (err);

    for (int i =0; i < cl_data->config->gpu.num_kernels; i++){
        free(cl_program_buffer[i]);
    }

    CHECK_CL_ERROR ( clBuildProgram( cl_data->program, 1, &cl_data->device_id[cl_data->gpu_id], cl_options, NULL, NULL ) );

    cl_data->corr_kernel = clCreateKernel( cl_data->program, "corr", &err );
    CHECK_CL_ERROR(err);

    cl_data->offset_accumulate_kernel = clCreateKernel( cl_data->program, "offsetAccumulateElements", &err );
    CHECK_CL_ERROR(err);

    cl_data->preseed_kernel = clCreateKernel( cl_data->program, "preseed", &err );
    CHECK_CL_ERROR(err);

    // Create command queues
    for (int i = 0; i < NUM_QUEUES; ++i) {
        cl_data->queue[i] = clCreateCommandQueue( cl_data->context, cl_data->device_id[cl_data->gpu_id], CL_QUEUE_PROFILING_ENABLE, &err );
        CHECK_CL_ERROR(err);
    }

    // TODO create a struct to contain all of these (including events) to make this memory allocation cleaner. 
    // 

    // Setup device input buffers
    cl_data->device_input_buffer = (cl_mem *) malloc(cl_data->in_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(cl_data->device_input_buffer);
    for (int i = 0; i < cl_data->in_buf->num_buffers; ++i) {
        cl_data->device_input_buffer[i] = clCreateBuffer(cl_data->context, CL_MEM_READ_ONLY, cl_data->in_buf->aligned_buffer_size, NULL, &err);
        CHECK_CL_ERROR(err);
    }

    // Array used to zero the output memory on the device.
    // TODO should this be in it's own function?
    err = posix_memalign((void **) &cl_data->accumulate_zeros, PAGESIZE_MEM, cl_data->aligned_accumulate_len);
    if ( err != 0 ) {
        ERROR("Error creating aligned memory for accumulate zeros");
        exit(err);
    }

    // Ask that all pages be kept in memory
    err = mlock((void *) cl_data->accumulate_zeros, cl_data->aligned_accumulate_len);
    if ( err == -1 ) {
        ERROR("Error locking memory - check ulimit -a to check memlock limits");
        exit(errno);
    }
    memset(cl_data->accumulate_zeros, 0, cl_data->aligned_accumulate_len );

    // Setup device accumulate buffers.
    cl_data->device_accumulate_buffer = (cl_mem *) malloc(cl_data->in_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(cl_data->device_accumulate_buffer);
    for (int i = 0; i < cl_data->in_buf->num_buffers; ++i) {
        cl_data->device_accumulate_buffer[i] = clCreateBuffer(cl_data->context, CL_MEM_READ_WRITE, cl_data->aligned_accumulate_len, NULL, &err);
        CHECK_CL_ERROR(err);
    }

    // Setup device output buffers.
    cl_data->device_output_buffer = (cl_mem *) malloc(cl_data->out_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(cl_data->device_output_buffer);
    for (int i = 0; i < cl_data->out_buf->num_buffers; ++i) {
        cl_data->device_output_buffer[i] = clCreateBuffer(cl_data->context, CL_MEM_WRITE_ONLY, cl_data->out_buf->aligned_buffer_size, NULL, &err);
        CHECK_CL_ERROR(err);
    }

    cl_data->cb_data = malloc(cl_data->in_buf->num_buffers * sizeof(struct callBackData));
    CHECK_MEM(cl_data->cb_data);

    cl_data->host_buffer_ready = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->host_buffer_ready);

    cl_data->offset_accumulate_finished = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->offset_accumulate_finished);

    cl_data->preseed_finished = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->preseed_finished); 

    cl_data->corr_finished = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->corr_finished);

    cl_data->input_data_written = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->input_data_written);

    cl_data->accumulate_data_zeroed = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->accumulate_data_zeroed);

    cl_data->read_finished = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->read_finished);

    // Create lookup tables 

    //upper triangular address mapping --converting 1d addresses to 2d addresses
    unsigned int global_id_x_map[cl_data->num_blocks];
    unsigned int global_id_y_map[cl_data->num_blocks];

    //TODO: p260 OpenCL in Action has a clever while loop that changes 1 D addresses to X & Y indices for an upper triangle.  
    // Time Test kernels using them compared to the lookup tables for NUM_ELEM = 256
    int largest_num_blocks_1D = cl_data->config->processing.num_adjusted_elements /cl_data->config->gpu.block_size;
    int index_1D = 0;
    for (int j = 0; j < largest_num_blocks_1D; j++){
        for (int i = j; i < largest_num_blocks_1D; i++){
            global_id_x_map[index_1D] = i;
            global_id_y_map[index_1D] = j;
            index_1D++;
        }
    }


    cl_mem id_x_map = clCreateBuffer(cl_data->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    cl_data->num_blocks * sizeof(cl_uint), global_id_x_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    cl_mem id_y_map = clCreateBuffer(cl_data->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    cl_data->num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    //set other parameters that will be fixed for the kernels (changeable parameters will be set in run loops)
    CHECK_CL_ERROR( clSetKernelArg(cl_data->corr_kernel, 
                                   2,
                                   sizeof(id_x_map),
                                   (void*) &id_x_map) ); //this should maybe be sizeof(void *)?

    CHECK_CL_ERROR( clSetKernelArg(cl_data->corr_kernel, 
                                   3,
                                   sizeof(id_y_map),
                                   (void*) &id_y_map) );
    CHECK_CL_ERROR( clSetKernelArg(cl_data->corr_kernel,
                                   4,
                                   8*8*4 * sizeof(cl_uint),
                                   NULL) );

    CHECK_CL_ERROR( clSetKernelArg(cl_data->preseed_kernel,
                                   2,
                                   sizeof(id_x_map),
                                   (void*) &id_x_map) ); //this should maybe be sizeof(void *)?

    CHECK_CL_ERROR( clSetKernelArg(cl_data->preseed_kernel,
                                   3,
                                   sizeof(id_y_map),
                                   (void*) &id_y_map) );

    CHECK_CL_ERROR( clSetKernelArg(cl_data->preseed_kernel,
                                   4,
                                   64* sizeof(cl_uint),
                                   NULL) );

    CHECK_CL_ERROR( clSetKernelArg(cl_data->preseed_kernel,
                                   5,
                                   64* sizeof(cl_uint),
                                   NULL) );

    // Number of compressed accumulations.
    cl_data->num_accumulations = cl_data->config->processing.samples_per_data_set/256;

    // Accumulation kernel global and local work space sizes.
    cl_data->gws_accum[0] = 64*cl_data->config->processing.num_data_sets;
    cl_data->gws_accum[1] = (int)ceil(cl_data->config->processing.num_adjusted_elements *
        cl_data->config->processing.num_adjusted_local_freq/256.0);
    cl_data->gws_accum[2] = cl_data->config->processing.samples_per_data_set/1024;

    cl_data->lws_accum[0] = 64;
    cl_data->lws_accum[1] = 1;
    cl_data->lws_accum[2] = 1;

    // Pre-seed kernel global and local work space sizes.
    cl_data->gws_preseed[0] = 8*cl_data->config->processing.num_data_sets;
    cl_data->gws_preseed[1] = 8*cl_data->config->processing.num_adjusted_local_freq;
    cl_data->gws_preseed[2] = cl_data->config->processing.num_blocks;

    cl_data->lws_preseed[0] = 8;
    cl_data->lws_preseed[1] = 8;
    cl_data->lws_preseed[2] = 1;

    // Correlation kernel global and local work space sizes.
    cl_data->gws_corr[0] = 8*cl_data->config->processing.num_data_sets;
    cl_data->gws_corr[1] = 8*cl_data->config->processing.num_adjusted_local_freq;
    cl_data->gws_corr[2] = cl_data->num_blocks*cl_data->num_accumulations;

    cl_data->lws_corr[0] = 8;
    cl_data->lws_corr[1] = 8;
    cl_data->lws_corr[2] = 1;

}

void close_open_cl(struct OpenCLData * cl_data)
{
    CHECK_CL_ERROR( clReleaseKernel(cl_data->corr_kernel) );
    CHECK_CL_ERROR( clReleaseProgram(cl_data->program) );

    for (int i = 0; i < NUM_QUEUES; ++i) {
        CHECK_CL_ERROR( clReleaseCommandQueue(cl_data->queue[i]) );
    }

    for (int i = 0; i < cl_data->in_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(cl_data->device_input_buffer[i]) );
    }
    free(cl_data->device_input_buffer);

    for (int i = 0; i < cl_data->in_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(cl_data->device_accumulate_buffer[i]) );
    }
    free(cl_data->device_accumulate_buffer);

    for (int i = 0; i < cl_data->out_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(cl_data->device_output_buffer[i]) );
    }
    free(cl_data->device_output_buffer);

    free(cl_data->host_buffer_ready);
    free(cl_data->host_buffer_ready);
    free(cl_data->input_data_written);
    free(cl_data->accumulate_data_zeroed);
    free(cl_data->read_finished);
    free(cl_data->corr_finished);
    free(cl_data->offset_accumulate_finished);
    free(cl_data->preseed_finished);

    free(cl_data->cb_data);

    CHECK_CL_ERROR( clReleaseContext(cl_data->context) );
}
