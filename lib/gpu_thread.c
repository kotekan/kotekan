

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
#include "util.h"
#include "beamforming.h"

#include "pthread.h"
#include <unistd.h>

pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;

void setup_open_cl(struct OpenCLData * cl_data);

void close_open_cl(struct OpenCLData * cl_data);

void add_queue_set(struct OpenCLData * cl_data, int buffer_id);

void release_events_for_buffer(struct OpenCLData * cl_data, int buffer_id);

void add_write_to_queue_set(struct OpenCLData * cl_data, int buffer_id);

void setup_beamform_kernel_worksize(struct OpenCLData * cl_data);

void init_command_queue(struct gpuThreadArgs * args, struct OpenCLData * cl_data) {
    int num_initial_command_queues = args->in_buf->num_buffers;
    if (args->config->gpu.use_time_shift) {
        num_initial_command_queues -= cl_data->num_links;
    }
    for (int i = 0; i < num_initial_command_queues; ++i) {
        add_queue_set(cl_data, i);
    }
}

void* gpu_thread(void* arg)
{
    struct gpuThreadArgs * args = (struct gpuThreadArgs *) arg;

    struct OpenCLData cl_data;

    cl_data.in_buf = args->in_buf;
    cl_data.out_buf = args->out_buf;
    cl_data.beamforming_out_buf = args->beamforming_out_buf;

    cl_data.config = args->config;

    cl_data.accumulate_len = cl_data.config->processing.num_adjusted_local_freq *
    cl_data.config->processing.num_adjusted_elements * 2 * cl_data.config->processing.num_data_sets * sizeof(cl_int);
    cl_data.aligned_accumulate_len = PAGESIZE_MEM * (ceil((double)cl_data.accumulate_len / (double)PAGESIZE_MEM));
    assert(cl_data.aligned_accumulate_len >= cl_data.accumulate_len);

    cl_data.gpu_id = args->gpu_id;
    cl_data.num_links = num_links_per_gpu(cl_data.config, cl_data.gpu_id);
    INFO("gpu_thread: gpu_id: %d; num_links: %d", cl_data.gpu_id, cl_data.num_links);

    cl_data.stream_info = malloc(cl_data.num_links * sizeof(struct StreamINFO));
    CHECK_MEM(cl_data.stream_info);

    setup_open_cl(&cl_data);

    CHECK_ERROR( pthread_mutex_lock(&args->lock) );
    args->started = 1;
    CHECK_ERROR( pthread_mutex_unlock(&args->lock) );

    // Signal consumer (main thread in this case).
    CHECK_ERROR( pthread_cond_broadcast(&args->cond) );

    // Just wait on one buffer.
    int buffer_list[1] = {0};
    int bufferID = 0;
    int links_started = 0;

    // Main loop for collecting data from producers.
    for(EVER) {

        // Wait for data, this call will block.
        bufferID = get_full_buffer_from_list(args->in_buf, buffer_list, 1);

        //INFO("gpu_thread; got buffer on gpu %d in buffer (%d,%d)", args->gpu_id, args->gpu_id, bufferID);

        // We need the next buffer to be empty (i.e. for the GPU to have already processed it).
        if (args->config->gpu.use_time_shift) {
            wait_for_empty_buffer(args->in_buf, mod(buffer_list[0] + cl_data.num_links, args->in_buf->num_buffers));
        }

        // When the system starts we need to get one buffer from each network link to
        // know the seq number, stream_i, and absolute time to initialize the command queues.
        if (links_started < cl_data.num_links) {

            // Set the stream ID for the link.
            int32_t stream_id = get_streamID(args->in_buf, bufferID);
            assert(stream_id != -1);
            cl_data.stream_info[links_started].stream_id = extract_stream_id(stream_id);

            // Todo get/set time information here as well.

            links_started++;
            if (links_started == cl_data.num_links) {
                // Setup the beamforming options that required stream information.
                if (cl_data.config->gpu.use_beamforming == 1)
                    setup_beamform_kernel_worksize(&cl_data);
                // Create the inital command queue.
                init_command_queue(args, &cl_data);
                // Start the GPU transfers/kernels for the data we already have.
                for (int i = 0; i < cl_data.num_links; ++i) {
                    CHECK_CL_ERROR( clSetUserEventStatus(cl_data.host_buffer_ready[i], CL_SUCCESS) );
                }
            }

            buffer_list[0] = (buffer_list[0] + 1) % args->in_buf->num_buffers;
            continue;
        }

        // If buffer id is -1, then all the producers are done.
        if (bufferID == -1) {
            break;
        }

        // Wait for the output buffer to be empty as well.
        // We also require that the N+1 buffer be empty, so that the N+1 user event
        // has been cleared.
        // This should almost never block, since the output buffer should clear quickly.
        wait_for_empty_buffer(args->out_buf, bufferID);
        if (args->config->gpu.use_time_shift) {
            wait_for_empty_buffer(args->out_buf, mod(bufferID + cl_data.num_links, args->out_buf->num_buffers));
        }
        if (args->config->gpu.use_beamforming) {
            wait_for_empty_buffer(args->beamforming_out_buf, bufferID);
        }

        //INFO("GPU Kernel started on gpu %d in buffer (%d,%d)", args->gpu_id, args->gpu_id, bufferID);

        CHECK_CL_ERROR( clSetUserEventStatus(cl_data.host_buffer_ready[bufferID], CL_SUCCESS) );

        buffer_list[0] = (buffer_list[0] + 1) % args->in_buf->num_buffers;
    }

    DEBUG("Closing OpenCL, thread: %d", args->gpu_id);

    // *** BUG ***
    // This function blocks, and prevents the system from shutting down.  However without this
    // function the program will have memory leaks if gpu_thread is restarted!
    // *** BUG ***
    //close_open_cl(&cl_data);

    mark_producer_done(args->out_buf, 0);

    DEBUG("Closed OpenCL, thread: %d", args->gpu_id);

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
    if (cb_data->cl_data->config->gpu.use_beamforming) {
        copy_buffer_info(cb_data->cl_data->in_buf, cb_data->buffer_id,
                         cb_data->cl_data->beamforming_out_buf, cb_data->buffer_id);
    }

    // Copy the information contained in the input buffer
    move_buffer_info(cb_data->cl_data->in_buf, cb_data->buffer_id,
                     cb_data->cl_data->out_buf, cb_data->buffer_id);

    // Mark the input buffer as "empty" so that it can be reused.
    mark_buffer_empty(cb_data->cl_data->in_buf, cb_data->buffer_id);

    // Mark the output buffer as full, so it can be processed.
    mark_buffer_full(cb_data->cl_data->out_buf, cb_data->buffer_id);

    // Mark the beamforming buffer as full.
    if (cb_data->cl_data->config->gpu.use_beamforming) {
        mark_buffer_full(cb_data->cl_data->beamforming_out_buf, cb_data->buffer_id);
    }

    // Relase the events from the last queue set run.
    release_events_for_buffer(cb_data->cl_data, cb_data->buffer_id);

    if (cb_data->cl_data->config->gpu.use_time_shift == 1) {
        // We add the (n - links) queue set, since it will create the n write buffer.
        add_queue_set(cb_data->cl_data, mod(cb_data->buffer_id - cb_data->cl_data->num_links, cb_data->cl_data->in_buf->num_buffers));
    } else {
        add_queue_set(cb_data->cl_data, cb_data->buffer_id);
    }
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

    if (cl_data->config->gpu.use_time_shift) {
        assert(cl_data->time_shift_finished[buffer_id] != NULL);
        clReleaseEvent(cl_data->time_shift_finished[buffer_id]);
    }

    if (cl_data->config->gpu.use_beamforming) {
        assert(cl_data->beamform_finished[buffer_id] != NULL);
        clReleaseEvent(cl_data->beamform_finished[buffer_id]);
    }
}

void add_write_to_queue_set(struct OpenCLData * cl_data, int buffer_id)
{
    cl_int err;

    cl_data->host_buffer_ready[buffer_id] = clCreateUserEvent(cl_data->context, &err);

    CHECK_CL_ERROR(err);

    if (cl_data->config->gpu.use_time_shift == 1) {

        // TODO Remove this requirement!!
        assert(cl_data->in_buf->aligned_buffer_size == cl_data->in_buf->buffer_size);

        // Data transfer to GPU
        CHECK_CL_ERROR( clEnqueueWriteBuffer(cl_data->queue[0],
                                            cl_data->device_input_buffer[buffer_id % cl_data->num_links],
                                            CL_FALSE,
                                            cl_data->in_buf->buffer_size * (buffer_id / cl_data->num_links), //offset
                                            cl_data->in_buf->buffer_size,
                                            cl_data->in_buf->data[buffer_id],
                                            1,
                                            &cl_data->host_buffer_ready[buffer_id], // Wait on this user event (network finished).
                                            &cl_data->input_data_written[buffer_id]) );

    } else {
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
    }

    CHECK_CL_ERROR( clEnqueueWriteBuffer(cl_data->queue[0],
                                         cl_data->device_accumulate_buffer[buffer_id],
                                         CL_FALSE,
                                         0,
                                         cl_data->aligned_accumulate_len,
                                         cl_data->accumulate_zeros,
                                         1,
                                         &cl_data->input_data_written[buffer_id],
                                         &cl_data->accumulate_data_zeroed[buffer_id]) );
}

void add_time_shift_kernel(struct OpenCLData * cl_data, int buffer_id) {

    cl_event mem_copy_in_finished_list[2];
    mem_copy_in_finished_list[0] = cl_data->accumulate_data_zeroed[buffer_id];
    mem_copy_in_finished_list[1] = cl_data->accumulate_data_zeroed[mod(buffer_id + cl_data->num_links, cl_data->in_buf->num_buffers)];

    CHECK_CL_ERROR( clSetKernelArg(cl_data->time_shift_kernel,
                                   0,
                                   sizeof(void *),
                                   (void *) &cl_data->device_input_buffer[buffer_id % cl_data->num_links]) );

    CHECK_CL_ERROR( clSetKernelArg(cl_data->time_shift_kernel,
                                   1,
                                   sizeof(void *),
                                   (void *) &cl_data->device_time_shifted_buffer) );

    CHECK_CL_ERROR( clSetKernelArg(cl_data->time_shift_kernel,
                                   2,
                                   sizeof(cl_int),
                                   &cl_data->config->gpu.ts_num_elem_to_shift) );
    CHECK_CL_ERROR( clSetKernelArg(cl_data->time_shift_kernel,
                                   3,
                                   sizeof(cl_int),
                                   &cl_data->config->gpu.ts_element_offset) );
    CHECK_CL_ERROR( clSetKernelArg(cl_data->time_shift_kernel,
                                   4,
                                   sizeof(cl_int),
                                   &cl_data->config->gpu.ts_samples_to_shift) );
    int adjusted_buffer_id = buffer_id / cl_data->num_links;
    CHECK_CL_ERROR( clSetKernelArg(cl_data->time_shift_kernel,
                                   5,
                                   sizeof(cl_int),
                                   &adjusted_buffer_id) );

    CHECK_CL_ERROR( clEnqueueNDRangeKernel(cl_data->queue[1],
                                           cl_data->time_shift_kernel,
                                           3,
                                           NULL,
                                           cl_data->gws_time_shift,
                                           cl_data->lws_time_shift,
                                           2,
                                           mem_copy_in_finished_list,
                                           &cl_data->time_shift_finished[buffer_id]) );

}

void add_queue_set(struct OpenCLData * cl_data, int buffer_id)
{
    // This function can be called from a call back, so it must be thread safe to avoid
    // having queues out of order.
    pthread_mutex_lock(&queue_lock);

    // Set call back data
    cl_data->cb_data[buffer_id].buffer_id = buffer_id;
    cl_data->cb_data[buffer_id].cl_data = cl_data;

    // The input data host to device copies.
    if (cl_data->config->gpu.use_time_shift == 1) {
        add_write_to_queue_set(cl_data, mod(buffer_id + cl_data->num_links, cl_data->in_buf->num_buffers));
    } else {
        add_write_to_queue_set(cl_data, buffer_id);
    }

    cl_mem device_kernel_input_data = cl_data->device_input_buffer[buffer_id];
    cl_event * input_data_ready_event = &cl_data->accumulate_data_zeroed[buffer_id];

    // The time shift kernel
    if (cl_data->config->gpu.use_time_shift == 1) {
        add_time_shift_kernel(cl_data, buffer_id);
        device_kernel_input_data = cl_data->device_time_shifted_buffer;
        input_data_ready_event = &cl_data->time_shift_finished[buffer_id];
    }

    // The offset accumulate kernel args.
    // Set 2 arguments--input array and zeroed output array

    CHECK_CL_ERROR( clSetKernelArg(cl_data->offset_accumulate_kernel,
                                        0,
                                        sizeof(void*),
                                       (void*) &device_kernel_input_data) );

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
                                            input_data_ready_event,
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
                                       (void*) &device_kernel_input_data) );

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

    // The beamforming kernel.
    cl_event last_event = cl_data->read_finished[buffer_id];
    if (cl_data->config->gpu.use_beamforming) {


        CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                       0,
                                       sizeof(void *),
                                       (void*) &device_kernel_input_data) );

        CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                       1,
                                       sizeof(void *),
                                       (void*) &cl_data->device_beamform_output_buffer[buffer_id]) );

        CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                       2,
                                       sizeof(cl_mem),
                                       (void*) &cl_data->device_freq_map[buffer_id % cl_data->num_links]) );

        CHECK_CL_ERROR( clEnqueueNDRangeKernel(cl_data->queue[1],
                                               cl_data->beamform_kernel,
                                               3,
                                               NULL,
                                               cl_data->gws_beamforming,
                                               cl_data->lws_beamforming,
                                               1,
                                               &cl_data->corr_finished[buffer_id],
                                               &cl_data->beamform_finished[buffer_id]) );

        // Read the results
        CHECK_CL_ERROR( clEnqueueReadBuffer(cl_data->queue[2],
                                            cl_data->device_beamform_output_buffer[buffer_id],
                                            CL_FALSE,
                                            0,
                                            cl_data->beamforming_out_buf->aligned_buffer_size,
                                            cl_data->beamforming_out_buf->data[buffer_id],
                                            1,
                                            &cl_data->beamform_finished[buffer_id],
                                            &cl_data->beamform_read_finished[buffer_id]) );

        last_event = cl_data->beamform_read_finished[buffer_id];
    }
    // Setup call back.
    CHECK_CL_ERROR( clSetEventCallback(last_event,
                                            CL_COMPLETE,
                                            &read_complete,
                                            &cl_data->cb_data[buffer_id]) );

    pthread_mutex_unlock(&queue_lock);
}

void setup_beamform_kernel_worksize(struct OpenCLData * cl_data) {

    cl_int err;
    cl_data->device_freq_map = malloc(cl_data->num_links * sizeof(cl_mem));
    CHECK_MEM(cl_data->device_freq_map);
    float freq[cl_data->config->processing.num_local_freq];
    stream_id_t stream_id;

    for (int i = 0; i < cl_data->num_links; ++i) {

        stream_id = cl_data->stream_info[i].stream_id;

        for (int j = 0; j < cl_data->config->processing.num_local_freq; ++j) {
            freq[j] = freq_from_bin(bin_number(&stream_id, j));
        }
        cl_data->device_freq_map[i] = clCreateBuffer(cl_data->context,
                                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                cl_data->config->processing.num_local_freq
                                                    * sizeof(float),
                                                freq,
                                                &err);
        CHECK_CL_ERROR(err);
        INFO("creating buffer for freq step %d", i);
    }

    INFO("setup_beamform_kernel_worksize, setting bit shift factor to %d",
         cl_data->config->beamforming.bit_shift_factor);
    CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                   5,
                                   sizeof(int),
                                   &cl_data->config->beamforming.bit_shift_factor) );

    // Beamforming kernel global and local work space sizes.
    cl_data->gws_beamforming[0] = cl_data->config->processing.num_elements / 4;
    cl_data->gws_beamforming[1] = cl_data->config->processing.num_local_freq;
    cl_data->gws_beamforming[2] = cl_data->config->processing.samples_per_data_set / 32;

    cl_data->lws_beamforming[0] = 64;
    cl_data->lws_beamforming[1] = 1;
    cl_data->lws_beamforming[2] = 1;
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
    sprintf(cl_options, "-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du -D NUM_BUFFERS=%du",
            cl_data->config->processing.num_elements, cl_data->config->processing.num_local_freq,
            cl_data->config->processing.num_adjusted_elements,
            cl_data->config->processing.num_adjusted_local_freq,
            cl_data->config->processing.num_blocks,
            cl_data->config->processing.samples_per_data_set,
            cl_data->config->processing.buffer_depth);
    INFO("Kernel options: %s", cl_options);

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

    cl_data->time_shift_kernel = clCreateKernel( cl_data->program, "time_shift", &err );
    CHECK_CL_ERROR(err);

    cl_data->beamform_kernel = clCreateKernel( cl_data->program, "gpu_beamforming", &err );
    CHECK_CL_ERROR(err);

    // Create command queues
    for (int i = 0; i < NUM_QUEUES; ++i) {
        cl_data->queue[i] = clCreateCommandQueue( cl_data->context, cl_data->device_id[cl_data->gpu_id], CL_QUEUE_PROFILING_ENABLE, &err );
        CHECK_CL_ERROR(err);
    }

    // TODO create a struct to contain all of these (including events) to make this memory allocation cleaner. 
    // 

    // Setup device input buffers.
    int num_input_buffers = cl_data->in_buf->num_buffers;
    int input_buffer_size = cl_data->in_buf->aligned_buffer_size;

    cl_data->num_links = num_links_per_gpu(cl_data->config, cl_data->gpu_id);
    if (cl_data->config->gpu.use_time_shift) {
        num_input_buffers = cl_data->num_links;
        input_buffer_size = cl_data->in_buf->buffer_size * cl_data->config->processing.buffer_depth;
    }

    cl_data->device_input_buffer = (cl_mem *) malloc(num_input_buffers * sizeof(cl_mem));
    INFO("setup_open_cl, gpu id %d, numlinks %d", cl_data->gpu_id, cl_data->num_links);
    CHECK_MEM(cl_data->device_input_buffer);

    for (int i = 0; i < num_input_buffers; ++i) {
        cl_data->device_input_buffer[i] = clCreateBuffer(cl_data->context,
                                                          CL_MEM_READ_ONLY,
                                                          input_buffer_size,
                                                          NULL,
                                                          &err);
        CHECK_CL_ERROR(err);
    }

    if (cl_data->config->gpu.use_time_shift) {
        // Create the time_shifted buffer.
        cl_data->device_time_shifted_buffer = clCreateBuffer(cl_data->context,
                                                            CL_MEM_READ_WRITE,
                                                            cl_data->in_buf->buffer_size,
                                                            NULL,
                                                            &err);
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
        cl_data->device_accumulate_buffer[i] = clCreateBuffer(cl_data->context,
                                                              CL_MEM_READ_WRITE,
                                                              cl_data->aligned_accumulate_len,
                                                              NULL, &err);
        CHECK_CL_ERROR(err);
    }

    // Setup device output buffers.
    cl_data->device_output_buffer = (cl_mem *) malloc(cl_data->out_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(cl_data->device_output_buffer);
    for (int i = 0; i < cl_data->out_buf->num_buffers; ++i) {
        cl_data->device_output_buffer[i] = clCreateBuffer(cl_data->context,
                                                          CL_MEM_WRITE_ONLY,
                                                          cl_data->out_buf->aligned_buffer_size,
                                                          NULL, &err);
        CHECK_CL_ERROR(err);
    }

    // Setup beamforming output buffers.
    if (cl_data->config->gpu.use_beamforming == 1) {
        cl_data->device_beamform_output_buffer = (cl_mem *) malloc(cl_data->beamforming_out_buf->num_buffers * sizeof(cl_mem));
        CHECK_MEM(cl_data->device_beamform_output_buffer);
        for (int i = 0; i < cl_data->beamforming_out_buf->num_buffers; ++i) {
            cl_data->device_beamform_output_buffer[i] = clCreateBuffer(cl_data->context,
                                                                    CL_MEM_WRITE_ONLY,
                                                                    cl_data->beamforming_out_buf->aligned_buffer_size,
                                                                    NULL, &err);
            CHECK_CL_ERROR(err);
        }
    }

    cl_data->cb_data = malloc(cl_data->in_buf->num_buffers * sizeof(struct callBackData));
    CHECK_MEM(cl_data->cb_data);

    cl_data->host_buffer_ready = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->host_buffer_ready);

    cl_data->offset_accumulate_finished = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->offset_accumulate_finished);

    cl_data->time_shift_finished = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->time_shift_finished);

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

    cl_data->beamform_finished = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->beamform_finished);

    cl_data->beamform_read_finished = malloc(cl_data->in_buf->num_buffers * sizeof(cl_event));
    CHECK_MEM(cl_data->beamform_read_finished);

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

    cl_int *zeros=calloc(cl_data->num_blocks*cl_data->config->processing.num_local_freq,
                         sizeof(cl_int)); // block locking

    cl_data->device_block_lock = clCreateBuffer (cl_data->context,
                                        CL_MEM_COPY_HOST_PTR,
                                        cl_data->num_blocks*cl_data->config->processing.num_local_freq*sizeof(cl_int),
                                        zeros,
                                        &err);
    CHECK_CL_ERROR(err);

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
                                   sizeof(void*),
                                   (void*)&cl_data->device_block_lock) );

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

    // Setup beamforming output.

    // TODO update this at the ~1 minute time basis.
    float phases[cl_data->config->processing.num_elements];
    /*get_delays(cl_data->config->beamforming.ra,
               cl_data->config->beamforming.dec,
               cl_data->config,
               cl_data->config->beamforming.element_positions,
               phases);*/
    cl_mem device_phases = clCreateBuffer(cl_data->context,
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          cl_data->config->processing.num_elements * sizeof(cl_uint),
                                          phases,
                                          &err);
    CHECK_CL_ERROR(err);

    CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                   3,
                                   sizeof(cl_mem),
                                   (void*) &device_phases) );

    unsigned char mask[cl_data->config->processing.num_adjusted_elements];
    for (int i = 0; i < cl_data->config->processing.num_adjusted_elements; ++i) {
        mask[i] = 1;
    }
    for (int i = 0; i < cl_data->config->beamforming.num_masked_elements; ++i) {
        int mask_position = cl_data->config->beamforming.element_mask[i];
        mask_position = cl_data->config->processing.inverse_product_remap[mask_position];
        mask[mask_position] = 0;
    }
    //for (int i = 0; i < cl_data->config->processing.num_adjusted_elements; ++i) {
    //    INFO("MASK[%d] = %d", i, mask[i]);
    //}
    cl_mem device_mask = clCreateBuffer(cl_data->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        cl_data->config->processing.num_elements * sizeof(unsigned char),
                                        mask,
                                        &err);

    CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                   4,
                                   sizeof(cl_mem),
                                   (void*) &device_mask) );

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

    cl_data->gws_time_shift[0] = cl_data->config->processing.num_elements;
    cl_data->gws_time_shift[1] = cl_data->config->processing.num_local_freq;
    cl_data->gws_time_shift[2] = cl_data->config->processing.samples_per_data_set *
                                    cl_data->config->processing.num_data_sets;

    // Time shift kernel global and local work space sizes.
    if (cl_data->config->processing.num_elements > 64) {
        cl_data->lws_time_shift[0] = 64;
        cl_data->lws_time_shift[1] = 1;
    }
    else{
        cl_data->lws_time_shift[0] = cl_data->config->processing.num_elements;
        cl_data->lws_time_shift[1] = 64/cl_data->config->processing.num_elements;
    }
    cl_data->lws_time_shift[2] = 1;

}

void close_open_cl(struct OpenCLData * cl_data)
{
    for (int i = 0; i < NUM_QUEUES; ++i) {
        CHECK_CL_ERROR( clReleaseCommandQueue(cl_data->queue[i]) );
    }

    CHECK_CL_ERROR( clReleaseKernel(cl_data->time_shift_kernel) );
    CHECK_CL_ERROR( clReleaseKernel(cl_data->preseed_kernel) );
    CHECK_CL_ERROR( clReleaseKernel(cl_data->offset_accumulate_kernel) );
    CHECK_CL_ERROR( clReleaseKernel(cl_data->corr_kernel) );
    CHECK_CL_ERROR( clReleaseProgram(cl_data->program) );

    int num_input_buffers = cl_data->in_buf->num_buffers;
    if (cl_data->config->gpu.use_time_shift) {
        num_input_buffers = cl_data->num_links;
    }
    for (int i = 0; i < num_input_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(cl_data->device_input_buffer[i]) );
    }
    free(cl_data->device_input_buffer);
    CHECK_CL_ERROR( clReleaseMemObject(cl_data->device_time_shifted_buffer) );
    CHECK_CL_ERROR( clReleaseMemObject(cl_data->device_block_lock) );

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
    free(cl_data->beamform_finished);
    free(cl_data->beamform_read_finished);

    free(cl_data->cb_data);

    CHECK_CL_ERROR( clReleaseContext(cl_data->context) );
}

