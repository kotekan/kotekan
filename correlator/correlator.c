#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>

#include "errors.h"
#include "buffers.h"
#include "gpu_thread.h"
#include "network_dna.h"
#include "file_write.h"


#define NUM_LINKS 7
#define BUFFER_DEPTH 3

#define BLOCK_SIZE 32

#define NUM_ELEMENTS 32  // Must be >= 32
#define NUM_TIMESAMPLES 64*1024 //32 initally best
#define NUM_FREQ 64
// Since we are using the kernels for the 256-element correlator for a
// 16-element correlator, we pack more than one full correlator product into each
// work group to make it efficent.  This means the paramaters for the 16-element
// correlator don't reflect the true number of frequencies and elements per FPGA link.
// These true values are set here.
#define ACTUAL_NUM_ELEMENTS 16u
#define ACTUAL_NUM_FREQUENCIES 128u

int main(int argc, char ** argv) {

    // TODO process argc/argv in a nice way.

    // Setup syslog
    openlog ("correlator", LOG_CONS | LOG_PID | LOG_NDELAY | LOG_PERROR, LOG_LOCAL1);

    // Create buffers.
    struct Buffer input_buffer;
    struct Buffer output_buffer;

    // TODO We are also doing this math in the OpenCL thread, this should be avoided.
    int num_blocks = (NUM_ELEMENTS / BLOCK_SIZE) * (NUM_ELEMENTS / BLOCK_SIZE + 1) / 2.;
    cl_int output_len = NUM_FREQ*num_blocks*(BLOCK_SIZE*BLOCK_SIZE)*2.;

    createBuffer(&input_buffer, NUM_LINKS * BUFFER_DEPTH, NUM_TIMESAMPLES * NUM_ELEMENTS * NUM_FREQ, NUM_LINKS);
    createBuffer(&output_buffer, NUM_LINKS * BUFFER_DEPTH, output_len * sizeof(cl_int), 1);
    DEBUG("%d %d %d %d", NUM_TIMESAMPLES * NUM_ELEMENTS * NUM_FREQ, output_len, num_blocks);

    // Create Open CL thread.
    pthread_t gpu_t;
    struct gpu_thread_args gpu_args;
    gpu_args.in_buf = &input_buffer;
    gpu_args.out_buf = &output_buffer;
    gpu_args.block_size = BLOCK_SIZE;
    gpu_args.num_elements = NUM_ELEMENTS;
    gpu_args.num_timesamples = NUM_TIMESAMPLES;
    gpu_args.actual_num_elements = ACTUAL_NUM_ELEMENTS;
    gpu_args.actual_num_freq = ACTUAL_NUM_FREQUENCIES;
    gpu_args.num_freq = NUM_FREQ;
    gpu_args.gpu_id = atoi(argv[1]);
    gpu_args.started = 0;
    CHECK_ERROR( pthread_mutex_init(&gpu_args.lock, NULL) );
    CHECK_ERROR( pthread_cond_init(&gpu_args.cond, NULL) );

    DEBUG("Setting up GPU thread\n");

    CHECK_ERROR( pthread_create(&gpu_t, NULL, (void *) &gpu_thread, (void *)&gpu_args ) );

    // Block until the OpenCL thread is read to read data.
    wait_for_gpu_thread_ready(&gpu_args);

    CHECK_ERROR( pthread_mutex_destroy(&gpu_args.lock) );
    CHECK_ERROR( pthread_cond_destroy(&gpu_args.cond) );

    DEBUG("Starting up network threads\n");

    // Create network threads
    pthread_t network_t[NUM_LINKS];
    struct network_thread_arg network_args[NUM_LINKS];
    for (int i = 0; i < NUM_LINKS; ++i) {
        network_args[i].buf = &input_buffer;
        network_args[i].bufferDepth = BUFFER_DEPTH;
        //Sets how long takes data.  in GB.  ~4800 is ~1hr
        network_args[i].data_limit = 2400;
        network_args[i].portNumber = 41000;
        network_args[i].numLinks = NUM_LINKS;
        network_args[i].link_id = i;
        char * ip_address = malloc(100*sizeof(char));
        snprintf(ip_address, 100, "dna%d", i);
        network_args[i].ip_address = ip_address;

        CHECK_ERROR( pthread_create(&network_t[i], NULL, (void *) &network_thread, (void *)&network_args[i] ) );
    }

    // Create consumer thread (i.e. file write thread).
    pthread_t file_write_t;
    struct file_write_thread_arg file_write_args;
    file_write_args.buf = &output_buffer;
    file_write_args.bufferDepth = BUFFER_DEPTH;
    file_write_args.data_dir = "results";
    file_write_args.dataset_name = "test_data";
    file_write_args.num_links = NUM_LINKS;
    CHECK_ERROR( pthread_create(&file_write_t, NULL, (void *) &file_write_thread, (void *)&file_write_args ) );

    // Join with threads.
    int * ret;

    CHECK_ERROR( pthread_join(gpu_t, (void **) &ret) );
    for (int i = 0; i < NUM_LINKS; ++i) {
        CHECK_ERROR( pthread_join(network_t[i], (void **) &ret) );
    }

    CHECK_ERROR( pthread_join(file_write_t, (void **) &ret) );

    closelog();

    return 0;
}
