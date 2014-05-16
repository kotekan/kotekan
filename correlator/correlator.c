#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <getopt.h>
#include <assert.h>
#include <string.h>

#include "errors.h"
#include "buffers.h"
#include "gpu_thread.h"
#include "network_dna.h"
#include "file_write.h"
#include "error_correction.h"
#include "ch_acq_uplink.h"

#define BLOCK_SIZE 32

#define NUM_ELEMENTS 32  // Must be >= 32
#define NUM_FREQ 64
// Since we are using the kernels for the 256-element correlator for a
// 16-element correlator, we pack more than one full correlator product into each
// work group to make it efficent.  This means the paramaters for the 16-element
// correlator don't reflect the true number of frequencies and elements per FPGA link.
// These true values are set here.
#define ACTUAL_NUM_ELEMENTS 16u
#define ACTUAL_NUM_FREQUENCIES 128u

#define TIMESAMPLES_PER_PACKET 4

#define CH_ACQ_PORT 41001

// The total number of frequencies across the entire system.
#define TOTAL_FREQUENCIES 1024

void print_help() {
    printf("usage: correlator [opts]\n\n");
    printf("Options:\n");
    printf("    --gpu (-g) [number]         The id of the GPU to use.\n");
    printf("    --use-ch-acq (-a) [ip address]");
    printf("    --read-file (-f) [file or dir]  Read a packet dump file or dir instead of using the network.\n");
    printf("    --num-links (-l) [number]       Number of links to listen to (default 8).\n");
    printf("    --spf (-s) [number]             Number of time samples per GPU frame x1024 (default 32).\n");
    printf("    --num-data-sets (-n) [number]   Number of sub data sets per GPU frame.\n");
    printf("    --gpu-spf (-S) [number]         Number of GPU frames to intergrate per final frame (default 1).\n");
    printf("    --buffer-depth (-d) [number]    The number of buffers to keep per link (default/min 3).\n\n");
    printf("Testing Options:\n");
    printf("    --no-network-test (-t)          Generates fake data for testing.\n\n");
}

int main(int argc, char ** argv) {

    // Setup syslog
    openlog ("correlator", LOG_CONS | LOG_PID | LOG_NDELAY | LOG_PERROR, LOG_LOCAL1);

    // Default values:
    int opt_val = 0;
    int gpu_id = 0;
    int no_network_test = 0;
    char * ch_acq_ip_address;
    int use_ch_acq = 0;
    int read_file = 0;
    char * file_name;
    int num_links = 8;
    int num_timesamples = 32;
    int gpu_spf = 1;
    int buffer_depth = 3;
    int num_data_sets = 1;
 
    for (;;) {
        static struct option long_options[] = {
            {"gpu", required_argument, 0, 'g'},
            {"no-network-test", no_argument, 0, 't'},
            {"use-ch-acq", required_argument, 0, 'a'},
            {"read-file", required_argument, 0, 'f'},
            {"num-links", required_argument, 0, 'l'},
            {"spf", required_argument, 0, 's'},
            {"gpu-spf", required_argument, 0, 'S'},
            {"buffer-depth", required_argument, 0 , 'd'},
            {"num-data-sets", required_argument, 0, 'n'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        opt_val = getopt_long (argc, argv, "g:tha:f:l:s:S:d:n:",
                               long_options, &option_index);

        // End of args
        if (opt_val == -1) {
            break;
        }

        switch (opt_val) {
            case 'h':
                print_help();
                return 0;
                break;
            case 't':
                no_network_test = 1;
                break;
            case 'g':
                gpu_id = atoi(optarg);
                break;
            case 'l':
                num_links = atoi(optarg);
                break;
            case 'a':
                ch_acq_ip_address = strdup(optarg);
                use_ch_acq = 1;
                break;
            case 'f':
                read_file = 1;
                file_name = strdup(optarg);
                break;
            case 's':
                num_timesamples = atoi(optarg);
                break;
            case 'S':
                gpu_spf = atoi(optarg);
                break;
            case 'd':
                buffer_depth = atoi(optarg);
                break;
            case 'n':
                num_data_sets = atoi(optarg);
                break;
            default:
                printf("Invalid option, run with -h to see options");
                return -1;
                break;
        }
    }

    // If we are reading from a file, force the number of links to be 1.
    if (read_file == 1) {
        num_links = 1;
    }

    // We want the number of time samples to be a multiple of 1024
    num_timesamples = num_timesamples * 1024;

    // Create buffers.
    struct Buffer input_buffer;
    struct Buffer output_buffer;

    // TODO We are also doing this math in the OpenCL thread, this should be avoided.
    int num_blocks = (NUM_ELEMENTS / BLOCK_SIZE) * (NUM_ELEMENTS / BLOCK_SIZE + 1) / 2.;
    cl_int output_len = NUM_FREQ*num_blocks*(BLOCK_SIZE*BLOCK_SIZE)*2.;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool pool;
    create_info_pool(&pool, 2 * num_links * buffer_depth, NUM_FREQ, NUM_ELEMENTS);

    create_buffer(&input_buffer, num_links * buffer_depth, num_timesamples * NUM_ELEMENTS * NUM_FREQ * num_data_sets, num_links, &pool);
    create_buffer(&output_buffer, num_links * buffer_depth, output_len * num_data_sets * sizeof(cl_int), 1, &pool);

    INFO("num_links = %d", num_links);
    INFO("num_timesamples = %d", num_timesamples);
    INFO("num_data_sets = %d", num_data_sets);
    INFO("input_buffer_size = %d", input_buffer.buffer_size);
    INFO("output_buffer_size = %d", output_buffer.buffer_size);
    INFO("gpu_spf = %d", gpu_spf);

    // Create Open CL thread.
    pthread_t gpu_t;
    struct gpuThreadArgs gpu_args;
    gpu_args.in_buf = &input_buffer;
    gpu_args.out_buf = &output_buffer;
    gpu_args.block_size = BLOCK_SIZE;
    gpu_args.num_elements = NUM_ELEMENTS;
    gpu_args.num_timesamples = num_timesamples;
    gpu_args.actual_num_elements = ACTUAL_NUM_ELEMENTS;
    gpu_args.actual_num_freq = ACTUAL_NUM_FREQUENCIES;
    gpu_args.num_freq = NUM_FREQ;
    gpu_args.gpu_id = gpu_id;
    gpu_args.started = 0;
    gpu_args.num_data_sets = num_data_sets;
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
    pthread_t network_t[num_links];
    struct networkThreadArg network_args[num_links];
    for (int i = 0; i < num_links; ++i) {
        network_args[i].buf = &input_buffer;
        network_args[i].buffer_depth = buffer_depth;
        //Sets how long takes data.  in GB.  ~4800 is ~1hr
        // 0 = unlimited
        network_args[i].data_limit = 0;
        network_args[i].port_number = 41000;
        network_args[i].num_links = num_links;
        network_args[i].link_id = i;
        char * ip_address = malloc(100*sizeof(char));
        snprintf(ip_address, 100, "dna%d", i);
        network_args[i].ip_address = ip_address;

        // Args used for testing.
        network_args[i].actual_num_elements = ACTUAL_NUM_ELEMENTS;
        network_args[i].actual_num_freq = ACTUAL_NUM_FREQUENCIES;
        network_args[i].num_timesamples = num_timesamples;

        // Args for reading from file
        network_args[i].read_from_file = read_file;
        network_args[i].file_name = file_name;

        if (no_network_test == 0) {
            CHECK_ERROR( pthread_create(&network_t[i], NULL, (void *) &network_thread, (void *)&network_args[i] ) );
        } else {
            CHECK_ERROR( pthread_create(&network_t[i], NULL, (void *) &test_network_thread, (void *)&network_args[i] ) );
        }
    }

    pthread_t output_consumer_t;
    if (use_ch_acq == 1) {
        // Consumer thread which sends data to ch_acq server to be written to disk.
        struct ch_acqUplinkThreadArg ch_acq_uplink_args;
        ch_acq_uplink_args.buf = &output_buffer;
        ch_acq_uplink_args.num_links = num_links;
        ch_acq_uplink_args.buffer_depth = buffer_depth;
        ch_acq_uplink_args.ch_acq_ip_addr = ch_acq_ip_address;
        ch_acq_uplink_args.ch_acq_port_num = CH_ACQ_PORT;
        ch_acq_uplink_args.actual_num_elements = ACTUAL_NUM_ELEMENTS;
        ch_acq_uplink_args.actual_num_freq = ACTUAL_NUM_FREQUENCIES;
        ch_acq_uplink_args.total_num_freq = TOTAL_FREQUENCIES;
        ch_acq_uplink_args.num_data_sets = num_data_sets;
        ch_acq_uplink_args.num_timesamples = num_timesamples;
        ch_acq_uplink_args.timesamples_per_packet = TIMESAMPLES_PER_PACKET;
        CHECK_ERROR( pthread_create(&output_consumer_t, NULL, (void *) &ch_acq_uplink_thread, (void *)&ch_acq_uplink_args ) );
    } else  {
        // Create consumer thread (i.e. file write thread).
        struct fileWriteThreadArg file_write_args;
        file_write_args.buf = &output_buffer;
        file_write_args.buffer_depth = buffer_depth;
        file_write_args.data_dir = "results";
        file_write_args.dataset_name = "test_data";
        file_write_args.num_links = num_links;
        file_write_args.actual_num_elements = ACTUAL_NUM_ELEMENTS;
        file_write_args.actual_num_freq = ACTUAL_NUM_FREQUENCIES;
        file_write_args.num_data_sets = num_data_sets;
        file_write_args.num_timesamples = num_timesamples;
        CHECK_ERROR( pthread_create(&output_consumer_t, NULL, (void *) &file_write_thread, (void *)&file_write_args ) );
    }

    // Join with threads.
    int * ret;

    CHECK_ERROR( pthread_join(gpu_t, (void **) &ret) );
    for (int i = 0; i < num_links; ++i) {
        CHECK_ERROR( pthread_join(network_t[i], (void **) &ret) );
    }

    CHECK_ERROR( pthread_join(output_consumer_t, (void **) &ret) );

    closelog();

    return 0;
}
