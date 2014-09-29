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
#include "link_mapping.h"
#include "config.h"

void print_help() {
    printf("usage: kotekan [opts]\n\n");
    printf("Options:\n");
    printf("    --config (-c) [file]            The local JSON config file to use\n\n");
    printf("Testing Options:\n");
    printf("    --no-network-test (-t)          Generates fake data for testing.\n");
    printf("    --read-file (-f) [file or dir]  Read a packet dump file or dir instead of using the network.\n");
    printf("    --write-local (-l)              Write the data locally.\n");
}

int main(int argc, char ** argv) {

    // Setup syslog
    openlog ("kotekan", LOG_CONS | LOG_PID | LOG_NDELAY | LOG_PERROR, LOG_LOCAL1);

    int use_ch_acq = 1;
    int read_file = 0;
    char * input_file_name = NULL;
    int opt_val = 0;
    int no_network_test = 0;
    char * config_file_name = "../../kotekan/kotekan.conf";
    struct Config config;
    int error = 0;

    for (;;) {
        static struct option long_options[] = {
            {"no-network-test", no_argument, 0, 't'},
            {"read-file", required_argument, 0, 'f'},
            {"config", required_argument, 0, 'c'},
            {"write-local", no_argument, 0, 'l'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        opt_val = getopt_long (argc, argv, "htlf:c:",
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
            case 'l':
                use_ch_acq = 0;
                break;
            case 'f':
                read_file = 1;
                input_file_name = strdup(optarg);
                break;
            case 'c':
                config_file_name = strdup(optarg);
                break;
            default:
                printf("Invalid option, run with -h to see options");
                return -1;
                break;
        }
    }

    // Load configuration file.
    error = load_config_from_file(&config, config_file_name);
    if (error) {
        ERROR("Error processing config file, exiting...");
        return -1;
    }

    INFO("Parsed config:");
    print_config(&config);

    // If we are reading from a file, force the number of links to be 1.
    if (read_file == 1) {
        config.fpga_network.num_links = 1;
    }

    // Create buffers.
    struct Buffer input_buffer[config.gpu.num_gpus];
    struct Buffer output_buffer[config.gpu.num_gpus];

    cl_int output_len = config.processing.num_adjusted_local_freq * config.processing.num_blocks*
                        (config.gpu.block_size*config.gpu.block_size)*2.;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool pool[config.gpu.num_gpus];

    pthread_t gpu_t[config.gpu.num_gpus];
    struct gpuThreadArgs gpu_args[config.gpu.num_gpus];

    for (int i = 0; i < config.gpu.num_gpus; ++i) {

        INFO ("Creating buffers");

        // TODO Figure out why this value currently needs to be constant.
        int links_per_gpu = 3; // num_links_per_gpu(&config, i);

        create_info_pool(&pool[i], 2 * links_per_gpu * config.processing.buffer_depth,
                                    config.processing.num_adjusted_local_freq,
                                    config.processing.num_adjusted_elements);

        create_buffer(&input_buffer[i], links_per_gpu * config.processing.buffer_depth,
                      config.processing.samples_per_data_set * config.processing.num_adjusted_elements *
                      config.processing.num_adjusted_local_freq * config.processing.num_data_sets,
                      links_per_gpu, &pool[i]);
        create_buffer(&output_buffer[i], links_per_gpu * config.processing.buffer_depth,
                      output_len * config.processing.num_data_sets * sizeof(cl_int), 1, &pool[i]);

        gpu_args[i].in_buf = &input_buffer[i];
        gpu_args[i].out_buf = &output_buffer[i];
        gpu_args[i].gpu_id = i;
        gpu_args[i].started = 0;
        gpu_args[i].config = &config;

        CHECK_ERROR( pthread_mutex_init(&gpu_args[i].lock, NULL) );
        CHECK_ERROR( pthread_cond_init(&gpu_args[i].cond, NULL) );

        DEBUG("Setting up GPU thread %d\n", i);

        CHECK_ERROR( pthread_create(&gpu_t[i], NULL, (void *) &gpu_thread, (void *)&gpu_args[i] ) );

        // Block until the OpenCL thread is read to read data.
        wait_for_gpu_thread_ready(&gpu_args[i]);

        CHECK_ERROR( pthread_mutex_destroy(&gpu_args[i].lock) );
        CHECK_ERROR( pthread_cond_destroy(&gpu_args[i].cond) );
    }

    //sleep(5);
    DEBUG("Starting up network threads\n");

    // Create network threads
    pthread_t network_t[config.fpga_network.num_links];
    struct networkThreadArg network_args[config.fpga_network.num_links];

    for (int i = 0; i < config.fpga_network.num_links; ++i) {
        DEBUG("Link %d being assigned to buffer %d", i, config.fpga_network.link_map[i].gpu_id);
        network_args[i].buf = &input_buffer[config.fpga_network.link_map[i].gpu_id];
        //Sets how long takes data.  in GB.  ~4800 is ~1hr
        // 0 = unlimited
        network_args[i].data_limit = 0;
        network_args[i].ip_address = config.fpga_network.link_map[i].link_name;
        network_args[i].link_id = config.fpga_network.link_map[i].link_id;
        network_args[i].frequency_id = i;
        network_args[i].num_links_in_group = num_links_in_group(&config, i);

        // Args for reading from file
        network_args[i].read_from_file = read_file;
        network_args[i].file_name = input_file_name;

        network_args[i].config = &config;

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
        ch_acq_uplink_args.buf = output_buffer;
        ch_acq_uplink_args.config = &config;
        CHECK_ERROR( pthread_create(&output_consumer_t, NULL, (void *) &ch_acq_uplink_thread, (void *)&ch_acq_uplink_args ) );
    } else  {
        // Create consumer thread (i.e. file write thread).
        // TODO: only works with one GPU, fix to work with more than one?
        struct fileWriteThreadArg file_write_args;
        file_write_args.config = &config;
        file_write_args.buf = &output_buffer[0];
        file_write_args.data_dir = "results";
        file_write_args.dataset_name = "test_data";
        CHECK_ERROR( pthread_create(&output_consumer_t, NULL, (void *) &file_write_thread, (void *)&file_write_args ) );
    }

    // Join with threads.
    int * ret;

    for (int i = 0; i < config.gpu.num_gpus; ++i) {
        CHECK_ERROR( pthread_join(gpu_t[i], (void **) &ret) );
    }
    for (int i = 0; i < config.fpga_network.num_links; ++i) {
        CHECK_ERROR( pthread_join(network_t[i], (void **) &ret) );
    }

    CHECK_ERROR( pthread_join(output_consumer_t, (void **) &ret) );

    closelog();

    return 0;
}
