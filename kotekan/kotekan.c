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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "errors.h"
#include "buffers.h"
#include "gpu_thread.h"
#include "network_dna.h"
#include "file_write.h"
#include "error_correction.h"
#include "ch_acq_uplink.h"
#include "config.h"

void print_help() {
    printf("usage: kotekan [opts]\n\n");
    printf("Options:\n");
    printf("    --config (-c) [file]            The local JSON config file to use\n");
    printf("    --log-to-screen (-s)            Output log messages to stderr\n");
    printf("    --log-level (-l) [level]        0 = errors only, 1 = +warnings, 2 = +info, 3 = +debug\n\n");
    printf("    --daemon -d [root]              Run in daemon mode, disables output to stderr");
    printf("Testing Options:\n");
    printf("    --no-network-test (-t)          Generates fake data for testing.\n");
    printf("    --read-file (-f) [file or dir]  Read a packet dump file or dir instead of using the network.\n");
    printf("    --write-local (-w)              Write the data locally.\n");
}

void daemonize(char * root_dir, int log_options) {
    // Process ID and Session ID.
    pid_t pid, sid;

    // Fork off the parent process.
    pid = fork();
    if (pid < 0) {
        exit(EXIT_FAILURE);
    }

    // Exit the parrent proces.
    if (pid > 0) {
        FILE * pid_fd = fopen("/var/run/kotekan.pid", "w");
        if (!pid_fd) {
            perror("failed to open PID file");
            exit(EXIT_FAILURE);
        }
        fprintf(pid_fd, "%d\n", pid);
        fclose(pid_fd);
        exit(EXIT_SUCCESS);
    }

    // Change the file mode mask.
    umask(0);

    // Open logs.
    openlog ("kotekan", log_options, LOG_LOCAL1);

    // Create a new SID for the child process
    sid = setsid();
    if (sid < 0) {
        ERROR("Cannot set SID!");
        exit(EXIT_FAILURE);
    }

    // Change the current working directory.
    if ((chdir(root_dir)) < 0) {
        ERROR("Cannot change directory to %s", root_dir);
        exit(EXIT_FAILURE);
    }

    // Close standard file descriptors
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    // TODO Handle signals.
}

int main(int argc, char ** argv) {

    int use_ch_acq = 1;
    int read_file = 0;
    char * input_file_name = NULL;
    int opt_val = 0;
    int no_network_test = 0;
    char * config_file_name = "kotekan.conf";
    struct Config config;
    int error = 0;
    int log_options = LOG_CONS | LOG_PID | LOG_NDELAY;
    int make_daemon = 0;
    char * kotekan_root_dir = "./";

    for (;;) {
        static struct option long_options[] = {
            {"no-network-test", no_argument, 0, 't'},
            {"read-file", required_argument, 0, 'f'},
            {"config", required_argument, 0, 'c'},
            {"write-local", no_argument, 0, 'w'},
            {"log-level", required_argument, 0, 'l'},
            {"daemon", required_argument, 0, 'd'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        int option_index = 0;
        int option_value = 0;

        opt_val = getopt_long (argc, argv, "htwsf:c:l:d:",
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
            case 'w':
                use_ch_acq = 0;
                break;
            case 's':
                log_options = log_options | LOG_PERROR;
                break;
            case 'f':
                read_file = 1;
                input_file_name = strdup(optarg);
                break;
            case 'c':
                config_file_name = strdup(optarg);
                break;
            case 'd':
                make_daemon = 1;
                kotekan_root_dir = strdup(optarg);
                break;
            case 'l':
                option_value = atoi(optarg);
                switch (option_value) {
                    case 3:
                        log_level_debug = 1;
                    case 2:
                        log_level_info = 1;
                    case 1:
                        log_level_warn = 1;
                    default:
                        break;
                }
                break;
            default:
                printf("Invalid option, run with -h to see options");
                return -1;
                break;
        }
    }

    if (make_daemon) {
        // Do not use LOG_CONS or LOG_PERROR, since stderr does not exist in daemon mode.
        daemonize(kotekan_root_dir, LOG_PID | LOG_NDELAY);
    } else {
        // Setup syslog
        openlog ("kotekan", log_options, LOG_LOCAL1);
    }

    INFO("kotekan starting..."); /// TODO Include git commit id.

    // Load configuration file.
    error = load_config_from_file(&config, config_file_name);
    if (error) {
        ERROR("Error processing config file, exiting...");
        return -1;
    }

    INFO("Parsed config file %s:", config_file_name);
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

    INFO("Starting GPU threads...");

    for (int i = 0; i < config.gpu.num_gpus; ++i) {

        DEBUG("Creating buffers...");

        // TODO Figure out why this value currently needs to be constant.
        int links_per_gpu = 3; // num_links_per_gpu(&config, i);

        create_info_pool(&pool[i], 2 * links_per_gpu * config.processing.buffer_depth,
                                    config.processing.num_adjusted_local_freq,
                                    config.processing.num_adjusted_elements);

        create_buffer(&input_buffer[i], links_per_gpu * config.processing.buffer_depth,
                      config.processing.samples_per_data_set * config.processing.num_adjusted_elements *
                      config.processing.num_adjusted_local_freq * config.processing.num_data_sets,
                      num_links_per_gpu(&config, i), &pool[i]);
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

        INFO("GPU thread %d ready.", i);
    }

    //sleep(5);
    INFO("Starting up network threads...");

    // Create network threads
    pthread_t network_t[config.fpga_network.num_links];
    struct networkThreadArg network_args[config.fpga_network.num_links];

    for (int i = 0; i < config.fpga_network.num_links; ++i) {
        INFO("Link %d being assigned to buffer %d", i, config.fpga_network.link_map[i].gpu_id);
        network_args[i].buf = &input_buffer[config.fpga_network.link_map[i].gpu_id];
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

    INFO("kotekan shutdown successfully.");

    closelog();

    return 0;
}
